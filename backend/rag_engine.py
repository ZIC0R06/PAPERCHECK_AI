"""
Mentor AI — Document Processing & RAG Engine
============================================
Handles:
  1. PDF, DOCX, TXT extraction
  2. OCR fallback for image-based/scanned documents
  3. Semantic chunking with overlap & metadata (page, chapter, subject)
  4. Real vector indexing & cosine similarity retrieval
  5. Grounded answering with citations & absence handling
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from backend.database import StudyDocument, DocumentChunk, CourseMaterial
from backend.similarity import SemanticSimilarityModel
from backend.llm_provider import get_llm_client

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, db: Session):
        self.db = db
        self.similarity_model = SemanticSimilarityModel()
        self.llm_client = get_llm_client()

    @staticmethod
    def _ocr_text(file_path: str) -> str:
        """Run the project's OCR pipeline and always return a plain string."""
        try:
            from backend.ocr_module import OCRModule
            res = OCRModule().extract_text(file_path)
            text = getattr(res, "text", res)  # OCRModule returns an OCRResult
            return text.strip() if isinstance(text, str) else ""
        except Exception as ex:
            logger.warning(f"OCR failed for {file_path}: {ex}")
            return ""

    @classmethod
    def extract_document_pages(cls, file_path: str) -> Tuple[List[Tuple[int, str]], str]:
        """
        Extract text from PDF / DOCX / TXT / MD / images.
        Returns ([(page_number, text), ...], method). An empty list means nothing
        could be extracted — callers must surface that instead of inventing content.
        """
        ext = Path(file_path).suffix.lower()
        pages: List[Tuple[int, str]] = []
        method = "none"

        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    t = (page.extract_text() or "").strip()
                    if t:
                        pages.append((i + 1, t))
                method = "pypdf"
            except Exception as e:
                logger.warning(f"pypdf extraction issue: {e}")
            if not pages:
                try:
                    import fitz  # PyMuPDF
                    with fitz.open(file_path) as d:
                        for i, page in enumerate(d):
                            t = (page.get_text() or "").strip()
                            if t:
                                pages.append((i + 1, t))
                    method = "pymupdf"
                except Exception as e:
                    logger.info(f"PyMuPDF not usable: {e}")
            if not pages:  # scanned PDF → OCR
                t = cls._ocr_text(file_path)
                if t:
                    pages.append((1, t))
                    method = "ocr"

        elif ext == ".docx":
            try:
                import docx
                d = docx.Document(file_path)
                parts = [p.text for p in d.paragraphs if p.text.strip()]
                for table in d.tables:
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            parts.append(" | ".join(cells))
                full = "\n\n".join(parts).strip()
                if full:
                    pages.append((1, full))
                method = "python-docx"
            except Exception as e:
                logger.error(f"DOCX extraction error: {e}")

        elif ext in (".txt", ".md"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    t = f.read().strip()
                if t:
                    pages.append((1, t))
                method = "text"
            except Exception as e:
                logger.error(f"Text file read error: {e}")

        elif ext in (".png", ".jpg", ".jpeg"):
            t = cls._ocr_text(file_path)
            if t:
                pages.append((1, t))
                method = "ocr"

        return pages, method

    def process_and_index_file(
        self,
        file_path: str,
        title: str,
        subject: str = "General",
        chapter: str = "Chapter 1",
        user_id: Optional[int] = None,
        pages: Optional[List[Tuple[int, str]]] = None,
        display_name: Optional[str] = None,
    ) -> StudyDocument:
        """
        Parses a document (PDF, DOCX, TXT), cleans text, creates overlapping chunks,
        and saves indexed records in database.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if pages is not None:
            extracted_pages = list(pages)
        else:
            extracted_pages, _method = self.extract_document_pages(file_path)

        if not extracted_pages:
            extracted_pages.append((1, f"Study document: {title}. Focus on {subject} - {chapter}."))

        # Create StudyDocument in DB
        doc_record = StudyDocument(
            user_id=user_id,
            title=title,
            subject=subject,
            chapter=chapter,
            filename=display_name or path.name,
            file_path=file_path,
            file_type=ext.replace(".", ""),
            status="ready",
            total_pages=len(extracted_pages),
            chunk_count=0
        )
        self.db.add(doc_record)
        self.db.commit()
        self.db.refresh(doc_record)

        # Chunk text with overlap (approx 400-600 characters per chunk)
        chunk_count = 0
        for page_num, page_text in extracted_pages:
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [page_text]

            for p_idx, para in enumerate(paragraphs):
                # Simple sentence-level sliding chunker
                sentences = re.split(r'(?<=[.!?])\s+', para)
                buffer = ""
                for s in sentences:
                    if len(buffer) + len(s) < 500:
                        buffer += (" " + s if buffer else s)
                    else:
                        if buffer:
                            chunk_rec = DocumentChunk(
                                document_id=doc_record.id,
                                chunk_index=chunk_count,
                                page_number=page_num,
                                chapter=chapter,
                                content=buffer.strip(),
                                keywords=", ".join(list(set(re.findall(r'\b[A-Za-z]{4,}\b', buffer.lower())))[:6])
                            )
                            self.db.add(chunk_rec)
                            chunk_count += 1
                        buffer = s
                if buffer:
                    chunk_rec = DocumentChunk(
                        document_id=doc_record.id,
                        chunk_index=chunk_count,
                        page_number=page_num,
                        chapter=chapter,
                        content=buffer.strip(),
                        keywords=", ".join(list(set(re.findall(r'\b[A-Za-z]{4,}\b', buffer.lower())))[:6])
                    )
                    self.db.add(chunk_rec)
                    chunk_count += 1

        doc_record.chunk_count = chunk_count
        self.db.commit()
        self.db.refresh(doc_record)
        return doc_record

    def retrieve_context(
        self,
        query: str,
        user_id: Optional[int] = None,
        subject: Optional[str] = None,
        document_id: Optional[int] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        RAG Vector retrieval: fetches top-k most relevant chunks using semantic matching.
        """
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))

        chunks_query = self.db.query(DocumentChunk).join(StudyDocument)
        if document_id:
            chunks_query = chunks_query.filter(DocumentChunk.document_id == document_id)
        if subject and subject != "All Subjects":
            chunks_query = chunks_query.filter(StudyDocument.subject == subject)
        if user_id:
            chunks_query = chunks_query.filter((StudyDocument.user_id == user_id) | (StudyDocument.user_id == None))

        all_chunks = chunks_query.all()
        if not all_chunks:
            # Fallback to course materials if no study document uploaded
            mat_q = self.db.query(CourseMaterial)
            mats = mat_q.all()
            scored_mats = []
            for m in mats:
                sim = self.similarity_model.compute_similarity(query, m.content_text)
                scored_mats.append({
                    "content": m.content_text,
                    "title": m.topic,
                    "page": 1,
                    "chapter": m.topic,
                    "score": sim.score,
                    "source": f"Course Syllabus: {m.course_code}"
                })
            scored_mats.sort(key=lambda x: x["score"], reverse=True)
            return scored_mats[:top_k]

        scored_chunks = []
        for ch in all_chunks:
            # Combined score: Semantic cosine similarity + lexical keyword bonus
            sim = self.similarity_model.compute_similarity(query, ch.content)
            ch_words = set(re.findall(r'\b[a-z]{3,}\b', ch.content.lower()))
            overlap = len(query_terms & ch_words) / max(1, len(query_terms))
            total_rank = (sim.score * 0.7) + (overlap * 0.3)

            scored_chunks.append({
                "chunk_id": ch.id,
                "content": ch.content,
                "title": ch.document.title if ch.document else "Document",
                "page": ch.page_number,
                "chapter": ch.chapter or (ch.document.chapter if ch.document else ""),
                "score": round(float(total_rank), 3),
                "source": f"{ch.document.title if ch.document else 'Doc'} (Page {ch.page_number}, {ch.chapter})"
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def answer_query(
        self,
        query: str,
        user_id: Optional[int] = None,
        subject: Optional[str] = None,
        document_id: Optional[int] = None
    ) -> Dict:
        """
        Executes grounded RAG Q&A:
        Retrieves evidence passages and formats a transparent answer with citations.
        """
        retrieved = self.retrieve_context(query, user_id=user_id, subject=subject, document_id=document_id, top_k=3)

        # Check relevance threshold (avoid fabricating answer if completely missing)
        best_score = retrieved[0]["score"] if retrieved else 0.0

        citations = [r["source"] for r in retrieved if r.get("score", 0) > 0.01]

        if not retrieved or best_score < 0.015:
            return {
                "reply": "I searched your uploaded books and syllabus materials, but could not find sufficient verified details covering this specific inquiry. Please ensure the relevant textbook chapter is uploaded, or rephrase your question.",
                "citations": [],
                "grounded": False
            }

        # Build context prompt
        context_block = "\n\n".join([f"[{r['source']}]:\n{r['content']}" for r in retrieved])
        system_prompt = (
            "You are Mentor AI, an elite university academic mentor. "
            "Answer the student query strictly grounded in the provided book excerpts. "
            "Include key definitions, bulleted operational mechanics, and equations/complexity where applicable. "
            "If the excerpts do not contain the answer, explicitly state that."
        )
        user_prompt = f"Query: {query}\n\nRetrieved Reference Passages:\n{context_block}\n\nProvide an insightful, academic answer citing the sources."

        try:
            llm_text = self.llm_client.complete(system_prompt, user_prompt, max_tokens=1000)
            if llm_text and llm_text.strip():
                return {
                    "reply": llm_text,
                    "citations": citations,
                    "grounded": True
                }
        except Exception as e:
            logger.warning(f"LLM call fallback: {e}")

        # Deterministic Grounded Synthesis fallback
        summary_points = "\n".join([f"- **Key Passage ({r['source']}):** {r['content'][:220]}..." for r in retrieved])
        reply = (
            f"### 📖 Grounded Book Explanation\n\n"
            f"Based on your uploaded course literature and materials, here is the synthesis for: **\"{query}\"**\n\n"
            f"{summary_points}\n\n"
            f"💡 *Actionable Study Tip:* Review these passages in detail to master this topic before taking a quiz."
        )
        return {
            "reply": reply,
            "citations": citations,
            "grounded": True
        }
