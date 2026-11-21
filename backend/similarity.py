"""
AI Paper Checking — Semantic Similarity Module
===============================================
Computes semantic similarity between student answers and model answers.
Uses Sentence-BERT (all-MiniLM-L6-v2) as primary neural embedding model,
with a robust scikit-learn TF-IDF + n-gram cosine similarity fallback
when SentenceTransformer or PyTorch is not loaded.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    score: float          # 0.0 – 1.0 cosine similarity
    student_embedding: list
    teacher_embedding: list
    method: str = "sbert"


class SemanticSimilarityModel:
    """
    Wraps SentenceTransformer to produce cosine-similarity scores
    between student and teacher answers, with zero-downtime TF-IDF fallback.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        import os
        self._model_name = model_name or os.getenv("SBERT_MODEL", self.DEFAULT_MODEL)
        self._model = None
        self._sbert_available = True

    def _load(self):
        if self._model is None and self._sbert_available:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                logger.info("SentenceTransformer loaded successfully.")
            except Exception as e:
                logger.warning(f"SentenceTransformer not available ({e}). Using scikit-learn TF-IDF fallback.")
                self._sbert_available = False

    def _tfidf_fallback(self, student_answer: str, teacher_answer: str) -> SimilarityResult:
        """
        Fast, zero-download TF-IDF + bi-gram cosine similarity fallback.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vect = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
            tfidf_matrix = vect.fit_transform([student_answer, teacher_answer])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            sim = max(0.0, min(1.0, sim))

            # Bonus for word overlap on domain keywords
            s_words = set(student_answer.lower().split())
            t_words = set(teacher_answer.lower().split())
            jaccard = len(s_words & t_words) / max(1, len(s_words | t_words))
            blended = 0.75 * sim + 0.25 * jaccard

            return SimilarityResult(
                score=round(float(blended), 4),
                student_embedding=[],
                teacher_embedding=[],
                method="tfidf_fallback"
            )
        except Exception as ex:
            logger.error(f"TF-IDF fallback failed: {ex}")
            # Ultra-safe lexical fallback
            s_words = set(student_answer.lower().split())
            t_words = set(teacher_answer.lower().split())
            overlap = len(s_words & t_words) / max(1, len(t_words))
            return SimilarityResult(score=round(min(1.0, overlap), 4), student_embedding=[], teacher_embedding=[], method="lexical")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def compute_similarity(
        self,
        student_answer: str,
        teacher_answer: str
    ) -> SimilarityResult:
        """
        Returns cosine similarity between student and teacher answers.
        Score range: 0.0 (unrelated) to 1.0 (identical meaning).
        """
        if not student_answer or not student_answer.strip():
            return SimilarityResult(score=0.0, student_embedding=[], teacher_embedding=[], method="empty")
        if not teacher_answer or not teacher_answer.strip():
            return SimilarityResult(score=0.0, student_embedding=[], teacher_embedding=[], method="empty")

        self._load()

        if self._sbert_available and self._model is not None:
            try:
                from sentence_transformers import util
                embeddings = self._model.encode(
                    [student_answer, teacher_answer],
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
                cosine_score = max(0.0, min(1.0, cosine_score))

                return SimilarityResult(
                    score=round(cosine_score, 4),
                    student_embedding=embeddings[0].cpu().tolist(),
                    teacher_embedding=embeddings[1].cpu().tolist(),
                    method="sbert"
                )
            except Exception as e:
                logger.warning(f"SBERT inference failed: {e}. Falling back to TF-IDF.")
                return self._tfidf_fallback(student_answer, teacher_answer)
        else:
            return self._tfidf_fallback(student_answer, teacher_answer)

    def compute_sentence_level(
        self,
        student_answer: str,
        teacher_answer: str
    ) -> dict:
        """
        Compute similarity at sentence level for granular feedback.
        """
        import re
        student_sents = [s.strip() for s in re.split(r'[.!?]+', student_answer) if s.strip()]
        teacher_sents = [s.strip() for s in re.split(r'[.!?]+', teacher_answer) if s.strip()]

        if not student_sents or not teacher_sents:
            return {"sentence_scores": [], "average": 0.0, "matched_pairs": []}

        scores = []
        matched_pairs = []
        for s_idx, s_text in enumerate(student_sents):
            best_sim = 0.0
            best_t = ""
            for t_text in teacher_sents:
                sim = self.compute_similarity(s_text, t_text).score
                if sim > best_sim:
                    best_sim = sim
                    best_t = t_text
            scores.append(best_sim)
            matched_pairs.append({
                "student_sentence": s_text,
                "best_matching_reference": best_t,
                "similarity": best_sim
            })

        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            "sentence_scores": scores,
            "average": round(avg_score, 4),
            "matched_pairs": matched_pairs
        }
