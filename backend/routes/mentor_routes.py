"""
AI Paper Checking — Personalized AI Mentor Chatbot Routes
==========================================================
Interactive AI mentor endpoints grounded in student's verified exam evaluations,
marking rubrics, and approved course materials (RAG).
"""

import json
import os
import queue
import threading
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.database import (
    get_db, SessionLocal, User, Student, ChatSession, ChatMessage, StudyDocument, StudentBooklet, CourseMaterial
)
from backend import mentor_files as MF
from backend.rag_engine import RAGEngine
from backend.auth import get_current_user
from backend.schemas import ChatRequestSchema, QuizRequestSchema
from backend.mentor_service import AIMentorService

router = APIRouter()


def _get_student_for_user(user: User, db: Session) -> Student:
    student = None
    if user.student_profile:
        student = user.student_profile
    else:
        student = db.query(Student).filter(
            (Student.user_id == user.id) |
            (Student.student_code == user.username) |
            (Student.registration_number == user.registration_number)
        ).first()

    if not student:
        # Create student profile on the fly if needed
        student = Student(
            user_id=user.id,
            student_code=user.registration_number or user.username,
            name=user.full_name,
            email=user.email,
            registration_number=user.registration_number or user.username,
            department=user.department or "Computer Science & Engineering",
            is_active=True
        )
        db.add(student)
        db.commit()
        db.refresh(student)
    return student


def _validate_chat(data: ChatRequestSchema):
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if len(data.message) > 8000:
        raise HTTPException(status_code=413, detail="Message is too long (max 8000 characters).")
    data.message = data.message.strip()


def _query_kwargs(data: ChatRequestSchema) -> dict:
    return dict(message=data.message, course_code=data.course_code, booklet_id=data.booklet_id,
                question_number=data.question_number, session_id=data.session_id,
                document_ids=data.document_ids, response_style=data.response_style, action=data.action,
                response_language=data.response_language, regenerate=data.regenerate)


@router.post("/chat", summary="Chat with AI academic mentor")
def chat_with_mentor(
    data: ChatRequestSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    _validate_chat(data)
    student = _get_student_for_user(current_user, db)
    service = AIMentorService(db)
    return service.handle_student_query(current_user=current_user, student=student, **_query_kwargs(data))


@router.post("/chat/stream", summary="Chat with streaming (Server-Sent Events)")
def chat_with_mentor_stream(
    data: ChatRequestSchema,
    current_user: User = Depends(get_current_user),
):
    """
    SSE events:  token {"t": "..."} · reset {} (a new LLM answer starts) · done {full result} · error {"detail"}.
    Tokens are real LLM deltas; replies built without an LLM arrive only in the `done` event.
    """
    _validate_chat(data)
    q: "queue.Queue" = queue.Queue()
    user_id = current_user.id

    def emit(tok):
        if isinstance(tok, dict) and tok.get("reset"):
            q.put(("reset", {}))
        elif tok:
            q.put(("token", {"t": tok}))

    def worker():
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            student = _get_student_for_user(user, db)
            svc = AIMentorService(db)
            svc.on_token = emit
            q.put(("done", svc.handle_student_query(current_user=user, student=student, **_query_kwargs(data))))
        except Exception as e:  # pragma: no cover - surfaced to the client
            q.put(("error", {"detail": f"Mentor error: {type(e).__name__}"}))
        finally:
            db.close()
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                item = q.get(timeout=240)
            except queue.Empty:
                yield "event: error\ndata: " + json.dumps({"detail": "The mentor took too long to respond."}) + "\n\n"
                return
            if item is None:
                return
            kind, payload = item
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/sessions", summary="List student chat sessions")
def list_chat_sessions(
    q: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    query = db.query(ChatSession).filter(ChatSession.student_id == student.id)
    if q and q.strip():
        like = f"%{q.strip()[:100]}%"
        hit_ids = db.query(ChatMessage.session_id).filter(ChatMessage.content.ilike(like))
        query = query.filter(or_(ChatSession.title.ilike(like), ChatSession.id.in_(hit_ids)))
    sessions = query.order_by(ChatSession.id.desc()).limit(200).all()

    return [
        {
            "id": s.id,
            "title": s.title,
            "course_code": s.course_code,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "message_count": len(s.messages)
        }
        for s in sessions
    ]


@router.get("/session/{session_id}/messages", summary="Get message history for chat session")
def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.student_id == student.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return [
        {
            "id": m.id,
            "sender": m.sender,
            "content": m.content,
            "sources_cited": m.sources_cited,
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in session.messages
    ]


@router.post("/generate-quiz", summary="Generate personalized practice quiz on weak topics")
def generate_quiz(
    data: QuizRequestSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    service = AIMentorService(db)
    topics = [t for t in (data.weak_topics or []) if t and t.strip()]
    if not topics:  # use real evidence only — never placeholder topics
        ctx = service._gather_rag_context(student, data.course_code, None, None, "")
        topics = [t for t, _ in service._collect_weak(current_user, ctx)]
    if not topics:
        raise HTTPException(status_code=422, detail="No weak topics found yet in your published results or quizzes.")
    quiz = service.generate_practice_quiz(
        student=student,
        course_code=data.course_code,
        weak_topics=topics
    )
    return quiz


# ══════════════════════════════════════════════════════════════════════════════
# AI Mentor dashboard: sessions, attachments, context
# ══════════════════════════════════════════════════════════════════════════════

@router.delete("/session/{session_id}", summary="Delete one of my chat sessions")
def delete_chat_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.student_id == student.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    db.delete(session)
    db.commit()
    return {"status": "deleted", "session_id": session_id}


def _file_payload(d: StudyDocument) -> dict:
    try:
        size = os.path.getsize(d.file_path)
    except OSError:
        size = 0
    chars = sum(len(c.content or "") for c in d.chunks)
    return {
        "id": d.id,
        "name": d.filename,
        "file_type": d.file_type,
        "size_bytes": size,
        "size": MF.human_size(size),
        "pages": d.total_pages,
        "chunks": d.chunk_count,
        "chars": chars,
        "status": d.status,
        "folder": d.folder or "",
        "uploaded_at": d.created_at.isoformat() + "Z" if d.created_at else None,
    }


@router.post("/files", summary="Attach a file to the AI Mentor chat (validated, text-extracted, indexed)")
async def upload_mentor_file(
    file: UploadFile = File(...),
    folder: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    content = await file.read(MF.MAX_BYTES + 1)
    display = MF.safe_display_name(file.filename)
    try:
        ext = MF.validate_upload(display, content)
    except MF.FileValidationError as e:
        code = 413 if "limit" in str(e) else 415 if "Unsupported" in str(e) else 422
        raise HTTPException(status_code=code, detail=str(e))

    path = MF.store_upload(current_user.id, ext, content)
    pages, method = await run_in_threadpool(RAGEngine.extract_document_pages, str(path))
    if not pages or sum(len(t) for _, t in pages) < 20:
        try:
            os.remove(path)
        except OSError:
            pass
        hint = (" It looks like a scanned/image file and OCR (Tesseract/PaddleOCR) is not available on the server."
                if ext in (".png", ".jpg", ".jpeg", ".pdf") else "")
        raise HTTPException(status_code=422, detail=f"No readable text found in '{display}'.{hint}")

    rag = RAGEngine(db)
    doc = rag.process_and_index_file(
        file_path=str(path), title=display, subject=MF.CHAT_UPLOAD_SUBJECT,
        chapter=os.path.splitext(display)[0][:150], user_id=current_user.id,
        pages=pages, display_name=display,
    )
    import re as _re
    folder = _re.sub(r"[<>/\\{}`$;]", "", folder or "").strip()[:100]
    if folder:
        doc.folder = folder
        db.commit()
    payload = _file_payload(doc)
    payload["extraction"] = method
    payload["preview"] = pages[0][1][:280]
    return payload


@router.get("/files", summary="List my AI Mentor files (most recent first)")
def list_mentor_files(
    limit: int = 20,
    folder: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    qq = db.query(StudyDocument).filter(StudyDocument.user_id == current_user.id)
    if folder is not None:
        qq = qq.filter(StudyDocument.folder == folder)
    docs = qq.order_by(StudyDocument.id.desc()).limit(max(1, min(limit, 200))).all()
    return [_file_payload(d) for d in docs]


@router.delete("/files/{doc_id}", summary="Delete one of my AI Mentor files")
def delete_mentor_file(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(StudyDocument).filter(
        StudyDocument.id == doc_id, StudyDocument.user_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except OSError:
        pass
    db.delete(doc)
    db.commit()
    return {"status": "deleted", "id": doc_id}


@router.get("/context", summary="AI Mentor dashboard context for the signed-in student")
def mentor_context(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    service = AIMentorService(db)
    published = db.query(StudentBooklet).filter(
        StudentBooklet.student_id == student.id, StudentBooklet.is_published == True  # noqa: E712
    ).all()
    courses = sorted({b.course_code for b in published if b.course_code})
    notes = db.query(CourseMaterial).filter(CourseMaterial.course_code.in_(courses)).count() if courses else 0
    files = db.query(StudyDocument).filter(StudyDocument.user_id == current_user.id).count()
    return {
        "name": student.name or current_user.full_name or current_user.username,
        "department": student.department,
        "programme": getattr(student, "programme", None),
        "semester": getattr(student, "semester", None),
        "llm_online": service.llm_available(),
        "provider": getattr(service.llm_client, "active_provider", None) if service.llm_available() else "offline",
        "rag_sources": {
            "course_notes": notes,
            "published_results": len(published),
            "my_files": files,
            "courses": courses,
        },
        "max_file_mb": MF.MAX_MB,
        "allowed_types": sorted(MF.ALLOWED_EXT),
    }



class SessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


@router.patch("/session/{session_id}", summary="Rename one of my chat sessions")
def rename_chat_session(session_id: int, data: SessionRename, current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    student = _get_student_for_user(current_user, db)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.student_id == student.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session.title = data.title.strip()
    db.commit()
    return {"id": session.id, "title": session.title}


class FileFolderUpdate(BaseModel):
    folder: str = Field("", max_length=100, pattern=r"^[^<>/\\{}`$;]*$")


@router.patch("/files/{doc_id}", summary="Move one of my files into a folder")
def move_mentor_file(doc_id: int, data: FileFolderUpdate, current_user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    doc = db.query(StudyDocument).filter(StudyDocument.id == doc_id, StudyDocument.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    doc.folder = data.folder.strip()
    db.commit()
    return _file_payload(doc)


@router.get("/folders", summary="My study-material folders with file counts")
def list_folders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    counts = {}
    for (f,) in db.query(StudyDocument.folder).filter(StudyDocument.user_id == current_user.id).all():
        counts[f or ""] = counts.get(f or "", 0) + 1
    return [{"folder": k, "files": v} for k, v in sorted(counts.items(), key=lambda kv: (kv[0] == "", kv[0].lower()))]


@router.get("/files/{doc_id}/preview", summary="Text preview of one of my files")
def preview_mentor_file(doc_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = db.query(StudyDocument).filter(StudyDocument.id == doc_id, StudyDocument.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    pages = {}
    for c in sorted(doc.chunks, key=lambda c: c.chunk_index):
        pages.setdefault(c.page_number or 1, []).append(c.content)
    preview = [{"page": p, "text": " ".join(t)[:1200]} for p, t in sorted(pages.items())[:3]]
    return {**_file_payload(doc), "preview_pages": preview}
