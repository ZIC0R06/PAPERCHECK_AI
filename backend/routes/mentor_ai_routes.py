"""
Mentor AI & Learning Operations Routes
=======================================
Provides endpoints for:
  • Study Book Upload, Text Extraction & RAG Indexing
  • Grounded RAG Question Answering with Citations
  • Comprehensive Chapter Explanations (Summary, Concepts, Formulas, Hinglish/English)
  • Dynamic Book-Grounded Quiz Generator
  • Quiz Submission, Score Calculation & Weak-Topic Tracking
  • Personalized Adaptive Study Roadmap
  • Learning Analytics & Dashboard Stats
  • Question Bank Management (CRUD & Paper Assembly)
"""

import json
import os
import uuid
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database import (
    get_db, User, Student, StudyDocument, DocumentChunk, QuizAttempt, WeakTopic,
    StudyRoadmap, CourseMaterial, Question, ExamPaper, ExamQuestion, GeneratedQuiz
)
from backend import learning_service as LS
from backend.auth import get_current_user, require_role
from backend.rag_engine import RAGEngine
from backend.llm_provider import get_llm_client

router = APIRouter()

# ── Pydantic Request Schemas ──────────────────────────────────────────────────

class RAGChatRequest(BaseModel):
    query: str
    subject: Optional[str] = None
    document_id: Optional[int] = None
    general_knowledge_fallback: bool = False

class ChapterExplainRequest(BaseModel):
    subject: str = "All Subjects"
    book_title: Optional[str] = None
    chapter: str = ""
    style: str = "Detailed" # Detailed, Simple English, Hinglish, Formulas & Exam Points
    document_id: Optional[int] = None

class QuizGenerateRequest(BaseModel):
    subject: str = "All Subjects"
    chapter: str = ""
    difficulty: str = "Medium" # Easy, Medium, Hard, Mixed
    question_type: str = "MCQ" # MCQ, True/False, Mixed
    num_questions: int = Field(4, ge=1, le=20)
    document_id: Optional[int] = None

class QuizSubmitRequest(BaseModel):
    quiz_id: int                                   # graded against answers stored on the server
    answers: List[dict]                            # [{"question_id": int, "selected_answer": str}]
    subject: Optional[str] = None                  # ignored (kept for older clients)
    topic: Optional[str] = None

class RoadmapGenerateRequest(BaseModel):
    target_exam: str = "University Semester Finals"
    target_date: str = "2026-12-15"
    daily_hours: float = 2.5
    subjects: List[str] = ["Design & Analysis of Algorithms", "Database Management Systems"]
    preferred_language: str = "English & Hinglish"

class QuestionBankItemCreate(BaseModel):
    question_text: str
    question_type: str = "open_ended" # open_ended, mcq, true_false, numerical
    teacher_answer: str = ""
    max_marks: float = 5.0
    correct_option: Optional[str] = None
    mcq_options: Optional[str] = None # JSON string or options
    exam_name: Optional[str] = "General Question Bank"

# ── 1. Study Books & Document Processing ───────────────────────────────────────

@router.post("/documents/upload", summary="Upload study textbook or notes for RAG")
async def upload_study_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    subject: str = Form("General"),
    chapter: str = Form("Chapter 1"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ext = Path(file.filename).suffix.lower()
    allowed = [".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg"]
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {', '.join(allowed)}")

    upload_dir = Path("uploads/study_materials")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    save_path = upload_dir / safe_name

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    rag = RAGEngine(db)
    doc_record = rag.process_and_index_file(
        file_path=str(save_path),
        title=title,
        subject=subject,
        chapter=chapter,
        user_id=current_user.id
    )

    return {
        "status": "success",
        "document_id": doc_record.id,
        "title": doc_record.title,
        "subject": doc_record.subject,
        "chapter": doc_record.chapter,
        "chunks_indexed": doc_record.chunk_count,
        "total_pages": doc_record.total_pages
    }

@router.get("/documents", summary="List student uploaded study books")
def list_study_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    docs = db.query(StudyDocument).filter(
        (StudyDocument.user_id == current_user.id) | (StudyDocument.user_id == None)
    ).order_by(StudyDocument.id.desc()).all()

    return [
        {
            "id": d.id,
            "title": d.title,
            "subject": d.subject,
            "chapter": d.chapter,
            "filename": d.filename,
            "file_type": d.file_type,
            "total_pages": d.total_pages,
            "chunks_count": d.chunk_count,
            "status": d.status,
            "created_at": d.created_at.strftime("%Y-%m-%d %H:%M") if d.created_at else None
        }
        for d in docs
    ]

@router.delete("/documents/{doc_id}", summary="Delete study document")
def delete_study_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = db.query(StudyDocument).filter(
        StudyDocument.id == doc_id,
        (StudyDocument.user_id == current_user.id) | (current_user.role == "admin")
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    
    # Remove file on disk if exists
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception:
        pass

    db.delete(doc)
    db.commit()
    return {"status": "success", "message": f"Document {doc_id} deleted."}

# ── 2. Real RAG Question Answering ─────────────────────────────────────────────

@router.post("/rag/ask", summary="Ask question grounded in uploaded books & syllabus")
def ask_book_rag(
    data: RAGChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    rag = RAGEngine(db)
    result = rag.answer_query(
        query=data.query,
        user_id=current_user.id,
        subject=data.subject,
        document_id=data.document_id
    )
    return result

# ── 3. Comprehensive Book & Chapter Explanations ───────────────────────────────

@router.post("/explain/chapter", summary="Explain a chapter from the student's own material (grounded)")
def explain_chapter(
    data: ChapterExplainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        passages = LS.gather_passages(db, current_user, data.subject, data.chapter, data.document_id)
    except LS.NoMaterialError as e:
        raise HTTPException(status_code=404, detail=str(e))
    text, sources, mode = LS.explain(passages, data.subject, data.chapter, data.style)
    return {
        "chapter": data.chapter,
        "subject": data.subject,
        "style": data.style,
        "explanation": text,
        "citations": sources,
        "mode": mode,
    }

# ── 4. Book-Based AI Quiz Center ───────────────────────────────────────────────

@router.post("/quiz/generate", summary="Generate a quiz from the student's study material")
def generate_book_quiz(
    data: QuizGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if data.document_id:
        owned = db.query(StudyDocument).filter(
            StudyDocument.id == data.document_id,
            (StudyDocument.user_id == current_user.id) | (StudyDocument.user_id == None)  # noqa: E711
        ).first()
        if not owned:
            raise HTTPException(status_code=404, detail="Document not found")
    try:
        passages = LS.gather_passages(db, current_user, data.subject, data.chapter, data.document_id)
    except LS.NoMaterialError as e:
        raise HTTPException(status_code=404, detail=str(e))

    questions, mode = LS.generate_mcqs(passages, data.num_questions, data.difficulty, data.chapter)
    if not questions:
        raise HTTPException(status_code=422, detail="This material doesn't have enough distinct content to build questions. "
                                                    "Try a larger chapter or another document.")
    for i, q in enumerate(questions, 1):
        q["id"] = i
        q["type"] = "MCQ"
    sources = LS._sources(passages)
    rec = GeneratedQuiz(user_id=current_user.id, subject=data.subject, chapter=data.chapter or "",
                        document_id=data.document_id, difficulty=data.difficulty, mode=mode,
                        questions_json=json.dumps(questions), sources_json=json.dumps(sources))
    db.add(rec)
    db.commit()
    db.refresh(rec)
    public = [{k: q[k] for k in ("id", "type", "question", "options", "topic")} for q in questions]
    return {
        "quiz_id": rec.id,
        "subject": data.subject,
        "chapter": data.chapter,
        "difficulty": data.difficulty,
        "mode": mode,
        "questions": public,           # correct answers are NOT sent to the browser
        "citations": sources,
        "note": None if mode == "llm" else
                "Offline mode (Ollama not reachable): fill-in-the-blank questions built from sentences in your material; "
                "difficulty cannot be adjusted.",
    }

@router.post("/quiz/submit", summary="Submit answers; graded on the server against the stored quiz")
def submit_quiz_attempt(
    data: QuizSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    quiz = db.query(GeneratedQuiz).filter(GeneratedQuiz.id == data.quiz_id,
                                          GeneratedQuiz.user_id == current_user.id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    if quiz.attempt_id:
        raise HTTPException(status_code=409, detail="This quiz was already submitted. Generate a new quiz to try again.")
    questions = json.loads(quiz.questions_json)
    chosen = {int(a.get("question_id")): str(a.get("selected_answer", "")) for a in data.answers
              if str(a.get("question_id", "")).isdigit()}
    total = len(questions)
    correct_count = 0
    graded_answers = []
    for q in questions:
        sel = chosen.get(q["id"], "")
        is_correct = sel.strip().lower() == q["correct_answer"].strip().lower()
        correct_count += int(is_correct)
        topic = (q.get("topic") or quiz.chapter or quiz.subject or "General")[:150]
        graded_answers.append({
            "question_id": q["id"], "question": q["question"], "selected_answer": sel or None,
            "correct_answer": q["correct_answer"], "is_correct": is_correct,
            "explanation": q.get("explanation", ""), "topic": topic,
        })

        wt = db.query(WeakTopic).filter_by(user_id=current_user.id, topic_name=topic).first()
        if not wt:
            wt = WeakTopic(user_id=current_user.id, subject=quiz.subject, topic_name=topic,
                           accuracy=100.0 if is_correct else 0.0, attempts_count=1,
                           status="Mastered" if is_correct else "Needs Practice",
                           remedy_notes=f"From quiz #{quiz.id} on {datetime.utcnow().strftime('%Y-%m-%d')}.")
            db.add(wt)
        else:
            wt.attempts_count += 1
            new_acc = ((wt.accuracy * (wt.attempts_count - 1)) + (100.0 if is_correct else 0.0)) / wt.attempts_count
            wt.accuracy = round(new_acc, 1)
            wt.status = "Mastered" if new_acc >= 80.0 else ("Improving" if new_acc >= 50.0 else "Needs Practice")
            wt.updated_at = datetime.utcnow()

    pct = round((correct_count / total) * 100.0, 1) if total else 0.0
    attempt = QuizAttempt(user_id=current_user.id, subject=quiz.subject, topic=quiz.chapter or quiz.subject,
                          total_score=float(correct_count), max_score=float(total), percentage=pct,
                          quiz_data=json.dumps(graded_answers))
    db.add(attempt)
    db.flush()
    quiz.attempt_id = attempt.id
    from backend.access import notify
    notify(db, current_user.id, "quiz", f"Quiz result: {correct_count}/{total} ({pct}%)",
           f"{quiz.subject} — {quiz.chapter or 'practice quiz'}")
    db.commit()
    db.refresh(attempt)
    return {"attempt_id": attempt.id, "quiz_id": quiz.id, "score": float(correct_count), "max_score": float(total),
            "percentage": pct, "answers": graded_answers}

@router.get("/quiz/attempts", summary="My previous quiz attempts")
def list_quiz_attempts(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    rows = db.query(QuizAttempt).filter(QuizAttempt.user_id == current_user.id).order_by(QuizAttempt.id.desc()).limit(50).all()
    return [{"id": a.id, "subject": a.subject, "topic": a.topic, "score": a.total_score, "max_score": a.max_score,
             "percentage": a.percentage, "created_at": a.created_at.isoformat() if a.created_at else None} for a in rows]

@router.get("/subjects", summary="Subjects, chapters and documents available to me")
def my_subjects(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return LS.list_subjects(db, current_user)

# ── 5. Weak Topics & Adaptive Study Roadmap ───────────────────────────────────

@router.get("/weak-topics", summary="Get evidence-grounded weak topics for student")
def get_student_weak_topics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    topics = db.query(WeakTopic).filter_by(user_id=current_user.id).order_by(WeakTopic.accuracy.asc()).all()
    return [
        {
            "id": t.id,
            "topic_name": t.topic_name,
            "subject": t.subject,
            "accuracy": t.accuracy,
            "attempts_count": t.attempts_count,
            "status": t.status,
            "remedy_notes": t.remedy_notes,
            "last_updated": t.updated_at.strftime("%Y-%m-%d") if t.updated_at else None
        }
        for t in topics
    ]

@router.post("/roadmap/generate", summary="Generate a study roadmap from my real results and quiz history")
def generate_study_roadmap(
    data: RoadmapGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from backend.routes.mentor_routes import _get_student_for_user
    from backend.mentor_service import AIMentorService
    published_weak: List[str] = []
    if current_user.role == "student":
        student = _get_student_for_user(current_user, db)
        ctx = AIMentorService(db)._gather_rag_context(student, None, None, None, "")
        published_weak = ctx.get("weak_topics", [])
    weak, strong = LS.weak_and_strong(db, current_user, published_weak)
    if not weak:
        raise HTTPException(status_code=422, detail=(
            "No weak topics found yet — there are no missed concepts in your published results and no low quiz "
            "scores. Take a quiz in the Quiz Center first; the roadmap is built from that evidence."))
    tasks = LS.build_roadmap(weak, data.daily_hours)

    roadmap = db.query(StudyRoadmap).filter_by(user_id=current_user.id).first()
    if not roadmap:
        roadmap = StudyRoadmap(user_id=current_user.id)
        db.add(roadmap)
    roadmap.target_exam = data.target_exam
    roadmap.target_date = data.target_date
    roadmap.daily_hours = data.daily_hours
    roadmap.plan_data = json.dumps({"tasks": tasks, "weak": weak, "strong": strong})
    roadmap.progress_pct = 0.0
    roadmap.updated_at = datetime.utcnow()
    from backend.access import notify
    notify(db, current_user.id, "roadmap", "New learning roadmap generated",
           f"{len(tasks)} tasks focusing on: " + ", ".join(w["topic"] for w in weak[:3]))
    db.commit()
    db.refresh(roadmap)
    return {"status": "success", "target_exam": roadmap.target_exam, "target_date": roadmap.target_date,
            "daily_hours": roadmap.daily_hours, "tasks": tasks, "weak_topics": weak, "strong_topics": strong,
            "progress_pct": 0.0}

def _plan(roadmap) -> dict:
    raw = json.loads(roadmap.plan_data) if roadmap.plan_data else []
    return raw if isinstance(raw, dict) else {"tasks": raw, "weak": [], "strong": []}

@router.get("/roadmap", summary="Get existing study roadmap")
def get_study_roadmap(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    roadmap = db.query(StudyRoadmap).filter_by(user_id=current_user.id).first()
    if not roadmap:
        return None
    plan = _plan(roadmap)
    return {
        "target_exam": roadmap.target_exam,
        "target_date": roadmap.target_date,
        "daily_hours": roadmap.daily_hours,
        "progress_pct": roadmap.progress_pct,
        "tasks": plan.get("tasks", []),
        "weak_topics": plan.get("weak", []),
        "strong_topics": plan.get("strong", []),
    }

class RoadmapTaskUpdate(BaseModel):
    completed: bool

@router.patch("/roadmap/task/{index}", summary="Mark a roadmap task done / not done")
def update_roadmap_task(index: int, data: RoadmapTaskUpdate, db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)):
    roadmap = db.query(StudyRoadmap).filter_by(user_id=current_user.id).first()
    if not roadmap:
        raise HTTPException(status_code=404, detail="No roadmap yet")
    plan = _plan(roadmap)
    tasks = plan.get("tasks", [])
    if not 0 <= index < len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")
    tasks[index]["completed"] = data.completed
    roadmap.progress_pct = round(100.0 * sum(1 for t in tasks if t.get("completed")) / len(tasks), 1)
    roadmap.plan_data = json.dumps(plan)
    roadmap.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "success", "progress_pct": roadmap.progress_pct, "task": tasks[index]}

# ── 6. Question Bank CRUD (Fixes Admin Question Management) ────────────────────

@router.get("/questions/bank", summary="List questions from the institutional question bank")
def list_question_bank(
    exam_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Question)
    if exam_name:
        q = q.filter(Question.exam_name == exam_name)
    items = q.order_by(Question.id.desc()).all()
    return [
        {
            "id": item.id,
            "exam_name": item.exam_name,
            "question_number": item.question_number,
            "question_text": item.question_text,
            "question_type": item.question_type,
            "teacher_answer": item.teacher_answer,
            "correct_option": item.correct_option,
            "max_marks": item.max_marks,
            "created_at": item.created_at.strftime("%Y-%m-%d") if item.created_at else None
        }
        for item in items
    ]

@router.post("/questions/bank", summary="Add question to the institutional question bank")
def create_question_bank_item(
    data: QuestionBankItemCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "faculty"]))
):
    count = db.query(Question).count()
    q = Question(
        exam_name=data.exam_name or "General Question Bank",
        question_number=count + 1,
        question_text=data.question_text,
        question_type=data.question_type,
        teacher_answer=data.teacher_answer,
        correct_option=data.correct_option,
        max_marks=data.max_marks
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    return {"status": "success", "question_id": q.id, "message": "Question added to Question Bank."}

@router.delete("/questions/bank/{q_id}", summary="Delete question from question bank")
def delete_question_bank_item(
    q_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "faculty"]))
):
    q = db.query(Question).filter(Question.id == q_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    db.delete(q)
    db.commit()
    return {"status": "success", "message": f"Question {q_id} deleted."}
