"""Student AI Examination Assistant (prefix /student-assistant). Only the student's own PUBLISHED results."""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth import require_role
from backend.database import User, get_db
from backend.llm_provider import get_llm_client
from backend.routes.student_routes import _get_student_for_user
from backend.student_exam_assistant import HELP, StudentExamAssistant, StudentRepo

logger = logging.getLogger(__name__)
router = APIRouter()
STUDENT = Depends(require_role(["student"]))


class Turn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Turn] = []
    response_language: Optional[str] = "auto"


@router.get("/capabilities")
def capabilities(user: User = STUDENT, db: Session = Depends(get_db)):
    h = get_llm_client().health()
    st = _get_student_for_user(user, db)
    sheets = StudentRepo(db, st).sheets(published_only=False)
    return {"role": "student", "help": HELP, "student": {"name": st.name, "registration_number": st.registration_number},
            "published": sum(1 for s in sheets if s["published"]), "in_progress": sum(1 for s in sheets if not s["published"]),
            "ai": {"provider": h.get("provider"), "model": h.get("model"),
                   "ready": bool(h.get("reachable") and h.get("model_available")), "error": h.get("error")}}


@router.get("/results")
def results(user: User = STUDENT, db: Session = Depends(get_db)):
    st = _get_student_for_user(user, db)
    out = []
    for s in StudentRepo(db, st).sheets(published_only=False):
        out.append({**{k: v for k, v in s.items() if k != "questions"},
                    "questions": [{k: q[k] for k in ("question_number", "awarded", "max_marks", "deductions",
                                                     "missing_concepts", "strengths", "teacher_comment", "feedback")}
                                  for q in s["questions"]],
                    "evaluated_url": f"/student/answer-sheet/{s['booklet_id']}/evaluated" if s["published"] else None,
                    "original_url": f"/student/answer-sheet/{s['booklet_id']}/original"})
    return out


@router.post("/chat")
def chat(data: ChatIn, user: User = STUDENT, db: Session = Depends(get_db)):
    st = _get_student_for_user(user, db)
    return StudentExamAssistant(StudentRepo(db, st), get_llm_client()).chat(
        data.message, [{"role": t.role, "content": t.content} for t in data.history][-8:], data.response_language or "auto")
