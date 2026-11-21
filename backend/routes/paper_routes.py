"""
Official question papers, model answers and marking schemes (prefix /exam-papers)
=================================================================================
Teachers manage papers only for subjects they are assigned to (admin: all). A paper is identified by
subject + examination name, so papers of different examinations never overwrite each other. Each question has:
number, text, marks, type (open_ended / short_answer / numerical / diagram / fill_blank / mcq / true_false),
model answer, correct option (MCQ / true-false) and a marking scheme (criteria with marks).

Answer sheets of the same subject + examination that were uploaded before the paper existed are linked
automatically whenever the paper is saved.
"""

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend import access as AC
from backend.auth import require_role
from backend.database import Course, ExamPaper, ExamQuestion, Rubric, StudentBooklet, User, get_db

router = APIRouter()
STAFF = Depends(require_role(["admin", "faculty"]))
QTYPES = {"open_ended", "short_answer", "numerical", "diagram", "fill_blank", "mcq", "true_false"}


class CriterionIn(BaseModel):
    criterion: str = Field(..., min_length=2, max_length=200)
    marks: float = Field(..., gt=0, le=100)


class QuestionIn(BaseModel):
    question_number: int = Field(..., ge=1, le=200)
    question_text: str = Field("", max_length=5000)
    marks: float = Field(..., gt=0, le=200)
    question_type: str = "open_ended"
    model_answer: str = Field("", max_length=20000)
    correct_option: str = Field("", max_length=5)
    rubric: List[CriterionIn] = []

    @field_validator("question_type")
    @classmethod
    def _qt(cls, v):
        v = (v or "open_ended").strip().lower()
        if v not in QTYPES:
            raise ValueError(f"question_type must be one of {sorted(QTYPES)}")
        return v


class PaperIn(BaseModel):
    course_code: str = Field(..., min_length=2, max_length=20)
    exam_name: str = Field(..., min_length=2, max_length=200)
    duration_hours: Optional[float] = Field(None, ge=0, le=24)
    questions: List[QuestionIn] = []


def _can(db, user, course_code: str) -> bool:
    return user.role == "admin" or any(c == course_code.upper() for c, _ in AC.teacher_pairs(db, user))


def _paper_or_404(db, user, pid: int) -> ExamPaper:
    p = db.query(ExamPaper).filter(ExamPaper.id == pid).first()
    if not p or not _can(db, user, p.course_code or ""):
        raise HTTPException(status_code=404, detail="Question paper not found")
    return p


def _slug(course: str, exam: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", f"{course}_{exam}").strip("_")
    return base[:180]


def _rubric(db, q: ExamQuestion):
    return [{"criterion": r.criterion, "marks": r.marks}
            for r in db.query(Rubric).filter(Rubric.exam_question_id == q.id).order_by(Rubric.id).all()]


def _warnings(db, q: ExamQuestion) -> List[str]:
    w = []
    rb = _rubric(db, q)
    if rb and abs(sum(r["marks"] for r in rb) - (q.marks or 0)) > 0.01:
        w.append(f"Q{q.question_number}: marking-scheme marks add up to {sum(r['marks'] for r in rb):g}, "
                 f"question carries {q.marks:g} (criteria will be scaled).")
    if q.question_type in ("mcq", "true_false") and not q.correct_option:
        w.append(f"Q{q.question_number}: set the correct option.")
    elif q.question_type not in ("mcq", "true_false") and not (q.teacher_answer or "").strip() and not rb:
        w.append(f"Q{q.question_number}: no model answer or marking scheme — it can't be AI-evaluated.")
    return w


def paper_dict(db, p: ExamPaper) -> dict:
    qs = (db.query(ExamQuestion).filter(ExamQuestion.paper_id_fk == p.id, ExamQuestion.is_or_option == False)  # noqa: E712
          .order_by(ExamQuestion.question_number).all())
    warnings = [w for q in qs for w in _warnings(db, q)]
    linked = db.query(StudentBooklet).filter(StudentBooklet.exam_paper_id == p.id).count()
    return {"id": p.id, "paper_id": p.paper_id, "course_code": p.course_code, "course_name": p.course_name,
            "exam_name": p.exam_name, "duration_hours": p.duration_hours,
            "total_marks": round(sum(q.marks or 0 for q in qs), 2), "num_questions": len(qs),
            "answer_sheets_linked": linked, "ready_for_evaluation": bool(qs) and not warnings, "warnings": warnings,
            "questions": [{"id": q.id, "question_number": q.question_number, "question_text": q.question_text or "",
                           "marks": q.marks, "question_type": q.question_type or "open_ended",
                           "model_answer": q.teacher_answer or "", "correct_option": q.correct_option or "",
                           "rubric": _rubric(db, q)} for q in qs]}


def _apply_question(db, p: ExamPaper, data: QuestionIn) -> ExamQuestion:
    q = db.query(ExamQuestion).filter(ExamQuestion.paper_id_fk == p.id, ExamQuestion.question_number == data.question_number,
                                      ExamQuestion.is_or_option == False).first()  # noqa: E712
    if not q:
        q = ExamQuestion(paper_id_fk=p.id, question_number=data.question_number, is_or_option=False, part_name="")
        db.add(q)
        db.flush()
    q.question_text, q.marks, q.question_type = data.question_text.strip(), data.marks, data.question_type
    q.teacher_answer = data.model_answer.strip() or None
    opt = data.correct_option.strip().upper()
    if data.question_type == "true_false" and opt[:1] in ("T", "F"):
        opt = "TRUE" if opt.startswith("T") else "FALSE"
    q.correct_option = opt or None
    db.query(Rubric).filter(Rubric.exam_question_id == q.id).delete()
    for c in data.rubric:
        db.add(Rubric(exam_question_id=q.id, criterion=c.criterion.strip(), marks=c.marks))
    return q


def _finish(db, p: ExamPaper, user, action: str, details: dict) -> dict:
    db.flush()
    p.total_marks = round(sum(q.marks or 0 for q in p.questions if not q.is_or_option), 2)
    # link answer sheets of this subject + exam that were uploaded before the paper existed
    for b in db.query(StudentBooklet).filter(StudentBooklet.course_code == p.course_code,
                                             StudentBooklet.exam_paper_id.is_(None)).all():
        if (b.exam_name or "").strip().lower() == (p.exam_name or "").strip().lower():
            b.exam_paper_id = p.id
            for a in b.answers:
                if not a.exam_question_id:
                    eq = db.query(ExamQuestion).filter_by(paper_id_fk=p.id, question_number=a.question_number,
                                                          is_or_option=False).first()
                    a.exam_question_id = eq.id if eq else None
    AC.audit(db, user, action, "exam_paper", p.id, details)
    db.commit()
    db.refresh(p)
    return paper_dict(db, p)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("", summary="Question papers I can manage (filter by subject)")
def list_papers(course_code: Optional[str] = None, db: Session = Depends(get_db), user: User = STAFF):
    q = db.query(ExamPaper)
    if course_code:
        q = q.filter(ExamPaper.course_code == course_code.strip().upper())
    out = []
    for p in q.order_by(ExamPaper.id.desc()).all():
        if _can(db, user, p.course_code or ""):
            d = paper_dict(db, p)
            d.pop("questions")
            out.append(d)
    return out


@router.get("/{pid}", summary="Paper with questions, model answers and marking scheme")
def get_paper(pid: int, db: Session = Depends(get_db), user: User = STAFF):
    return paper_dict(db, _paper_or_404(db, user, pid))


@router.post("", summary="Create (or update) the paper for a subject + examination")
def save_paper(data: PaperIn, db: Session = Depends(get_db), user: User = STAFF):
    code, exam = data.course_code.strip().upper(), data.exam_name.strip()
    if not _can(db, user, code):
        raise HTTPException(status_code=403, detail=f"You are not assigned to {code}.")
    nums = [q.question_number for q in data.questions]
    if len(nums) != len(set(nums)):
        raise HTTPException(status_code=422, detail="Question numbers must be unique.")
    p = db.query(ExamPaper).filter(ExamPaper.course_code == code, ExamPaper.exam_name == exam).first()
    created = p is None
    if created:
        slug = _slug(code, exam)
        if db.query(ExamPaper).filter(ExamPaper.paper_id == slug).first():
            slug = f"{slug}_{uuid.uuid4().hex[:4]}"
        course = db.query(Course).filter(Course.code == code).first()
        p = ExamPaper(paper_id=slug, course_code=code, exam_name=exam, course_name=course.name if course else code,
                      parsed_at=datetime.utcnow())
        db.add(p)
        db.flush()
    if data.duration_hours is not None:
        p.duration_hours = data.duration_hours
    for qd in data.questions:
        _apply_question(db, p, qd)
    return _finish(db, p, user, "EXAM_PAPER_CREATED" if created else "EXAM_PAPER_UPDATED",
                   {"course": code, "exam": exam, "questions": nums})


@router.put("/{pid}/questions/{qn}", summary="Add or edit one question (text, marks, model answer, marking scheme)")
def save_question(pid: int, qn: int, data: QuestionIn, db: Session = Depends(get_db), user: User = STAFF):
    p = _paper_or_404(db, user, pid)
    if data.question_number != qn:
        raise HTTPException(status_code=422, detail="Question number in the URL and body differ.")
    _apply_question(db, p, data)
    return _finish(db, p, user, "EXAM_QUESTION_SAVED", {"question": qn})


@router.delete("/{pid}/questions/{qn}", summary="Remove a question")
def delete_question(pid: int, qn: int, db: Session = Depends(get_db), user: User = STAFF):
    p = _paper_or_404(db, user, pid)
    q = db.query(ExamQuestion).filter(ExamQuestion.paper_id_fk == p.id, ExamQuestion.question_number == qn).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    from backend.database import StudentAnswerText
    if db.query(StudentAnswerText).filter(StudentAnswerText.exam_question_id == q.id).count():
        raise HTTPException(status_code=409, detail="Answer sheets were already read against this question; it can't be removed.")
    db.query(Rubric).filter(Rubric.exam_question_id == q.id).delete()
    db.delete(q)
    return _finish(db, p, user, "EXAM_QUESTION_DELETED", {"question": qn})


@router.delete("/{pid}", summary="Delete a paper (only when no answer sheet uses it)")
def delete_paper(pid: int, db: Session = Depends(get_db), user: User = STAFF):
    p = _paper_or_404(db, user, pid)
    if db.query(StudentBooklet).filter(StudentBooklet.exam_paper_id == p.id).count():
        raise HTTPException(status_code=409, detail="Answer sheets are linked to this paper; it can't be deleted.")
    for q in p.questions:
        db.query(Rubric).filter(Rubric.exam_question_id == q.id).delete()
    AC.audit(db, user, "EXAM_PAPER_DELETED", "exam_paper", p.id, {"course": p.course_code, "exam": p.exam_name})
    db.delete(p)
    db.commit()
    return {"status": "deleted"}


def _save_upload(data: bytes, prefix: str) -> str:
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="Upload a PDF.")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB).")
    from backend.api import UPLOAD_DIR
    path = Path(UPLOAD_DIR) / f"{prefix}_{uuid.uuid4().hex}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


@router.post("/import-pdf", summary="Create the paper from a question-paper PDF (questions + marks are extracted)")
async def import_pdf(file: UploadFile = File(...), course_code: str = Form(...), exam_name: str = Form(...),
                     db: Session = Depends(get_db), user: User = STAFF):
    code, exam = course_code.strip().upper(), exam_name.strip()
    if not _can(db, user, code):
        raise HTTPException(status_code=403, detail=f"You are not assigned to {code}.")
    path = _save_upload(await file.read(), "qpaper")
    from backend.api import _parse_paper_in_thread
    try:
        parsed = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None, _parse_paper_in_thread, path), 180)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read the question paper: {e}")
    qs = [QuestionIn(question_number=q.question_number, question_text=q.question_text or "", marks=q.marks or 1,
                     question_type=q.question_type if (q.question_type or "") in QTYPES else "open_ended")
          for q in parsed.questions if not q.is_or_option]
    if not qs:
        raise HTTPException(status_code=422, detail="No questions with marks were found in this PDF. Add them manually.")
    out = save_paper(PaperIn(course_code=code, exam_name=exam, duration_hours=parsed.duration_hours, questions=[]), db, user)
    p = db.query(ExamPaper).filter(ExamPaper.id == out["id"]).first()
    p.pdf_path, p.raw_text = path, (parsed.raw_text or "")[:10000]
    for q in qs:
        existing = db.query(ExamQuestion).filter_by(paper_id_fk=p.id, question_number=q.question_number, is_or_option=False).first()
        if existing:     # keep model answers / schemes already entered; refresh text + marks
            existing.question_text, existing.marks = q.question_text, q.marks
        else:
            _apply_question(db, p, q)
    return _finish(db, p, user, "EXAM_PAPER_IMPORTED", {"questions": [q.question_number for q in qs]})


@router.post("/{pid}/answer-key-pdf", summary="Attach model answers from an answer-key PDF")
async def answer_key_pdf(pid: int, file: UploadFile = File(...), db: Session = Depends(get_db), user: User = STAFF):
    p = _paper_or_404(db, user, pid)
    path = _save_upload(await file.read(), "anskey")
    from backend.api import _parse_answer_key_in_thread
    try:
        answers = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None, _parse_answer_key_in_thread, path), 180)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read the answer key: {e}")
    linked = []
    for qn, text in (answers or {}).items():
        q = db.query(ExamQuestion).filter_by(paper_id_fk=p.id, question_number=int(qn), is_or_option=False).first()
        if q and (text or "").strip():
            q.teacher_answer = text.strip()
            linked.append(int(qn))
    if not linked:
        raise HTTPException(status_code=422, detail="No model answers matching this paper's question numbers were found "
                                                    "(answers should start with 'Q1', '1.' …). Type them instead.")
    out = _finish(db, p, user, "EXAM_ANSWER_KEY_IMPORTED", {"questions": linked})
    out["answers_linked"] = linked
    return out
