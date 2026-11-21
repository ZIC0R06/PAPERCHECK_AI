"""
Answer sheet → question-wise answers → justified marks (shared by upload, scanner, review and publishing)
======================================================================================================
* create_booklet()           — registers an uploaded / scanned sheet (no placeholder answers)
* ensure_answers_extracted() — reads the handwriting (backend.handwriting_reader: vision AI, else local OCR),
                               creates one answer row per question of the official paper (unanswered → empty)
                               with the answer's location on the pages
* grade_booklet() / grade_row() — marks each answer against the official question, model answer and marking
                               scheme (backend.answer_grader) and records breakdown, deductions, corrections, flags
* question_evaluations()     — the marks as finally awarded (teacher override > AI), used by APIs and PDFs
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from backend.database import (Course, CourseMaterial, ExamPaper, ExamQuestion, Result, Rubric, Student,
                              StudentAnswerText, StudentBooklet, Submission, User)
from backend import access as AC

logger = logging.getLogger(__name__)

PLACEHOLDER_PREFIXES = ("Uploaded answer sheet (", "Student submitted answer script (",
                        "Student submission registered for evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Paper lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_exam_paper(db: Session, course_code: str, exam_name: str = "") -> Optional[ExamPaper]:
    """The official paper for this subject + examination. Falls back to the subject's only paper (never guesses
    between several papers of different examinations)."""
    q = db.query(ExamPaper).filter(ExamPaper.course_code == (course_code or "").upper())
    if exam_name:
        exact = q.filter(ExamPaper.exam_name == exam_name).order_by(ExamPaper.id.desc()).first()
        if exact:
            return exact
        ci = [p for p in q.all() if (p.exam_name or "").strip().lower() == exam_name.strip().lower()]
        if ci:
            return ci[-1]
    papers = q.order_by(ExamPaper.id.desc()).all()
    if not exam_name:
        return papers[0] if len(papers) == 1 else None
    # a paper saved without an examination name (old PDF imports) is used only if it is the subject's only paper
    unnamed = [p for p in papers if not (p.exam_name or "").strip()]
    if len(papers) == 1 and unnamed:
        return unnamed[0]
    return None


def paper_questions(db: Session, paper_id: Optional[int]) -> List[ExamQuestion]:
    if not paper_id:
        return []
    return (db.query(ExamQuestion).filter(ExamQuestion.paper_id_fk == paper_id, ExamQuestion.is_or_option == False)  # noqa: E712
            .order_by(ExamQuestion.question_number).all())


def create_booklet(db: Session, student: Student, course_code: str, exam_name: str, pdf_path: str,
                   user: Optional[User], total_pages: int = 0) -> StudentBooklet:
    course = db.query(Course).filter(Course.code == course_code.upper()).first()
    paper = find_exam_paper(db, course_code, exam_name)
    b = StudentBooklet(
        student_id=student.id, exam_paper_id=paper.id if paper else None, pdf_path=pdf_path,
        total_pages=total_pages, course_code=course_code.upper(), course_name=course.name if course else course_code,
        exam_name=exam_name, roll_number=student.student_code, register_number=student.registration_number,
        department=student.department, programme=student.programme, semester=student.semester,
        approval_status="pending", is_published=False,
    )
    db.add(b)
    db.flush()
    AC.audit(db, user, "ANSWER_SHEET_UPLOADED", "booklet", b.id,
             {"student": student.registration_number, "course": course_code, "exam": exam_name, "pdf": pdf_path})
    db.commit()
    db.refresh(b)
    return b


def _is_placeholder(booklet: StudentBooklet) -> bool:
    return bool(booklet.answers) and all(
        (a.answer_text or "").startswith(PLACEHOLDER_PREFIXES) for a in booklet.answers)


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

def ensure_answers_extracted(db: Session, booklet: StudentBooklet, force: bool = False) -> Dict:
    """Read the sheet if it has no real answer text yet (or force=True). Returns a short report."""
    if booklet.answers and not _is_placeholder(booklet) and not force:
        return {"ran": False, "answers": len(booklet.answers), "engine": booklet.ocr_engine or ""}
    for a in list(booklet.answers):           # drop placeholders / previous reading (and their marks)
        if a.submission:
            db.delete(a.submission)
        db.delete(a)
    db.flush()
    if not booklet.pdf_path or not os.path.exists(booklet.pdf_path):
        db.commit()
        return {"ran": False, "reason": "answer-sheet file is missing on the server"}
    if not booklet.exam_paper_id:
        paper = find_exam_paper(db, booklet.course_code, booklet.exam_name)
        booklet.exam_paper_id = paper.id if paper else None
    qrows = paper_questions(db, booklet.exam_paper_id)
    qdicts = [{"question_number": q.question_number, "question_text": q.question_text, "marks": q.marks}
              for q in qrows]

    from backend.handwriting_reader import read_answer_sheet
    try:
        reading = read_answer_sheet(booklet.pdf_path, qdicts)
    except Exception as e:
        logger.exception("Answer-sheet reading failed")
        db.commit()
        return {"ran": True, "reason": f"reading failed: {type(e).__name__}: {e}"}

    found = {a.question_number: a for a in reading.answers}
    # Single-question paper and no marker written: everything on the sheet answers that question.
    if len(qrows) == 1 and not found and reading.unassigned_text:
        from backend.handwriting_reader import ReadAnswer
        found[qrows[0].question_number] = ReadAnswer(qrows[0].question_number, reading.unassigned_text,
                                                     [{"page": 1, "y0": 0.0, "y1": 1.0}], 0.5, reading.engine)
        reading.unassigned_text = ""

    booklet.total_pages = reading.pages
    booklet.raw_full_ocr = reading.full_text[:50000]
    booklet.ocr_engine = reading.engine[:80]
    booklet.ocr_report = json.dumps(reading.report())
    booklet.parsed_at = datetime.utcnow()

    def add(qn, eq, ra):
        db.add(StudentAnswerText(
            booklet_id=booklet.id, exam_question_id=eq.id if eq else None, question_number=qn,
            part_name=(eq.part_name if eq else "") or "", is_or_option=False,
            answer_text=(ra.text if ra else ""), raw_ocr_text=(ra.text if ra else ""),
            page_refs=json.dumps(ra.page_refs if ra else []), ocr_confidence=(ra.confidence if ra else None),
            ocr_engine=((ra.engine if ra else reading.engine) or "")[:80], evaluated=False))

    for eq in qrows:                       # every official question gets a row (unanswered → empty → 0 marks)
        add(eq.question_number, eq, found.pop(eq.question_number, None))
    for qn, ra in sorted(found.items()):   # answers to numbers that are not on the paper (or no paper linked)
        add(qn, None, ra)
    db.commit()
    db.refresh(booklet)
    answered = sum(1 for a in booklet.answers if (a.answer_text or "").strip())
    report = {"ran": True, "pages": reading.pages, "engine": reading.engine, "answers": len(booklet.answers),
              "answered": answered, "linked_to_paper": bool(booklet.exam_paper_id), "warnings": reading.warnings}
    if reading.unassigned_text:
        report["unassigned_text"] = reading.unassigned_text[:500]
    if not answered:
        report["reason"] = ("text was read but no question numbers were found (answers should start with 'Q1', "
                            "'Ans 2', '3)' …)" if reading.unassigned_text else "no handwriting could be read")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Grading
# ─────────────────────────────────────────────────────────────────────────────

def question_payload(db: Session, booklet: StudentBooklet, ans: StudentAnswerText) -> Dict:
    eq = ans.exam_question
    rubric = []
    if eq:
        rubric = [{"criterion": r.criterion, "marks": r.marks} for r in
                  db.query(Rubric).filter(Rubric.exam_question_id == eq.id).order_by(Rubric.id).all()]
    model = (eq.teacher_answer or "") if eq else ""
    note = ""
    if not model and not rubric and booklet.course_code and not (eq and eq.correct_option):
        mat = db.query(CourseMaterial).filter_by(course_code=booklet.course_code).first()
        if mat and (mat.content_text or "").strip():
            model = mat.content_text[:6000]
            note = "Evaluated against course notes because no model answer was uploaded — teacher must verify."
    marks = eq.marks if eq and eq.marks else None
    if marks is None:
        note = (note + " Maximum marks not found in the question paper; 10 assumed.").strip()
    if not eq and booklet.exam_paper_id:
        note = (note + f" Q{ans.question_number} is not on the official question paper.").strip()
    return {"question_number": ans.question_number, "question_text": (eq.question_text if eq else "") or "",
            "question_type": (eq.question_type if eq else None) or "open_ended", "marks": marks or 10.0,
            "model_answer": model, "correct_option": (eq.correct_option if eq else None) or "",
            "rubric": rubric, "reference_note": note}


def has_reference(q: Dict) -> bool:
    return bool((q.get("model_answer") or "").strip() or q.get("rubric") or q.get("correct_option"))


def grade_row(db: Session, booklet: StudentBooklet, ans: StudentAnswerText, llm_client=None) -> Optional[Result]:
    """Mark one answer; returns the Result or None when the question has no model answer / scheme."""
    from backend.answer_grader import grade_answer
    q = question_payload(db, booklet, ans)
    if not has_reference(q):
        return None
    g = grade_answer(q, ans.answer_text or "", ocr_confidence=ans.ocr_confidence, ocr_engine=ans.ocr_engine or "",
                     llm_client=llm_client)
    sub = ans.submission
    if not sub:
        sub = Submission(student_id=booklet.student_id, image_path=booklet.pdf_path, exam_paper_id=booklet.exam_paper_id,
                         exam_question_id=ans.exam_question_id)
        db.add(sub)
        db.flush()
        ans.submission_id = sub.id
        ans.submission = sub      # relationship in sync before commit (totals read it)
    sub.extracted_text = ans.answer_text
    sub.ocr_confidence = ans.ocr_confidence
    res = sub.result
    if not res:
        res = Result(submission_id=sub.id)
        sub.result = res          # keep the relationship in sync so totals see it before a commit
        db.add(res)
    res.final_score = g.score
    res.llm_score = g.score if g.method == "ai" else 0.0
    res.max_marks = g.max_marks
    res.question_type = q["question_type"][:20]
    res.confidence = g.confidence
    res.feedback = g.feedback
    res.strengths = json.dumps(g.strengths)
    res.missing_concepts = json.dumps(g.missing_concepts)
    res.evaluation_json = g.evaluation_json()
    res.examiner_approved = False
    res.examiner_score = None
    res.flagged_for_review = bool(g.flags)
    res.flag_reason = " | ".join(dict.fromkeys(g.flags))[:255]
    ans.evaluated = True
    return res


def final_score(r: Optional[Result]) -> float:
    if not r:
        return 0.0
    return float(r.examiner_score if r.examiner_approved and r.examiner_score is not None else (r.final_score or 0.0))


def recompute_totals(booklet: StudentBooklet) -> None:
    ai = sum(float(a.submission.result.final_score or 0) for a in booklet.answers if a.submission and a.submission.result)
    fin = sum(final_score(a.submission.result) for a in booklet.answers if a.submission and a.submission.result)
    booklet.total_ai_score = round(ai, 2)
    booklet.total_final_score = round(fin, 2)


def grade_booklet(db: Session, booklet: StudentBooklet) -> Dict:
    from backend.llm_provider import get_llm_client
    client = get_llm_client()
    skipped, flagged, total_max = [], 0, 0.0
    for ans in booklet.answers:
        res = grade_row(db, booklet, ans, client)
        if res is None:
            skipped.append(ans.question_number)
            continue
        flagged += int(bool(res.flagged_for_review))
        total_max += float(res.max_marks or 0)
    db.flush()
    recompute_totals(booklet)
    return {"skipped": skipped, "flagged": flagged, "max_marks": round(total_max, 2),
            "grader": client.active_provider if client.has_real_provider else "offline (marking-point matching)"}


# ─────────────────────────────────────────────────────────────────────────────
# Read-out (APIs, PDFs)
# ─────────────────────────────────────────────────────────────────────────────

def _jl(v, default):
    if not v:
        return default
    try:
        x = json.loads(v)
        return x if isinstance(x, type(default)) else default
    except Exception:
        return [v] if isinstance(default, list) else default


def question_evaluations(booklet: StudentBooklet, include_unevaluated: bool = True) -> List[Dict]:
    out = []
    for ans in sorted(booklet.answers, key=lambda a: (a.question_number or 0, a.id)):
        r = ans.submission.result if ans.submission else None
        if not r and not include_unevaluated:
            continue
        eq = ans.exam_question
        ev = _jl(r.evaluation_json, {}) if r else {}
        awarded = final_score(r) if r else None
        teacher_changed = bool(r and r.examiner_approved and r.examiner_score is not None
                               and abs((r.examiner_score or 0) - (r.final_score or 0)) > 1e-9)
        deductions = list(ev.get("deductions", []))
        if teacher_changed:
            diff = round(float(r.examiner_score) - float(r.final_score or 0), 2)
            if diff < 0:
                deductions.append({"marks": -diff, "criterion": "Teacher's adjustment",
                                   "reason": (r.examiner_comments or "Marks reduced by the teacher.")})
            else:   # teacher restored marks: AI deductions are reduced accordingly
                deductions.append({"marks": -diff, "criterion": "Teacher's adjustment",
                                   "reason": f"+{diff:g} restored by the teacher: " + (r.examiner_comments or "")})
        out.append({
            "answer_id": ans.id, "result_id": r.id if r else None, "question_number": ans.question_number,
            "question_text": (eq.question_text if eq else "") or "", "question_type": (r.question_type if r else
                                                                                     (eq.question_type if eq else "")),
            "student_answer": ans.answer_text or "", "page_refs": _jl(ans.page_refs, []),
            "ocr_confidence": ans.ocr_confidence, "ocr_engine": ans.ocr_engine or "",
            "max_marks": float(r.max_marks if r and r.max_marks else (eq.marks if eq and eq.marks else 0) or 0),
            "ai_score": float(r.final_score) if r and r.final_score is not None else None,
            "awarded": awarded, "teacher_verified": bool(r and r.examiner_approved),
            "teacher_changed": teacher_changed, "teacher_comment": (r.examiner_comments if r else "") or "",
            "feedback": (r.feedback if r else "") or "", "strengths": _jl(r.strengths, []) if r else [],
            "missing_concepts": _jl(r.missing_concepts, []) if r else [],
            "rubric_breakdown": ev.get("rubric_breakdown", []), "deductions": deductions,
            "corrections": ev.get("corrections", []), "method": ev.get("method", ""), "provider": ev.get("provider", ""),
            "confidence": r.confidence if r else None, "flagged": bool(r and r.flagged_for_review),
            "flag_reason": (r.flag_reason if r else "") or "",
        })
    return out
