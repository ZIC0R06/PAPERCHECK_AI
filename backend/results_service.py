"""
Results service — one source of truth for marks, grades and analytics
=====================================================================
Marks always come from the database:
  • subjective  = StudentBooklet answers → Result (AI score, teacher score kept separately)
  • objective   = OmrResponse (verified)
  • consolidated= FinalResult (OMR + subjective), approved by a teacher, then published

Grading (10-point scale, configurable pass mark PASS_PERCENT, default 40):
  O ≥90 (10) · A+ ≥80 (9) · A ≥70 (8) · B+ ≥60 (7) · B ≥50 (6) · C ≥45 (5) · P ≥pass (4) · F (0)
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.database import (
    Batch, Course, FinalResult, OmrKey, OmrResponse, Student, StudentBooklet, TeacherAssignment, TeacherProfile, User,
)

PASS_PERCENT = float(os.getenv("PASS_PERCENT", "40"))
GRADES = [(90, "O", 10), (80, "A+", 9), (70, "A", 8), (60, "B+", 7), (50, "B", 6), (45, "C", 5)]


def grade_for(pct: Optional[float]) -> Tuple[str, float]:
    if pct is None:
        return "", 0.0
    for cut, g, gp in GRADES:
        if pct >= cut:
            return g, float(gp)
    return ("P", 4.0) if pct >= PASS_PERCENT else ("F", 0.0)


def _score(r) -> Optional[float]:
    if r is None:
        return None
    return r.examiner_score if r.examiner_approved and r.examiner_score is not None else r.final_score


def booklet_totals(b: StudentBooklet) -> Dict:
    """AI total, teacher-approved total and max for a booklet (only evaluated answers count)."""
    ai = final = mx = 0.0
    evaluated = 0
    for a in b.answers:
        r = a.submission.result if a.submission else None
        if r is None:
            continue
        evaluated += 1
        mx += r.max_marks or 0.0
        ai += r.final_score or 0.0
        final += _score(r) or 0.0
    return {"ai": round(ai, 2), "final": round(final, 2), "max": round(mx, 2), "evaluated": evaluated,
            "answers": len(b.answers)}


def booklet_status_label(b: StudentBooklet) -> str:
    return {"pending": "Awaiting AI evaluation", "in_review": "AI evaluated — awaiting teacher review",
            "approved": "Teacher approved — not yet published", "published": "Published"}.get(b.approval_status, b.approval_status or "")


# ── consolidated result ──────────────────────────────────────────────────────

def compile_final_result(db: Session, student: Student, course_code: str, exam_name: str) -> FinalResult:
    """Create/refresh the consolidated result from the approved booklet and verified OMR (whichever exist)."""
    course_code = course_code.upper()
    booklet = (db.query(StudentBooklet).filter(StudentBooklet.student_id == student.id,
                                               StudentBooklet.course_code == course_code,
                                               StudentBooklet.exam_name == exam_name)
               .order_by(StudentBooklet.id.desc()).first())
    key = db.query(OmrKey).filter(OmrKey.course_code == course_code, OmrKey.exam_name == exam_name).first()
    omr = db.query(OmrResponse).filter(OmrResponse.key_id == key.id, OmrResponse.student_id == student.id).first() if key else None

    problems = []
    if booklet is None and omr is None:
        raise ValueError("No answer sheet or OMR sheet exists for this student, subject and examination.")
    if booklet is not None and booklet.approval_status not in ("approved", "published"):
        problems.append("the written answer sheet is not yet approved by a teacher")
    if omr is not None and omr.needs_verification:
        problems.append("the OMR sheet has readings that need manual verification")
    if problems:
        raise ValueError("Cannot compile yet: " + "; ".join(problems) + ".")

    fr = db.query(FinalResult).filter_by(student_id=student.id, course_code=course_code, exam_name=exam_name).first()
    if fr is None:
        fr = FinalResult(student_id=student.id, course_code=course_code, exam_name=exam_name)
        db.add(fr)
    if fr.status == "published":
        raise ValueError("This result is already published.")
    course = db.query(Course).filter(Course.code == course_code).first()
    fr.course_name = course.name if course else (booklet.course_name if booklet else course_code)
    total = mx = 0.0
    if booklet is not None:
        t = booklet_totals(booklet)
        fr.booklet_id, fr.subjective_score, fr.subjective_max = booklet.id, t["final"], t["max"]
        total += t["final"]; mx += t["max"]
    else:
        fr.booklet_id = fr.subjective_score = fr.subjective_max = None
    if omr is not None:
        fr.omr_response_id, fr.omr_score, fr.omr_max = omr.id, omr.score, omr.max_score
        total += omr.score; mx += omr.max_score
    else:
        fr.omr_response_id = fr.omr_score = fr.omr_max = None
    fr.total, fr.max_total = round(total, 2), round(mx, 2)
    fr.percentage = round(100.0 * total / mx, 2) if mx else 0.0
    fr.grade, fr.grade_point = grade_for(fr.percentage)
    fr.passed = fr.percentage >= PASS_PERCENT
    fr.status = "draft"
    fr.approved_by_id = fr.approved_at = None
    fr.updated_at = datetime.utcnow()
    db.flush()
    return fr


def final_result_dict(fr: FinalResult) -> Dict:
    s = fr.student
    return {
        "id": fr.id, "registration_number": s.registration_number if s else "", "student_name": s.name if s else "",
        "course_code": fr.course_code, "course_name": fr.course_name, "exam_name": fr.exam_name,
        "omr_score": fr.omr_score, "omr_max": fr.omr_max, "subjective_score": fr.subjective_score,
        "subjective_max": fr.subjective_max, "total": fr.total, "max_total": fr.max_total,
        "percentage": fr.percentage, "grade": fr.grade, "grade_point": fr.grade_point, "passed": fr.passed,
        "status": fr.status, "booklet_id": fr.booklet_id, "omr_response_id": fr.omr_response_id,
        "approved_at": fr.approved_at.isoformat() if fr.approved_at else None,
        "published_at": fr.published_at.isoformat() if fr.published_at else None,
    }


# ── per-subject rows (class results, analytics, 360° view) ───────────────────

def subject_rows(db: Session, students: Iterable[Student], course_code: Optional[str] = None,
                 exam_name: Optional[str] = None, published_only: bool = False,
                 allowed_pairs: Optional[set] = None) -> List[Dict]:
    """
    One row per student × subject × exam. A FinalResult wins over its booklet; booklets that have not been
    compiled still appear (subjective marks only, with their evaluation status).
    allowed_pairs = {(course_code, batch_id)} limits rows for teachers.
    """
    rows = []
    for st in students:
        batch = st.batch_rel
        base = {"registration_number": st.registration_number or st.student_code, "student_name": st.name,
                "batch_code": batch.batch_code if batch else "", "section": st.section or (batch.section if batch else ""),
                "semester": st.semester, "programme": st.programme, "academic_session": st.academic_session or (batch.academic_session if batch else "")}
        seen = set()
        frs = db.query(FinalResult).filter(FinalResult.student_id == st.id)
        if course_code:
            frs = frs.filter(FinalResult.course_code == course_code.upper())
        if exam_name:
            frs = frs.filter(FinalResult.exam_name == exam_name)
        for fr in frs.all():
            if allowed_pairs is not None and (fr.course_code, st.batch_id) not in allowed_pairs:
                continue
            if published_only and fr.status != "published":
                continue
            seen.add((fr.course_code, fr.exam_name))
            if fr.booklet_id:
                seen.add(("#b", fr.booklet_id))
            rows.append({**base, "subject_code": fr.course_code, "subject_name": fr.course_name, "exam_name": fr.exam_name,
                         "omr_marks": fr.omr_score, "subjective_marks": fr.subjective_score, "marks": fr.total,
                         "max_marks": fr.max_total, "percentage": fr.percentage, "grade": fr.grade,
                         "grade_point": fr.grade_point, "passed": fr.passed,
                         "ai_evaluation_status": "Compiled (OMR + subjective)", "final_approved_marks":
                         fr.total if fr.status in ("approved", "published") else None, "status": fr.status,
                         "source": "final_result"})
        bq = db.query(StudentBooklet).filter(StudentBooklet.student_id == st.id)
        if course_code:
            bq = bq.filter(StudentBooklet.course_code == course_code.upper())
        if exam_name:
            bq = bq.filter(StudentBooklet.exam_name == exam_name)
        for b in bq.all():
            if ("#b", b.id) in seen or (b.course_code, b.exam_name) in seen:
                continue
            if allowed_pairs is not None and (b.course_code, st.batch_id) not in allowed_pairs:
                continue
            if published_only and not b.is_published:
                continue
            t = booklet_totals(b)
            approved = b.approval_status in ("approved", "published")
            pct = round(100.0 * t["final"] / t["max"], 2) if (approved and t["max"]) else None
            g, gp = grade_for(pct)
            rows.append({**base, "subject_code": b.course_code, "subject_name": b.course_name, "exam_name": b.exam_name,
                         "omr_marks": None, "subjective_marks": t["final"] if t["evaluated"] else None,
                         "marks": t["final"] if approved else None, "max_marks": t["max"] or None,
                         "percentage": pct, "grade": g, "grade_point": gp if pct is not None else None,
                         "passed": (pct >= PASS_PERCENT) if pct is not None else None,
                         "ai_marks": t["ai"] if t["evaluated"] else None,
                         "ai_evaluation_status": booklet_status_label(b),
                         "final_approved_marks": t["final"] if approved else None,
                         "status": "published" if b.is_published else b.approval_status, "source": "booklet",
                         "booklet_id": b.id})
    return rows


def summarize(rows: List[Dict]) -> Dict:
    """Pass/fail and score statistics over rows that have an approved percentage."""
    graded = [r for r in rows if r.get("percentage") is not None]
    pcts = [r["percentage"] for r in graded]
    passed = sum(1 for r in graded if r.get("passed"))
    return {
        "records": len(rows), "graded": len(graded), "students": len({r["registration_number"] for r in rows}),
        "passed": passed, "failed": len(graded) - passed,
        "pass_percentage": round(100.0 * passed / len(graded), 1) if graded else None,
        "fail_percentage": round(100.0 * (len(graded) - passed) / len(graded), 1) if graded else None,
        "average_percentage": round(mean(pcts), 2) if pcts else None,
        "highest_percentage": max(pcts) if pcts else None, "lowest_percentage": min(pcts) if pcts else None,
        "average_marks": round(mean([r["marks"] for r in graded if r.get("marks") is not None]), 2) if graded else None,
        "highest_marks": max((r["marks"] for r in graded if r.get("marks") is not None), default=None),
        "lowest_marks": min((r["marks"] for r in graded if r.get("marks") is not None), default=None),
        "pending": len(rows) - len(graded),
    }


def weak_topics_for(db: Session, booklet_ids: Iterable[int], top: int = 10) -> List[Dict]:
    """Most frequent missing concepts across evaluated answers (topic-wise weakness)."""
    c = Counter()
    ids = list(booklet_ids)
    if not ids:
        return []
    for b in db.query(StudentBooklet).filter(StudentBooklet.id.in_(ids)).all():
        for a in b.answers:
            r = a.submission.result if a.submission else None
            if not r or not r.missing_concepts:
                continue
            try:
                items = json.loads(r.missing_concepts) if r.missing_concepts.startswith("[") else [r.missing_concepts]
            except Exception:
                items = [r.missing_concepts]
            for m in items:
                m = str(m).strip()
                if m:
                    c[m.lower()] += 1
    return [{"topic": t, "occurrences": n} for t, n in c.most_common(top)]


def question_difficulty(db: Session, booklet_ids: Iterable[int]) -> List[Dict]:
    """Average % score per question number → identifies difficult questions."""
    agg = defaultdict(list)
    ids = list(booklet_ids)
    if not ids:
        return []
    for b in db.query(StudentBooklet).filter(StudentBooklet.id.in_(ids)).all():
        for a in b.answers:
            r = a.submission.result if a.submission else None
            if r and r.max_marks:
                agg[(b.course_code, a.question_number)].append(100.0 * (_score(r) or 0) / r.max_marks)
    out = [{"subject_code": k[0], "question_number": k[1], "attempts": len(v), "average_percentage": round(mean(v), 1)}
           for k, v in agg.items()]
    return sorted(out, key=lambda x: x["average_percentage"])


def cgpa(rows: List[Dict], credits: Dict[str, int]) -> Optional[float]:
    """Credit-weighted grade-point average over the latest graded result per subject."""
    latest = {}
    for r in rows:
        if r.get("grade_point") is not None and r.get("percentage") is not None:
            latest[r["subject_code"]] = r
    if not latest:
        return None
    num = sum(credits.get(c, 3) * r["grade_point"] for c, r in latest.items())
    den = sum(credits.get(c, 3) for c in latest)
    return round(num / den, 2) if den else None


def refresh_final_after_reevaluation(db: Session, booklet: StudentBooklet) -> Optional[FinalResult]:
    """
    A re-evaluation changed marks of an already compiled / published written paper: update the consolidated
    result in place (status kept — the change was decided by a teacher through the re-evaluation process).
    """
    fr = db.query(FinalResult).filter(FinalResult.booklet_id == booklet.id).first()
    if fr is None:
        return None
    t = booklet_totals(booklet)
    fr.subjective_score, fr.subjective_max = t["final"], t["max"]
    total = (fr.subjective_score or 0) + (fr.omr_score or 0)
    mx = (fr.subjective_max or 0) + (fr.omr_max or 0)
    fr.total, fr.max_total = round(total, 2), round(mx, 2)
    fr.percentage = round(100.0 * total / mx, 2) if mx else 0.0
    fr.grade, fr.grade_point = grade_for(fr.percentage)
    fr.passed = fr.percentage >= PASS_PERCENT
    fr.updated_at = datetime.utcnow()
    db.flush()
    return fr
