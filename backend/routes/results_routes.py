"""
Results, analytics, dashboards & notifications
==============================================
/results  — class results (+ CSV / Excel / PDF), analytics, OMR+subjective compilation, approval,
            publication, evaluation-report PDFs, teacher dashboard, student dashboard & downloads
/notifications — per-user notifications
"""

import csv
import io
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import access as AC
from backend import results_service as RS
from backend.auth import get_current_user, require_role
from backend.database import (
    Batch, Course, FinalResult, Notification, OmrResponse, QuizAttempt, StudyRoadmap, Student, StudentBooklet,
    TeacherProfile, User, WeakTopic, get_db,
)

router = APIRouter()
notifications_router = APIRouter()
STAFF = Depends(require_role(["admin", "faculty"]))


def _students_for_filters(db, user, batch_code=None, course_code=None, section=None, semester=None, session=None,
                          subject_code=None):
    q = AC.scoped_students(db, user, subject_code)
    if batch_code or course_code or section or semester or session:
        bq = db.query(Batch)
        if batch_code:
            bq = bq.filter(func.upper(Batch.batch_code) == batch_code.strip().upper())
        if course_code:
            bq = bq.filter(func.upper(Batch.course_code) == course_code.strip().upper())
        if section:
            bq = bq.filter(func.upper(Batch.section) == section.strip().upper())
        if semester:
            bq = bq.filter(Batch.semester.ilike(f"%{semester.strip()}%"))
        if session:
            bq = bq.filter(Batch.academic_session == session.strip())
        ids = [b.id for b in bq.all()]
        q = q.filter(Student.batch_id.in_(ids)) if ids else q.filter(False)
    return q.order_by(Student.registration_number).all()


def _allowed(db, user):
    return None if user.role == "admin" else set(AC.teacher_pairs(db, user))


# ── class results + export ────────────────────────────────────────────────────

EXPORT_COLS = ["registration_number", "student_name", "batch_code", "section", "semester", "subject_code",
               "subject_name", "exam_name", "omr_marks", "subjective_marks", "ai_marks", "final_approved_marks",
               "max_marks", "percentage", "grade", "passed", "ai_evaluation_status", "status"]


@router.get("/class", summary="Class results by batch / course / section / semester / subject")
def class_results(batch_code: Optional[str] = None, course_code: Optional[str] = None, section: Optional[str] = None,
                  semester: Optional[str] = None, session: Optional[str] = None, subject_code: Optional[str] = None,
                  exam_name: Optional[str] = None, format: str = Query("json", pattern="^(json|csv|xlsx|pdf)$"),
                  db: Session = Depends(get_db), user: User = STAFF):
    students = _students_for_filters(db, user, batch_code, course_code, section, semester, session, subject_code)
    rows = RS.subject_rows(db, students, course_code=subject_code, exam_name=exam_name, allowed_pairs=_allowed(db, user))
    summary = RS.summarize(rows)
    filters = {"batch_code": batch_code, "course_code": course_code, "section": section, "semester": semester,
               "session": session, "subject_code": subject_code, "exam_name": exam_name}
    if format == "json":
        return {"filters": filters, "summary": summary, "rows": rows}
    name = "class_result_" + "_".join(str(v) for v in filters.values() if v).replace(" ", "-")[:60] or "class_result"
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EXPORT_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "passed": "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "")})
        return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    if format == "xlsx":
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook()
        ws = wb.active
        ws.title = "Class result"
        ws.append([c.replace("_", " ").title() for c in EXPORT_COLS])
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1E2A6E")
        for r in rows:
            ws.append([("PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "")) if k == "passed"
                       else r.get(k) for k in EXPORT_COLS])
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(40, max(10, max(len(str(c.value or "")) for c in col) + 2))
        s2 = wb.create_sheet("Summary")
        for k, v in summary.items():
            s2.append([k.replace("_", " ").title(), v])
        out = io.BytesIO()
        wb.save(out)
        return Response(out.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})
    from backend.reports import class_result_pdf
    return Response(class_result_pdf(rows, filters, summary), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})


# ── analytics ─────────────────────────────────────────────────────────────────

GROUPS = {"batch": "batch_code", "section": "section", "semester": "semester", "subject": "subject_code",
          "session": "academic_session", "exam": "exam_name"}


@router.get("/analytics", summary="Compare performance by batch / section / semester / subject / teacher / session")
def analytics(group_by: str = Query("subject", pattern="^(batch|section|semester|subject|teacher|session|exam)$"),
              batch_code: Optional[str] = None, course_code: Optional[str] = None, section: Optional[str] = None,
              semester: Optional[str] = None, session: Optional[str] = None, subject_code: Optional[str] = None,
              db: Session = Depends(get_db), user: User = STAFF):
    students = _students_for_filters(db, user, batch_code, course_code, section, semester, session, subject_code)
    allowed = _allowed(db, user)
    rows = RS.subject_rows(db, students, course_code=subject_code, allowed_pairs=allowed)
    groups = {}
    if group_by == "teacher":
        st_batch = {s.registration_number: s.batch_id for s in students}
        for t in db.query(TeacherProfile).all():
            if user.role == "faculty" and t.user_id != user.id:
                continue
            pairs = {(a.course.code, a.batch_id) for a in t.assignments}
            sel = [r for r in rows if (r["subject_code"], st_batch.get(r["registration_number"])) in pairs]
            groups[f"{t.teacher_code} — {t.user.full_name}"] = sel
    else:
        key = GROUPS[group_by]
        for r in rows:
            groups.setdefault(r.get(key) or "—", []).append(r)
    booklet_ids = [r["booklet_id"] for r in rows if r.get("booklet_id")]
    booklet_ids += [fr.booklet_id for fr in db.query(FinalResult).filter(FinalResult.student_id.in_([s.id for s in students])).all() if fr.booklet_id]
    if allowed is not None:
        ok = {b.id for b in AC.scoped_booklets(db, user).all()}
        booklet_ids = [b for b in booklet_ids if b in ok]
    return {
        "group_by": group_by,
        "overall": RS.summarize(rows),
        "groups": [{"group": k, **RS.summarize(v)} for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))],
        "grade_distribution": _grade_dist(rows),
        "weak_topics": RS.weak_topics_for(db, booklet_ids),
        "difficult_questions": RS.question_difficulty(db, booklet_ids)[:10],
        "low_pass_subjects": [g for g in ({"subject": k, **RS.summarize(v)} for k, v in
                                          _by(rows, "subject_code").items()) if g["pass_percentage"] is not None and g["pass_percentage"] < 60],
    }


def _by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def _grade_dist(rows):
    d = {}
    for r in rows:
        if r.get("grade"):
            d[r["grade"]] = d.get(r["grade"], 0) + 1
    order = ["O", "A+", "A", "B+", "B", "C", "P", "F"]
    return [{"grade": g, "count": d[g]} for g in order if g in d]


# ── compile / approve / publish consolidated results ─────────────────────────

class CompileIn(BaseModel):
    subject_code: str
    exam_name: str
    registration_number: Optional[str] = None   # one student …
    batch_code: Optional[str] = None            # … or a whole batch


@router.post("/compile", summary="Combine OMR + subjective marks into consolidated results (draft)")
def compile_results(data: CompileIn, db: Session = Depends(get_db), user: User = STAFF):
    if not data.registration_number and not data.batch_code:
        raise HTTPException(status_code=422, detail="Give a registration number or a batch code.")
    q = AC.scoped_students(db, user, data.subject_code)
    if data.registration_number:
        q = q.filter(func.upper(Student.registration_number) == data.registration_number.strip().upper())
    else:
        b = db.query(Batch).filter(func.upper(Batch.batch_code) == data.batch_code.strip().upper()).first()
        q = q.filter(Student.batch_id == (b.id if b else -1))
    students = q.all()
    if not students:
        raise HTTPException(status_code=404, detail="No matching students that you are allowed to manage.")
    done, problems = [], []
    for st in students:
        if not AC.can_teach(db, user, st, data.subject_code):
            problems.append({"registration_number": st.registration_number, "error": "not assigned"})
            continue
        try:
            fr = RS.compile_final_result(db, st, data.subject_code, data.exam_name)
            done.append(RS.final_result_dict(fr))
        except ValueError as e:
            problems.append({"registration_number": st.registration_number, "error": str(e)})
    AC.audit(db, user, "RESULTS_COMPILED", "final_result", data.subject_code,
             {"exam": data.exam_name, "compiled": len(done), "problems": len(problems)})
    db.commit()
    return {"compiled": done, "not_compiled": problems}


def _final_or_404(db, user, fid) -> FinalResult:
    fr = db.query(FinalResult).filter(FinalResult.id == fid).first()
    if not fr or not AC.can_teach(db, user, fr.student, fr.course_code):
        raise HTTPException(status_code=404, detail="Result not found")
    return fr


@router.get("/final", summary="Consolidated results (filter by subject / exam / batch / status)")
def list_final(subject_code: Optional[str] = None, exam_name: Optional[str] = None, batch_code: Optional[str] = None,
               status: Optional[str] = None, db: Session = Depends(get_db), user: User = STAFF):
    ids = [s.id for s in _students_for_filters(db, user, batch_code=batch_code, subject_code=subject_code)]
    q = db.query(FinalResult).filter(FinalResult.student_id.in_(ids)) if ids else db.query(FinalResult).filter(False)
    if subject_code:
        q = q.filter(FinalResult.course_code == subject_code.upper())
    if exam_name:
        q = q.filter(FinalResult.exam_name == exam_name)
    if status:
        q = q.filter(FinalResult.status == status)
    allowed = _allowed(db, user)
    return [RS.final_result_dict(fr) for fr in q.order_by(FinalResult.id).all()
            if allowed is None or (fr.course_code, fr.student.batch_id) in allowed]


@router.post("/final/{fid}/approve", summary="Teacher approves a consolidated result")
def approve_final(fid: int, db: Session = Depends(get_db), user: User = STAFF):
    fr = _final_or_404(db, user, fid)
    if fr.status == "published":
        raise HTTPException(status_code=409, detail="Already published.")
    fr.status, fr.approved_by_id, fr.approved_at = "approved", user.id, datetime.utcnow()
    AC.audit(db, user, "FINAL_RESULT_APPROVED", "final_result", fr.id, {"total": fr.total, "max": fr.max_total})
    db.commit()
    return RS.final_result_dict(fr)


@router.post("/final/{fid}/publish", summary="Publish an approved consolidated result")
def publish_final(fid: int, db: Session = Depends(get_db), user: User = STAFF):
    fr = _final_or_404(db, user, fid)
    if fr.status != "approved":
        raise HTTPException(status_code=409, detail="Only approved results can be published.")
    fr.status, fr.published_at = "published", datetime.utcnow()
    if fr.booklet_id:  # the written part becomes visible with its question-wise breakdown
        b = db.query(StudentBooklet).get(fr.booklet_id)
        if b and not b.is_published:
            b.is_published, b.published_at, b.published_by_id, b.approval_status = True, fr.published_at, user.id, "published"
    AC.audit(db, user, "FINAL_RESULT_PUBLISHED", "final_result", fr.id, {"total": fr.total})
    AC.notify_student(db, fr.student, "result_published", f"{fr.course_code}: final result published",
                      f"{fr.exam_name}: {fr.total:g}/{fr.max_total:g} ({fr.percentage:g}%), grade {fr.grade}.")
    db.commit()
    return RS.final_result_dict(fr)


# ── evaluation report PDF ────────────────────────────────────────────────────

@router.get("/booklet/{booklet_id}/report.pdf", summary="Detailed evaluation report (marks analysis) PDF")
def booklet_report(booklet_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from backend.reports import evaluation_report_pdf
    b = db.query(StudentBooklet).filter(StudentBooklet.id == booklet_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Booklet not found")
    if user.role == "student":
        st = db.query(Student).filter(Student.user_id == user.id).first()
        if not st or b.student_id != st.id or not b.is_published:
            raise HTTPException(status_code=404, detail="Booklet not found")
    elif not AC.can_access_booklet(db, user, b):
        raise HTTPException(status_code=404, detail="Booklet not found")
    fr = db.query(FinalResult).filter(FinalResult.booklet_id == b.id).first()
    pdf = evaluation_report_pdf(b, fr if fr and (user.role != "student" or fr.status == "published") else None)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="evaluation_{b.register_number}_{b.course_code}.pdf"'})


# ── teacher dashboard ────────────────────────────────────────────────────────

@router.get("/teacher/dashboard", summary="Teacher dashboard (only assigned batches & subjects)")
def teacher_dashboard(db: Session = Depends(get_db), user: User = Depends(require_role(["faculty"]))):
    from backend.routes.academic_routes import teacher_stats, _teacher_dict
    prof = AC.teacher_profile(db, user)
    if not prof:
        return {"profile": None, "message": "No teacher profile yet — ask the examination department to register "
                                            "your teacher ID and assign subjects/batches."}
    booklets = AC.scoped_booklets(db, user).all()
    st = {}
    for b in booklets:
        st[b.approval_status or "pending"] = st.get(b.approval_status or "pending", 0) + 1
    pairs = set(AC.teacher_pairs(db, user))
    omr_pending = [o for o in db.query(OmrResponse).filter(OmrResponse.needs_verification == True).all()  # noqa: E712
                   if (o.key.course_code, o.student.batch_id) in pairs]
    return {"profile": _teacher_dict(db, prof), "stats": teacher_stats(db, prof),
            "answer_sheets": {"total": len(booklets), "pending_evaluation": st.get("pending", 0),
                              "awaiting_review": st.get("in_review", 0), "approved_unpublished": st.get("approved", 0),
                              "published": st.get("published", 0)},
            "omr_needing_verification": len(omr_pending),
            "results_awaiting_approval": sum(1 for fr in db.query(FinalResult).filter(FinalResult.status == "draft").all()
                                             if (fr.course_code, fr.student.batch_id) in pairs)}


# ── student dashboard & downloads ────────────────────────────────────────────

def _me(db, user) -> Student:
    st = db.query(Student).filter(Student.user_id == user.id).first() or \
        db.query(Student).filter(Student.registration_number == user.registration_number).first()
    if not st:
        raise HTTPException(status_code=404, detail="Student record not found")
    return st


@router.get("/my/dashboard", summary="Student dashboard (own published records only)")
def my_dashboard(db: Session = Depends(get_db), user: User = Depends(require_role(["student"]))):
    from backend.routes.academic_routes import student_profile_dict
    st = _me(db, user)
    published = RS.subject_rows(db, [st], published_only=True)
    credits = {r.course.code: r.course.credits for r in st.course_registrations if r.course}
    sem = {}
    for r in published:
        sem.setdefault(r["semester"] or "—", []).append(r)
    pending = [{"subject_code": b.course_code, "exam_name": b.exam_name, "status": RS.booklet_status_label(b)}
               for b in db.query(StudentBooklet).filter(StudentBooklet.student_id == st.id,
                                                        StudentBooklet.is_published == False).all()]  # noqa: E712
    quizzes = db.query(QuizAttempt).filter(QuizAttempt.user_id == user.id).all()
    roadmap = db.query(StudyRoadmap).filter(StudyRoadmap.user_id == user.id).first()
    weak = db.query(WeakTopic).filter(WeakTopic.user_id == user.id, WeakTopic.status != "Mastered").order_by(WeakTopic.accuracy).limit(8).all()
    return {
        "profile": student_profile_dict(db, st),
        "total_subjects": len(credits),
        "results": published,
        "summary": RS.summarize(published),
        "semester_wise": [{"semester": k, **RS.summarize(v)} for k, v in sem.items()],
        "cgpa": RS.cgpa(published, credits),
        "evaluation_in_progress": pending,
        "quiz": {"attempts": len(quizzes),
                 "average_percentage": round(sum(q.percentage for q in quizzes) / len(quizzes), 1) if quizzes else None},
        "roadmap": {"exists": bool(roadmap), "progress_pct": roadmap.progress_pct if roadmap else None},
        "weak_topics": [{"topic": w.topic_name, "accuracy": w.accuracy} for w in weak],
        "pass_mark_percent": RS.PASS_PERCENT,
    }


@router.get("/my/final.pdf", summary="Download my final approved result PDF")
def my_final_pdf(db: Session = Depends(get_db), user: User = Depends(require_role(["student"]))):
    from backend.reports import final_result_pdf
    st = _me(db, user)
    rows = RS.subject_rows(db, [st], published_only=True)
    if not rows:
        raise HTTPException(status_code=404, detail="No published results yet.")
    credits = {r.course.code: r.course.credits for r in st.course_registrations if r.course}
    return Response(final_result_pdf(st, rows, RS.cgpa(rows, credits)), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="final_result_{st.registration_number}.pdf"'})


# ── notifications ─────────────────────────────────────────────────────────────

@notifications_router.get("", summary="My notifications")
def my_notifications(unread_only: bool = False, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = db.query(Notification).filter(Notification.user_id == user.id)
    if unread_only:
        q = q.filter(Notification.is_read == False)  # noqa: E712
    items = q.order_by(Notification.id.desc()).limit(50).all()
    return {"unread": db.query(Notification).filter(Notification.user_id == user.id, Notification.is_read == False).count(),  # noqa: E712
            "items": [{"id": n.id, "kind": n.kind, "title": n.title, "body": n.body, "is_read": n.is_read,
                       "created_at": n.created_at.isoformat() + "Z"} for n in items]}


@notifications_router.post("/read-all", summary="Mark all as read")
def read_all(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.query(Notification).filter(Notification.user_id == user.id).update({"is_read": True})
    db.commit()
    return {"status": "ok"}


@notifications_router.post("/{nid}/read", summary="Mark one as read")
def read_one(nid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    n = db.query(Notification).filter(Notification.id == nid, Notification.user_id == user.id).first()
    if not n:
        raise HTTPException(status_code=404, detail="Not found")
    n.is_read = True
    db.commit()
    return {"status": "ok"}
