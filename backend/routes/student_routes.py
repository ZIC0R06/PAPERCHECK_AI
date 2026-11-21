"""
AI Paper Checking — Student Portal Routes
==========================================
Secure, authenticated endpoints for students:
  • Profile & registered academic courses
  • Published examination results & question-wise marks breakdown
  • Download original and evaluated annotated answer sheets
  • Re-evaluation request submission and dispute tracking
  • Strict backend authorization: students can only access their own records!
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import (
    get_db, User, Student, StudentBooklet, StudentAnswerText,
    Submission, Result, CourseRegistration, ReEvaluationRequest, Course, AuditLog
)
from backend.auth import get_current_user, verify_student_access
from backend.schemas import ReEvaluationCreateSchema
from backend.evaluated_sheet_generator import generate_evaluated_sheet_pdf
from backend import access as AC


def _check_booklet_access(db: Session, user: User, booklet: StudentBooklet, require_published: bool = True):
    """Students: only their own (published) booklets. Teachers: only assigned. Admin: all."""
    if user.role == "student":
        verify_student_access(user, booklet.student)
        if require_published and not booklet.is_published:
            raise HTTPException(status_code=403, detail="This examination result is still under evaluation.")
    elif not AC.can_access_booklet(db, user, booklet):
        raise HTTPException(status_code=404, detail="Booklet not found")

router = APIRouter()


def _get_student_for_user(user: User, db: Session) -> Student:
    """Resolves student record for current user, verifying existence."""
    student = None
    if user.student_profile:
        student = user.student_profile
    else:
        student = db.query(Student).filter(
            (Student.user_id == user.id) |
            (Student.student_code == user.username) |
            (Student.registration_number == user.registration_number)
        ).first()

    if not student and user.role == "student":
        # Create student profile on the fly if user was created without one
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

    if not student and user.role != "admin":
        raise HTTPException(status_code=404, detail="Student academic record not found.")
    return student


# ─────────────────────────────────────────────────────────────
# 1. Student Profile & Registered Courses
# ─────────────────────────────────────────────────────────────

@router.get("/profile", summary="Student academic profile")
def get_student_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    verify_student_access(current_user, student)

    # Registered courses
    courses = []
    for reg in student.course_registrations:
        c = reg.course
        if c:
            courses.append({
                "code": c.code,
                "name": c.name,
                "credits": c.credits,
                "semester": reg.semester
            })

    return {
        "student_id": student.id,
        "name": student.name,
        "registration_number": student.registration_number or student.student_code,
        "roll_number": student.student_code,
        "department": student.department,
        "programme": student.programme,
        "semester": student.semester,
        "batch": student.batch,
        "batch_code": student.batch_rel.batch_code if student.batch_rel else "",
        "section": student.section or (student.batch_rel.section if student.batch_rel else ""),
        "academic_session": student.academic_session or (student.batch_rel.academic_session if student.batch_rel else ""),
        "course_code": student.batch_rel.course_code if student.batch_rel else "",
        "email": student.email,
        "registered_courses": courses
    }


# ─────────────────────────────────────────────────────────────
# 2. Published Examination Results
# ─────────────────────────────────────────────────────────────

@router.get("/results", summary="Published examination results for student")
def get_student_published_results(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns only published exam results for the authenticated student.
    Unpublished drafts are strictly hidden from students.
    """
    student = _get_student_for_user(current_user, db)
    verify_student_access(current_user, student)

    booklets = db.query(StudentBooklet).filter(
        StudentBooklet.student_id == student.id,
        StudentBooklet.is_published == True
    ).order_by(StudentBooklet.id.desc()).all()

    return [
        {
            "booklet_id": b.id,
            "course_code": b.course_code,
            "course_name": b.course_name,
            "exam_name": b.exam_name,
            "semester": b.semester,
            "total_score": b.total_final_score or b.total_ai_score,
            "published_at": b.published_at.isoformat() if b.published_at else None,
            "approval_status": b.approval_status
        }
        for b in booklets
    ]


@router.get("/result/{booklet_id}", summary="Detailed question-wise examination result")
def get_student_result_detail(
    booklet_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    booklet = db.query(StudentBooklet).filter(StudentBooklet.id == booklet_id).first()

    if not booklet:
        raise HTTPException(status_code=404, detail="Exam result record not found")

    # Strict authorization enforcement
    _check_booklet_access(db, current_user, booklet)

    from backend.booklet_ocr import question_evaluations
    questions = []
    for q in question_evaluations(booklet):
        if q["result_id"] is None:
            continue
        eq = next((a.exam_question for a in booklet.answers if a.id == q["answer_id"]), None)
        questions.append({
            "question_number": q["question_number"],
            "question_text": q["question_text"],
            "student_answer": q["student_answer"],
            "expected_answer": (eq.teacher_answer or "") if eq else "",
            "ai_score": q["ai_score"],
            "marks_awarded": q["awarded"],
            "max_marks": q["max_marks"],
            "marks_deducted": round(q["max_marks"] - (q["awarded"] or 0), 2),
            "part_name": eq.part_name if eq else "",
            "question_type": q["question_type"] or "open_ended",
            "strengths": q["strengths"],
            "missing_concepts": q["missing_concepts"],
            "rubric_breakdown": q["rubric_breakdown"],
            "deductions": q["deductions"],
            "corrections": q["corrections"],
            "feedback": q["feedback"],
            "examiner_comments": q["teacher_comment"] if q["teacher_verified"] else "",
            "is_examiner_certified": q["teacher_verified"],
            "teacher_changed_marks": q["teacher_changed"],
            "page_refs": q["page_refs"],
        })
    total_awarded = sum(q["marks_awarded"] or 0 for q in questions)
    total_max = sum(q["max_marks"] for q in questions)
    pct = round((total_awarded / total_max * 100), 2) if total_max > 0 else 0.0

    return {
        "booklet_id": booklet.id,
        "course_code": booklet.course_code,
        "course_name": booklet.course_name,
        "exam_name": booklet.exam_name,
        "total_awarded": round(total_awarded, 2),
        "total_max": round(total_max, 2),
        "percentage": pct,
        "published_at": booklet.published_at.isoformat() if booklet.published_at else None,
        "questions": questions
    }


# ─────────────────────────────────────────────────────────────
# 2B. Student Answer Sheet Submission
# ─────────────────────────────────────────────────────────────

@router.post("/answer-sheet/upload", summary="Student upload answer sheet for examination")
async def upload_student_answer_sheet(
    file: UploadFile = File(...),
    course_code: str = Form(...),
    exam_name: str = Form("End Semester Examination 2026"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    verify_student_access(current_user, student)

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, and PNG files are accepted.")

    # Check for duplicate submission for this course and exam
    c_code = course_code.strip().upper()
    existing = db.query(StudentBooklet).filter(
        StudentBooklet.student_id == student.id,
        StudentBooklet.course_code == c_code,
        StudentBooklet.exam_name == exam_name
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"You have already submitted an answer sheet for {c_code} ({exam_name}). Duplicate submissions are not permitted."
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    clean_reg = student.registration_number or student.student_code or "STUDENT"
    filename = f"{clean_reg}_{c_code}_{uuid.uuid4().hex[:6]}{ext}"
    save_path = upload_dir / filename

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum file size is 20MB.")

    with open(save_path, "wb") as f:
        f.write(content)

    course = db.query(Course).filter(Course.code == c_code).first()
    c_name = course.name if course else c_code

    booklet = StudentBooklet(
        student_id=student.id,
        pdf_path=str(save_path),
        course_code=c_code,
        course_name=c_name,
        exam_name=exam_name,
        roll_number=student.student_code,
        register_number=student.registration_number,
        department=student.department,
        programme=student.programme,
        semester=student.semester,
        approval_status="pending",
        is_published=False
    )
    db.add(booklet)
    db.commit()
    db.refresh(booklet)

    audit = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        action="STUDENT_ANSWER_SHEET_SUBMITTED",
        target_type="booklet",
        target_id=str(booklet.id),
        details=json.dumps({"course_code": c_code, "filename": filename})
    )
    db.add(audit)
    db.commit()

    return {
        "status": "success",
        "booklet_id": booklet.id,
        "message": f"Answer sheet for {c_code} uploaded successfully. Evaluation is pending faculty review.",
        "course_code": c_code
    }


# ─────────────────────────────────────────────────────────────
# 3. Answer Sheet Viewer & Download (Original vs Evaluated)
# ─────────────────────────────────────────────────────────────

@router.get("/answer-sheet/{booklet_id}/original", summary="Download original uploaded answer sheet")
def download_original_sheet(
    booklet_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    booklet = db.query(StudentBooklet).filter(StudentBooklet.id == booklet_id).first()

    if not booklet:
        raise HTTPException(status_code=404, detail="Booklet not found")

    _check_booklet_access(db, current_user, booklet, require_published=False)

    if not booklet.pdf_path or not os.path.exists(booklet.pdf_path):
        raise HTTPException(status_code=404, detail="Original answer sheet file not found on server.")

    filename = os.path.basename(booklet.pdf_path)
    return FileResponse(
        path=booklet.pdf_path,
        filename=f"original_{filename}",
        media_type="application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
    )


@router.get("/answer-sheet/{booklet_id}/evaluated", summary="Download official evaluated answer sheet with annotations")
def download_evaluated_sheet(
    booklet_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    booklet = db.query(StudentBooklet).filter(StudentBooklet.id == booklet_id).first()

    if not booklet:
        raise HTTPException(status_code=404, detail="Booklet not found")

    _check_booklet_access(db, current_user, booklet, require_published=True)

    # Built from the database on every request, so it always shows the latest (re-evaluated) marks.
    from fastapi.responses import Response
    from backend.evaluated_sheet_generator import build_evaluated_pdf_for_booklet
    try:
        pdf = build_evaluated_pdf_for_booklet(booklet)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("evaluated PDF failed")
        raise HTTPException(status_code=500, detail=f"Failed to render evaluated answer sheet ({type(e).__name__}).")
    AC.audit(db, current_user, "EVALUATED_SHEET_DOWNLOADED", "booklet", booklet.id, {})
    db.commit()
    fname = f"evaluated_script_{booklet.course_code}_{booklet.register_number}.pdf"
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ─────────────────────────────────────────────────────────────
# 4. Student Grade Dispute & Re-evaluation Requests
# ─────────────────────────────────────────────────────────────

@router.post("/re-evaluation", summary="Submit re-evaluation dispute")
def submit_re_evaluation_request(
    data: ReEvaluationCreateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    booklet = db.query(StudentBooklet).filter(StudentBooklet.id == data.booklet_id).first()

    if not booklet:
        raise HTTPException(status_code=404, detail="Booklet not found")

    verify_student_access(current_user, booklet.student)
    if not booklet.is_published:
        raise HTTPException(status_code=409, detail="Re-evaluation can be requested only after the result is published.")
    target = None
    if data.question_number:
        target = next((a for a in booklet.answers if a.question_number == data.question_number), None)
        if not target:
            raise HTTPException(status_code=404, detail=f"Question {data.question_number} is not part of this answer sheet.")

    # Check for existing pending request for this question/booklet
    existing = db.query(ReEvaluationRequest).filter(
        ReEvaluationRequest.booklet_id == data.booklet_id,
        ReEvaluationRequest.question_number == data.question_number,
        ReEvaluationRequest.status == "pending"
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="A re-evaluation request is already pending for this item.")

    # Current marks of the disputed question (or of the whole paper)
    from backend.booklet_ocr import final_score
    curr_score = (final_score(target.submission.result) if target and target.submission else
                  (booklet.total_final_score or booklet.total_ai_score))

    req = ReEvaluationRequest(
        booklet_id=booklet.id,
        student_id=student.id,
        question_number=data.question_number,
        reason=data.reason.strip(),
        status="pending",
        score_before=curr_score
    )
    db.add(req)
    db.flush()
    AC.audit(db, current_user, "REEVALUATION_REQUESTED", "re_evaluation", req.id,
             {"booklet_id": booklet.id, "question_number": data.question_number})
    try:
        from backend.database import TeacherAssignment
        st = booklet.student
        course = db.query(Course).filter(Course.code == booklet.course_code).first()
        tas = db.query(TeacherAssignment).filter(TeacherAssignment.course_id == course.id).all() if course else []
        for ta in tas:
            if st is None or ta.batch_id == st.batch_id:
                AC.notify(db, ta.teacher.user_id, "reevaluation", f"Re-evaluation request: {booklet.course_code}",
                          f"{st.registration_number if st else ''} disputes Q{data.question_number or 'all'}: "
                          f"{data.reason.strip()[:200]}")
    except Exception:  # notification is best-effort
        pass
    db.commit()
    db.refresh(req)

    return {
        "status": "success",
        "request_id": req.id,
        "message": "Re-evaluation request submitted to Chief Examiner."
    }


@router.get("/re-evaluations", summary="List student re-evaluation history")
def list_student_re_evaluations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    verify_student_access(current_user, student)

    requests = db.query(ReEvaluationRequest).filter(
        ReEvaluationRequest.student_id == student.id
    ).order_by(ReEvaluationRequest.id.desc()).all()

    return [
        {
            "id": r.id,
            "course_code": r.booklet.course_code if r.booklet else "N/A",
            "question_number": r.question_number or "All Questions",
            "reason": r.reason,
            "status": r.status,
            "examiner_remarks": r.examiner_remarks,
            "score_before": r.score_before,
            "score_after": r.score_after,
            "submitted_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None
        }
        for r in requests
    ]


# ─────────────────────────────────────────────────────────────
# 5. Performance Trends & Analytics
# ─────────────────────────────────────────────────────────────

@router.get("/performance-trends", summary="Student historical performance trends")
def get_student_performance_trends(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    student = _get_student_for_user(current_user, db)
    verify_student_access(current_user, student)

    booklets = db.query(StudentBooklet).filter(
        StudentBooklet.student_id == student.id,
        StudentBooklet.is_published == True
    ).all()

    trends = []
    for b in booklets:
        score = b.total_final_score or b.total_ai_score
        trends.append({
            "exam_name": b.exam_name or b.course_code,
            "course_code": b.course_code,
            "score": score,
            "date": b.date_of_exam or (b.published_at.strftime("%Y-%m-%d") if b.published_at else "")
        })

    return {"student_name": student.name, "trends": trends}
