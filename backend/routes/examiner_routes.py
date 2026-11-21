"""
AI Paper Checking — Faculty and Examiner Dashboard Routes
==========================================================
Empowers university faculty and examiners to:
  • Review AI-generated grades and deduction reasoning
  • Override marks and edit feedback before finalisation
  • Flag uncertain evaluations or poor OCR for manual verification
  • Authorize and publish official exam results
  • Resolve student grade disputes and re-evaluation requests
  • Maintain an immutable audit log of all grading modifications
"""

import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import uuid
import re
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from backend.database import (
    get_db, User, Student, StudentBooklet, StudentAnswerText,
    Submission, Result, ReEvaluationRequest, AuditLog, Course, CourseMaterial, ExamQuestion
)
from backend.auth import require_role
from backend import access as AC
import logging
logger = logging.getLogger(__name__)
from backend.schemas import ExaminerReviewSchema, ReEvaluationResolveSchema, AnswerTextUpdateSchema
from backend.evaluated_sheet_generator import generate_evaluated_sheet_pdf
from backend.llm_evaluator import LLMEvaluator

router = APIRouter(dependencies=[Depends(require_role(["faculty", "admin"]))])


# ─────────────────────────────────────────────────────────────
# 1. Examiner Booklet Review Queue
# ─────────────────────────────────────────────────────────────

@router.get("/booklets", summary="Examiner review queue")
def list_booklets_for_review(
    approval_status: Optional[str] = None,
    course_code: Optional[str] = None,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    query = AC.scoped_booklets(db, examiner)
    if approval_status:
        query = query.filter(StudentBooklet.approval_status == approval_status)
    if course_code:
        query = query.filter(StudentBooklet.course_code == course_code)

    booklets = query.order_by(StudentBooklet.id.desc()).all()

    return [
        {
            "id": b.id,
            "student_name": b.student.name if b.student else "Unassigned",
            "registration_number": b.register_number or (b.student.registration_number if b.student else "N/A"),
            "course_code": b.course_code,
            "course_name": b.course_name,
            "exam_name": b.exam_name,
            "total_ai_score": b.total_ai_score,
            "total_final_score": b.total_final_score,
            "approval_status": b.approval_status,
            "is_published": b.is_published,
            "answers_count": len(b.answers),
            "parsed_at": b.parsed_at.isoformat() if b.parsed_at else None
        }
        for b in booklets
    ]


@router.get("/booklet/{booklet_id}", summary="Detailed booklet evaluation for examiner review")
def get_booklet_review_details(booklet_id: int, db: Session = Depends(get_db),
                               examiner: User = Depends(require_role(["faculty", "admin"]))):
    from backend.booklet_ocr import question_evaluations, question_payload
    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    qe = question_evaluations(booklet)
    by_id = {a.id: a for a in booklet.answers}
    questions_data = []
    for q in qe:
        ans = by_id[q["answer_id"]]
        ref = question_payload(db, booklet, ans)
        questions_data.append({
            **q,
            # names kept for the existing UI
            "student_answer_text": q["student_answer"], "raw_ocr_text": ans.raw_ocr_text or "",
            "expected_answer": ref["model_answer"] if not ref.get("reference_note", "").startswith("Evaluated against course notes") else "",
            "marking_criteria": ref["rubric"], "correct_option": ref["correct_option"],
            "part_name": ans.part_name, "ai_score": q["ai_score"] if q["ai_score"] is not None else 0.0,
            "examiner_score": ans.submission.result.examiner_score if ans.submission and ans.submission.result else None,
            "examiner_comments": q["teacher_comment"], "examiner_approved": q["teacher_verified"],
            "flagged_for_review": q["flagged"], "confidence": q["confidence"] or 0.0,
        })
    max_total = sum(q["max_marks"] for q in qe)
    try:
        ocr_report = json.loads(booklet.ocr_report or "{}")
    except Exception:
        ocr_report = {}
    return {
        "booklet_id": booklet.id,
        "student": {
            "name": booklet.student.name if booklet.student else "Student",
            "registration_number": booklet.register_number,
            "roll_number": booklet.roll_number,
            "department": booklet.department,
            "programme": booklet.programme,
            "semester": booklet.semester,
        },
        "course_code": booklet.course_code,
        "course_name": booklet.course_name,
        "exam_name": booklet.exam_name,
        "exam_paper_id": booklet.exam_paper_id,
        "total_pages": booklet.total_pages,
        "ocr_engine": booklet.ocr_engine or "",
        "ocr_report": ocr_report,
        "approval_status": booklet.approval_status,
        "is_published": booklet.is_published,
        "total_ai_score": round(booklet.total_ai_score or 0, 2),
        "total_final_score": round(sum((q["awarded"] or 0) for q in qe), 2),
        "max_total_marks": round(max_total, 2),
        "questions": questions_data
    }


# ─────────────────────────────────────────────────────────────
# 2. Examiner Marks Editing & Uncertainty Flagging
# ─────────────────────────────────────────────────────────────

@router.post("/review-question", summary="Examiner edit marks and comments")
def review_question_result(
    data: ExaminerReviewSchema,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    result = db.query(Result).filter(Result.id == data.result_id).first()
    ans_row = db.query(StudentAnswerText).filter(StudentAnswerText.submission_id == result.submission_id).first() if result else None
    if not result or not ans_row or not AC.can_access_booklet(db, examiner, ans_row.booklet):
        raise HTTPException(status_code=404, detail="Result record not found")
    if ans_row.booklet.is_published:
        raise HTTPException(status_code=409, detail="This result is already published. Use the re-evaluation process to change it.")
    max_m = result.max_marks or 10.0
    if data.examiner_score < 0 or data.examiner_score > max_m:
        raise HTTPException(status_code=422, detail=f"Marks must be between 0 and {max_m:g}.")
    old_score = result.examiner_score if result.examiner_score is not None else result.final_score
    if abs((old_score or 0) - data.examiner_score) > 1e-9 and not (data.examiner_comments or "").strip():
        raise HTTPException(status_code=422, detail="Please record a reason when changing the AI-awarded marks.")

    result.examiner_score = data.examiner_score
    result.examiner_comments = data.examiner_comments
    result.examiner_approved = True
    result.flagged_for_review = data.flag_for_review
    result.flag_reason = data.flag_reason or ""
    result.approved_by_id = examiner.id
    result.approved_at = datetime.utcnow()
    from backend.booklet_ocr import recompute_totals
    recompute_totals(ans_row.booklet)
    if ans_row.booklet.approval_status in ("pending", "approved"):
        ans_row.booklet.approval_status = "in_review"   # a change after approval needs re-approval

    # Log audit trail
    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="MARKS_MODIFICATION",
        target_type="result",
        target_id=str(result.id),
        details=json.dumps({
            "previous_score": old_score,
            "ai_score": result.final_score,
            "new_score": data.examiner_score,
            "booklet_id": ans_row.booklet_id,
            "question_number": ans_row.question_number,
            "examiner_comments": data.examiner_comments,
            "flagged": data.flag_for_review
        })
    )
    db.add(audit)
    db.commit()

    return {
        "status": "success",
        "result_id": result.id,
        "examiner_score": result.examiner_score,
        "examiner_approved": result.examiner_approved
    }


@router.patch("/booklet/{booklet_id}/answer/{answer_id}", summary="Correct OCR student answer text and re-evaluate")
def update_answer_ocr_text(
    booklet_id: int,
    answer_id: int,
    data: AnswerTextUpdateSchema,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    """
    Allows faculty to inspect and correct extracted OCR handwriting text,
    re-running evaluation on the corrected text.
    """
    AC.booklet_or_404(db, examiner, booklet_id)
    ans = db.query(StudentAnswerText).filter(
        StudentAnswerText.id == answer_id,
        StudentAnswerText.booklet_id == booklet_id
    ).first()
    if not ans:
        raise HTTPException(status_code=404, detail="Student answer record not found")
    if ans.booklet.is_published:
        raise HTTPException(status_code=409, detail="This result is already published.")

    from backend.booklet_ocr import grade_row, recompute_totals
    old_text = ans.answer_text
    ans.answer_text = data.corrected_text
    ans.ocr_confidence = 1.0            # typed / checked by the teacher
    ans.ocr_engine = "teacher-corrected"
    try:
        res = grade_row(db, ans.booklet, ans)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if res is None:
        db.rollback()
        raise HTTPException(status_code=422, detail="No model answer or marking scheme exists for this question, so it "
                                                    "can't be evaluated automatically. Add the answer key first.")
    db.flush()
    booklet = ans.booklet
    recompute_totals(booklet)
    if booklet.approval_status == "approved":
        booklet.approval_status = "in_review"

    class _E:  # keep the response/audit shape
        score = res.final_score
    eval_res = _E()

    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="OCR_TEXT_CORRECTED",
        target_type="student_answer_text",
        target_id=str(ans.id),
        details=json.dumps({"question_number": ans.question_number, "old_length": len(old_text or ""), "new_length": len(data.corrected_text), "new_ai_score": eval_res.score})
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "answer_id": ans.id, "new_ai_score": eval_res.score}


@router.post("/evaluate-booklet/{booklet_id}", summary="Read the handwriting (if needed) and AI-evaluate every answer")
def evaluate_booklet_by_examiner(
    booklet_id: int,
    reread: bool = False,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    """
    * Reads the sheet if it has no answer text yet (or `reread=true`): vision AI when a key is configured,
      otherwise local OCR (flagged for verification). One answer row per question of the official paper.
    * Marks every answer against the official question, model answer and marking scheme with a
      criterion-wise breakdown, deductions and corrections. Unanswered questions get 0 ("not attempted").
    * Questions with no model answer / marking scheme are NOT scored — they are listed in `skipped_questions`.
    """
    from backend.booklet_ocr import ensure_answers_extracted, grade_booklet

    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    if booklet.is_published:
        raise HTTPException(status_code=409, detail="Published results can't be re-evaluated here; use re-evaluation.")

    extracted = ensure_answers_extracted(db, booklet, force=reread)
    answered = [a for a in booklet.answers if (a.answer_text or "").strip()]
    if not answered:
        detail = ("No answer text could be read from this answer sheet "
                  f"({extracted.get('reason', 'no handwriting could be read')}). Re-scan the sheet with better "
                  "lighting, or type the answers with 'Correct OCR text' before evaluating.")
        if not booklet.exam_paper_id:
            detail += (f" Also, no question paper is linked to {booklet.course_code} / {booklet.exam_name}: "
                       "set up the question paper first so answers can be matched to its questions.")
        raise HTTPException(status_code=422, detail=detail)

    g = grade_booklet(db, booklet)
    if g["max_marks"] == 0:
        db.commit()
        raise HTTPException(status_code=422, detail=(
            "None of the questions could be evaluated: add the question paper with model answers / marking "
            f"scheme for {booklet.course_code or 'this course'} first. Questions without a reference: {g['skipped']}"))

    booklet.approval_status = "in_review"
    AC.audit(db, examiner, "AI_EVALUATION_RUN", "booklet", booklet.id,
             {"total_ai_score": booklet.total_ai_score, "max": g["max_marks"], "skipped_questions": g["skipped"],
              "flagged": g["flagged"], "grader": g["grader"], "ocr": {k: v for k, v in extracted.items() if k != "warnings"}})
    AC.notify_student(db, booklet.student, "sheet_evaluated", f"{booklet.course_code}: answer sheet evaluated",
                      "Your answer sheet has been evaluated by AI and is awaiting teacher verification.")
    db.commit()

    return {
        "status": "success",
        "booklet_id": booklet.id,
        "total_ai_score": booklet.total_ai_score,
        "max_marks": g["max_marks"],
        "approval_status": booklet.approval_status,
        "skipped_questions": g["skipped"],
        "flagged_questions": g["flagged"],
        "grader": g["grader"],
        "reader": booklet.ocr_engine,
        "ocr": extracted,
    }


@router.get("/booklet/{booklet_id}/page/{page_no}.png", summary="One page of the scanned sheet (optionally cropped to an answer)")
def booklet_page_image(booklet_id: int, page_no: int, y0: float = 0.0, y1: float = 1.0,
                       db: Session = Depends(get_db), examiner: User = Depends(require_role(["faculty", "admin"]))):
    from fastapi.responses import Response
    from backend.evaluated_sheet_generator import render_page_png
    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    png = render_page_png(booklet.pdf_path, page_no, y0, y1)
    if png is None:
        raise HTTPException(status_code=404, detail="Page not found")
    return Response(png, media_type="image/png")


@router.get("/booklet/{booklet_id}/evaluated.pdf", summary="Preview the annotated evaluated answer sheet (teacher)")
def booklet_evaluated_preview(booklet_id: int, db: Session = Depends(get_db),
                              examiner: User = Depends(require_role(["faculty", "admin"]))):
    from fastapi.responses import Response
    from backend.evaluated_sheet_generator import build_evaluated_pdf_for_booklet
    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    if not booklet.answers:
        raise HTTPException(status_code=409, detail="Run the AI evaluation first.")
    pdf = build_evaluated_pdf_for_booklet(booklet)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="evaluated_preview_{booklet_id}.pdf"'})


@router.post("/booklet/upload", summary="Faculty upload single student answer booklet")
async def faculty_upload_booklet(
    file: UploadFile = File(...),
    student_registration: str = Form(...),
    course_code: str = Form(...),
    exam_name: str = Form("End Semester Examination 2026"),
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, and PNG files are accepted.")

    reg = student_registration.strip().upper()
    student = db.query(Student).filter(
        (Student.registration_number.ilike(reg)) | (Student.student_code.ilike(reg))
    ).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"No registered student found with registration number: {reg}")
    c_code = course_code.strip().upper()
    AC.require_can_teach(db, examiner, student, c_code)

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB).")
    head = content[:8]
    if (ext == ".pdf" and not content[:1024].lstrip().startswith(b"%PDF")) or \
       (ext == ".png" and head != b"\x89PNG\r\n\x1a\n") or (ext in (".jpg", ".jpeg") and not head.startswith(b"\xff\xd8\xff")):
        raise HTTPException(status_code=422, detail="The file content does not match its extension.")
    dup = db.query(StudentBooklet).filter(StudentBooklet.student_id == student.id, StudentBooklet.course_code == c_code,
                                         StudentBooklet.exam_name == exam_name).first()
    if dup:
        raise HTTPException(status_code=409, detail=f"An answer sheet for {reg} / {c_code} / {exam_name} already exists (booklet #{dup.id}).")

    upload_dir = Path("uploads") / "answer_sheets"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{student.registration_number}_{c_code}_{uuid.uuid4().hex[:6]}{ext}"
    with open(save_path, "wb") as f:
        f.write(content)
    if ext != ".pdf":  # store every answer sheet as PDF
        from PIL import Image
        pdf_path = save_path.with_suffix(".pdf")
        Image.open(save_path).convert("RGB").save(pdf_path, "PDF", resolution=200.0)
        save_path = pdf_path

    from backend.booklet_ocr import create_booklet
    booklet = create_booklet(db, student, c_code, exam_name, str(save_path), examiner)
    return {
        "status": "success",
        "booklet_id": booklet.id,
        "student_name": student.name,
        "registration_number": student.registration_number,
        "course_code": booklet.course_code,
        "answers_found": len(booklet.answers),
        "message": "Uploaded. Run AI evaluation to read and score the answers.",
    }


@router.post("/save-draft/{booklet_id}", summary="Save evaluation draft without publishing")
def save_evaluation_draft(
    booklet_id: int,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    booklet = AC.booklet_or_404(db, examiner, booklet_id)

    booklet.approval_status = "in_review"
    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="DRAFT_SAVED",
        target_type="booklet",
        target_id=str(booklet.id),
        details=json.dumps({"status": "draft_saved"})
    )
    db.add(audit)
    db.commit()
    return {"status": "success", "approval_status": "in_review", "message": "Evaluation draft saved."}


# ─────────────────────────────────────────────────────────────
# 3. Booklet Approval & Official Publication
# ─────────────────────────────────────────────────────────────

@router.post("/approve-booklet/{booklet_id}", summary="Examiner approve full booklet evaluation")
def approve_booklet(
    booklet_id: int,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    if booklet.is_published:
        raise HTTPException(status_code=409, detail="Already published.")
    unevaluated = [a.question_number for a in booklet.answers if not (a.submission and a.submission.result)]
    if not booklet.answers or unevaluated:
        raise HTTPException(status_code=409, detail=f"Every answer must be evaluated before approval. Not evaluated: {unevaluated or 'all'}")
    unresolved = [a.question_number for a in booklet.answers
                  if a.submission.result.flagged_for_review and not a.submission.result.examiner_approved]
    if unresolved:
        raise HTTPException(status_code=409, detail=(
            f"Question(s) {unresolved} are flagged for verification (e.g. handwriting read by local OCR, low confidence). "
            "Open each one, check it against the original page and save your marks before approving."))

    total_final = 0.0
    for ans in booklet.answers:
        if ans.submission and ans.submission.result:
            r = ans.submission.result
            score = r.examiner_score if r.examiner_approved and r.examiner_score is not None else r.final_score
            total_final += (score or 0.0)

    booklet.total_final_score = round(total_final, 2)
    booklet.approval_status = "approved"

    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="BOOKLET_APPROVED",
        target_type="booklet",
        target_id=str(booklet.id),
        details=json.dumps({"total_final_score": booklet.total_final_score})
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "approval_status": "approved", "total_final_score": booklet.total_final_score}


@router.post("/publish-booklet/{booklet_id}", summary="Publish verified exam result to student portal")
def publish_booklet(
    booklet_id: int,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    booklet = AC.booklet_or_404(db, examiner, booklet_id)
    if booklet.approval_status != "approved":
        raise HTTPException(status_code=409, detail="A teacher must review and approve the marks before the result can be published.")

    booklet.is_published = True
    booklet.published_at = datetime.utcnow()
    booklet.published_by_id = examiner.id
    booklet.approval_status = "published"

    from backend.booklet_ocr import recompute_totals
    recompute_totals(booklet)
    db.flush()

    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="RESULT_PUBLISHED",
        target_type="booklet",
        target_id=str(booklet.id),
        details=json.dumps({"published_at": booklet.published_at.isoformat()})
    )
    db.add(audit)
    AC.notify_student(db, booklet.student, "result_published", f"{booklet.course_code}: result published",
                      f"Your result for {booklet.exam_name} is now available.")
    db.commit()

    return {"status": "success", "is_published": True, "message": "Result officially published to Student Portal."}


# ─────────────────────────────────────────────────────────────
# 4. Student Grade Dispute / Re-evaluation Resolution
# ─────────────────────────────────────────────────────────────

@router.get("/re-evaluations", summary="List pending re-evaluation requests")
def list_re_evaluations(db: Session = Depends(get_db),
                        examiner: User = Depends(require_role(["faculty", "admin"]))):
    allowed = {b.id for b in AC.scoped_booklets(db, examiner).all()}
    requests = [r for r in db.query(ReEvaluationRequest).order_by(ReEvaluationRequest.id.desc()).all()
                if r.booklet_id in allowed]
    return [
        {
            "id": r.id,
            "booklet_id": r.booklet_id,
            "student_name": r.student.name if r.student else "Student",
            "registration_number": r.student.registration_number if r.student else "N/A",
            "course_code": r.booklet.course_code if r.booklet else "N/A",
            "question_number": r.question_number,
            "reason": r.reason,
            "status": r.status,
            "examiner_remarks": r.examiner_remarks,
            "score_before": r.score_before,
            "score_after": r.score_after,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in requests
    ]


@router.post("/resolve-reevaluation", summary="Resolve re-evaluation request")
def resolve_re_evaluation(
    data: ReEvaluationResolveSchema,
    db: Session = Depends(get_db),
    examiner: User = Depends(require_role(["faculty", "admin"]))
):
    req = db.query(ReEvaluationRequest).filter(ReEvaluationRequest.id == data.request_id).first()
    if not req or not AC.can_access_booklet(db, examiner, req.booklet):
        raise HTTPException(status_code=404, detail="Re-evaluation request not found")
    if req.status not in ("pending", "in_review"):
        raise HTTPException(status_code=409, detail=f"This request is already {req.status}.")
    if data.status not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="Decision must be 'approved' or 'rejected'.")
    if not (data.examiner_remarks or "").strip():
        raise HTTPException(status_code=422, detail="Please record the reason for your decision.")
    target = None
    if req.question_number and req.booklet:
        target = db.query(StudentAnswerText).filter(StudentAnswerText.booklet_id == req.booklet_id,
                                                    StudentAnswerText.question_number == req.question_number).first()
    if data.status == "approved" and data.score_after is not None:
        if not (target and target.submission and target.submission.result):
            raise HTTPException(status_code=422, detail="Approving with new marks needs a specific evaluated question.")
        mx = target.submission.result.max_marks or 0
        if data.score_after < 0 or data.score_after > mx:
            raise HTTPException(status_code=422, detail=f"New marks must be between 0 and {mx:g}.")

    req.status = data.status  # "approved" or "rejected"
    req.examiner_remarks = data.examiner_remarks
    req.score_after = data.score_after
    req.resolved_by_id = examiner.id
    req.updated_at = datetime.utcnow()

    # If score changed, update booklet & result
    if data.status == "approved" and data.score_after is not None:
        if req.question_number and req.booklet:
            ans = db.query(StudentAnswerText).filter(
                StudentAnswerText.booklet_id == req.booklet_id,
                StudentAnswerText.question_number == req.question_number
            ).first()
            if ans and ans.submission and ans.submission.result:
                r = ans.submission.result
                r.examiner_score = data.score_after
                r.examiner_approved = True
                r.examiner_comments = ((r.examiner_comments or "") + f" [Re-evaluation: {data.examiner_remarks}]").strip()
                r.approved_by_id, r.approved_at = examiner.id, datetime.utcnow()

        # Re-tally booklet final score; refresh the consolidated (OMR + written) result if one exists
        if req.booklet:
            from backend.booklet_ocr import recompute_totals
            recompute_totals(req.booklet)
            try:
                from backend.results_service import refresh_final_after_reevaluation
                refresh_final_after_reevaluation(db, req.booklet)
            except ImportError:
                pass

    audit = AuditLog(
        user_id=examiner.id,
        username=examiner.username,
        role=examiner.role,
        action="REEVALUATION_RESOLVED",
        target_type="re_evaluation",
        target_id=str(req.id),
        details=json.dumps({"status": data.status, "remarks": data.examiner_remarks, "score_after": data.score_after})
    )
    db.add(audit)
    AC.notify_student(db, req.student, "feedback", "Re-evaluation decision available",
                      f"Your re-evaluation request was {data.status}. Remarks: {data.examiner_remarks}")
    db.commit()

    return {"status": "success", "re_evaluation_status": req.status}
