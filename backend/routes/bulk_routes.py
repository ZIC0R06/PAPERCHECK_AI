"""
AI Robot bulk examination checking (prefix /bulk) — Faculty & Admin
====================================================================
1. POST /bulk/jobs                       one batch = ONE question paper (subject + examination), reused for every sheet
2. POST /bulk/jobs/{id}/sheets           upload many answer sheets (PDF / JPG / PNG); each is identified to a student
   POST /bulk/jobs/{id}/scan             camera pages of ONE sheet → processed → combined PDF → identified
3. POST /bulk/items/{id}/assign          manual student verification for unclear sheets
4. POST /bulk/jobs/{id}/start            real background checking (OCR → per-question grading → totals)
   GET  /bulk/jobs/{id}                  live progress (current student / question, done / pending) — polled by the robot UI
5. GET  /bulk/jobs/{id}/report(.pdf)     class performance; /results.zip = checked sheets + feedback per student
6. POST /bulk/jobs/{id}/approve          teacher approval (same rules as Examiner Review: flagged answers must be verified)
   POST /bulk/jobs/{id}/publish          publish approved results to students
"""
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend import access as AC
from backend import bulk_evaluation as BE
from backend.auth import require_role
from backend.database import EvaluationJob, EvaluationJobItem, ExamPaper, Student, User, get_db

router = APIRouter()
STAFF = Depends(require_role(["admin", "faculty"]))


class JobIn(BaseModel):
    exam_paper_id: Optional[int] = None
    course_code: Optional[str] = Field(None, max_length=20)
    exam_name: Optional[str] = Field(None, max_length=200)
    title: Optional[str] = Field("", max_length=200)


class AssignIn(BaseModel):
    registration_number: str = Field(..., min_length=2, max_length=50)


class BookletsIn(BaseModel):
    booklet_ids: Optional[List[int]] = None


def _item_or_404(db: Session, user: User, item_id: int) -> EvaluationJobItem:
    it = db.query(EvaluationJobItem).filter(EvaluationJobItem.id == item_id).first()
    if not it:
        raise HTTPException(status_code=404, detail="Answer sheet not found")
    BE.job_or_404(db, user, it.job_id)
    return it


@router.get("/papers", summary="Question papers usable for bulk checking (with marking readiness)")
def papers(db: Session = Depends(get_db), user: User = STAFF):
    out = []
    for p in db.query(ExamPaper).order_by(ExamPaper.id.desc()).all():
        if not BE.can_manage_course(db, user, p.course_code or ""):
            continue
        rd = BE.paper_readiness(db, p)
        rd.pop("detail", None)
        out.append({**rd, "course_name": p.course_name, "label": f"{p.course_code} · {p.exam_name or '(no exam name)'}"})
    return out


@router.get("/papers/{pid}/readiness", summary="Which questions can be marked")
def readiness(pid: int, db: Session = Depends(get_db), user: User = STAFF):
    p = db.query(ExamPaper).filter(ExamPaper.id == pid).first()
    if not p or not BE.can_manage_course(db, user, p.course_code or ""):
        raise HTTPException(status_code=404, detail="Question paper not found")
    return BE.paper_readiness(db, p)


@router.post("/jobs", summary="Create a checking batch for ONE question paper")
def create_job(data: JobIn, db: Session = Depends(get_db), user: User = STAFF):
    p = None
    if data.exam_paper_id:
        p = db.query(ExamPaper).filter(ExamPaper.id == data.exam_paper_id).first()
    elif data.course_code and data.exam_name:
        from backend.booklet_ocr import find_exam_paper
        p = find_exam_paper(db, data.course_code, data.exam_name)
    if not p:
        raise HTTPException(status_code=404, detail=(
            "Question paper not found. Upload the official question paper for this subject + examination once "
            "(Question Papers & Model Answers), then create the checking batch."))
    if not BE.can_manage_course(db, user, p.course_code or ""):
        raise HTTPException(status_code=403, detail=f"You are not assigned to {p.course_code}.")
    if not (p.exam_name or "").strip():
        raise HTTPException(status_code=422, detail="Give the question paper an examination name first.")
    rd = BE.paper_readiness(db, p)
    if not rd["questions"]:
        raise HTTPException(status_code=422, detail="The question paper has no questions yet.")
    j = EvaluationJob(created_by_id=user.id, exam_paper_id=p.id, course_code=(p.course_code or "").upper(),
                      exam_name=p.exam_name, title=(data.title or "").strip(), status="draft",
                      message="Add answer sheets (upload or camera), then start checking.")
    db.add(j)
    db.flush()
    AC.audit(db, user, "BULK_JOB_CREATED", "evaluation_job", j.id, {"paper": p.id, "course": j.course_code, "exam": j.exam_name})
    db.commit()
    db.refresh(j)
    return {**BE.job_dict(j), "readiness": rd}


@router.get("/jobs", summary="My checking batches (admin: all)")
def list_jobs(db: Session = Depends(get_db), user: User = STAFF):
    q = db.query(EvaluationJob)
    if not AC.is_admin(user):
        q = q.filter(EvaluationJob.created_by_id == user.id)
    return [BE.job_dict(j, items=False) for j in q.order_by(EvaluationJob.id.desc()).limit(100).all()
            if AC.is_admin(user) or BE.can_manage_course(db, user, j.course_code)]


@router.get("/jobs/{jid}", summary="Batch with every sheet and live progress")
def get_job(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    return {**BE.job_dict(j), "worker_alive": BE.is_running(j.id)}


@router.get("/jobs/{jid}/students", summary="Students who may be assigned to sheets of this batch")
def job_students(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    taken = {i.student_id for i in j.items if i.student_id}
    return [{"registration_number": s.registration_number, "name": s.name, "batch": s.batch, "assigned": s.id in taken}
            for s in sorted(BE.eligible_students(db, user, j.course_code), key=lambda s: s.registration_number or "")]


@router.post("/jobs/{jid}/sheets", summary="Upload answer sheets (PDF / JPG / PNG, many at once)")
async def upload_sheets(jid: int, files: List[UploadFile] = File(...), db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    added, rejected = [], []
    for f in files:
        data = await f.read(BE.MAX_FILE_MB * 1024 * 1024 + 1)
        try:
            it = await run_in_threadpool(BE.add_sheet, db, user, j, f.filename or "sheet", data, "upload")
            added.append(BE.item_dict(it))
        except BE.SheetError as e:
            db.rollback()
            rejected.append({"filename": f.filename, "error": str(e), "status": e.status})
    db.refresh(j)
    return {"added": added, "rejected": rejected, "job": BE.job_dict(j, items=False)}


@router.post("/jobs/{jid}/scan", summary="Camera scan of ONE answer sheet: pages in order → combined PDF")
async def scan_sheet(jid: int, pages: List[UploadFile] = File(...), rotations: str = Form("[]"),
                     registration_number: str = Form(""), db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    if len(pages) > 60:
        raise HTTPException(status_code=409, detail="At most 60 pages per answer sheet.")
    imgs = []
    for p in pages:
        d = await p.read(15 * 1024 * 1024 + 1)
        if len(d) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="A page image is larger than 15 MB.")
        if not (d[:3] == b"\xff\xd8\xff" or d[:8] == b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=415, detail="Camera pages must be JPG or PNG images.")
        imgs.append(d)
    try:
        rots = [int(x) for x in json.loads(rotations or "[]")]
    except Exception:
        rots = []
    try:
        pdf, n = await run_in_threadpool(BE.scan_to_pdf, imgs, rots)
    except BE.SheetError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process the captured pages. Retake them with better light.")
    name = f"camera_scan_{(registration_number or 'sheet').strip().upper()}_{len(j.items) + 1}.pdf"
    try:
        it = await run_in_threadpool(BE.add_sheet, db, user, j, name, pdf, "camera", registration_number, pdf, n)
    except BE.SheetError as e:
        db.rollback()
        raise HTTPException(status_code=e.status, detail=str(e))
    return {"item": BE.item_dict(it), "job": BE.job_dict(j, items=False)}


@router.post("/items/{iid}/assign", summary="Manual verification: assign the sheet to a student")
def assign(iid: int, data: AssignIn, db: Session = Depends(get_db), user: User = STAFF):
    it = _item_or_404(db, user, iid)
    j = it.job
    if it.status not in ("needs_identification", "ready"):
        raise HTTPException(status_code=409, detail=f"This sheet is already {it.status}; its student can't be changed here.")
    reg = BE.norm_reg(data.registration_number)
    st = next((s for s in BE.eligible_students(db, user, j.course_code)
               if BE.norm_reg(s.registration_number) == reg or BE.norm_reg(s.student_code) == reg), None)
    if not st:
        raise HTTPException(status_code=404, detail=f"{reg} is not a student you can check for {j.course_code}.")
    why = BE._conflict_reason(db, j, st, it.id)
    if why:
        raise HTTPException(status_code=409, detail=why)
    old = it.student.registration_number if it.student else None
    detail = json.loads(it.id_detail or "{}")
    detail.pop("problem", None)
    detail["verified_by"] = user.username
    it.student_id, it.id_method, it.status, it.id_detail = st.id, "manual", "ready", json.dumps(detail)
    BE._event(it, f"Verified manually by {user.username} as {st.registration_number} ({st.name}).")
    AC.audit(db, user, "BULK_SHEET_IDENTIFIED", "evaluation_job_item", it.id,
             {"job": j.id, "from": old, "to": st.registration_number})
    db.commit()
    return BE.item_dict(it)


@router.delete("/items/{iid}", summary="Remove a sheet that has not been checked")
def remove_item(iid: int, db: Session = Depends(get_db), user: User = STAFF):
    it = _item_or_404(db, user, iid)
    if it.job.status in BE.ACTIVE:
        raise HTTPException(status_code=409, detail="Checking is running; cancel it first.")
    if it.booklet_id:
        raise HTTPException(status_code=409, detail="This sheet was already checked; manage it in Examiner Review.")
    AC.audit(db, user, "BULK_SHEET_REMOVED", "evaluation_job_item", it.id, {"job": it.job_id, "file": it.filename})
    db.delete(it)              # files stay on disk (originals are preserved)
    db.commit()
    return {"status": "removed"}


@router.get("/items/{iid}/original", summary="The original uploaded file, unchanged")
def original(iid: int, db: Session = Depends(get_db), user: User = STAFF):
    it = _item_or_404(db, user, iid)
    return FileResponse(it.original_path, filename=it.filename or "answer_sheet")


@router.get("/items/{iid}/page1.png", summary="First page preview (for verifying the student)")
def page1(iid: int, db: Session = Depends(get_db), user: User = STAFF):
    from backend.evaluated_sheet_generator import render_page_png
    it = _item_or_404(db, user, iid)
    png = render_page_png(it.pdf_path, 1, 0.0, 1.0, dpi=70)
    if png is None:
        raise HTTPException(status_code=404, detail="Page not available")
    return Response(png, media_type="image/png")


@router.get("/items/{iid}/evaluated.pdf", summary="Checked answer sheet with marks beside each answer")
def evaluated(iid: int, db: Session = Depends(get_db), user: User = STAFF):
    from backend.evaluated_sheet_generator import build_evaluated_pdf_for_booklet
    it = _item_or_404(db, user, iid)
    if not it.booklet or not it.booklet.answers:
        raise HTTPException(status_code=409, detail="This sheet has not been checked yet.")
    reg = it.student.registration_number if it.student else f"sheet{it.id}"
    return Response(build_evaluated_pdf_for_booklet(it.booklet), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{reg}_{it.job.course_code}_checked.pdf"'})


@router.post("/jobs/{jid}/start", summary="Start the robot: check every identified sheet in the background")
def start(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    return BE.start_job(db, user, j)


@router.post("/jobs/{jid}/cancel", summary="Stop after the current question")
def cancel(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    if j.status not in BE.ACTIVE:
        return {"status": j.status}
    j.status, j.message = "cancelling", "Stopping after the current question…"
    AC.audit(db, user, "BULK_CHECKING_CANCELLED", "evaluation_job", j.id, {})
    db.commit()
    return {"status": "cancelling"}


@router.get("/jobs/{jid}/report", summary="Class performance report (JSON)")
def report(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    return BE.class_report(db, BE.job_or_404(db, user, jid))


@router.get("/jobs/{jid}/report.pdf", summary="Class performance report (PDF)")
def report_pdf(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    return Response(BE.class_report_pdf(db, j), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="class_report_{j.course_code}_{j.id}.pdf"'})


@router.get("/jobs/{jid}/results.zip", summary="Checked answer sheets + feedback reports of every student")
def results_zip(jid: int, db: Session = Depends(get_db), user: User = STAFF):
    j = BE.job_or_404(db, user, jid)
    return Response(BE.results_zip(db, j), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="checked_sheets_{j.course_code}_{j.id}.zip"'})


def _per_booklet(db, user, j, ids, fn):
    out = []
    for it in j.items:
        if not it.booklet_id or (ids and it.booklet_id not in ids):
            continue
        try:
            r = fn(it.booklet_id, db=db, examiner=user)
            out.append({"item_id": it.id, "booklet_id": it.booklet_id, "ok": True, "result": r})
        except HTTPException as e:
            db.rollback()
            out.append({"item_id": it.id, "booklet_id": it.booklet_id, "ok": False, "error": e.detail})
    return out


@router.post("/jobs/{jid}/approve", summary="Teacher approval of checked sheets (flagged answers must be verified first)")
def approve(jid: int, data: BookletsIn, db: Session = Depends(get_db), user: User = STAFF):
    from backend.routes.examiner_routes import approve_booklet
    j = BE.job_or_404(db, user, jid)
    return {"results": _per_booklet(db, user, j, set(data.booklet_ids or []), approve_booklet)}


@router.post("/jobs/{jid}/publish", summary="Publish approved results to students")
def publish(jid: int, data: BookletsIn, db: Session = Depends(get_db), user: User = STAFF):
    from backend.routes.examiner_routes import publish_booklet
    j = BE.job_or_404(db, user, jid)
    return {"results": _per_booklet(db, user, j, set(data.booklet_ids or []), publish_booklet)}
