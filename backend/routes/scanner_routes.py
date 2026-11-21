"""
Answer-sheet scanner (prefix /scanner) — teachers & admin
=========================================================
One session = one student × subject × exam. Pages are captured (camera) or uploaded, auto-cropped,
perspective-corrected, enhanced and quality-checked, can be rotated / reordered / deleted, previewed
as a PDF, and finally submitted: the PDF is saved under the student's registration number and an
answer booklet is created for OCR + AI evaluation. Pages can never move between sessions.
"""

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import access as AC
from backend import scanner as SC
from backend.auth import require_role
from backend.booklet_ocr import create_booklet
from backend.database import ScanPage, ScanSession, Student, StudentBooklet, User, get_db

router = APIRouter()
STAFF = Depends(require_role(["admin", "faculty"]))
SCAN_ROOT = Path(os.getenv("SCAN_DIR", "uploads/scans"))
MAX_PAGE_MB = 15
MAX_PAGES = 60


class SessionIn(BaseModel):
    registration_number: str = Field(..., min_length=2, max_length=50)
    subject_code: str = Field(..., min_length=2, max_length=20)
    exam_name: str = Field(..., min_length=2, max_length=200)


class OrderIn(BaseModel):
    page_ids: List[int]


class RotateIn(BaseModel):
    degrees: int = Field(..., description="90, 180 or 270")


def _session_or_404(db: Session, user: User, sid: int) -> ScanSession:
    s = db.query(ScanSession).filter(ScanSession.id == sid).first()
    if not s or not AC.can_teach(db, user, s.student, s.course_code):
        raise HTTPException(status_code=404, detail="Scan session not found")
    return s


def _open(s: ScanSession):
    if s.status != "open":
        raise HTTPException(status_code=409, detail=f"This session is {s.status}; start a new one.")


def _page_dict(p: ScanPage) -> dict:
    return {"id": p.id, "position": p.position, "quality": json.loads(p.quality_json or "{}"),
            "image_url": f"/scanner/sessions/{p.session_id}/pages/{p.id}/image",
            "original_url": f"/scanner/sessions/{p.session_id}/pages/{p.id}/image?kind=original"}


def _session_dict(s: ScanSession) -> dict:
    st = s.student
    return {"id": s.id, "status": s.status, "registration_number": st.registration_number, "student_name": st.name,
            "batch_code": st.batch_rel.batch_code if st.batch_rel else "", "subject_code": s.course_code,
            "exam_name": s.exam_name, "booklet_id": s.booklet_id, "pdf_path": s.pdf_path,
            "created_at": s.created_at.isoformat() + "Z" if s.created_at else None,
            "pages": [_page_dict(p) for p in sorted(s.pages, key=lambda p: p.position)],
            "warnings": sum(1 for p in s.pages if not json.loads(p.quality_json or "{}").get("ok", True))}


def _renumber(s: ScanSession):
    for i, p in enumerate(sorted(s.pages, key=lambda p: p.position)):
        p.position = i + 1


@router.post("/sessions", summary="Start scanning one student's answer sheet")
def start_session(data: SessionIn, db: Session = Depends(get_db), user: User = STAFF):
    reg, subj = data.registration_number.strip().upper(), data.subject_code.strip().upper()
    st = db.query(Student).filter((func.upper(Student.registration_number) == reg) |
                                  (func.upper(Student.student_code) == reg)).first()
    if not st:
        raise HTTPException(status_code=404, detail=f"No student with registration number {reg}")
    AC.require_can_teach(db, user, st, subj)
    if db.query(StudentBooklet).filter(StudentBooklet.student_id == st.id, StudentBooklet.course_code == subj,
                                       StudentBooklet.exam_name == data.exam_name.strip()).first():
        raise HTTPException(status_code=409, detail=f"An answer sheet for {reg} / {subj} / {data.exam_name} already exists.")
    existing = db.query(ScanSession).filter(ScanSession.student_id == st.id, ScanSession.course_code == subj,
                                            ScanSession.exam_name == data.exam_name.strip(),
                                            ScanSession.status == "open").first()
    if existing:
        return {**_session_dict(existing), "resumed": True}
    s = ScanSession(created_by_id=user.id, student_id=st.id, course_code=subj, exam_name=data.exam_name.strip())
    db.add(s)
    db.flush()
    AC.audit(db, user, "SCAN_SESSION_START", "scan_session", s.id, {"student": reg, "subject": subj, "exam": data.exam_name})
    db.commit()
    db.refresh(s)
    return {**_session_dict(s), "resumed": False}


@router.get("/sessions", summary="My scan sessions")
def list_sessions(status: str = "open", db: Session = Depends(get_db), user: User = STAFF):
    q = db.query(ScanSession)
    if status != "all":
        q = q.filter(ScanSession.status == status)
    return [_session_dict(s) for s in q.order_by(ScanSession.id.desc()).limit(100).all()
            if AC.can_teach(db, user, s.student, s.course_code)]


@router.get("/sessions/{sid}", summary="Session with pages")
def get_session(sid: int, db: Session = Depends(get_db), user: User = STAFF):
    return _session_dict(_session_or_404(db, user, sid))


@router.post("/sessions/{sid}/pages", summary="Add a captured / uploaded page (JPG or PNG)")
async def add_page(sid: int, file: UploadFile = File(...), db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    if len(s.pages) >= MAX_PAGES:
        raise HTTPException(status_code=409, detail=f"A session can hold at most {MAX_PAGES} pages.")
    data = await file.read(MAX_PAGE_MB * 1024 * 1024 + 1)
    if len(data) > MAX_PAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_PAGE_MB} MB).")
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:4] in (b"RIFF",)):
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG photo/scan of the page.")
    digest = hashlib.sha256(data).hexdigest()
    dup = db.query(ScanPage).filter(ScanPage.sha256 == digest).first()
    if dup:
        if dup.session_id == s.id:
            raise HTTPException(status_code=409, detail=f"This exact image is already page {dup.position} of this session.")
        other = dup.session
        raise HTTPException(status_code=409, detail=(
            "This exact image was already scanned for another answer sheet "
            f"({other.student.registration_number} / {other.course_code}). Pages are never shared between students."))
    try:
        processed, q = await run_in_threadpool(SC.process_page, data)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read this image. Try another photo.")
    folder = SCAN_ROOT / f"session_{s.id}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex[:10]
    orig, proc = folder / f"{stem}_original.jpg", folder / f"{stem}_page.jpg"
    orig.write_bytes(data)
    proc.write_bytes(processed)
    p = ScanPage(session_id=s.id, position=len(s.pages) + 1, original_path=str(orig), processed_path=str(proc),
                 sha256=digest, quality_json=json.dumps(q))
    db.add(p)
    db.commit()
    db.refresh(p)
    return _page_dict(p)


@router.get("/sessions/{sid}/pages/{pid}/image", summary="Processed (or original) page image")
def page_image(sid: int, pid: int, kind: str = "processed", db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    p = next((x for x in s.pages if x.id == pid), None)
    if not p:
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(p.original_path if kind == "original" else p.processed_path, media_type="image/jpeg")


@router.post("/sessions/{sid}/pages/{pid}/rotate", summary="Rotate a page")
def rotate_page(sid: int, pid: int, data: RotateIn, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    if data.degrees % 360 not in (90, 180, 270):
        raise HTTPException(status_code=422, detail="Rotate by 90, 180 or 270 degrees.")
    p = next((x for x in s.pages if x.id == pid), None)
    if not p:
        raise HTTPException(status_code=404, detail="Page not found")
    Path(p.processed_path).write_bytes(SC.rotate_jpeg(Path(p.processed_path).read_bytes(), data.degrees))
    db.commit()
    return _page_dict(p)


@router.put("/sessions/{sid}/order", summary="Reorder pages")
def reorder(sid: int, data: OrderIn, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    ids = [p.id for p in s.pages]
    if sorted(ids) != sorted(data.page_ids):
        raise HTTPException(status_code=422, detail="Send every page id of this session exactly once.")
    pos = {pid: i + 1 for i, pid in enumerate(data.page_ids)}
    for p in s.pages:
        p.position = pos[p.id]
    db.commit()
    db.refresh(s)
    return _session_dict(s)


@router.delete("/sessions/{sid}/pages/{pid}", summary="Delete a page")
def delete_page(sid: int, pid: int, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    p = next((x for x in s.pages if x.id == pid), None)
    if not p:
        raise HTTPException(status_code=404, detail="Page not found")
    for f in (p.original_path, p.processed_path):
        try:
            os.remove(f)
        except OSError:
            pass
    s.pages.remove(p)
    db.delete(p)
    _renumber(s)
    db.commit()
    db.refresh(s)
    return _session_dict(s)


@router.get("/sessions/{sid}/preview.pdf", summary="Review the combined PDF before submitting")
def preview_pdf(sid: int, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    if s.pdf_path and os.path.exists(s.pdf_path):
        return FileResponse(s.pdf_path, media_type="application/pdf")
    pages = sorted(s.pages, key=lambda p: p.position)
    if not pages:
        raise HTTPException(status_code=409, detail="No pages yet.")
    import io
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        SC.images_to_pdf([p.processed_path for p in pages], tmp.name)
        data = Path(tmp.name).read_bytes()
    os.remove(tmp.name)
    return Response(data, media_type="application/pdf")


@router.post("/sessions/{sid}/submit", summary="Combine pages into one PDF and submit for evaluation")
def submit(sid: int, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    pages = sorted(s.pages, key=lambda p: p.position)
    if not pages:
        raise HTTPException(status_code=409, detail="Add at least one page before submitting.")
    st = s.student
    if db.query(StudentBooklet).filter(StudentBooklet.student_id == st.id, StudentBooklet.course_code == s.course_code,
                                       StudentBooklet.exam_name == s.exam_name).first():
        raise HTTPException(status_code=409, detail="An answer sheet for this student/subject/exam already exists.")
    out_dir = Path("uploads") / "answer_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", s.exam_name).strip("-")[:40]
    pdf = out_dir / f"{st.registration_number}_{s.course_code}_{slug}_{datetime.utcnow():%Y%m%d%H%M%S}.pdf"
    SC.images_to_pdf([p.processed_path for p in pages], str(pdf))
    booklet = create_booklet(db, st, s.course_code, s.exam_name, str(pdf), user, total_pages=len(pages))
    s.status, s.pdf_path, s.booklet_id, s.submitted_at = "submitted", str(pdf), booklet.id, datetime.utcnow()
    AC.audit(db, user, "SCAN_SUBMITTED", "scan_session", s.id,
             {"pages": len(pages), "booklet_id": booklet.id, "pdf": str(pdf)})
    db.commit()
    return {**_session_dict(s), "message": f"Saved {len(pages)} page(s) as {pdf.name}. Run AI evaluation from the review queue."}


@router.post("/sessions/{sid}/cancel", summary="Discard an open session")
def cancel(sid: int, db: Session = Depends(get_db), user: User = STAFF):
    s = _session_or_404(db, user, sid)
    _open(s)
    s.status = "cancelled"
    AC.audit(db, user, "SCAN_SESSION_CANCEL", "scan_session", s.id, {})
    db.commit()
    return {"status": "cancelled"}
