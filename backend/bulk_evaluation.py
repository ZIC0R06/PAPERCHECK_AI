"""
AI Robot bulk examination checking
==================================
Faculty / Admin upload ONE official question paper (with answer key / marking scheme) for a subject +
examination, then many students' answer sheets (PDF / images / camera scans). This module:

* add_sheet()        — stores an untouched copy of every sheet, converts images to PDF, rejects exact duplicates
* identify_sheet()   — finds WHICH student a sheet belongs to from the file name and from the registration number
                       written / printed on the first page (OCR). A sheet is assigned automatically only when every
                       source that found a registration number agrees on exactly one student in the faculty's scope.
                       Anything unclear (nothing found, two different students, student already has a sheet, same
                       student twice in the batch) goes to MANUAL verification — marks are never given to a guess.
* run_job()          — background worker. For every identified sheet: create the answer booklet → read the answers
                       (handwriting_reader: vision AI on Ollama when available, else local Tesseract) → grade every
                       question against the SAME paper / marking scheme (answer_grader) → total. Progress (current
                       student, current question, done / pending) is written to the database after every question, so
                       what the robot UI shows is the real backend state.
* class_report()     — class performance statistics; class_report_pdf() / results_zip() for downloads.

Marks produced here are AI marks in status "in_review": a teacher must still verify flagged answers and approve
(examiner_routes.approve_booklet) before anything can be published to students.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import uuid
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend import access as AC
from backend.database import (CourseRegistration, Course, EvaluationJob, EvaluationJobItem, ExamPaper, ExamQuestion,
                              Rubric, SessionLocal, Student, StudentBooklet, User)

logger = logging.getLogger(__name__)

BULK_ROOT = Path(os.getenv("BULK_DIR", "uploads/bulk"))
MAX_FILE_MB = int(os.getenv("BULK_MAX_FILE_MB", "40"))
MAX_ITEMS = int(os.getenv("BULK_MAX_SHEETS", "300"))
PASS_PERCENT = float(os.getenv("PASS_PERCENT", "40"))

ACTIVE = ("queued", "running", "cancelling")
STARTABLE = ("draft", "interrupted", "cancelled", "completed", "failed")
_threads: Dict[int, threading.Thread] = {}
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.utcnow()


def norm_reg(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


_FOLD = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2",
                       "G": "6"})


def fold_reg(s: str) -> str:
    """Registration number with common OCR confusions folded (O→0, I/L→1, S→5, B→8 …)."""
    return norm_reg(s).translate(_FOLD)


def reg_tokens(text: str) -> List[str]:
    """Registration-number-like tokens: ≥5 characters, at least one digit."""
    out = []
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]{3,}[A-Za-z0-9]", text or ""):
        n = norm_reg(t)
        if len(n) >= 5 and re.search(r"\d", n) and n not in out:
            out.append(n)
    # numbers split by OCR into "21BCS 101" → also try adjacent pairs
    parts = re.findall(r"[A-Za-z0-9]+", text or "")
    for a, b in zip(parts, parts[1:]):
        n = norm_reg(a + b)
        if 6 <= len(n) <= 16 and re.search(r"\d", n) and n not in out:
            out.append(n)
    return out


def _event(item: EvaluationJobItem, text: str) -> None:
    try:
        ev = json.loads(item.events or "[]")
    except Exception:
        ev = []
    ev.append({"t": _now().strftime("%H:%M:%S"), "m": text[:300]})
    item.events = json.dumps(ev[-40:])


def paper_readiness(db: Session, paper: ExamPaper) -> Dict:
    """How many questions of the paper can actually be marked (model answer, marking scheme or MCQ key)."""
    qs = (db.query(ExamQuestion).filter(ExamQuestion.paper_id_fk == paper.id, ExamQuestion.is_or_option == False)  # noqa: E712
          .order_by(ExamQuestion.question_number).all())
    rows, ready = [], 0
    for q in qs:
        n_rub = db.query(Rubric).filter(Rubric.exam_question_id == q.id).count()
        ok = bool((q.teacher_answer or "").strip() or n_rub or (q.correct_option or "").strip())
        ready += int(ok)
        rows.append({"question_number": q.question_number, "marks": q.marks, "has_model_answer": bool((q.teacher_answer or "").strip()),
                     "marking_points": n_rub, "mcq_key": bool((q.correct_option or "").strip()), "ready": ok,
                     "question_text": (q.question_text or "")[:160]})
    return {"paper_id": paper.id, "course_code": paper.course_code, "exam_name": paper.exam_name,
            "questions": len(qs), "ready_questions": ready, "total_marks": round(sum(float(q.marks or 0) for q in qs), 2),
            "ready": bool(qs) and ready == len(qs), "partially_ready": 0 < ready < len(qs), "detail": rows}


def can_manage_course(db: Session, user: User, course_code: str) -> bool:
    if AC.is_admin(user):
        return True
    return user.role == "faculty" and (course_code or "").upper() in {c for c, _ in AC.teacher_pairs(db, user)}


def job_or_404(db: Session, user: User, job_id: int) -> EvaluationJob:
    from fastapi import HTTPException
    j = db.query(EvaluationJob).filter(EvaluationJob.id == job_id).first()
    if not j or not (AC.is_admin(user) or (j.created_by_id == user.id and can_manage_course(db, user, j.course_code))):
        raise HTTPException(status_code=404, detail="Checking batch not found")
    return j


def eligible_students(db: Session, user: User, course_code: str) -> List[Student]:
    """Students this user may check for this subject (faculty: assigned batches only)."""
    q = AC.scoped_students(db, user, course_code).filter(Student.is_active == True)  # noqa: E712
    if AC.is_admin(user):
        course = db.query(Course).filter(Course.code == course_code.upper()).first()
        if course:
            reg_ids = [r.student_id for r in db.query(CourseRegistration).filter(CourseRegistration.course_id == course.id).all()]
            if reg_ids:          # admin: students registered for the subject (all students if no registrations recorded)
                q = q.filter(Student.id.in_(reg_ids))
    return q.all()


# ─────────────────────────────────────────────────────────────────────────────
# reading the registration number from the sheet
# ─────────────────────────────────────────────────────────────────────────────

def first_page_text(pdf_path: str) -> Tuple[str, str]:
    """Text of the first page: embedded text layer if present, else Tesseract on the header area then the page."""
    try:
        import pymupdf
    except Exception:          # pragma: no cover
        import fitz as pymupdf  # type: ignore
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return "", f"could not open PDF ({type(e).__name__})"
    try:
        if doc.page_count == 0:
            return "", "PDF has no pages"
        page = doc[0]
        txt = page.get_text() or ""
        if len(txt.strip()) >= 15:
            return txt, "pdf-text"
        try:
            from backend.tesseract_locator import configure
            if not configure().get("available"):
                return "", "Tesseract OCR not installed"
            import pytesseract
            from PIL import Image
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
            header = img.crop((0, 0, img.width, int(img.height * 0.38)))
            t = pytesseract.image_to_string(header, config="--psm 6")
            if not reg_tokens(t):
                t += "\n" + pytesseract.image_to_string(img, config="--psm 6")
            return t, "tesseract"
        except Exception as e:
            return "", f"OCR failed ({type(e).__name__}: {e})"
    finally:
        doc.close()


def identify_sheet(db: Session, user: User, job: EvaluationJob, filename: str, pdf_path: str,
                   entered_reg: str = "") -> Dict:
    """Return {"student": Student|None, "method", "detail"}; student is set only when the evidence is unambiguous."""
    pool = eligible_students(db, user, job.course_code)
    exact: Dict[str, List[Student]] = {}
    folded: Dict[str, List[Student]] = {}
    for s in pool:
        for code in {norm_reg(s.registration_number), norm_reg(s.student_code)} - {""}:
            exact.setdefault(code, []).append(s)
            folded.setdefault(fold_reg(code), []).append(s)

    def match(tokens: List[str]) -> Dict[int, Student]:
        found: Dict[int, Student] = {}
        for t in tokens:
            hits = exact.get(t) or []
            if not hits:
                f = folded.get(fold_reg(t)) or []
                hits = f if len({s.id for s in f}) == 1 else []      # folded match only if unambiguous
            for s in hits:
                found[s.id] = s
        return found

    detail: Dict = {"pool": len(pool)}
    sources: Dict[str, Dict[int, Student]] = {}
    if entered_reg.strip():
        e = match([norm_reg(entered_reg)])
        detail["entered"] = norm_reg(entered_reg)
        if not e:
            detail["problem"] = (f"Registration number {norm_reg(entered_reg)} is not a student you can check for "
                                 f"{job.course_code}.")
        sources["entered"] = e
    stem = Path(filename or "").stem
    ftoks = reg_tokens(stem.replace("_", " ")) + ([norm_reg(stem)] if len(norm_reg(stem)) >= 5 else [])
    sources["filename"] = match(ftoks)
    text, how = first_page_text(pdf_path)
    detail["sheet_reader"] = how
    stoks = reg_tokens(text)
    detail["sheet_tokens"] = stoks[:12]
    sources["sheet"] = match(stoks)
    for k, v in sources.items():
        detail[k + "_matches"] = [s.registration_number for s in v.values()]

    if "problem" in detail:
        return {"student": None, "method": "", "detail": detail}
    union: Dict[int, Student] = {}
    for v in sources.values():
        union.update(v)
    if len(union) == 1:
        st = next(iter(union.values()))
        used = [k for k, v in sources.items() if v]
        return {"student": st, "method": "+".join(used), "detail": detail}
    if not union:
        detail["problem"] = ("No registration number of a student in your classes was found in the file name or on the "
                             "first page. Select the student manually.")
    else:
        detail["problem"] = ("Conflicting identification: " + ", ".join(
            f"{k} says {', '.join(s.registration_number for s in v.values())}" for k, v in sources.items() if v)
            + ". Verify the sheet and select the correct student.")
    return {"student": None, "method": "", "detail": detail}


def _conflict_reason(db: Session, job: EvaluationJob, student: Student, item_id: Optional[int]) -> str:
    other = (db.query(EvaluationJobItem).filter(EvaluationJobItem.job_id == job.id, EvaluationJobItem.student_id == student.id,
                                               EvaluationJobItem.status != "skipped")
             .filter(EvaluationJobItem.id != (item_id or -1)).first())
    if other:
        return (f"Sheet #{other.id} ({other.filename}) in this batch is already assigned to {student.registration_number}. "
                "One student can have only one answer sheet per examination.")
    existing = db.query(StudentBooklet).filter(StudentBooklet.student_id == student.id,
                                               StudentBooklet.course_code == job.course_code.upper(),
                                               StudentBooklet.exam_name == job.exam_name).first()
    if existing:
        linked = db.query(EvaluationJobItem).filter(EvaluationJobItem.booklet_id == existing.id).first()
        if not linked or linked.id != item_id:
            return (f"{student.registration_number} already has an answer sheet for {job.course_code} / {job.exam_name} "
                    f"(booklet #{existing.id}). Remove or re-evaluate that one instead.")
    return ""


def apply_identification(db: Session, job: EvaluationJob, item: EvaluationJobItem, student: Optional[Student],
                         method: str, detail: Dict) -> None:
    if student is not None:
        why = _conflict_reason(db, job, student, item.id)
        if why:
            detail = {**detail, "problem": why, "suggested": student.registration_number}
            student = None
    item.id_detail = json.dumps(detail, default=str)
    if student is None:
        item.student_id, item.id_method, item.status = None, "", "needs_identification"
        _event(item, "Needs manual verification: " + detail.get("problem", "student not identified"))
    else:
        item.student_id, item.id_method, item.status = student.id, method, "ready"
        _event(item, f"Identified as {student.registration_number} ({student.name}) from {method}.")


# ─────────────────────────────────────────────────────────────────────────────
# adding sheets
# ─────────────────────────────────────────────────────────────────────────────

class SheetError(ValueError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def _sniff(data: bytes) -> str:
    if data[:1024].lstrip().startswith(b"%PDF"):
        return ".pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    return ""


def add_sheet(db: Session, user: User, job: EvaluationJob, filename: str, data: bytes, source: str = "upload",
              entered_reg: str = "", pdf_bytes: Optional[bytes] = None, pages: int = 0) -> EvaluationJobItem:
    """Store one sheet (original preserved) and identify its student."""
    if job.status in ACTIVE:
        raise SheetError(409, "Checking is running for this batch; add sheets after it finishes or cancel it.")
    if len(job.items) >= MAX_ITEMS:
        raise SheetError(409, f"A batch can hold at most {MAX_ITEMS} answer sheets.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise SheetError(413, f"{filename}: file too large (max {MAX_FILE_MB} MB).")
    ext = _sniff(data)
    if not ext:
        raise SheetError(415, f"{filename}: only PDF, JPG or PNG answer sheets are accepted (content check).")
    digest = hashlib.sha256(data).hexdigest()
    dup = db.query(EvaluationJobItem).filter(EvaluationJobItem.sha256 == digest, EvaluationJobItem.status != "skipped").first()
    if dup:
        raise SheetError(409, f"{filename}: this exact file was already added (batch #{dup.job_id}, sheet #{dup.id}, "
                              f"{dup.filename}). Sheets are never shared between students.")
    folder = BULK_ROOT / f"job_{job.id}"
    (folder / "originals").mkdir(parents=True, exist_ok=True)
    (folder / "pdf").mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex[:10]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename or "sheet").name)[:80] or "sheet"
    orig = folder / "originals" / f"{stem}_{safe}"
    if not orig.suffix:
        orig = orig.with_suffix(ext)
    orig.write_bytes(data)
    try:
        os.chmod(orig, 0o444)           # originals are read-only on disk
    except Exception:
        pass
    pdf = folder / "pdf" / f"{stem}.pdf"
    if pdf_bytes is not None:
        pdf.write_bytes(pdf_bytes)
    elif ext == ".pdf":
        shutil.copyfile(orig, pdf)
    else:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        im.save(pdf, "PDF", resolution=200.0)
    try:
        import pymupdf
        with pymupdf.open(str(pdf)) as d:
            pages = d.page_count
    except Exception:
        if pdf.exists() and ext == ".pdf":
            pdf.unlink()
            raise SheetError(422, f"{filename}: the PDF could not be opened.")
    item = EvaluationJobItem(job_id=job.id, source=source, filename=Path(filename or safe).name[:300],
                             original_path=str(orig), pdf_path=str(pdf), sha256=digest, pages=pages)
    db.add(item)
    db.flush()
    _event(item, f"Received {item.filename} ({pages} page{'s' if pages != 1 else ''}, {source}).")
    ident = identify_sheet(db, user, job, filename, str(pdf), entered_reg)
    apply_identification(db, job, item, ident["student"], ident["method"], ident["detail"])
    AC.audit(db, user, "BULK_SHEET_ADDED", "evaluation_job", job.id,
             {"item": item.id, "file": item.filename, "source": source, "status": item.status,
              "student": ident["student"].registration_number if ident["student"] else None})
    db.commit()
    db.refresh(item)
    return item


def scan_to_pdf(images: List[bytes], rotations: List[int]) -> Tuple[bytes, int]:
    """Camera pages → processed (crop / perspective / enhance) → one PDF."""
    from backend import scanner as SC
    from PIL import Image
    pages = []
    for i, raw in enumerate(images):
        processed, _q = SC.process_page(raw)
        rot = (rotations[i] if i < len(rotations) else 0) % 360
        if rot in (90, 180, 270):
            processed = SC.rotate_jpeg(processed, rot)
        pages.append(Image.open(io.BytesIO(processed)).convert("RGB"))
    if not pages:
        raise SheetError(400, "No pages were captured.")
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", resolution=200.0, save_all=True, append_images=pages[1:])
    return buf.getvalue(), len(pages)


# ─────────────────────────────────────────────────────────────────────────────
# dictionaries for the API / UI
# ─────────────────────────────────────────────────────────────────────────────

def item_dict(item: EvaluationJobItem) -> Dict:
    st = item.student
    try:
        detail = json.loads(item.id_detail or "{}")
    except Exception:
        detail = {}
    b = item.booklet
    return {"id": item.id, "filename": item.filename, "source": item.source, "pages": item.pages, "status": item.status,
            "student": ({"id": st.id, "registration_number": st.registration_number, "name": st.name} if st else None),
            "id_method": item.id_method, "id_problem": detail.get("problem", ""), "id_suggested": detail.get("suggested"),
            "id_evidence": {k: detail.get(k) for k in ("filename_matches", "sheet_matches", "entered_matches", "sheet_reader")},
            "booklet_id": item.booklet_id, "booklet_status": (b.approval_status if b else None),
            "is_published": bool(b and b.is_published),
            "current_question": item.current_question, "questions_done": item.questions_done,
            "questions_total": item.questions_total, "total_ai": item.total_ai, "max_marks": item.max_marks,
            "flagged": item.flagged, "error": item.error or "",
            "events": json.loads(item.events or "[]")[-14:],
            "original_url": f"/bulk/items/{item.id}/original", "page_url": f"/bulk/items/{item.id}/page1.png",
            "evaluated_url": (f"/bulk/items/{item.id}/evaluated.pdf" if item.booklet_id and item.status == "evaluated" else None)}


def job_dict(job: EvaluationJob, items: bool = True) -> Dict:
    its = list(job.items)
    c = Counter(i.status for i in its)
    done = c.get("evaluated", 0) + c.get("needs_review", 0) + c.get("failed", 0)
    workable = sum(1 for i in its if i.status not in ("needs_identification", "skipped"))
    cur = next((i for i in its if i.id == job.current_item_id), None)
    q_total = sum(i.questions_total or 0 for i in its if i.status not in ("needs_identification", "skipped"))
    q_done = sum(i.questions_done or 0 for i in its)
    out = {"id": job.id, "title": job.title or f"{job.course_code} · {job.exam_name}", "course_code": job.course_code,
           "exam_name": job.exam_name, "exam_paper_id": job.exam_paper_id, "status": job.status, "message": job.message,
           "created_by": job.created_by.username if job.created_by else "",
           "created_at": job.created_at.isoformat() + "Z" if job.created_at else None,
           "started_at": job.started_at.isoformat() + "Z" if job.started_at else None,
           "finished_at": job.finished_at.isoformat() + "Z" if job.finished_at else None,
           "counts": dict(c), "sheets": len(its), "ready": c.get("ready", 0),
           "needs_identification": c.get("needs_identification", 0), "completed": done, "workable": workable,
           "pending": c.get("ready", 0) + c.get("reading", 0) + c.get("evaluating", 0),
           "progress_pct": round(100.0 * q_done / q_total, 1) if q_total else (100.0 if workable and done == workable else 0.0),
           "current": item_dict(cur) if cur else None}
    if items:
        out["items"] = [item_dict(i) for i in its]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the worker
# ─────────────────────────────────────────────────────────────────────────────

def start_job(db: Session, user: User, job: EvaluationJob, background: bool = True) -> Dict:
    from fastapi import HTTPException
    if job.status in ACTIVE:
        return {"started": False, "already_running": True, "job": job_dict(job, items=False)}
    if job.status not in STARTABLE:
        raise HTTPException(status_code=409, detail=f"Batch is {job.status}.")
    paper = db.query(ExamPaper).filter(ExamPaper.id == job.exam_paper_id).first()
    if not paper:
        raise HTTPException(status_code=409, detail="The question paper of this batch no longer exists.")
    rd = paper_readiness(db, paper)
    if not rd["ready_questions"]:
        raise HTTPException(status_code=422, detail=(
            "The question paper has no model answers / marking scheme yet, so nothing can be marked. Upload the answer "
            "key or marking scheme for the paper first."))
    ready = [i for i in job.items if i.status == "ready"]
    # sheets interrupted mid-way are checked again from the start of their grading
    for i in job.items:
        if i.status in ("reading", "evaluating"):
            i.status = "ready"
            ready.append(i)
    if not ready:
        waiting = sum(1 for i in job.items if i.status == "needs_identification")
        raise HTTPException(status_code=409, detail=(
            "There are no identified answer sheets waiting to be checked." +
            (f" {waiting} sheet(s) need manual student verification first." if waiting else
             " Upload or scan answer sheets first.")))
    job.status, job.started_at, job.finished_at = "queued", _now(), None
    job.message = f"Queued {len(ready)} answer sheet(s)."
    AC.audit(db, user, "BULK_CHECKING_STARTED", "evaluation_job", job.id,
             {"sheets": len(ready), "paper": paper.id, "readiness": f"{rd['ready_questions']}/{rd['questions']}"})
    db.commit()
    if background:
        launch(job.id, user.id)
    else:
        run_job(job.id, user.id)
    db.refresh(job)
    return {"started": True, "sheets": len(ready), "readiness": rd, "job": job_dict(job, items=False)}


def launch(job_id: int, user_id: int) -> None:
    with _lock:
        t = _threads.get(job_id)
        if t and t.is_alive():
            return
        t = threading.Thread(target=run_job, args=(job_id, user_id), name=f"bulk-check-{job_id}", daemon=True)
        _threads[job_id] = t
        t.start()


def is_running(job_id: int) -> bool:
    t = _threads.get(job_id)
    return bool(t and t.is_alive())


def _cancel_requested(db: Session, job_id: int) -> bool:
    st = db.query(EvaluationJob.status).filter(EvaluationJob.id == job_id).scalar()
    return st == "cancelling"


def run_job(job_id: int, user_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(EvaluationJob).get(job_id)
        user = db.query(User).get(user_id)
        if not job or not user:
            return
        job.status, job.message = "running", "Starting the checking robot…"
        db.commit()
        from backend.llm_provider import get_llm_client
        client = get_llm_client()
        for item_id in [i.id for i in job.items if i.status == "ready"]:
            if _cancel_requested(db, job_id):
                break
            item = db.query(EvaluationJobItem).get(item_id)
            try:
                _check_one(db, user, job, item, client)
            except Exception as e:      # one bad sheet never stops the batch
                logger.exception("bulk item %s failed", item_id)
                db.rollback()
                item = db.query(EvaluationJobItem).get(item_id)
                item.status, item.error, item.finished_at = "failed", f"{type(e).__name__}: {e}"[:1000], _now()
                _event(item, "Failed: " + item.error)
                db.commit()
        job = db.query(EvaluationJob).get(job_id)
        cancelled = job.status == "cancelling"
        c = Counter(i.status for i in job.items)
        job.status = "cancelled" if cancelled else "completed"
        job.current_item_id, job.finished_at = None, _now()
        job.message = (("Cancelled. " if cancelled else "Checking finished. ") +
                       f"{c.get('evaluated', 0)} evaluated, {c.get('needs_review', 0)} need review, "
                       f"{c.get('failed', 0)} failed, {c.get('needs_identification', 0)} need student verification. "
                       "Marks are AI marks awaiting teacher approval.")
        AC.audit(db, user, "BULK_CHECKING_FINISHED", "evaluation_job", job.id, {"counts": dict(c), "cancelled": cancelled})
        db.commit()
    except Exception:
        logger.exception("bulk job %s crashed", job_id)
        db.rollback()
        try:
            job = db.query(EvaluationJob).get(job_id)
            if job:
                job.status, job.message, job.finished_at = "failed", "The checking worker stopped unexpectedly; see server log.", _now()
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


def _check_one(db: Session, user: User, job: EvaluationJob, item: EvaluationJobItem, client) -> None:
    from backend.booklet_ocr import create_booklet, ensure_answers_extracted, grade_row, recompute_totals
    st = item.student
    if st is None:                         # never mark a sheet without a verified student
        item.status = "needs_identification"
        db.commit()
        return
    why = _conflict_reason(db, job, st, item.id)
    if why:
        apply_identification(db, job, item, None, "", {"problem": why, "suggested": st.registration_number})
        db.commit()
        return
    job.current_item_id = item.id
    job.message = f"Reading the answer sheet of {st.registration_number} ({st.name})…"
    item.status, item.error, item.questions_done, item.current_question = "reading", "", 0, None
    _event(item, "Reading answers (OCR)…")
    db.commit()

    booklet = item.booklet
    if booklet is None:
        booklet = create_booklet(db, st, job.course_code, job.exam_name, item.pdf_path, user, total_pages=item.pages)
        booklet.exam_paper_id = job.exam_paper_id         # the batch's paper, never a guessed one
        item.booklet_id = booklet.id
        db.commit()
    elif booklet.is_published:
        item.status, item.error = "needs_review", "Already published — use re-evaluation instead."
        db.commit()
        return
    rep = ensure_answers_extracted(db, booklet)
    db.refresh(booklet)
    answered = [a for a in booklet.answers if (a.answer_text or "").strip()]
    _event(item, f"Read {len(answered)} answered question(s) of {len(booklet.answers)} using {booklet.ocr_engine or 'OCR'}.")
    if not answered:
        item.status = "needs_review"
        item.error = ("No answer text could be read (" + (rep.get("reason") or "no handwriting found") +
                      "). Re-scan the sheet or type the answers in Examiner Review.")
        item.finished_at = _now()
        _event(item, item.error)
        db.commit()
        return

    item.status = "evaluating"
    answers = sorted(booklet.answers, key=lambda a: (a.question_number or 0, a.id))
    item.questions_total = len(answers)
    db.commit()
    flagged, max_total, skipped = 0, 0.0, []
    for ans in answers:
        if _cancel_requested(db, job.id):
            item.status, item.error = "ready", "Cancelled before finishing — will be checked again on restart."
            item.current_question = None
            db.commit()
            return
        item.current_question = ans.question_number
        job.message = f"Evaluating {st.registration_number} — question {ans.question_number} ({item.questions_done + 1}/{len(answers)})"
        db.commit()
        res = grade_row(db, booklet, ans, client)
        if res is None:
            skipped.append(ans.question_number)
            _event(item, f"Q{ans.question_number}: skipped — no model answer / marking scheme on the paper.")
        else:
            flagged += int(bool(res.flagged_for_review))
            max_total += float(res.max_marks or 0)
            if not (ans.answer_text or "").strip():
                _event(item, f"Q{ans.question_number}: not attempted — 0/{res.max_marks:g}.")
            else:
                _event(item, f"Q{ans.question_number}: {res.final_score:g}/{res.max_marks:g}"
                             + (" (flagged for teacher verification)" if res.flagged_for_review else ""))
        item.questions_done += 1
        db.commit()
    recompute_totals(booklet)
    booklet.approval_status = "in_review"
    item.total_ai, item.max_marks, item.flagged = booklet.total_ai_score, round(max_total, 2), flagged
    item.current_question, item.finished_at = None, _now()
    item.status = "evaluated" if max_total > 0 else "needs_review"
    if max_total == 0:
        item.error = "None of the questions could be marked (no model answers / marking scheme)."
    _event(item, f"Total {booklet.total_ai_score:g}/{max_total:g}. {flagged} answer(s) flagged for verification."
                 + (f" Skipped: {skipped}." if skipped else ""))
    AC.audit(db, user, "AI_EVALUATION_RUN", "booklet", booklet.id,
             {"bulk_job": job.id, "item": item.id, "total_ai_score": booklet.total_ai_score, "max": max_total,
              "skipped_questions": skipped, "flagged": flagged,
              "grader": client.active_provider if client.has_real_provider else "offline (marking-point matching)"})
    AC.notify_student(db, st, "sheet_evaluated", f"{job.course_code}: answer sheet evaluated",
                      "Your answer sheet has been evaluated by AI and is awaiting teacher verification.")
    db.commit()


def recover_interrupted() -> int:
    """At server start: jobs that were running when the server stopped become 'interrupted' (restartable)."""
    db = SessionLocal()
    n = 0
    try:
        for j in db.query(EvaluationJob).filter(EvaluationJob.status.in_(ACTIVE)).all():
            if is_running(j.id):
                continue
            j.status, j.current_item_id = "interrupted", None
            j.message = "The server stopped during checking. Press Start to continue with the remaining sheets."
            for i in j.items:
                if i.status in ("reading", "evaluating"):
                    i.status = "ready"
            n += 1
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("recover_interrupted failed")
    finally:
        db.close()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# reports
# ─────────────────────────────────────────────────────────────────────────────

def class_report(db: Session, job: EvaluationJob) -> Dict:
    from backend.booklet_ocr import question_evaluations
    students, qstats, concepts = [], {}, Counter()
    for it in job.items:
        b = it.booklet
        if not b or it.status not in ("evaluated", "needs_review"):
            continue
        qe = question_evaluations(b)
        got = round(sum((q["awarded"] or 0) for q in qe if q["result_id"]), 2)
        mx = round(sum(q["max_marks"] for q in qe if q["result_id"]), 2)
        pct = round(100 * got / mx, 1) if mx else None
        students.append({"item_id": it.id, "booklet_id": b.id, "registration_number": it.student.registration_number,
                         "name": it.student.name, "marks": got, "max_marks": mx, "percentage": pct,
                         "passed": (pct is not None and pct >= PASS_PERCENT), "flagged": sum(1 for q in qe if q["flagged"]),
                         "status": b.approval_status, "teacher_verified": all(q["teacher_verified"] for q in qe if q["result_id"])})
        for q in qe:
            if not q["result_id"]:
                continue
            s = qstats.setdefault(q["question_number"], {"question_number": q["question_number"], "max_marks": q["max_marks"],
                                                         "sum": 0.0, "n": 0, "attempted": 0, "full": 0, "zero": 0})
            s["sum"] += q["awarded"] or 0
            s["n"] += 1
            s["attempted"] += int(bool((q["student_answer"] or "").strip()))
            s["full"] += int((q["awarded"] or 0) >= q["max_marks"] > 0)
            s["zero"] += int((q["awarded"] or 0) == 0)
            if (q["awarded"] or 0) < q["max_marks"]:
                concepts.update(str(c).strip().lower() for c in q["missing_concepts"] if str(c).strip())
    qrows = []
    for qn in sorted(qstats):
        s = qstats[qn]
        avg = s["sum"] / s["n"] if s["n"] else 0
        qrows.append({"question_number": qn, "max_marks": s["max_marks"], "average": round(avg, 2),
                      "average_pct": round(100 * avg / s["max_marks"], 1) if s["max_marks"] else None,
                      "attempt_rate_pct": round(100 * s["attempted"] / s["n"], 1) if s["n"] else None,
                      "full_marks": s["full"], "zero_marks": s["zero"], "students": s["n"]})
    pcts = [s["percentage"] for s in students if s["percentage"] is not None]
    passed = sum(1 for s in students if s["passed"])
    summary = {"students_checked": len(students), "average_pct": round(sum(pcts) / len(pcts), 1) if pcts else None,
               "highest_pct": max(pcts) if pcts else None, "lowest_pct": min(pcts) if pcts else None,
               "passed": passed, "failed": len(students) - passed,
               "pass_rate_pct": round(100 * passed / len(students), 1) if students else None,
               "pass_mark_pct": PASS_PERCENT,
               "needs_identification": sum(1 for i in job.items if i.status == "needs_identification"),
               "failed_sheets": sum(1 for i in job.items if i.status == "failed"),
               "flagged_answers": sum(s["flagged"] for s in students)}
    hardest = sorted([q for q in qrows if q["average_pct"] is not None], key=lambda q: q["average_pct"])[:3]
    return {"job": job_dict(job, items=False), "summary": summary, "students": sorted(students, key=lambda s: s["registration_number"]),
            "questions": qrows, "hardest_questions": [q["question_number"] for q in hardest],
            "missed_concepts": [{"concept": c, "answers": n} for c, n in concepts.most_common(12)]}


def class_report_pdf(db: Session, job: EvaluationJob) -> bytes:
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer
    from backend.reports import BODY, H2, SMALL, _doc, _footer, _grid, _header, _p
    rep = class_report(db, job)
    s = rep["summary"]
    buf = io.BytesIO()
    doc = _doc(buf, f"Class performance {job.course_code}")
    fmt = lambda v, suf="": "—" if v is None else f"{v:g}{suf}"   # noqa: E731
    el = _header(f"AI Robot Checking — Class Performance Report · {job.course_code} · {job.exam_name}")
    el += [Paragraph(f"Batch #{job.id} · paper #{job.exam_paper_id} · status: {job.status} · "
                     "<b>AI marks awaiting teacher approval unless marked as approved/published.</b>", BODY), Spacer(1, 6),
           Paragraph("Summary", H2),
           _grid([["Checked", "Average", "Highest", "Lowest", "Passed", "Failed", "Pass rate", "Flagged answers", "Need ID"],
                  [s["students_checked"], fmt(s["average_pct"], "%"), fmt(s["highest_pct"], "%"), fmt(s["lowest_pct"], "%"),
                   s["passed"], s["failed"], fmt(s["pass_rate_pct"], "%"), s["flagged_answers"], s["needs_identification"]]],
                 [20 * mm] * 9),
           Paragraph("Students", H2)]
    data = [["Reg. No.", "Name", "Marks", "Max", "%", "Result", "Flagged", "Status"]]
    for r in rep["students"]:
        data.append([r["registration_number"], _p(r["name"], SMALL), fmt(r["marks"]), fmt(r["max_marks"]),
                     fmt(r["percentage"]), "PASS" if r["passed"] else "FAIL", r["flagged"], r["status"]])
    el.append(_grid(data, [26 * mm, 50 * mm, 16 * mm, 14 * mm, 14 * mm, 16 * mm, 16 * mm, 28 * mm]))
    el.append(Paragraph("Question-wise performance", H2))
    qd = [["Q", "Max", "Average", "Avg %", "Attempted %", "Full marks", "Zero marks"]]
    for q in rep["questions"]:
        qd.append([f"Q{q['question_number']}", fmt(q["max_marks"]), fmt(q["average"]), fmt(q["average_pct"]),
                   fmt(q["attempt_rate_pct"]), q["full_marks"], q["zero_marks"]])
    el.append(_grid(qd, [14 * mm, 16 * mm, 22 * mm, 20 * mm, 26 * mm, 24 * mm, 24 * mm]))
    if rep["missed_concepts"]:
        el.append(Paragraph("Most frequently missed concepts", H2))
        el.append(_grid([["Concept", "Answers"]] + [[_p(c["concept"], SMALL), c["answers"]] for c in rep["missed_concepts"]],
                        [140 * mm, 25 * mm]))
    pending = [i for i in job.items if i.status in ("needs_identification", "failed")]
    if pending:
        el.append(Paragraph("Sheets not checked", H2))
        el.append(_grid([["Sheet", "File", "Status", "Reason"]] + [
            [f"#{i.id}", _p(i.filename, SMALL), i.status, _p((json.loads(i.id_detail or '{}').get('problem') or i.error or ''), SMALL)]
            for i in pending], [16 * mm, 50 * mm, 30 * mm, 84 * mm]))
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def results_zip(db: Session, job: EvaluationJob) -> bytes:
    from backend.evaluated_sheet_generator import build_evaluated_pdf_for_booklet
    from backend.reports import evaluation_report_pdf
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("class_performance_report.pdf", class_report_pdf(db, job))
        for it in job.items:
            if it.status != "evaluated" or not it.booklet:
                continue
            reg = re.sub(r"[^A-Za-z0-9]+", "", it.student.registration_number or f"item{it.id}")
            z.writestr(f"{reg}/{reg}_{job.course_code}_checked_answer_sheet.pdf", build_evaluated_pdf_for_booklet(it.booklet))
            try:
                z.writestr(f"{reg}/{reg}_{job.course_code}_feedback_report.pdf", evaluation_report_pdf(it.booklet))
            except Exception:
                logger.exception("feedback report failed for item %s", it.id)
    return buf.getvalue()
