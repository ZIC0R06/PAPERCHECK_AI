"""
OMR evaluation (prefix /omr) — teachers & admin
==============================================
Answer keys → printable personalised sheets → scan (camera photo / image / PDF) → bubble reading →
score against the key → flagged readings verified manually → OMR result per student.
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import access as AC
from backend import omr as O
from backend.auth import require_role
from backend.database import Batch, OmrKey, OmrResponse, Student, User, get_db

router = APIRouter()
STAFF = Depends(require_role(["admin", "faculty"]))
OMR_DIR = Path(os.getenv("OMR_DIR", "uploads/omr"))


class KeyIn(BaseModel):
    subject_code: str = Field(..., min_length=2, max_length=20)
    exam_name: str = Field(..., min_length=2, max_length=200)
    answers: str = Field(..., description="e.g. 'ABCDA…' or 'A,B,C,D' — one letter per question")
    num_options: int = Field(4, ge=2, le=5)
    marks_correct: float = Field(1.0, gt=0, le=100)
    marks_wrong: float = Field(0.0, ge=0, le=100)
    multiple_policy: str = Field("flag", pattern="^(flag|wrong|zero)$",
                                 description="two or more filled bubbles: flag for teacher / count as wrong / zero")
    allow_negative_total: bool = False


class VerifyIn(BaseModel):
    answers: Dict[int, str] = Field(..., description="{question_number: 'A'..'E' or '' for blank/invalid}")
    note: str = Field("", max_length=500)


def _parse_answers(raw: str, nopt: int) -> List[str]:
    letters = [x for x in re.sub(r"[\s,;|/-]+", "", raw.upper())]
    allowed = set(O.LETTERS[:nopt])
    bad = sorted({x for x in letters if x not in allowed})
    if bad:
        raise HTTPException(status_code=422, detail=f"Invalid option(s) {bad}; allowed: {''.join(sorted(allowed))}")
    if not 1 <= len(letters) <= 100:
        raise HTTPException(status_code=422, detail="An OMR key must have between 1 and 100 questions.")
    return letters


def _can_subject(db, user, subject):
    if user.role == "admin":
        return True
    return any(c == subject.upper() for c, _ in AC.teacher_pairs(db, user))


def _key_or_404(db, user, kid) -> OmrKey:
    k = db.query(OmrKey).filter(OmrKey.id == kid).first()
    if not k or not _can_subject(db, user, k.course_code):
        raise HTTPException(status_code=404, detail="OMR key not found")
    return k


def _key_dict(k: OmrKey) -> dict:
    return {"id": k.id, "subject_code": k.course_code, "exam_name": k.exam_name, "num_questions": k.num_questions,
            "num_options": k.num_options, "answers": "".join(json.loads(k.answers_json)),
            "marks_correct": k.marks_correct, "marks_wrong": k.marks_wrong,
            "multiple_policy": k.multiple_policy or "flag", "allow_negative_total": bool(k.allow_negative_total),
            "max_score": round(k.num_questions * k.marks_correct, 2)}


def _resp_dict(r: OmrResponse, detail: bool = False) -> dict:
    d = {"id": r.id, "key_id": r.key_id, "subject_code": r.key.course_code, "exam_name": r.key.exam_name,
         "registration_number": r.student.registration_number, "student_name": r.student.name,
         "correct": r.correct, "incorrect": r.incorrect, "unanswered": r.unanswered, "flagged": r.flagged,
         "score": r.score, "max_score": r.max_score, "needs_verification": r.needs_verification,
         "verified": r.verified_at is not None, "reg_detected": r.reg_detected,
         "created_at": r.created_at.isoformat() + "Z" if r.created_at else None}
    if detail:
        d["questions"] = json.loads(r.detected_json or "[]")
        d["image_url"] = f"/omr/responses/{r.id}/image"
    return d


def _apply_score(r: OmrResponse, key: OmrKey, questions: List[dict], overrides: Optional[Dict[int, str]] = None):
    s = O.score(questions, json.loads(key.answers_json), key.marks_correct, key.marks_wrong, overrides,
                multiple_policy=key.multiple_policy or "flag", allow_negative_total=bool(key.allow_negative_total))
    r.detected_json = json.dumps(s["detail"])
    r.correct, r.incorrect, r.unanswered, r.flagged = s["correct"], s["incorrect"], s["unanswered"], s["flagged"]
    r.score, r.max_score, r.needs_verification = s["score"], s["max_score"], s["flagged"] > 0


# ── keys ──────────────────────────────────────────────────────────────────────

@router.post("/keys", summary="Create / replace an OMR answer key")
def create_key(data: KeyIn, db: Session = Depends(get_db), user: User = STAFF):
    subj = data.subject_code.strip().upper()
    if not _can_subject(db, user, subj):
        raise HTTPException(status_code=403, detail=f"You are not assigned to {subj}.")
    answers = _parse_answers(data.answers, data.num_options)
    k = db.query(OmrKey).filter(OmrKey.course_code == subj, OmrKey.exam_name == data.exam_name.strip()).first()
    if k and db.query(OmrResponse).filter(OmrResponse.key_id == k.id).count() and len(answers) != k.num_questions:
        raise HTTPException(status_code=409, detail="Sheets were already scanned with this key; the number of questions can't change.")
    if not k:
        k = OmrKey(course_code=subj, exam_name=data.exam_name.strip(), created_by_id=user.id)
        db.add(k)
    k.num_questions, k.num_options, k.answers_json = len(answers), data.num_options, json.dumps(answers)
    k.marks_correct, k.marks_wrong = data.marks_correct, data.marks_wrong
    k.multiple_policy, k.allow_negative_total = data.multiple_policy, data.allow_negative_total
    db.flush()
    for r in db.query(OmrResponse).filter(OmrResponse.key_id == k.id).all():   # key corrected → rescore everyone
        qs = [{kk: v for kk, v in q.items() if kk in ("q", "marked", "light", "status", "fill", "darkness")} for q in json.loads(r.detected_json)]
        ov = {q["q"]: q["final_answer"] for q in json.loads(r.detected_json) if q.get("reading") == "verified"}
        _apply_score(r, k, qs, ov)
    AC.audit(db, user, "OMR_KEY_SAVED", "omr_key", k.id, {"subject": subj, "exam": data.exam_name, "questions": len(answers)})
    db.commit()
    return _key_dict(k)


@router.get("/keys", summary="OMR keys I can use")
def list_keys(db: Session = Depends(get_db), user: User = STAFF):
    return [_key_dict(k) for k in db.query(OmrKey).order_by(OmrKey.id.desc()).all() if _can_subject(db, user, k.course_code)]


@router.get("/keys/{kid}/sheets.pdf", summary="Printable OMR sheets (personalised per student of a batch, or blank)")
def sheets(kid: int, batch_code: Optional[str] = None, blank: int = 0, db: Session = Depends(get_db), user: User = STAFF):
    k = _key_or_404(db, user, kid)
    people = []
    if batch_code:
        b = db.query(Batch).filter(func.upper(Batch.batch_code) == batch_code.strip().upper()).first()
        if not b:
            raise HTTPException(status_code=404, detail="Batch not found")
        sts = AC.scoped_students(db, user, k.course_code).filter(Student.batch_id == b.id).order_by(Student.registration_number).all()
        if not sts:
            raise HTTPException(status_code=404, detail="No students you teach in this batch for this subject.")
        people = [(s.registration_number, s.name) for s in sts]
    people += [("", "")] * max(0, min(int(blank), 200))
    if not people:
        raise HTTPException(status_code=422, detail="Choose a batch or a number of blank sheets.")
    pdf = O.sheets_pdf(k.id, k.course_code, k.exam_name, k.num_questions, k.num_options, people,
                       os.getenv("UNIVERSITY_NAME", "OMR Answer Sheet"))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="omr_{k.course_code}_{k.id}.pdf"'})


# ── scanning ──────────────────────────────────────────────────────────────────

def _scan_one(db: Session, user: User, k: OmrKey, data: bytes, registration_number: str, replace: bool) -> dict:
    """Read one sheet image and store / replace the student's OMR response. Raises HTTPException on problems."""
    try:
        reading = O.read_sheet(data, k.num_questions, k.num_options)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # never surface a bare 500 to the teacher
        import logging
        logging.getLogger(__name__).exception("OMR read failed")
        raise HTTPException(status_code=422, detail=f"Could not read this OMR sheet ({type(e).__name__}). "
                                                    "Try a clearer, well-lit photo of the whole sheet.")
    qr_key, qr_reg = O.parse_qr(reading["qr"])
    if qr_key is not None and qr_key != k.id:
        raise HTTPException(status_code=409, detail=f"This sheet belongs to OMR key #{qr_key}, not #{k.id}.")
    typed = registration_number.strip().upper()
    if qr_reg and typed and qr_reg.upper() != typed:
        raise HTTPException(status_code=409, detail=f"The sheet's QR code says {qr_reg} but you entered {typed}.")
    reg = (qr_reg or typed).upper()
    if not reg:
        raise HTTPException(status_code=422, detail=(
            "The student could not be identified: the sheet's QR code was not readable (or it is a blank sheet). "
            "Type the student's registration number and read the sheet again. The bubbles themselves were read fine."))
    st = db.query(Student).filter(func.upper(Student.registration_number) == reg).first()
    if not st:
        raise HTTPException(status_code=404, detail=f"No student with registration number {reg}")
    AC.require_can_teach(db, user, st, k.course_code)
    r = db.query(OmrResponse).filter(OmrResponse.key_id == k.id, OmrResponse.student_id == st.id).first()
    if r and not replace:
        raise HTTPException(status_code=409, detail=f"An OMR sheet for {reg} is already recorded (score {r.score:g}). Tick 'replace' to rescan.")
    OMR_DIR.mkdir(parents=True, exist_ok=True)
    ext = ".pdf" if data[:4] == b"%PDF" else (".png" if data[:4] == b"\x89PNG" else ".jpg")
    path = OMR_DIR / f"{reg}_{k.course_code}_{k.id}_{uuid.uuid4().hex[:6]}{ext}"
    path.write_bytes(data)
    if not r:
        r = OmrResponse(key_id=k.id, student_id=st.id)
        db.add(r)
    r.image_path, r.reg_detected, r.verified_by_id, r.verified_at = str(path), qr_reg, None, None
    _apply_score(r, k, reading["questions"])
    db.flush()
    AC.audit(db, user, "OMR_SCANNED", "omr_response", r.id,
             {"student": reg, "score": r.score, "flagged": r.flagged, "qr": bool(qr_reg)})
    if not r.needs_verification:
        AC.notify_student(db, st, "sheet_evaluated", f"{k.course_code}: OMR sheet evaluated",
                          "Your OMR sheet has been read; the result will appear once published.")
    db.commit()
    return _resp_dict(r, detail=True)


def _check_upload(data: bytes, limit_mb: int):
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {limit_mb} MB).")
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:4] == b"%PDF"):
        raise HTTPException(status_code=415, detail="Upload a JPG/PNG photo or a PDF of the OMR sheet.")


@router.post("/keys/{kid}/scan", summary="Read one OMR sheet (photo / image / PDF)")
async def scan(kid: int, file: UploadFile = File(...), registration_number: str = Form(""), replace: bool = Form(False),
               db: Session = Depends(get_db), user: User = STAFF):
    k = _key_or_404(db, user, kid)
    data = await file.read(20 * 1024 * 1024 + 1)
    _check_upload(data, 20)
    return await run_in_threadpool(_scan_one, db, user, k, data, registration_number, replace)


def _pdf_pages_png(data: bytes, dpi: int = 200) -> List[bytes]:
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz
    with fitz.open(stream=data, filetype="pdf") as doc:
        return [p.get_pixmap(dpi=dpi).tobytes("png") for p in doc]


@router.post("/keys/{kid}/scan-batch", summary="Read many OMR sheets at once (several photos and/or a multi-page scanned PDF)")
async def scan_batch(kid: int, files: List[UploadFile] = File(...), replace: bool = Form(False),
                     db: Session = Depends(get_db), user: User = STAFF):
    """Each page / photo is identified by its QR code. Returns one line per sheet: success or the reason it failed."""
    k = _key_or_404(db, user, kid)
    items = []
    for f in files[:100]:
        data = await f.read(60 * 1024 * 1024 + 1)
        try:
            _check_upload(data, 60)
        except HTTPException as e:
            items.append({"source": f.filename, "ok": False, "error": e.detail}); continue
        if data[:4] == b"%PDF":
            try:
                pages = await run_in_threadpool(_pdf_pages_png, data)
            except Exception as e:
                items.append({"source": f.filename, "ok": False, "error": f"Could not open the PDF ({type(e).__name__})."}); continue
            items += [{"source": f"{f.filename} · page {i}", "data": p} for i, p in enumerate(pages, start=1)]
        else:
            items.append({"source": f.filename, "data": data})
    if len([i for i in items if "data" in i]) > 300:
        raise HTTPException(status_code=413, detail="At most 300 sheets per batch.")
    out = []
    for it in items:
        if "data" not in it:
            out.append(it); continue
        try:
            r = await run_in_threadpool(_scan_one, db, user, k, it["data"], "", replace)
            out.append({"source": it["source"], "ok": True, "registration_number": r["registration_number"],
                        "score": r["score"], "max_score": r["max_score"], "flagged": r["flagged"], "response_id": r["id"]})
        except HTTPException as e:
            db.rollback()
            out.append({"source": it["source"], "ok": False, "error": e.detail})
    return {"sheets": len(out), "read": sum(1 for o in out if o.get("ok")),
            "needs_verification": sum(1 for o in out if o.get("ok") and o.get("flagged")), "results": out}


@router.get("/responses", summary="OMR results (filter by key / verification)")
def responses(key_id: Optional[int] = None, needs_verification: Optional[bool] = None,
              db: Session = Depends(get_db), user: User = STAFF):
    q = db.query(OmrResponse)
    if key_id:
        q = q.filter(OmrResponse.key_id == key_id)
    if needs_verification is not None:
        q = q.filter(OmrResponse.needs_verification == needs_verification)
    return [_resp_dict(r) for r in q.order_by(OmrResponse.id.desc()).all()
            if AC.can_teach(db, user, r.student, r.key.course_code)]


@router.get("/responses/{rid}", summary="OMR result with per-question readings")
def response_detail(rid: int, db: Session = Depends(get_db), user: User = STAFF):
    r = db.query(OmrResponse).filter(OmrResponse.id == rid).first()
    if not r or not AC.can_teach(db, user, r.student, r.key.course_code):
        raise HTTPException(status_code=404, detail="Not found")
    return _resp_dict(r, detail=True)


@router.get("/responses/{rid}/image", summary="Scanned OMR sheet")
def response_image(rid: int, db: Session = Depends(get_db), user: User = STAFF):
    r = db.query(OmrResponse).filter(OmrResponse.id == rid).first()
    if not r or not AC.can_teach(db, user, r.student, r.key.course_code) or not os.path.exists(r.image_path or ""):
        raise HTTPException(status_code=404, detail="Not found")
    mt = {".pdf": "application/pdf", ".png": "image/png"}.get(os.path.splitext(r.image_path)[1].lower(), "image/jpeg")
    return FileResponse(r.image_path, media_type=mt)


@router.post("/responses/{rid}/verify", summary="Teacher resolves flagged / disputed readings")
def verify(rid: int, data: VerifyIn, db: Session = Depends(get_db), user: User = STAFF):
    r = db.query(OmrResponse).filter(OmrResponse.id == rid).first()
    if not r or not AC.can_teach(db, user, r.student, r.key.course_code):
        raise HTTPException(status_code=404, detail="Not found")
    k = r.key
    allowed = set(O.LETTERS[:k.num_options]) | {""}
    for qn, a in data.answers.items():
        if not 1 <= int(qn) <= k.num_questions or a.upper() not in allowed:
            raise HTTPException(status_code=422, detail=f"Invalid answer {a!r} for question {qn}.")
    prev = json.loads(r.detected_json or "[]")
    ov = {q["q"]: q["final_answer"] for q in prev if q.get("reading") == "verified"}
    ov.update({int(q): a.upper() for q, a in data.answers.items()})
    raw = [{kk: v for kk, v in q.items() if kk in ("q", "marked", "light", "status", "fill", "darkness")} for q in prev]
    before = r.score
    _apply_score(r, k, raw, ov)
    if r.flagged:
        unresolved = [q["q"] for q in json.loads(r.detected_json) if q["result"] == "flagged"]
        db.commit()
        return {**_resp_dict(r, detail=True), "message": f"Still needs a decision for question(s) {unresolved}."}
    r.verified_by_id, r.verified_at = user.id, datetime.utcnow()
    AC.audit(db, user, "OMR_VERIFIED", "omr_response", r.id,
             {"overrides": data.answers, "score_before": before, "score_after": r.score, "note": data.note})
    AC.notify_student(db, r.student, "sheet_evaluated", f"{k.course_code}: OMR sheet evaluated",
                      "Your OMR sheet has been verified; the result will appear once published.")
    db.commit()
    return _resp_dict(r, detail=True)
