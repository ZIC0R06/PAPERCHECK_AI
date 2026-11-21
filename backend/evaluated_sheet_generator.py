"""
Evaluated answer sheet (PDF)
============================
The original upload is never modified. The evaluated copy contains:

1. Result summary — student / exam details, status stamp (official vs preliminary), question-wise marks
   (AI marks, teacher-verified final marks), total and percentage.
2. The student's own pages, each with an examiner margin added on the right: for every answer on the page a
   red bracket beside the answer, the marks box ("Q2  6.5 / 10"), the deductions with marks lost, corrections
   and the teacher's comment. Nothing is drawn over the handwriting.
3. Question-wise evaluation — the transcription that was marked, the criterion-by-criterion breakdown
   (marks available / awarded / reason), deductions, corrections and feedback.

Built on demand from the database, so it always reflects the latest marks (re-evaluation included).
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RED = (0.78, 0.09, 0.09)
BLUE = (0.10, 0.25, 0.65)
GREEN = (0.08, 0.45, 0.20)
GREY = (0.35, 0.38, 0.45)
NAVY = (0.06, 0.15, 0.26)
MARGIN_W = 230.0          # examiner margin added to the right of every original page (points)

_FONT, _FONT_B = "Helvetica", "Helvetica-Bold"


def _fonts():
    """Use a Unicode TTF when available (handwriting transcriptions contain symbols); else Helvetica."""
    global _FONT, _FONT_B
    if _FONT != "Helvetica":
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                  ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
                  ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
                  ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")]
    for reg, bold in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("PCSans", reg))
                pdfmetrics.registerFont(TTFont("PCSans-Bold", bold))
                _FONT, _FONT_B = "PCSans", "PCSans-Bold"
                return
            except Exception:  # pragma: no cover
                pass


def _safe(t) -> str:
    t = "" if t is None else str(t)
    if _FONT == "Helvetica":
        t = t.encode("latin-1", "replace").decode("latin-1")
    return t


def _fmt(x) -> str:
    if x is None or x == "":
        return "—"
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


# ─────────────────────────────────────────────────────────────────────────────
# Page images
# ─────────────────────────────────────────────────────────────────────────────

def _open_pdf(path: str):
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz
    return fitz.open(path)


def original_page_images(path: str, dpi: int = 130) -> List[bytes]:
    """JPEG bytes of each page of the original PDF / image."""
    if not path or not os.path.exists(path):
        return []
    if not path.lower().endswith(".pdf"):
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        b = io.BytesIO(); im.save(b, "JPEG", quality=82)
        return [b.getvalue()]
    out = []
    with _open_pdf(path) as doc:
        for p in doc:
            out.append(p.get_pixmap(dpi=dpi).tobytes("jpeg", jpg_quality=82))
    return out


def render_page_png(path: str, page_no: int, y0: float = 0.0, y1: float = 1.0, dpi: int = 110) -> Optional[bytes]:
    """PNG of one page (1-based), optionally cropped to the vertical band y0..y1 (fractions)."""
    if not path or not os.path.exists(path):
        return None
    from PIL import Image
    imgs = original_page_images(path, dpi=dpi)
    if not 1 <= page_no <= len(imgs):
        return None
    im = Image.open(io.BytesIO(imgs[page_no - 1]))
    y0, y1 = max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))
    if y1 > y0 and (y0 > 0 or y1 < 1):
        h = im.height
        im = im.crop((0, max(0, int((y0 - 0.02) * h)), im.width, min(h, int((y1 + 0.02) * h))))
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers (canvas)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(c, text: str, font: str, size: float, width: float) -> List[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    lines = []
    for para in _safe(text).split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if stringWidth(t, font, size) <= width or not cur:
                cur = t
            else:
                lines.append(cur); cur = w
        lines.append(cur)
    return [l for l in lines if l is not None]


def _annotated_pages(original_path: str, qevals: List[Dict], official: bool) -> Optional[bytes]:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    pages = original_page_images(original_path)
    if not pages:
        return None
    by_page: Dict[int, List[tuple]] = {}
    for q in qevals:
        for i, ref in enumerate(q.get("page_refs") or []):
            by_page.setdefault(int(ref.get("page", 1)), []).append((q, ref, i == 0))
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for pno, jpg in enumerate(pages, start=1):
        ir = ImageReader(io.BytesIO(jpg))
        iw, ih = ir.getSize()
        W = 595.0                                   # scale the scan to A4 width, keep aspect
        H = W * ih / iw
        c.setPageSize((W + MARGIN_W, H))
        c.drawImage(ir, 0, 0, W, H)
        # examiner margin
        c.setFillColorRGB(0.985, 0.975, 0.96)
        c.rect(W, 0, MARGIN_W, H, stroke=0, fill=1)
        c.setStrokeColorRGB(*RED); c.setLineWidth(1.2)
        c.line(W, 0, W, H)
        c.setFillColorRGB(*GREY); c.setFont(_FONT, 6.5)
        c.drawString(W + 8, H - 12, _safe(f"Examiner's margin · page {pno}/{len(pages)} · "
                                          + ("official" if official else "preliminary")))
        next_free = H - 22                          # keep annotation blocks from overlapping
        for q, ref, first in sorted(by_page.get(pno, []), key=lambda t: t[1].get("y0", 0)):
            y_top = H * (1 - float(ref.get("y0", 0)))
            y_bot = H * (1 - float(ref.get("y1", 1)))
            # bracket beside the answer
            c.setStrokeColorRGB(*RED); c.setLineWidth(2)
            c.line(W - 10, y_top, W - 10, y_bot)
            c.line(W - 10, y_top, W - 18, y_top); c.line(W - 10, y_bot, W - 18, y_bot)
            y = min(y_top, next_free)
            x = W + 8
            box_w = MARGIN_W - 16
            # marks box
            c.setFillColorRGB(1, 1, 1); c.setStrokeColorRGB(*RED); c.setLineWidth(1.4)
            c.roundRect(x, y - 24, box_w, 22, 4, stroke=1, fill=1)
            c.setFillColorRGB(*RED); c.setFont(_FONT_B, 12)
            label = f"Q{q['question_number']}" + ("" if first else " (contd.)")
            c.drawString(x + 6, y - 18, _safe(label))
            if first:
                c.drawRightString(x + box_w - 6, y - 18, _safe(f"{_fmt(q.get('awarded'))} / {_fmt(q.get('max_marks'))}"))
            y -= 32
            if first:
                notes = []
                if q.get("teacher_changed"):
                    notes.append((BLUE, f"AI suggested {_fmt(q.get('ai_score'))}; marks set by teacher."))
                for d in (q.get("deductions") or [])[:4]:
                    notes.append((RED, f"−{_fmt(d.get('marks'))}: {d.get('criterion') or ''} — {d.get('reason') or ''}"))
                for cr in (q.get("corrections") or [])[:2]:
                    notes.append((RED, f"✗ \"{cr.get('student_wrote', '')}\" → {cr.get('correct_version', '')}"))
                if q.get("teacher_comment"):
                    notes.append((BLUE, f"Teacher: {q['teacher_comment']}"))
                if not notes and q.get("awarded") is not None and q.get("max_marks") and q["awarded"] >= q["max_marks"]:
                    notes.append((GREEN, "✓ Full marks."))
                for colr, text in notes:
                    lines = _wrap(c, text, _FONT, 7.2, box_w)[:5]
                    c.setFillColorRGB(*colr); c.setFont(_FONT, 7.2)
                    for ln in lines:
                        if y < 14:
                            break
                        c.drawString(x, y, ln); y -= 9
                    y -= 3
            next_free = y - 6
        c.showPage()
    c.save()
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Summary + detail (platypus)
# ─────────────────────────────────────────────────────────────────────────────

def _story(meta: Dict, qevals: List[Dict], official: bool, unattempted: List[int]):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    def P(t, size=9, bold=False, color=colors.black, lead=None, space=2):
        st = ParagraphStyle("p", fontName=_FONT_B if bold else _FONT, fontSize=size, leading=lead or size * 1.3,
                            textColor=color, spaceAfter=space)
        return Paragraph(escape(_safe(t)).replace("\n", "<br/>"), st)

    navy = colors.Color(*NAVY)
    s = []
    head = Table([[P("EVALUATED ANSWER SHEET", 16, True, colors.white),
                   P(("OFFICIAL — TEACHER APPROVED & PUBLISHED" if official else
                      "PRELIMINARY — NOT YET APPROVED / PUBLISHED"), 9, True, colors.white)]],
                 colWidths=[110 * mm, 70 * mm])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), navy),
                              ("BACKGROUND", (1, 0), (1, 0), colors.Color(*(GREEN if official else (0.70, 0.33, 0.04))))]
                             + [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    s += [head, Spacer(1, 6)]
    tot, mx = meta.get("total_awarded") or 0, meta.get("max_marks") or 0
    pct = (100.0 * tot / mx) if mx else 0
    info = [[P(f"Student: {meta.get('student_name', '')}", 9, True), P(f"Course: {meta.get('course_code', '')} — {meta.get('course_name', '')}", 9, True)],
            [P(f"Registration no.: {meta.get('registration_number', '')}"), P(f"Examination: {meta.get('exam_name', '')}")],
            [P(f"Roll no.: {meta.get('roll_number', '') or '—'}   Semester: {meta.get('semester', '') or '—'}"),
             P(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")]]
    t = Table(info, colWidths=[90 * mm, 90 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, -1), colors.Color(.97, .98, .99))]))
    s += [t, Spacer(1, 8)]
    s += [P(f"Total: {_fmt(tot)} / {_fmt(mx)}   ({pct:.1f}%)", 15, True, colors.Color(*NAVY), space=4)]
    if not official:
        s += [P("These marks are an AI evaluation awaiting teacher verification. They are not final.", 8.5, False, colors.Color(.6, .3, 0))]
    rows = [[P(x, 8, True) for x in ("Q", "Marks", "AI marks", "Verified", "Main deductions")]]
    for q in qevals:
        ded = "; ".join(f"−{_fmt(d.get('marks'))} {d.get('criterion') or d.get('reason') or ''}"[:110]
                        for d in (q.get("deductions") or [])[:3]) or ("Not evaluated" if q.get("awarded") is None else "—")
        rows.append([P(f"Q{q['question_number']}", 8, True), P(f"{_fmt(q.get('awarded'))} / {_fmt(q.get('max_marks'))}", 8, True),
                     P(_fmt(q.get("ai_score")), 8), P("Teacher" if q.get("teacher_verified") else "AI only", 8),
                     P(ded, 7.5)])
    t = Table(rows, colWidths=[12 * mm, 22 * mm, 18 * mm, 18 * mm, 110 * mm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.89, .91, .94)),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [t, Spacer(1, 6)]
    if unattempted:
        s += [P(f"Not attempted / no answer found: {', '.join('Q' + str(n) for n in unattempted)} (0 marks).", 8.5, False, colors.Color(*RED))]
    s += [P("The pages that follow are the student's original pages with the examiner's margin (marks, deductions, "
            "comments), then the question-wise evaluation.", 8, False, colors.Color(*GREY))]
    yield ("summary", s)

    d = []
    d += [P("QUESTION-WISE EVALUATION", 14, True, colors.Color(*NAVY), space=6)]
    for q in qevals:
        d += [Spacer(1, 6), P(f"Question {q['question_number']}  —  {_fmt(q.get('awarded'))} / {_fmt(q.get('max_marks'))} marks"
                              + ("   (teacher verified)" if q.get("teacher_verified") else "   (AI evaluated)"), 11, True, colors.Color(*NAVY))]
        if q.get("question_text"):
            d += [P(f"Question: {q['question_text']}", 8.5, False, colors.Color(*GREY))]
        d += [P("Answer as read from the sheet:", 8.5, True),
              P(q.get("student_answer") or "(no answer found on the sheet)", 8.5, False, colors.Color(.15, .15, .15))]
        rb = q.get("rubric_breakdown") or []
        if rb:
            rows = [[P(x, 7.5, True) for x in ("Marking point", "Avail.", "Given", "Status", "Reason / evidence")]]
            for r in rb:
                why = (r.get("reason") or "") + (f"  [\u201c{r['evidence']}\u201d]" if r.get("evidence") else "")
                rows.append([P(r.get("criterion", ""), 7.5), P(_fmt(r.get("max_marks")), 7.5), P(_fmt(r.get("awarded")), 7.5, True),
                             P(r.get("status", ""), 7.5), P(why, 7)])
            t = Table(rows, colWidths=[62 * mm, 13 * mm, 13 * mm, 17 * mm, 75 * mm], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.93, .94, .96))]))
            d += [t]
        for dd in q.get("deductions") or []:
            d += [P(f"−{_fmt(dd.get('marks'))}  {dd.get('criterion') or ''}: {dd.get('reason') or ''}", 8, False, colors.Color(*RED), space=1)]
        for cr in q.get("corrections") or []:
            d += [P(f"Correction: \u201c{cr.get('student_wrote', '')}\u201d → {cr.get('correct_version', '')}. {cr.get('explanation', '')}",
                    8, False, colors.Color(*RED), space=1)]
        if q.get("feedback"):
            d += [P(f"Feedback: {q['feedback']}", 8, False, colors.Color(*GREY))]
        if q.get("teacher_changed"):
            d += [P(f"AI suggested {_fmt(q.get('ai_score'))}; the teacher awarded {_fmt(q.get('awarded'))}.", 8, True, colors.Color(*BLUE))]
        if q.get("teacher_comment"):
            d += [P(f"Teacher's comment: {q['teacher_comment']}", 8.5, True, colors.Color(*BLUE))]
    sig = ("Verified and published by the examination department. Marks shown are final unless changed through "
           "re-evaluation." if official else "Preliminary copy — marks may change after teacher review.")
    d += [Spacer(1, 10), P(sig, 8, False, colors.Color(*GREY))]
    yield ("detail", d)


def build_evaluated_pdf(original_path: str, meta: Dict, qevals: List[Dict], official: bool) -> bytes:
    _fonts()
    import pypdf
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate

    unattempted = [q["question_number"] for q in qevals if not (q.get("student_answer") or "").strip()]
    parts = {}
    for name, story in _story(meta, qevals, official, unattempted):
        b = io.BytesIO()
        SimpleDocTemplate(b, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
                          title=f"Evaluated answer sheet {meta.get('registration_number', '')}").build(story)
        parts[name] = b.getvalue()
    annotated = None
    try:
        annotated = _annotated_pages(original_path, qevals, official)
    except Exception:
        logger.exception("Could not annotate the original pages")
    w = pypdf.PdfWriter()
    for blob in (parts["summary"], annotated, parts["detail"]):
        if blob:
            for p in pypdf.PdfReader(io.BytesIO(blob)).pages:
                w.add_page(p)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def booklet_meta(booklet) -> Dict:
    from backend.booklet_ocr import question_evaluations
    qe = question_evaluations(booklet)
    return {"student_name": booklet.student.name if booklet.student else "", "registration_number": booklet.register_number,
            "roll_number": booklet.roll_number, "course_code": booklet.course_code, "course_name": booklet.course_name,
            "exam_name": booklet.exam_name, "semester": booklet.semester,
            "total_awarded": round(sum((q["awarded"] or 0) for q in qe), 2),
            "max_marks": round(sum(q["max_marks"] for q in qe if q["result_id"]), 2)}, qe


def build_evaluated_pdf_for_booklet(booklet) -> bytes:
    meta, qe = booklet_meta(booklet)
    return build_evaluated_pdf(booklet.pdf_path, meta, qe, official=bool(booklet.is_published))


# ── backward compatible API (old callers / tests) ───────────────────────────────

def generate_evaluated_sheet_pdf(booklet_id: int, original_pdf_path: str, metadata: Dict,
                                 question_evaluations: List[Dict], output_dir: str) -> str:
    qe = [{"question_number": q.get("question_number"), "awarded": q.get("awarded"), "max_marks": q.get("max_marks"),
           "ai_score": q.get("awarded"), "teacher_verified": "Certified" in (q.get("examiner_note") or ""),
           "feedback": q.get("feedback", ""), "student_answer": q.get("student_answer", "(see original page)"),
           "deductions": [{"marks": "", "criterion": m, "reason": "missing"} for m in q.get("missing_concepts", [])],
           "strengths": q.get("strengths", []), "page_refs": q.get("page_refs", [])} for q in question_evaluations]
    pdf = build_evaluated_pdf(original_pdf_path, metadata, qe, official=bool(metadata.get("is_published")))
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"evaluated_booklet_{booklet_id}.pdf")
    with open(out, "wb") as f:
        f.write(pdf)
    return out
