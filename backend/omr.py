"""
OMR sheets — generation and bubble detection (OpenCV)
====================================================
The same layout() is used to PRINT the sheet and to READ it, so bubble positions always match.

Sheet: A4 with four solid corner markers, a QR code identifying the exam key and student,
up to 100 questions × 2–5 options laid out in 4 columns of 25.

Reading:
  1. find the 4 corner markers → perspective-correct to the canonical page
  2. decode the QR code (exam key + registration number)
  3. measure the dark-pixel ratio inside each bubble (inner disk, ignoring the printed outline)
  4. per question: marked / blank / multiple / uncertain (light or partial marks)
Uncertain and multiple readings are flagged for manual verification and score 0 until verified.
"""

import io
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PAGE_W, PAGE_H = 595.28, 841.89          # A4 in points
SCALE = 2.0                               # canonical pixels per point
FID = 24.0                                # corner marker side (pt)
FID_C = [(40, 40), (PAGE_W - 40, 40), (PAGE_W - 40, PAGE_H - 40), (40, PAGE_H - 40)]  # TL TR BR BL centres (top-left origin)
BUBBLE_R = 7.0
ROWS_PER_COL = 25
COL_X0, COL_W = 62.0, 132.0
GRID_Y0, ROW_H = 190.0, 23.0
OPT_DX0, OPT_DX = 34.0, 21.0
LETTERS = "ABCDE"

MARK_T = 0.42     # inner-disk dark ratio at/above which a bubble counts as filled
LIGHT_T = 0.16    # between LIGHT_T and MARK_T → uncertain (light / partial mark)
FAINT_REL = 0.55  # a filled bubble lighter than 55 % of the student's typical mark → uncertain (half-erased?)
SHADE_REL = 0.28  # an evenly shaded bubble at ≥ 28 % of the typical mark's darkness → uncertain (light pencil)


def layout(num_questions: int, num_options: int) -> List[Dict]:
    """Bubble centres (top-left origin, points) for every question/option."""
    out = []
    for q in range(num_questions):
        col, row = divmod(q, ROWS_PER_COL)
        x0 = COL_X0 + col * COL_W
        y = GRID_Y0 + row * ROW_H
        out.append({"q": q + 1, "label_xy": (x0, y), "bubbles": [(x0 + OPT_DX0 + k * OPT_DX, y) for k in range(num_options)]})
    return out


# ── printing ──────────────────────────────────────────────────────────────────

def sheets_pdf(key_id: int, subject: str, exam: str, num_questions: int, num_options: int,
               students: List[Tuple[str, str]], institution: str = "") -> bytes:
    """One page per (registration_number, name); an empty reg prints a generic sheet."""
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))
    Y = lambda y: PAGE_H - y  # noqa: E731
    lay = layout(num_questions, num_options)
    for reg, name in students:
        for (cx, cy) in FID_C:
            c.setFillColorRGB(0, 0, 0)
            c.rect(cx - FID / 2, Y(cy) - FID / 2, FID, FID, stroke=0, fill=1)
        # header text must never run into the QR code (x ≥ PAGE_W-145) — that makes the QR unreadable
        def fit(text, font, size, max_w=PAGE_W - 150 - 70):
            from reportlab.pdfbase.pdfmetrics import stringWidth
            while text and stringWidth(text, font, size) > max_w:
                text = text[:-2] + "…"
            return text
        c.setFont("Helvetica-Bold", 13)
        c.drawString(70, Y(72), fit(institution or "OMR Answer Sheet", "Helvetica-Bold", 13))
        c.setFont("Helvetica", 9.5)
        c.drawString(70, Y(92), fit(f"Subject: {subject}    Examination: {exam}", "Helvetica", 9.5))
        c.drawString(70, Y(110), fit(f"Registration No.: {reg or '______________'}    Name: {name or '________________________'}", "Helvetica", 9.5))
        c.setFont("Helvetica", 8)
        c.drawString(70, Y(128), fit("Fill ONE bubble per question completely with a dark pen.", "Helvetica", 8))
        c.drawString(70, Y(140), fit(f"Key #{key_id} · {num_questions} questions · options {LETTERS[:num_options]} · keep the corner squares clean", "Helvetica", 8))
        qr = QrCodeWidget(f"OMR:{key_id}:{reg}")
        b = qr.getBounds()
        size = 70.0
        d = Drawing(size, size, transform=[size / (b[2] - b[0]), 0, 0, size / (b[3] - b[1]), 0, 0])
        d.add(qr)
        renderPDF.draw(d, c, PAGE_W - 140, Y(150))
        # option letters above each column
        c.setFont("Helvetica-Bold", 7.5)
        for col in range((num_questions - 1) // ROWS_PER_COL + 1):
            x0 = COL_X0 + col * COL_W
            for k in range(num_options):
                c.drawCentredString(x0 + OPT_DX0 + k * OPT_DX, Y(GRID_Y0 - 15), LETTERS[k])
        c.setLineWidth(0.8)
        for item in lay:
            lx, ly = item["label_xy"]
            c.setFont("Helvetica", 8)
            c.drawRightString(lx + 20, Y(ly) - 3, str(item["q"]))
            for (bx, by) in item["bubbles"]:
                c.circle(bx, Y(by), BUBBLE_R, stroke=1, fill=0)
        c.showPage()
    c.save()
    return buf.getvalue()


# ── reading ───────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    if data[:4] == b"%PDF":
        # PyMuPDF first: pure pip install, no Poppler needed (Poppler is rarely present on Windows)
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.page_count == 0:
                    raise ValueError("The PDF has no pages.")
                pix = doc[0].get_pixmap(dpi=200)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else
                                    (cv2.COLOR_GRAY2BGR if pix.n == 1 else cv2.COLOR_RGB2BGR))
        except ImportError:
            pass
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(data, dpi=200, first_page=1, last_page=1)
            return cv2.cvtColor(np.array(pages[0].convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError("Could not open the PDF. Install PyMuPDF (pip install pymupdf) or upload a JPG/PNG photo instead.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unreadable image. Upload a JPG, PNG or PDF of the OMR sheet.")
    try:  # respect the phone's EXIF rotation
        from PIL import Image, ImageOps
        import io as _io
        pil = ImageOps.exif_transpose(Image.open(_io.BytesIO(data))).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return img


def _marker_candidates(gray: np.ndarray) -> list:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    side_lo, side_hi = 0.008 * max(h, w), 0.09 * max(h, w)
    binaries = [cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]]
    k = int(max(h, w) * 0.06) | 1           # local threshold copes with shadows / uneven light
    binaries.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, k, 15))
    out = []
    for bw in binaries:
        # RETR_LIST: with a dark desk around the page the markers are nested inside the background's hole
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if not (side_lo < cw < side_hi and side_lo < ch < side_hi) or not 0.6 < cw / ch < 1.6:
                continue
            area = cv2.contourArea(c)
            hull = cv2.contourArea(cv2.convexHull(c)) or 1
            if area / hull < 0.85 or area / (cw * ch) < 0.6:
                continue
            # must be solid: the inside is dark (excludes hollow squares such as QR finder rings' holes)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, -1)
            if bw[mask > 0].mean() < 200:
                continue
            m = cv2.moments(c)
            cands.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
        out.append(cands)
    return out


def _corner_sets(gray: np.ndarray):
    """Yield candidate (TL, TR, BR, BL in image space) marker quadruples."""
    h, w = gray.shape
    for cands in _marker_candidates(gray):
        if len(cands) < 4:
            continue
        chosen = []
        for cx, cy in [(0, 0), (w, 0), (w, h), (0, h)]:
            chosen.append(min(cands, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)[:2])
        pts = np.array(chosen, dtype="float32")
        if len({tuple(np.round(p)) for p in pts}) == 4 and cv2.contourArea(pts) > 0.1 * h * w:
            yield pts


def _find_markers(gray: np.ndarray) -> Optional[np.ndarray]:
    return next(_corner_sets(gray), None)


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Remove shadows / uneven lighting: divide by a background estimate wider than a bubble."""
    k = int(BUBBLE_R * SCALE * 3) | 1
    bg = cv2.medianBlur(cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), k if k <= 255 else 255)
    return cv2.divide(gray, bg, scale=255)


def _ink_mask(flat: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(flat, (3, 3), 0)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 130, 215))          # paper is ~255 after flattening; keep a sane window
    return (blur < t).astype(np.uint8) * 255


def _layout_score(bw: np.ndarray, num_questions: int, num_options: int) -> float:
    """
    Fraction of expected bubbles whose printed outline is actually there (1.0 = perfect match).
    Wrong orientation or a sheet printed for a different key layout scores low.
    """
    r = BUBBLE_R * SCALE
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    hits = total = 0
    lay = layout(num_questions, num_options)
    step = max(1, len(lay) // 40)
    h, w = bw.shape
    for item in lay[::step] + [lay[-1]]:
        for bx, by in item["bubbles"]:
            cx, cy = bx * SCALE, by * SCALE
            best_ring = 0.0
            for dr in (-1.5, -0.5, 0.5, 1.5):
                xs = np.clip((cx + (r + dr) * np.cos(ang)).astype(int), 0, w - 1)
                ys = np.clip((cy + (r + dr) * np.sin(ang)).astype(int), 0, h - 1)
                best_ring = max(best_ring, float((bw[ys, xs] > 0).mean()))
            xs = np.clip((cx + 1.55 * r * np.cos(ang)).astype(int), 0, w - 1)
            ys = np.clip((cy + 1.55 * r * np.sin(ang)).astype(int), 0, h - 1)
            off = float((bw[ys, xs] > 0).mean())
            total += 1
            hits += int(best_ring >= 0.5 and best_ring - off >= 0.3)
    return hits / max(1, total)


def _decode_qr(gray_src: np.ndarray, M_inv: np.ndarray, warped: np.ndarray) -> str:
    """Try hard to read the QR code: canonical crop, upscaled, binarised, and the full-resolution source crop."""
    det = cv2.QRCodeDetector()
    x0, y0, x1, y1 = PAGE_W - 175, 30, PAGE_W - 55, 165           # QR area in page points
    tries = []
    roi = warped[int(y0 * SCALE):int(y1 * SCALE), int(x0 * SCALE):int(x1 * SCALE)]
    tries.append(roi)
    # same area cut from the original photo (more pixels for high-resolution phone photos)
    corners = np.float32([[x0 * SCALE, y0 * SCALE], [x1 * SCALE, y0 * SCALE], [x1 * SCALE, y1 * SCALE], [x0 * SCALE, y1 * SCALE]])
    src_c = cv2.perspectiveTransform(corners[None], M_inv)[0]
    side = int(max(np.linalg.norm(src_c[1] - src_c[0]), np.linalg.norm(src_c[3] - src_c[0])))
    if side > roi.shape[1]:
        dst = np.float32([[0, 0], [side, 0], [side, side * (y1 - y0) / (x1 - x0)], [0, side * (y1 - y0) / (x1 - x0)]])
        tries.append(cv2.warpPerspective(gray_src, cv2.getPerspectiveTransform(src_c, dst), (side, int(dst[2][1]))))
    for base in list(tries):
        for f in (2, 3, 4):
            up = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            tries.append(up)
            tries.append(cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    tries.append(warped)
    for t in tries:
        try:
            txt, _, _ = det.detectAndDecode(t)
        except Exception:
            txt = ""
        if txt:
            return txt
    return ""


def read_sheet(data: bytes, num_questions: int, num_options: int) -> Dict:
    img = _decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W, H = int(PAGE_W * SCALE), int(PAGE_H * SCALE)
    dst = np.array([[x * SCALE, y * SCALE] for x, y in FID_C], dtype="float32")

    best = None   # (score, qr_text, warped, flat_bw, M)
    found_markers = False
    for pts in _corner_sets(gray):
        found_markers = True
        # the sheet may be photographed upside-down or sideways: try all 4 orientations
        for rot in range(4):
            src = np.roll(pts, -rot, axis=0)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(gray, M, (W, H))
            bw = _ink_mask(_flatten(warped))
            sc = _layout_score(bw, num_questions, num_options)
            if best is None or sc > best[0]:
                best = (sc, warped, bw, M)
        if best and best[0] >= 0.9:
            break
    if not found_markers:
        raise ValueError("Could not find the 4 black corner squares. Photograph the WHOLE sheet (all 4 corners "
                         "visible), flat, without fingers or glare over the corners.")
    score_, warped, bw, M = best
    if score_ < 0.92:
        src = cv2.perspectiveTransform(dst[None], np.linalg.inv(M))[0]      # marker centres in the photo
        sheet_px = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
        if sheet_px < 550:
            raise ValueError(f"The photo resolution is too low (the sheet is only about {int(sheet_px)} pixels tall). "
                             "Laptop webcams are usually too blurry for OMR — upload a phone photo or a scan instead.")
        raise ValueError("The bubble layout doesn't match this answer key (wrong key selected, or the sheet was printed "
                         "for a different exam). Choose the right key, or reprint the sheets from “Print sheets”.")
    qr_text = _decode_qr(gray, np.linalg.inv(M), warped)

    r_in = int(BUBBLE_R * SCALE * 0.62)
    yy, xx = np.ogrid[-r_in:r_in + 1, -r_in:r_in + 1]
    disk = (xx * xx + yy * yy) <= r_in * r_in
    flat = _flatten(warped)
    lay = layout(num_questions, num_options)
    ratios_all, dark_all = [], []
    for item in lay:
        ratios, darks = [], []
        for bx, by in item["bubbles"]:
            cx, cy = int(bx * SCALE), int(by * SCALE)
            patch = bw[cy - r_in:cy + r_in + 1, cx - r_in:cx + r_in + 1]
            g = flat[cy - r_in:cy + r_in + 1, cx - r_in:cx + r_in + 1]
            ok = patch.shape == disk.shape
            ratios.append(float((patch[disk] > 0).mean()) if ok else 0.0)
            darks.append(float(255 - g[disk].mean()) if ok else 0.0)
        ratios_all.append(ratios); dark_all.append(darks)
    # How dark this student's marks normally are. A filled bubble much fainter than the student's other
    # marks is typically a half-erased answer or a hesitant mark → flag it instead of trusting it.
    filled_dark = [d for rs, ds in zip(ratios_all, dark_all) for r, d in zip(rs, ds) if r >= MARK_T]
    typical = float(np.median(filled_dark)) if len(filled_dark) >= 5 else None
    results = []
    for item, ratios, darks in zip(lay, ratios_all, dark_all):
        faint = [LETTERS[i] for i, (r, d) in enumerate(zip(ratios, darks))
                 if typical and ((r >= MARK_T and d < FAINT_REL * typical)          # filled but much lighter
                                 or (r < LIGHT_T and d >= SHADE_REL * typical))]    # visibly shaded, below ink level
        marked = [LETTERS[i] for i, r in enumerate(ratios) if r >= MARK_T and LETTERS[i] not in faint]
        light = [LETTERS[i] for i, r in enumerate(ratios) if LIGHT_T <= r < MARK_T] + faint
        if len(marked) > 1:
            status = "multiple"
        elif len(marked) == 1 and not light:
            status = "ok"
        elif len(marked) == 1 and light:
            status = "uncertain"          # a clear mark plus a partial/erased one
        elif light:
            status = "uncertain"          # only light / partial marks
        else:
            status = "blank"
        results.append({"q": item["q"], "marked": marked, "light": light, "status": status,
                        "fill": [round(r, 2) for r in ratios], "darkness": [round(d / 255, 2) for d in darks]})
    return {"qr": qr_text or "", "questions": results, "layout_score": round(score_, 3)}


def parse_qr(text: str) -> Tuple[Optional[int], str]:
    if text and text.startswith("OMR:"):
        parts = text.split(":", 2)
        try:
            return int(parts[1]), (parts[2] if len(parts) > 2 else "").strip()
        except ValueError:
            pass
    return None, ""


def score(questions: List[Dict], key: List[str], marks_correct: float, marks_wrong: float,
          overrides: Optional[Dict[int, str]] = None, multiple_policy: str = "flag",
          allow_negative_total: bool = False) -> Dict:
    """
    overrides: {q: "A" | ""} final answers set by a teacher during verification.
    multiple_policy: "flag" → teacher decides; "wrong" → counts as an incorrect answer (negative marking applies);
                     "zero" → no marks and no penalty. Uncertain (light/partial) marks are always flagged.
    """
    overrides = overrides or {}
    correct = incorrect = unanswered = flagged = 0
    detail = []
    for q in questions:
        n = q["q"]
        if n in overrides:
            ans, st = overrides[n], "verified"
        elif q["status"] == "ok":
            ans, st = q["marked"][0], "ok"
        elif q["status"] == "blank":
            ans, st = "", "blank"
        elif q["status"] == "multiple" and multiple_policy in ("wrong", "zero"):
            ans, st = ("*" if multiple_policy == "wrong" else ""), "multiple"
        else:
            ans, st = None, q["status"]       # needs verification
        k = key[n - 1] if n - 1 < len(key) else ""
        if ans is None:
            flagged += 1
            res = "flagged"
        elif ans == "":
            unanswered += 1
            res = "invalid (multiple marks)" if st == "multiple" else "unanswered"
        elif ans == k:
            correct += 1
            res = "correct"
        else:
            incorrect += 1
            res = "incorrect"
        detail.append({**q, "final_answer": ans, "key": k, "result": res, "reading": st})
    total = round(correct * marks_correct - incorrect * marks_wrong, 2)
    return {"correct": correct, "incorrect": incorrect, "unanswered": unanswered, "flagged": flagged,
            "score": total if allow_negative_total else max(total, 0.0),
            "max_score": round(len(key) * marks_correct, 2), "detail": detail}
