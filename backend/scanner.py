"""
Document scanner image pipeline (OpenCV)
=======================================
process_page(bytes) → (processed JPEG bytes, quality report)

  1. EXIF orientation fix
  2. document boundary detection (largest 4-corner contour, Canny + contour approximation)
  3. perspective correction to a flat, upright page
  4. rotation: portrait orientation + Tesseract OSD for upside-down pages (when available)
  5. enhancement: background flattening, CLAHE contrast, light sharpening (keeps pencil/pen strokes)
  6. quality check: blur (Laplacian variance), brightness, contrast, resolution, boundary found
"""

import io
import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
MAX_SIDE = 2200  # output resolution cap (≈ 190 dpi for A4)


def _decode(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _order(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_document(img: np.ndarray):
    """Return the 4 page corners (in img coordinates) or None."""
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img.copy()
    scale = min(scale, 1.0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    best, best_area = None, 0
    area_min = 0.2 * small.shape[0] * small.shape[1]
    for lo, hi in ((50, 150), (25, 90), (75, 200)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            a = cv2.contourArea(approx)
            if len(approx) == 4 and a > area_min and a > best_area and cv2.isContourConvex(approx):
                best, best_area = approx, a
        if best is not None:
            break
    if best is None:
        return None
    return _order(best / scale)


def warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    wdt = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    hgt = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [wdt - 1, 0], [wdt - 1, hgt - 1], [0, hgt - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (wdt, hgt))


def _osd_rotation(gray: np.ndarray) -> int:
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(gray, config="--psm 0", output_type=pytesseract.Output.DICT)
        if float(osd.get("orientation_conf", 0)) >= 2.0:
            return int(osd.get("rotate", 0))
    except Exception:
        pass
    return 0


def enhance(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # flatten uneven lighting (shadows from phones) by dividing by a blurred background
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    flat = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    sharp = cv2.addWeighted(clahe, 1.4, cv2.GaussianBlur(clahe, (0, 0), 2), -0.4, 0)
    return sharp


def quality(img_gray: np.ndarray, boundary_found: bool) -> Dict:
    h, w = img_gray.shape[:2]
    norm = cv2.resize(img_gray, (int(w * 1000 / max(h, w)), int(h * 1000 / max(h, w))))
    blur = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    bright, contrast = float(norm.mean()), float(norm.std())
    warnings: List[str] = []
    if blur < 60:
        warnings.append("Image looks blurry — hold the camera steady and retake.")
    if bright < 80:
        warnings.append("Too dark — use more light.")
    if bright > 245:
        warnings.append("Overexposed — reduce glare.")
    if contrast < 25:
        warnings.append("Low contrast — text may be hard to read.")
    if min(h, w) < 900:
        warnings.append("Low resolution — move closer or use a higher camera resolution.")
    if not boundary_found:
        warnings.append("Page edges not detected — place the sheet on a darker background with all 4 corners visible.")
    score = 100
    score -= 35 if blur < 60 else (15 if blur < 120 else 0)
    score -= 20 if (bright < 80 or bright > 245) else 0
    score -= 15 if contrast < 25 else 0
    score -= 15 if min(h, w) < 900 else 0
    score -= 10 if not boundary_found else 0
    return {"score": max(0, score), "blur_variance": round(blur, 1), "brightness": round(bright, 1),
            "contrast": round(contrast, 1), "width": w, "height": h, "boundary_found": boundary_found,
            "warnings": warnings, "ok": score >= 60}


def process_page(data: bytes, detect_boundary: bool = True) -> Tuple[bytes, Dict]:
    img = _decode(data)
    quad = find_document(img) if detect_boundary else None
    page = warp(img, quad) if quad is not None else img
    if page.shape[1] > page.shape[0] * 1.15:          # landscape → portrait
        page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
    h, w = page.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        page = cv2.resize(page, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    raw_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    out = enhance(page)
    rot = _osd_rotation(out)
    if rot in (90, 180, 270):
        out = cv2.rotate(out, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot])
    q = quality(raw_gray, quad is not None)   # judge the photo itself, not the enhanced output
    q["auto_rotated"] = rot
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), q


def rotate_jpeg(data: bytes, degrees: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees % 360]
    ok, buf = cv2.imencode(".jpg", cv2.rotate(img, code), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def images_to_pdf(paths: List[str], out_path: str):
    pages = [Image.open(p).convert("RGB") for p in paths]
    if not pages:
        raise ValueError("No pages to combine.")
    pages[0].save(out_path, "PDF", resolution=200.0, save_all=True, append_images=pages[1:])
