"""Builds realistic phone-photo images of filled OMR sheets for tests."""
import random

import cv2
import numpy as np
from pdf2image import convert_from_bytes

from backend import omr as O


def photo_of_sheet(key_id, subject, exam, nq, nopt, reg, answers, multi=None, light=None, angle=5, seed=1):
    rng = random.Random(seed)
    pdf = O.sheets_pdf(key_id, subject, exam, nq, nopt, [(reg, "Student")])
    img = np.array(convert_from_bytes(pdf, dpi=150)[0].convert("RGB"))[:, :, ::-1].copy()
    px = 150 / 72.0
    for item in O.layout(nq, nopt):
        q = item["q"]
        marks = []
        if answers.get(q):
            marks.append((answers[q], 1))
        if multi and q in multi:
            marks.append((multi[q], 1))
        if light and q in light:
            marks.append((light[q], 0))
        for letter, full in marks:
            bx, by = item["bubbles"][O.LETTERS.index(letter)]
            cx, cy = int(bx * px + rng.uniform(-1.5, 1.5)), int(by * px + rng.uniform(-1.5, 1.5))
            if full:
                cv2.circle(img, (cx, cy), int(O.BUBBLE_R * px * 0.95), (30, 30, 60), -1)
            else:   # a faint, partial pencil stroke
                r = int(O.BUBBLE_R * px * 0.6)
                cv2.line(img, (cx - r, cy), (cx + r, cy), (90, 90, 90), max(2, r // 2))
    h, w = img.shape[:2]
    bg = np.full((int(h * 1.3), int(w * 1.3), 3), (70, 80, 95), np.uint8)
    oy, ox = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
    bg[oy:oy + h, ox:ox + w] = img
    H, W = bg.shape[:2]
    bg = cv2.warpAffine(bg, cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0), (W, H), borderValue=(70, 80, 95))
    bg = np.clip(bg + np.random.default_rng(seed).normal(0, 5, bg.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
