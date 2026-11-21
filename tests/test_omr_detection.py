"""OMR reading robustness on simulated phone photos (rotation, desk background, noise)."""
import random

import pytest

from backend import omr as O
from tests.omr_testkit import photo_of_sheet


@pytest.mark.parametrize("angle,seed", [(6, 1), (-9, 2), (3, 3), (0, 4)])
def test_reads_every_bubble(angle, seed):
    rnd = random.Random(seed)
    nq = 40
    answers = {q: rnd.choice("ABCD") for q in range(1, nq + 1)}
    for q in (5, 17):
        answers.pop(q)                                  # blanks
    multi = {30: "A" if answers[30] != "A" else "B"}    # double-marked
    light = {5: "C"}                                    # only a faint partial mark
    r = O.read_sheet(photo_of_sheet(3, "CS301", "Mid", nq, 4, "21BCS101", answers, multi=multi, light=light,
                                    angle=angle, seed=seed), nq, 4)
    assert O.parse_qr(r["qr"]) == (3, "21BCS101")
    for q in r["questions"]:
        n = q["q"]
        if n in multi:
            assert q["status"] == "multiple"
        elif n in light:
            assert q["status"] == "uncertain"
        elif n not in answers:
            assert q["status"] == "blank"
        else:
            assert q["status"] == "ok" and q["marked"] == [answers[n]], (n, q)


def test_scoring_with_negative_marks_and_verification():
    qs = [{"q": 1, "marked": ["A"], "light": [], "status": "ok", "fill": []},
          {"q": 2, "marked": ["C"], "light": [], "status": "ok", "fill": []},
          {"q": 3, "marked": [], "light": [], "status": "blank", "fill": []},
          {"q": 4, "marked": ["A", "B"], "light": [], "status": "multiple", "fill": []}]
    s = O.score(qs, list("ABCD"), 2.0, 0.5)
    assert (s["correct"], s["incorrect"], s["unanswered"], s["flagged"], s["score"]) == (1, 1, 1, 1, 1.5)
    s2 = O.score(qs, list("ABCD"), 2.0, 0.5, overrides={4: "D"})
    assert s2["flagged"] == 0 and s2["score"] == 3.5


def test_unreadable_image_is_rejected():
    import cv2, numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((800, 600, 3), 200, np.uint8))
    with pytest.raises(ValueError):
        O.read_sheet(buf.tobytes(), 10, 4)


# ── real-world capture conditions (regression for field failures) ─────────────

def _photo(transform, nq=30):
    import cv2, numpy as np
    from pdf2image import convert_from_bytes
    rnd = random.Random(7)
    ans = {q: rnd.choice("ABCD") for q in range(1, nq + 1)}
    img = np.array(convert_from_bytes(O.sheets_pdf(5, "CS301", "Mid", nq, 4, [("21BCS101", "A")]), dpi=250)[0].convert("RGB"))[:, :, ::-1].copy()
    px = 250 / 72
    for it in O.layout(nq, 4):
        bx, by = it["bubbles"]["ABCD".index(ans[it["q"]])]
        cv2.circle(img, (int(bx * px), int(by * px)), int(O.BUBBLE_R * px * .9), (40, 40, 40), -1)
    h, w = img.shape[:2]
    bg = np.full((int(h * 1.3), int(w * 1.3), 3), (70, 80, 95), np.uint8)
    bg[(bg.shape[0] - h) // 2:(bg.shape[0] - h) // 2 + h, (bg.shape[1] - w) // 2:(bg.shape[1] - w) // 2 + w] = img
    return transform(bg), ans


def _rot(a):
    import cv2
    def f(im):
        H, W = im.shape[:2]
        if a % 180 == 90:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE if a == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.warpAffine(im, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1), (W, H), borderValue=(70, 80, 95))
    return f


def _shadow(im):
    import numpy as np
    H, W = im.shape[:2]
    g = np.tile(np.linspace(0.55, 1.05, W), (H, 1)) * np.tile(np.linspace(1.0, 0.75, H)[:, None], (1, W))
    return np.clip(im * g[..., None], 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name,fn", [("upside_down", _rot(180)), ("sideways", _rot(90)), ("sideways_ccw", _rot(-90)),
                                     ("tilted", _rot(11)), ("shadow", _shadow)])
def test_real_world_photos(name, fn):
    import cv2
    img, ans = _photo(fn)
    r = O.read_sheet(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), 30, 4)
    assert O.parse_qr(r["qr"]) == (5, "21BCS101"), name
    wrong = [(q["q"], q["status"], q["marked"]) for q in r["questions"] if q["status"] != "ok" or q["marked"] != [ans[q["q"]]]]
    assert not wrong, (name, wrong[:5])


def test_pdf_scan_without_poppler(monkeypatch):
    """A scanned PDF must be readable even when Poppler (pdf2image) is not installed — common on Windows."""
    import io, sys, cv2
    from PIL import Image
    img, ans = _photo(lambda im: im)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("L").save(buf, "PDF", resolution=200)
    monkeypatch.setitem(sys.modules, "pdf2image", None)            # simulate: pdf2image/Poppler unavailable
    r = O.read_sheet(buf.getvalue(), 30, 4)
    assert all(q["marked"] == [ans[q["q"]]] for q in r["questions"])


def test_wrong_key_layout_is_rejected_not_misread():
    import cv2
    img, _ = _photo(lambda im: im, nq=30)
    with pytest.raises(ValueError, match="layout doesn't match"):
        O.read_sheet(cv2.imencode(".jpg", img)[1].tobytes(), 100, 5)


def test_long_names_do_not_cover_qr():
    import numpy as np, cv2
    from pdf2image import convert_from_bytes
    pdf = O.sheets_pdf(8, "CS301 Design and Analysis of Algorithms (Advanced Section)",
                       "End Semester Examination November December 2026 Main Paper", 20, 4,
                       [("21BCS101", "A Student With A Very Long Name Indeed")])
    img = np.array(convert_from_bytes(pdf, dpi=200)[0].convert("RGB"))[:, :, ::-1].copy()
    r = O.read_sheet(cv2.imencode(".png", img)[1].tobytes(), 20, 4)
    assert O.parse_qr(r["qr"]) == (8, "21BCS101")
