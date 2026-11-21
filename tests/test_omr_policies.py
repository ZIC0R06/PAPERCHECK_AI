"""
OMR: negative marking, multiple-mark policy, faint (half-erased) marks and batch scanning of a multi-page PDF.
Runs through the real API on the isolated test database.
"""
import io
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import omr as O
from backend.api import app
from backend.database import init_db

U = uuid.uuid4().hex[:4].upper()


def _sheet(key_id, reg, marks, nq=10, nopt=4, dpi=150):
    """marks: {q: [(letter, grey_level)]}; grey 40 = dark pen, 190 = faint/half-erased."""
    import pymupdf
    pdf = O.sheets_pdf(key_id, "CS301", "OMR policy", nq, nopt, [(reg, "Student")])
    p = pymupdf.open(stream=pdf, filetype="pdf")[0].get_pixmap(dpi=dpi)
    img = np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, p.n)[:, :, :3].copy()
    px = dpi / 72
    for it in O.layout(nq, nopt):
        for letter, grey in marks.get(it["q"], []):
            bx, by = it["bubbles"][O.LETTERS.index(letter)]
            cv2.circle(img, (int(bx * px), int(by * px)), int(O.BUBBLE_R * px * .92), (grey, grey, grey), -1)
    return img


def test_faint_mark_among_dark_marks_is_flagged_not_trusted():
    marks = {q: [("ABCD"[q % 4], 40)] for q in range(1, 10)}
    marks[10] = [("B", 185)]                           # half-erased looking mark
    img = _sheet(3, "X1", marks)
    r = O.read_sheet(cv2.imencode(".png", img)[1].tobytes(), 10, 4)
    by_q = {q["q"]: q for q in r["questions"]}
    assert all(by_q[q]["status"] == "ok" for q in range(1, 10))
    assert by_q[10]["status"] == "uncertain" and "B" in by_q[10]["light"]


def test_scoring_policies():
    qs = [{"q": 1, "marked": ["A"], "status": "ok"}, {"q": 2, "marked": ["A", "B"], "status": "multiple"},
          {"q": 3, "marked": ["C"], "status": "ok"}]
    key = ["A", "B", "D"]
    flag = O.score(qs, key, 4, 1)
    assert flag["flagged"] == 1 and flag["score"] == 3                     # 4 − 1 (Q3 wrong), Q2 waits for teacher
    wrong = O.score(qs, key, 4, 1, multiple_policy="wrong")
    assert wrong["flagged"] == 0 and wrong["incorrect"] == 2 and wrong["score"] == 2
    zero = O.score(qs, key, 4, 1, multiple_policy="zero")
    assert zero["unanswered"] == 1 and zero["score"] == 3
    assert zero["detail"][1]["result"] == "invalid (multiple marks)"
    all_wrong = [{"q": 1, "marked": ["B"], "status": "ok"}, {"q": 2, "marked": ["A"], "status": "ok"}]
    assert O.score(all_wrong, ["A", "B"], 1, 0.25)["score"] == 0
    assert O.score(all_wrong, ["A", "B"], 1, 0.25, allow_negative_total=True)["score"] == -0.5


@pytest.fixture(scope="module")
def c():
    init_db()
    with TestClient(app) as client:
        yield client


def _login(c, u, p):
    return {"Authorization": "Bearer " + c.post("/auth/login", json={"username": u, "password": p}).json()["access_token"]}


def test_batch_scan_of_multipage_pdf_with_negative_marking(c):
    A = _login(c, "admin", "admin123")
    assert c.post("/academic/batches", headers=A, json={"batch_code": f"OB{U}", "programme": "B.Tech", "course_code": "CSE",
                                                         "section": "A", "semester": "Semester 5", "academic_session": "2026-27"}).status_code == 200
    regs = [f"OB{U}1", f"OB{U}2"]
    for r in regs:
        assert c.post("/academic/students", headers=A, json={"registration_number": r, "name": "Omr " + r,
                                                             "email": f"{r}@x.edu", "batch_code": f"OB{U}",
                                                             "subject_codes": ["CS301"]}).status_code == 200
    k = c.post("/omr/keys", headers=A, json={"subject_code": "CS301", "exam_name": f"OMR policy {U}", "answers": "ABCDABCDAB",
                                             "num_options": 4, "marks_correct": 4, "marks_wrong": 1,
                                             "multiple_policy": "wrong"}).json()
    assert k["multiple_policy"] == "wrong"
    key = "ABCDABCDAB"
    s1 = {q: [(key[q - 1], 40)] for q in range(1, 11)}                     # all correct → 40
    s2 = {q: [(key[q - 1], 40)] for q in range(1, 7)}                      # 6 correct
    s2[7] = [("A", 40)]                                                     # wrong (key C) → −1
    s2[8] = [("A", 40), ("D", 40)]                                          # multiple → counted wrong → −1
    pages = [Image.fromarray(_sheet(k["id"], regs[0], s1)), Image.fromarray(_sheet(k["id"], regs[1], s2))]
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    r = c.post(f"/omr/keys/{k['id']}/scan-batch", headers=A, files=[("files", ("stack.pdf", buf.getvalue(), "application/pdf")),
                                                                    ("files", ("note.txt", b"hello", "text/plain"))])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["sheets"] == 3 and out["read"] == 2
    got = {x["registration_number"]: x for x in out["results"] if x.get("ok")}
    assert got[regs[0]]["score"] == 40 and got[regs[1]]["score"] == 22    # 6×4 − 1 − 1
    assert not [x for x in out["results"] if x.get("ok") and x["flagged"]]
    bad = [x for x in out["results"] if not x.get("ok")][0]
    assert "JPG/PNG" in bad["error"]
    # rescanning without "replace" is refused per sheet, not as a crash
    r2 = c.post(f"/omr/keys/{k['id']}/scan-batch", headers=A, files=[("files", ("stack.pdf", buf.getvalue(), "application/pdf"))]).json()
    assert r2["read"] == 0 and all("already recorded" in x["error"] for x in r2["results"])
