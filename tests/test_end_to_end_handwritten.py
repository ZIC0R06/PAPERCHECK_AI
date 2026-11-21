"""
Complete workflow on the isolated test DB, through the real API:

 admin: batch, teacher (+ subject assignment), student
 teacher: question paper with model answers + marking scheme (/exam-papers)
 teacher: uploads a real image-only handwritten answer sheet (tests/fixtures) → read + AI evaluation
 teacher: sees flags, original-handwriting crops, criterion-wise breakdown; must verify flagged answers;
          changes a mark with a reason; approves; publishes
 student: sees question-wise marks + deductions; downloads the ORIGINAL and the EVALUATED PDF (marks written
          beside the answers on the student's own pages)
 re-evaluation: student disputes Q3 → teacher approves new marks → evaluated PDF + totals update
 OMR + written: teacher scans an OMR sheet → compile → approve → publish → student dashboard shows the total
"""
import io
import os
import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import init_db

U = uuid.uuid4().hex[:4].upper()
SUBJ, EXAM, BATCH, REG = "CS301", f"E2E Handwritten {U}", f"EH{U}", f"EH{U}01"
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "handwritten_answer_sheet.pdf")


@pytest.fixture(scope="module")
def c():
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        os.environ[k] = ""                      # offline engines: deterministic, no network
    init_db()
    with TestClient(app) as client:
        yield client


def _login(c, u, p):
    r = c.post("/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def ctx(c):
    A = _login(c, "admin", "admin123")
    assert c.post("/academic/batches", headers=A, json={"batch_code": BATCH, "programme": "B.Tech CSE", "course_code": "CSE",
                                                         "section": "A", "semester": "Semester 5",
                                                         "academic_session": "2026-27"}).status_code == 200
    t = c.post("/academic/teachers", headers=A, json={"full_name": "Dr E2E", "email": f"e2e{U}@x.edu",
                                                      "department_code": "CSE"}).json()
    assert c.post(f"/academic/teachers/{t['teacher_code']}/assignments", headers=A,
                  json={"subject_code": SUBJ, "batch_code": BATCH}).status_code == 200
    s = c.post("/academic/students", headers=A, json={"registration_number": REG, "name": "Hand Writer",
                                                     "email": f"hw{U}@x.edu", "batch_code": BATCH,
                                                     "subject_codes": [SUBJ]}).json()
    T = _login(c, t["teacher_code"], t["temporary_password"])
    S = _login(c, REG, s["temporary_password"])
    return {"A": A, "T": T, "S": S, "teacher": t}


def test_full_handwritten_workflow(c, ctx):
    A, T, S = ctx["A"], ctx["T"], ctx["S"]

    # 1. question paper with model answers + marking scheme
    paper = c.post("/exam-papers", headers=T, json={"course_code": SUBJ, "exam_name": EXAM, "questions": [
        {"question_number": 1, "question_text": "Define primary key with an example.", "marks": 10,
         "model_answer": "A primary key uniquely identifies each record in a table. It must be unique and not null. Example: roll number.",
         "rubric": [{"criterion": "uniquely identifies each record", "marks": 4}, {"criterion": "unique and not null", "marks": 3},
                    {"criterion": "example such as roll number", "marks": 3}]},
        {"question_number": 2, "question_text": "Explain normalization with 1NF, 2NF and 3NF.", "marks": 10,
         "model_answer": "Normalization organises data to reduce redundancy. 1NF removes repeating groups, 2NF removes partial dependency, 3NF removes transitive dependency."},
        {"question_number": 3, "question_text": "What is a deadlock? How can it be prevented?", "marks": 10,
         "model_answer": "Deadlock: two or more transactions wait for each other's locks forever. Prevention: lock ordering, timeouts, wait-die."},
        {"question_number": 4, "question_text": "Which SQL clause filters groups?", "marks": 2, "question_type": "mcq",
         "correct_option": "C"},
    ]})
    assert paper.status_code == 200, paper.text
    p = paper.json()
    assert p["num_questions"] == 4 and p["total_marks"] == 32 and p["ready_for_evaluation"], p["warnings"]
    # another teacher's subject is refused
    assert c.post("/exam-papers", headers=T, json={"course_code": "MA101", "exam_name": EXAM, "questions": []}).status_code == 403

    # 2. upload the handwritten sheet (image-only PDF)
    up = c.post("/examiner/booklet/upload", headers=T, files={"file": ("sheet.pdf", open(FIX, "rb").read(), "application/pdf")},
                data={"student_registration": REG, "course_code": SUBJ, "exam_name": EXAM})
    assert up.status_code == 200, up.text
    bid = up.json()["booklet_id"]
    # student can't see anything yet
    assert c.get(f"/student/answer-sheet/{bid}/evaluated", headers=S).status_code == 403

    # 3. read + evaluate
    ev = c.post(f"/examiner/evaluate-booklet/{bid}", headers=T)
    assert ev.status_code == 200, ev.text
    e = ev.json()
    assert e["reader"] == "local-tesseract" and e["max_marks"] == 32 and e["skipped_questions"] == []
    d = c.get(f"/examiner/booklet/{bid}", headers=T).json()
    qs = {q["question_number"]: q for q in d["questions"]}
    assert set(qs) == {1, 2, 3, 4}
    assert "primary key" in qs[1]["student_answer"].lower() and "deadlock" in qs[3]["student_answer"].lower()
    assert qs[1]["page_refs"][0]["page"] == 1 and qs[3]["page_refs"][0]["page"] == 2
    assert qs[1]["ai_score"] >= 7 and len(qs[1]["rubric_breakdown"]) == 3          # criterion-wise
    assert qs[3]["ai_score"] < 10 and qs[3]["deductions"]                           # incomplete answer loses marks
    assert qs[4]["student_answer"] == "" and qs[4]["ai_score"] == 0 and "Not attempted" in qs[4]["deductions"][0]["reason"]
    assert all(q["flagged_for_review"] for q in (qs[1], qs[2], qs[3]))             # local OCR + offline marking → verify
    assert e["total_ai_score"] == round(sum(q["ai_score"] for q in qs.values()), 2)
    img = c.get(f"/examiner/booklet/{bid}/page/{qs[1]['page_refs'][0]['page']}.png", headers=T,
                params={"y0": qs[1]["page_refs"][0]["y0"], "y1": qs[1]["page_refs"][0]["y1"]})
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    st = c.get("/system/ai-status", headers=T).json()
    assert st["vision_ai"] is False and st["notes"]

    # 4. teacher review: flagged answers must be verified before approval
    r = c.post(f"/examiner/approve-booklet/{bid}", headers=T)
    assert r.status_code == 409 and "flagged" in r.json()["detail"]
    assert c.post("/examiner/review-question", headers=T, json={"result_id": qs[2]["result_id"], "examiner_score": 6,
                                                                "examiner_comments": "", "flag_for_review": False}).status_code in (200, 422)
    r = c.post("/examiner/review-question", headers=T, json={"result_id": qs[2]["result_id"], "examiner_score": 6,
                                                             "examiner_comments": "Did not explain 1NF properly", "flag_for_review": False})
    assert r.status_code == 200
    for n in (1, 3, 4):
        assert c.post("/examiner/review-question", headers=T, json={"result_id": qs[n]["result_id"], "examiner_score": qs[n]["ai_score"],
                                                                    "examiner_comments": "", "flag_for_review": False}).status_code == 200
    assert c.post(f"/examiner/publish-booklet/{bid}", headers=T).status_code == 409   # not approved yet
    r = c.post(f"/examiner/approve-booklet/{bid}", headers=T)
    assert r.status_code == 200, r.text
    expected_total = round(qs[1]["ai_score"] + 6 + qs[3]["ai_score"] + 0, 2)
    assert r.json()["total_final_score"] == expected_total
    assert c.post(f"/examiner/publish-booklet/{bid}", headers=T).status_code == 200

    # 5. student portal
    res = c.get("/student/results", headers=S).json()
    assert any(x["booklet_id"] == bid for x in res)
    det = c.get(f"/student/result/{bid}", headers=S).json()
    assert det["total_awarded"] == expected_total and det["total_max"] == 32
    q2 = next(q for q in det["questions"] if q["question_number"] == 2)
    assert q2["marks_awarded"] == 6 and q2["teacher_changed_marks"] and q2["examiner_comments"] == "Did not explain 1NF properly"
    assert any(dd["criterion"] == "Teacher's adjustment" for dd in q2["deductions"])
    orig = c.get(f"/student/answer-sheet/{bid}/original", headers=S)
    assert orig.status_code == 200 and orig.content == open(FIX, "rb").read()          # untouched original
    evp = c.get(f"/student/answer-sheet/{bid}/evaluated", headers=S)
    assert evp.status_code == 200 and evp.content[:4] == b"%PDF"
    doc = pymupdf.open(stream=evp.content, filetype="pdf")
    text = "\n".join(pg.get_text() for pg in doc)
    assert "OFFICIAL" in text and f"{expected_total:g} / 32" in text
    assert doc.page_count >= 4                                    # summary + 2 annotated original pages + details
    page1 = doc[1].get_text()
    assert "Q1" in page1 and "Q2" in page1 and "6 / 10" in page1 and "Did not explain 1NF properly" in page1
    assert "Q3" in doc[2].get_text()
    assert "Not attempted" in text
    # other students / unauthenticated users can't get it
    other = _login(c, "student", "student123")
    assert c.get(f"/student/answer-sheet/{bid}/evaluated", headers=other).status_code == 403
    assert c.get(f"/student/answer-sheet/{bid}/evaluated").status_code == 401

    # 6. re-evaluation: student disputes Q3, teacher grants 8 marks
    r = c.post("/student/re-evaluation", headers=S, json={"booklet_id": bid, "question_number": 3,
                                                          "reason": "I explained prevention with timeouts which is valid."})
    assert r.status_code == 200, r.text
    rid = r.json()["request_id"]
    reqs = c.get("/examiner/re-evaluations", headers=T).json()
    rq = next(x for x in reqs if x["id"] == rid)
    assert rq["score_before"] == qs[3]["ai_score"]
    assert c.post("/examiner/resolve-reevaluation", headers=T, json={"request_id": rid, "status": "approved",
                                                                     "examiner_remarks": "Valid", "score_after": 99}).status_code == 422
    r = c.post("/examiner/resolve-reevaluation", headers=T, json={"request_id": rid, "status": "approved",
                                                                  "examiner_remarks": "Timeouts accepted as prevention.", "score_after": 8})
    assert r.status_code == 200, r.text
    det2 = c.get(f"/student/result/{bid}", headers=S).json()
    assert det2["total_awarded"] == round(qs[1]["ai_score"] + 6 + 8, 2)
    ev2 = pymupdf.open(stream=c.get(f"/student/answer-sheet/{bid}/evaluated", headers=S).content, filetype="pdf")
    assert "8 / 10" in ev2[2].get_text() and "Timeouts accepted" in "\n".join(p.get_text() for p in ev2)
    assert c.post("/examiner/resolve-reevaluation", headers=T, json={"request_id": rid, "status": "approved",
                                                                     "examiner_remarks": "again", "score_after": 9}).status_code == 409
    notes = c.get("/notifications", headers=S).json()
    assert any("published" in n["title"] for n in notes["items"]) and any("Re-evaluation" in n["title"] for n in notes["items"])


def test_answer_sheet_uploaded_before_paper_is_linked_later(c, ctx):
    T = ctx["T"]
    exam = f"Late paper {U}"
    up = c.post("/examiner/booklet/upload", headers=T, files={"file": ("s.pdf", open(FIX, "rb").read(), "application/pdf")},
                data={"student_registration": REG, "course_code": SUBJ, "exam_name": exam}).json()
    r = c.post(f"/examiner/evaluate-booklet/{up['booklet_id']}", headers=T)
    # no paper yet: either refused, or marked against the subject's course notes and flagged for the teacher
    if r.status_code == 200:
        d = c.get(f"/examiner/booklet/{up['booklet_id']}", headers=T).json()
        assert all("course notes" in q["flag_reason"] for q in d["questions"] if q["result_id"])
    else:
        assert r.status_code == 422
    p = c.post("/exam-papers", headers=T, json={"course_code": SUBJ, "exam_name": exam, "questions": [
        {"question_number": n, "question_text": f"Question {n}", "marks": 5, "model_answer": "primary key unique row"} for n in (1, 2, 3)]}).json()
    assert p["answer_sheets_linked"] == 1
    r = c.post(f"/examiner/evaluate-booklet/{up['booklet_id']}", headers=T, params={"reread": "true"})
    assert r.status_code == 200 and r.json()["max_marks"] == 15


def test_combined_omr_and_written_result(c, ctx):
    import cv2
    import numpy as np
    from backend import omr as O
    T, S = ctx["T"], ctx["S"]
    k = c.post("/omr/keys", headers=T, json={"subject_code": SUBJ, "exam_name": EXAM, "answers": "ABCDA",
                                             "num_options": 4, "marks_correct": 2, "marks_wrong": 0.5}).json()
    pdf = O.sheets_pdf(k["id"], SUBJ, EXAM, 5, 4, [(REG, "Hand Writer")])
    pg = pymupdf.open(stream=pdf, filetype="pdf")[0].get_pixmap(dpi=150)
    img = np.frombuffer(pg.samples, np.uint8).reshape(pg.height, pg.width, pg.n)[:, :, :3].copy()
    px = 150 / 72
    for it, a in zip(O.layout(5, 4), "ABCDB"):                      # 4 correct, 1 wrong → 8 − 0.5 = 7.5
        bx, by = it["bubbles"]["ABCD".index(a)]
        cv2.circle(img, (int(bx * px), int(by * px)), int(O.BUBBLE_R * px * .9), (30, 30, 30), -1)
    r = c.post(f"/omr/keys/{k['id']}/scan", headers=T, files={"file": ("omr.png", cv2.imencode(".png", img)[1].tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["score"] == 7.5 and r.json()["registration_number"] == REG
    comp = c.post("/results/compile", headers=T, json={"subject_code": SUBJ, "exam_name": EXAM, "registration_number": REG})
    assert comp.status_code == 200, comp.text
    fr = c.get("/results/final", headers=T, params={"subject_code": SUBJ, "exam_name": EXAM}).json()
    mine = next(x for x in fr if x["registration_number"] == REG)
    assert mine["omr_score"] == 7.5 and mine["max_total"] == 42
    assert c.post(f"/results/final/{mine['id']}/approve", headers=T).status_code == 200
    assert c.post(f"/results/final/{mine['id']}/publish", headers=T).status_code == 200
    dash = c.get("/results/my/dashboard", headers=S).json()
    assert any(r.get("subject_code") == SUBJ and r.get("exam_name") == EXAM and r.get("marks") == mine["total"]
               for r in dash.get("results", dash.get("subjects", []))) or str(mine["total"]) in str(dash)
    pdf = c.get("/results/my/final.pdf", headers=S)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_scanner_photos_of_handwritten_pages_are_read(c, ctx):
    """Camera/phone path: page photos → scanner (crop, deskew, enhance) → PDF → handwriting reading → marks."""
    import cv2
    import numpy as np
    T = ctx["T"]
    exam = f"Scanner {U}"
    c.post("/exam-papers", headers=T, json={"course_code": SUBJ, "exam_name": exam, "questions": [
        {"question_number": 1, "marks": 5, "model_answer": "A primary key uniquely identifies each row and is not null."},
        {"question_number": 2, "marks": 5, "model_answer": "Normalization reduces redundancy."},
        {"question_number": 3, "marks": 5, "model_answer": "Deadlock is a circular wait between transactions."}]})
    sid = c.post("/scanner/sessions", headers=T, json={"registration_number": REG, "subject_code": SUBJ, "exam_name": exam}).json()["id"]
    doc = pymupdf.open(FIX)
    for i, page in enumerate(doc):                      # simulate a phone photo: page on a desk, slightly rotated
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        h, w = img.shape[:2]
        bg = np.full((int(h * 1.15), int(w * 1.15), 3), (60, 70, 85), np.uint8)
        bg[(bg.shape[0] - h) // 2:(bg.shape[0] - h) // 2 + h, (bg.shape[1] - w) // 2:(bg.shape[1] - w) // 2 + w] = img
        H, W = bg.shape[:2]
        bg = cv2.warpAffine(bg, cv2.getRotationMatrix2D((W / 2, H / 2), 3 if i == 0 else -2, 1.0), (W, H), borderValue=(60, 70, 85))
        r = c.post(f"/scanner/sessions/{sid}/pages", headers=T,
                   files={"file": (f"p{i}.jpg", cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
    sub = c.post(f"/scanner/sessions/{sid}/submit", headers=T)
    assert sub.status_code == 200, sub.text
    bid = sub.json()["booklet_id"]
    ev = c.post(f"/examiner/evaluate-booklet/{bid}", headers=T)
    assert ev.status_code == 200, ev.text
    qs = {q["question_number"]: q for q in c.get(f"/examiner/booklet/{bid}", headers=T).json()["questions"]}
    assert "primary key" in qs[1]["student_answer"].lower()
    assert "deadlock" in qs[3]["student_answer"].lower() and qs[3]["page_refs"][0]["page"] == 2
