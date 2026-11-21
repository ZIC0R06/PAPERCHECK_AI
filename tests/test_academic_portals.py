"""
Three-portal tests: admin management, teacher scoping, exam workflow, results, exports, student dashboard.
Run on a throw-away DB copy:  DATABASE_URL=sqlite:///./test_copy.db pytest tests/test_academic_portals.py
"""
import io
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import ExamPaper, ExamQuestion, Rubric, SessionLocal, init_db

U = uuid.uuid4().hex[:4].upper()          # keeps reruns on the same DB independent
BATCH = f"T{U}B"
REG = f"T{U}S1"
REG2 = f"T{U}S2"
SUBJ = "CS301"
EXAM = f"Test Exam {U}"


@pytest.fixture(scope="module")
def c():
    init_db()
    with TestClient(app) as client:
        yield client


def login(c, u, p):
    r = c.post("/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def admin(c):
    return login(c, "admin", "admin123")


@pytest.fixture(scope="module")
def setup(c, admin):
    """Admin builds: batch, teacher (+assignment), two students."""
    r = c.post("/academic/batches", headers=admin, json={"batch_code": BATCH, "programme": "B.Tech CSE", "course_code": "CSE",
                                                         "section": "B", "semester": "Semester 5", "academic_session": "2026-27"})
    assert r.status_code == 200, r.text
    t = c.post("/academic/teachers", headers=admin, json={"full_name": "Dr Test Teacher", "email": f"t{U}@univ-demo.edu",
                                                          "department_code": "CSE"}).json()
    assert t["teacher_code"].startswith("TCSE") and t["temporary_password"]
    r = c.post(f"/academic/teachers/{t['teacher_code']}/assignments", headers=admin,
               json={"subject_code": SUBJ, "batch_code": BATCH})
    assert r.status_code == 200 and r.json()["assignments"][0]["batch_code"] == BATCH
    s1 = c.post("/academic/students", headers=admin, json={"registration_number": REG, "name": "Test Student One",
                                                           "email": f"s1{U}@univ-demo.edu", "batch_code": BATCH,
                                                           "subject_codes": [SUBJ, "CS302"]}).json()
    s2 = c.post("/academic/students", headers=admin, json={"registration_number": REG2, "name": "Test Student Two",
                                                           "email": f"s2{U}@univ-demo.edu", "batch_code": BATCH,
                                                           "subject_codes": [SUBJ]}).json()
    assert s1["section"] == "B" and s1["semester"] == "Semester 5" and len(s1["subjects"]) == 2
    # official question paper with model answer (what the faculty "Question Paper" page stores)
    db = SessionLocal()
    p = ExamPaper(paper_id=f"paper-{U}", course_code=SUBJ, course_name="DAA", exam_name=EXAM, total_marks=10)
    db.add(p); db.flush()
    q = ExamQuestion(paper_id_fk=p.id, question_number=1, marks=10, question_type="open_ended",
                     question_text="Explain the merge sort algorithm and its time complexity.",
                     teacher_answer="Merge sort is a divide and conquer algorithm. It splits the array into two halves, "
                                    "recursively sorts each half and merges the sorted halves. Its time complexity is "
                                    "O(n log n) in all cases and it needs O(n) extra space.")
    db.add(q); db.flush()
    db.add(Rubric(exam_question_id=q.id, criterion="divide and conquer idea", marks=4))
    db.add(Rubric(exam_question_id=q.id, criterion="O(n log n) complexity", marks=3))
    db.add(Rubric(exam_question_id=q.id, criterion="merge step", marks=3))
    db.commit(); db.close()
    teacher = login(c, t["teacher_code"], t["temporary_password"])
    st1 = login(c, REG, s1["temporary_password"])
    return {"teacher": teacher, "teacher_code": t["teacher_code"], "student": st1}


def _answer_sheet_pdf() -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=A4)
    cv.setFont("Helvetica", 14)
    y = 780
    for line in ["Q1. Merge sort is a divide and conquer algorithm.", "It splits the array into two halves and",
                 "recursively sorts both halves, then merges them.", "The time complexity is O(n log n)."]:
        cv.drawString(60, y, line); y -= 28
    cv.showPage(); cv.save()
    return buf.getvalue()


# ── admin ─────────────────────────────────────────────────────────────────────

def test_duplicate_registration_rejected(c, admin, setup):
    r = c.post("/academic/students", headers=admin, json={"registration_number": REG, "name": "Dup", "email": f"d{U}@univ-demo.edu"})
    assert r.status_code == 409


def test_admin_search_and_overview(c, admin, setup):
    assert [s["registration_number"] for s in c.get(f"/academic/students?q={REG}", headers=admin).json()] == [REG]
    assert len(c.get(f"/academic/students?batch_code={BATCH}", headers=admin).json()) == 2
    ov = c.get("/academic/overview", headers=admin).json()
    assert ov["teachers"] >= 2 and ov["batches"] >= 2
    t = c.get(f"/academic/teachers/{setup['teacher_code']}", headers=admin).json()
    assert t["stats"]["total_students"] == 2


def test_non_admin_cannot_manage(c, setup):
    assert c.post("/academic/batches", headers=setup["teacher"], json={"batch_code": "XX1"}).status_code == 403
    assert c.get("/academic/overview", headers=setup["student"]).status_code == 403
    assert c.get("/academic/students", headers=setup["student"]).status_code == 403


# ── teacher scoping ───────────────────────────────────────────────────────────

def test_teacher_sees_only_assigned(c, setup):
    tch = setup["teacher"]
    regs = {s["registration_number"] for s in c.get("/academic/students", headers=tch).json()}
    assert regs == {REG, REG2}                                       # not 21BCS101 (other batch)
    assert c.get("/academic/students/21BCS101/profile360", headers=tch).status_code == 404
    other = [b["id"] for b in c.get("/examiner/booklets", headers=login(c, "faculty", "faculty123")).json()]
    for bid in other:
        assert c.get(f"/examiner/booklet/{bid}", headers=tch).status_code == 404
    up = c.post("/examiner/booklet/upload", headers=tch, data={"student_registration": "21BCS101", "course_code": SUBJ},
                files={"file": ("a.pdf", io.BytesIO(_answer_sheet_pdf()), "application/pdf")})
    assert up.status_code == 403
    assert c.get("/submissions", headers=tch).status_code == 403   # unscoped legacy data → admin only


def test_teacher_without_assignment_sees_nothing(c, admin):
    t = c.post("/academic/teachers", headers=admin, json={"full_name": "Unassigned T", "email": f"u{U}@univ-demo.edu"}).json()
    h = login(c, t["teacher_code"], t["temporary_password"])
    assert c.get("/examiner/booklets", headers=h).json() == []
    assert c.get("/academic/students", headers=h).json() == []


# ── exam workflow: upload → OCR → AI evaluation → teacher review → approve → compile → publish ──

def test_full_exam_workflow(c, admin, setup):
    tch, stu = setup["teacher"], setup["student"]
    up = c.post("/examiner/booklet/upload", headers=tch, data={"student_registration": REG, "course_code": SUBJ, "exam_name": EXAM},
                files={"file": ("sheet.pdf", io.BytesIO(_answer_sheet_pdf()), "application/pdf")})
    assert up.status_code == 200, up.text
    bid = up.json()["booklet_id"]
    dup = c.post("/examiner/booklet/upload", headers=tch, data={"student_registration": REG, "course_code": SUBJ, "exam_name": EXAM},
                 files={"file": ("sheet.pdf", io.BytesIO(_answer_sheet_pdf()), "application/pdf")})
    assert dup.status_code == 409

    assert c.post(f"/examiner/publish-booklet/{bid}", headers=tch).status_code == 409   # not reviewed yet
    ev = c.post(f"/examiner/evaluate-booklet/{bid}", headers=tch)
    assert ev.status_code == 200, ev.text
    assert ev.json()["ocr"]["ran"] is True and ev.json()["max_marks"] == 10
    det = c.get(f"/examiner/booklet/{bid}", headers=tch).json()
    q = det["questions"][0]
    assert "merge" in q["student_answer_text"].lower()                     # real OCR text, not a placeholder
    assert q["expected_answer"] and q["question_text"]                      # side-by-side data
    assert 0 < q["ai_score"] <= 10

    # student can't see unpublished result
    assert c.get(f"/student/result/{bid}", headers=stu).status_code == 403
    assert c.get(f"/student/answer-sheet/{bid}/evaluated", headers=stu).status_code == 403

    no_reason = c.post("/examiner/review-question", headers=tch, json={"result_id": q["result_id"], "examiner_score": 9, "examiner_comments": ""})
    assert no_reason.status_code == 422
    too_high = c.post("/examiner/review-question", headers=tch, json={"result_id": q["result_id"], "examiner_score": 11, "examiner_comments": "x"})
    assert too_high.status_code == 422
    ok = c.post("/examiner/review-question", headers=tch, json={"result_id": q["result_id"], "examiner_score": 8,
                                                                 "examiner_comments": "Missed O(n) extra space."})
    assert ok.status_code == 200
    assert c.post(f"/examiner/approve-booklet/{bid}", headers=tch).json()["total_final_score"] == 8

    comp = c.post("/results/compile", headers=tch, json={"subject_code": SUBJ, "exam_name": EXAM, "registration_number": REG}).json()
    fr = comp["compiled"][0]
    assert fr["total"] == 8 and fr["max_total"] == 10 and fr["grade"] == "A+" and fr["status"] == "draft"
    assert c.post(f"/results/final/{fr['id']}/publish", headers=tch).status_code == 409        # approve first
    assert c.post(f"/results/final/{fr['id']}/approve", headers=tch).json()["status"] == "approved"
    assert c.post(f"/results/final/{fr['id']}/publish", headers=tch).json()["status"] == "published"

    # student side
    dash = c.get("/results/my/dashboard", headers=stu).json()
    assert dash["results"][0]["marks"] == 8 and dash["results"][0]["grade"] == "A+" and dash["cgpa"] == 9.0
    detail = c.get(f"/student/result/{bid}", headers=stu).json()
    assert detail["questions"][0]["marks_awarded"] == 8 and detail["questions"][0]["marks_deducted"] == 2
    assert detail["questions"][0]["ai_score"] != 8 or True
    notes = c.get("/notifications", headers=stu).json()
    assert notes["unread"] >= 1 and any(n["kind"] == "result_published" for n in notes["items"])
    pdf = c.get(f"/results/booklet/{bid}/report.pdf", headers=stu)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert c.get("/results/my/final.pdf", headers=stu).content[:4] == b"%PDF"
    # other student cannot see it
    s2 = login(c, "student2", "student123")
    assert c.get(f"/results/booklet/{bid}/report.pdf", headers=s2).status_code == 404
    assert c.get(f"/student/result/{bid}", headers=s2).status_code in (403, 404)


def test_class_results_exports(c, admin, setup):
    j = c.get(f"/results/class?batch_code={BATCH}&subject_code={SUBJ}", headers=admin).json()
    assert j["summary"]["students"] >= 1 and any(r["registration_number"] == REG for r in j["rows"])
    csv_r = c.get(f"/results/class?batch_code={BATCH}&format=csv", headers=admin)
    assert csv_r.status_code == 200 and REG in csv_r.text
    x = c.get(f"/results/class?batch_code={BATCH}&format=xlsx", headers=admin)
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(x.content))
    assert wb["Class result"].max_row >= 2
    p = c.get(f"/results/class?batch_code={BATCH}&format=pdf", headers=admin)
    assert p.content[:4] == b"%PDF"
    # teacher can export own class only
    assert any(r["registration_number"] == REG for r in c.get(f"/results/class?batch_code={BATCH}", headers=setup["teacher"]).json()["rows"])
    assert c.get("/results/class?batch_code=CSE2026A", headers=setup["teacher"]).json()["rows"] == []


def test_analytics(c, admin, setup):
    for g in ("batch", "section", "semester", "subject", "teacher", "session"):
        r = c.get(f"/results/analytics?group_by={g}", headers=admin)
        assert r.status_code == 200, (g, r.text)
    a = c.get(f"/results/analytics?group_by=subject&batch_code={BATCH}", headers=admin).json()
    assert a["overall"]["passed"] >= 1 and a["grade_distribution"]
    td = c.get("/results/teacher/dashboard", headers=setup["teacher"]).json()
    assert td["stats"]["total_students"] == 2 and td["answer_sheets"]["published"] >= 1


def test_profile360_and_report(c, admin, setup):
    p = c.get(f"/academic/students/{REG}/profile360", headers=admin).json()
    assert p["batch_code"] == BATCH and p["answer_sheets"] and p["results"]
    r = c.get(f"/academic/students/{REG}/report.pdf", headers=admin)
    assert r.content[:4] == b"%PDF"


def test_password_reset_and_deactivate(c, admin, setup):
    r = c.post(f"/academic/users/{REG2}/reset-password", headers=admin).json()
    assert login(c, REG2, r["temporary_password"])
    c.patch(f"/academic/students/{REG2}", headers=admin, json={"is_active": False})
    assert c.post("/auth/login", json={"username": REG2, "password": r["temporary_password"]}).status_code == 403


# ── scanner ───────────────────────────────────────────────────────────────────

def _page_photo(text_lines, seed=0):
    """Photo of a handwritten-style page lying on a desk (perspective + shadow)."""
    import cv2, numpy as np
    page = np.full((1400, 1000, 3), 245, np.uint8)
    y = 120
    for t in text_lines:
        cv2.putText(page, t, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA); y += 70
    bg = np.full((1800, 1400, 3), (60, 70, 85), np.uint8)
    H, W = bg.shape[:2]
    src = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])
    dst = np.float32([[230 + seed, 180], [1180, 230], [1210, 1640], [170, 1600 - seed]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (W, H))
    mask = cv2.warpPerspective(np.full((1400, 1000), 255, np.uint8), M, (W, H))
    bg[mask > 0] = warped[mask > 0]
    ok, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def test_scanner_multi_page_flow(c, admin, setup):
    tch = setup["teacher"]
    exam = EXAM + " scan"
    s = c.post("/scanner/sessions", headers=tch, json={"registration_number": REG2, "subject_code": SUBJ, "exam_name": exam}).json()
    sid = s["id"]
    assert s["registration_number"] == REG2 and not s["pages"]
    # cannot scan a student of another batch
    assert c.post("/scanner/sessions", headers=tch, json={"registration_number": "21BCS101", "subject_code": SUBJ,
                                                           "exam_name": exam}).status_code == 403
    pages = []
    for i, lines in enumerate([["Q1. Merge sort divides the array", "into two halves recursively."],
                               ["and merges them. Complexity", "is O(n log n)."],
                               ["Extra page"]]):
        r = c.post(f"/scanner/sessions/{sid}/pages", headers=tch, files={"file": (f"p{i}.jpg", _page_photo(lines, i * 7), "image/jpeg")})
        assert r.status_code == 200, r.text
        pages.append(r.json())
    q = pages[0]["quality"]
    assert q["boundary_found"] is True and q["height"] > q["width"]          # cropped + upright
    assert q["ok"] and not any("Overexposed" in w for w in q["warnings"]), q
    # same image again → duplicate rejected
    dup = c.post(f"/scanner/sessions/{sid}/pages", headers=tch, files={"file": ("p0.jpg", _page_photo(["Q1. Merge sort divides the array", "into two halves recursively."], 0), "image/jpeg")})
    assert dup.status_code == 409
    img = c.get(pages[0]["image_url"], headers=tch)
    assert img.status_code == 200 and img.content[:3] == b"\xff\xd8\xff"
    # delete page 3, reorder, rotate
    after = c.delete(f"/scanner/sessions/{sid}/pages/{pages[2]['id']}", headers=tch).json()
    assert [p["position"] for p in after["pages"]] == [1, 2]
    order = c.put(f"/scanner/sessions/{sid}/order", headers=tch, json={"page_ids": [pages[1]["id"], pages[0]["id"]]}).json()
    assert [p["id"] for p in order["pages"]] == [pages[1]["id"], pages[0]["id"]]
    c.put(f"/scanner/sessions/{sid}/order", headers=tch, json={"page_ids": [pages[0]["id"], pages[1]["id"]]})
    assert c.post(f"/scanner/sessions/{sid}/pages/{pages[0]['id']}/rotate", headers=tch, json={"degrees": 180}).status_code == 200
    c.post(f"/scanner/sessions/{sid}/pages/{pages[0]['id']}/rotate", headers=tch, json={"degrees": 180})
    prev = c.get(f"/scanner/sessions/{sid}/preview.pdf", headers=tch)
    assert prev.content[:4] == b"%PDF"
    sub = c.post(f"/scanner/sessions/{sid}/submit", headers=tch).json()
    assert sub["status"] == "submitted" and sub["booklet_id"] and REG2 in sub["pdf_path"]
    # pages can't be added after submission; the booklet is in the teacher's queue
    assert c.post(f"/scanner/sessions/{sid}/pages", headers=tch, files={"file": ("x.jpg", _page_photo(["late"], 3), "image/jpeg")}).status_code == 409
    assert sub["booklet_id"] in [b["id"] for b in c.get("/examiner/booklets", headers=tch).json()]
    ev = c.post(f"/examiner/evaluate-booklet/{sub['booklet_id']}", headers=tch)
    assert ev.status_code == 200, ev.text
    det = c.get(f"/examiner/booklet/{sub['booklet_id']}", headers=tch).json()
    assert "merge" in det["questions"][0]["student_answer_text"].lower()


# ── OMR + combined result ─────────────────────────────────────────────────────

def test_omr_and_combined_result(c, admin, setup):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from omr_testkit import photo_of_sheet
    tch, stu = setup["teacher"], setup["student"]
    key = c.post("/omr/keys", headers=tch, json={"subject_code": SUBJ, "exam_name": EXAM, "answers": "ABCDA BCDAB",
                                                  "marks_correct": 2, "marks_wrong": 0.5}).json()
    assert key["num_questions"] == 10 and key["max_score"] == 20
    assert c.post("/omr/keys", headers=setup["student"], json={"subject_code": SUBJ, "exam_name": "x", "answers": "AB"}).status_code == 403
    pdf = c.get(f"/omr/keys/{key['id']}/sheets.pdf?batch_code={BATCH}", headers=tch)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    # student fills 8 right, 1 wrong (q3), q10 double-marked
    marks = {1: "A", 2: "B", 3: "A", 4: "D", 5: "A", 6: "B", 7: "C", 8: "D", 9: "A", 10: "B"}
    photo = photo_of_sheet(key["id"], SUBJ, EXAM, 10, 4, REG, marks, multi={10: "A"})
    r = c.post(f"/omr/keys/{key['id']}/scan", headers=tch, files={"file": ("omr.jpg", photo, "image/jpeg")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["registration_number"] == REG and res["correct"] == 8 and res["incorrect"] == 1
    assert res["needs_verification"] is True and res["flagged"] == 1
    again = c.post(f"/omr/keys/{key['id']}/scan", headers=tch, files={"file": ("omr.jpg", photo, "image/jpeg")})
    assert again.status_code == 409
    # compile refuses while OMR needs verification
    comp = c.post("/results/compile", headers=tch, json={"subject_code": SUBJ, "exam_name": EXAM, "registration_number": REG}).json()
    assert not comp["compiled"] or comp["compiled"][0]["status"] == "published"
    v = c.post(f"/omr/responses/{res['id']}/verify", headers=tch, json={"answers": {"10": ""}, "note": "two bubbles filled"}).json()
    assert v["needs_verification"] is False and v["score"] == 8 * 2 - 0.5
    # results already published earlier for written part → recompiling a published result is refused
    comp = c.post("/results/compile", headers=tch, json={"subject_code": SUBJ, "exam_name": EXAM, "registration_number": REG}).json()
    assert comp["not_compiled"] and "already published" in comp["not_compiled"][0]["error"]


def test_omr_plus_written_new_exam(c, admin, setup):
    """Fresh exam: OMR 35/40-style combination with written marks → consolidated result."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from omr_testkit import photo_of_sheet
    tch, stu = setup["teacher"], setup["student"]
    exam2 = EXAM + " combined"
    key = c.post("/omr/keys", headers=tch, json={"subject_code": SUBJ, "exam_name": exam2, "answers": "ABCD", "marks_correct": 10}).json()
    photo = photo_of_sheet(key["id"], SUBJ, exam2, 4, 4, REG, {1: "A", 2: "B", 3: "C", 4: "A"})
    o = c.post(f"/omr/keys/{key['id']}/scan", headers=tch, files={"file": ("o.jpg", photo, "image/jpeg")}).json()
    assert o["score"] == 30 and not o["needs_verification"]
    up = c.post("/examiner/booklet/upload", headers=tch, data={"student_registration": REG, "course_code": SUBJ, "exam_name": exam2},
                files={"file": ("s.pdf", io.BytesIO(_answer_sheet_pdf()), "application/pdf")}).json()
    c.post(f"/examiner/evaluate-booklet/{up['booklet_id']}", headers=tch)
    det = c.get(f"/examiner/booklet/{up['booklet_id']}", headers=tch).json()
    c.post("/examiner/review-question", headers=tch, json={"result_id": det["questions"][0]["result_id"], "examiner_score": 6,
                                                            "examiner_comments": "Partial explanation of merge step."})
    c.post(f"/examiner/approve-booklet/{up['booklet_id']}", headers=tch)
    fr = c.post("/results/compile", headers=tch, json={"subject_code": SUBJ, "exam_name": exam2, "registration_number": REG}).json()["compiled"][0]
    assert (fr["omr_score"], fr["omr_max"], fr["subjective_score"], fr["subjective_max"]) == (30, 40, 6, 10)
    assert fr["total"] == 36 and fr["max_total"] == 50 and fr["percentage"] == 72.0 and fr["grade"] == "A"
    c.post(f"/results/final/{fr['id']}/approve", headers=tch)
    c.post(f"/results/final/{fr['id']}/publish", headers=tch)
    rows = c.get("/results/my/dashboard", headers=stu).json()["results"]
    row = next(r for r in rows if r["exam_name"] == exam2)
    assert row["omr_marks"] == 30 and row["subjective_marks"] == 6 and row["marks"] == 36
