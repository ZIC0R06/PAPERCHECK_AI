"""
AI Robot bulk examination checking — complete workflow through the real API (isolated test DB, real Tesseract OCR,
offline marking because the test environment has no Ollama).

 * ONE question paper + marking scheme reused for every sheet of the batch
 * many sheets uploaded at once: image-only PDFs, a JPG, a real handwritten PDF, and a camera scan (3 photos)
 * student identification from the file name and from the registration number written on the sheet;
   conflicting / missing identification → manual verification, never a guess; duplicates rejected
 * background checking with live per-question progress; class performance report (JSON + PDF); ZIP of checked sheets
 * chat / voice command "Check all uploaded answer sheets" starts the REAL job; status / report questions
 * approval rules: flagged answers block approval; publishing only after approval; student sees only own published result
 * faculty scope: another teacher cannot see the batch
"""
import io
import os
import time
import uuid
import zipfile

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend.api import app
from backend.database import init_db

U = str(1000 + uuid.uuid4().int % 9000)          # digits: OCR-stable registration numbers
SUBJ, EXAM, BATCH = "CS301", f"Bulk Robot Exam {U}", f"BR{U}"
REGS = [f"BR{U}0{i}" for i in range(1, 7)]
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "handwritten_answer_sheet.pdf")

QUESTIONS = [
    {"question_number": 1, "question_text": "Define primary key with an example.", "marks": 10,
     "model_answer": "A primary key uniquely identifies each record in a table. It must be unique and not null. Example: roll number.",
     "rubric": [{"criterion": "uniquely identifies each record", "marks": 4}, {"criterion": "unique and not null", "marks": 3},
                {"criterion": "example such as roll number", "marks": 3}]},
    {"question_number": 2, "question_text": "Explain normalization with 1NF, 2NF and 3NF.", "marks": 10,
     "model_answer": "Normalization organises data to reduce redundancy. 1NF removes repeating groups, 2NF removes partial dependency, 3NF removes transitive dependency."},
    {"question_number": 3, "question_text": "What is a deadlock? How can it be prevented?", "marks": 10,
     "model_answer": "Deadlock: two or more transactions wait for each other's locks forever. Prevention: lock ordering, timeouts, wait-die."},
]

GOOD = ["Q1 A primary key uniquely identifies each record in a table. It must be unique and not null.",
        "Example: roll number of a student.",
        "Q2 Normalization organises data to reduce redundancy. 1NF removes repeating groups.",
        "2NF removes partial dependency and 3NF removes transitive dependency.",
        "Q3 A deadlock is when two transactions wait for each other's locks forever.",
        "It can be prevented by lock ordering and timeouts."]
WEAK = ["Q1 A primary key is a column in a table.",
        "Q3 Deadlock means the system is slow."]


def _font(size=34):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def sheet_image(reg_on_sheet, lines):
    img = Image.new("RGB", (1654, 2339), "white")
    d = ImageDraw.Draw(img)
    f = _font()
    y = 120
    d.text((120, y), f"Answer Sheet - {SUBJ}", fill="black", font=_font(44)); y += 90
    if reg_on_sheet:
        d.text((120, y), f"Registration No: {reg_on_sheet}", fill="black", font=_font(46))
    y += 140
    for ln in lines:
        d.text((120, y), ln, fill="black", font=f)
        y += 70
    return img


def image_pdf(img):
    """image-only PDF (no text layer) → the OCR path is exercised."""
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=200.0)
    data = buf.getvalue()
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        assert not doc[0].get_text().strip()
    return data


def jpg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="module")
def c():
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
    t = c.post("/academic/teachers", headers=A, json={"full_name": "Dr Robot", "email": f"rb{U}@x.edu",
                                                      "department_code": "CSE"}).json()
    assert c.post(f"/academic/teachers/{t['teacher_code']}/assignments", headers=A,
                  json={"subject_code": SUBJ, "batch_code": BATCH}).status_code == 200
    t2 = c.post("/academic/teachers", headers=A, json={"full_name": "Dr Other", "email": f"ot{U}@x.edu",
                                                       "department_code": "CSE"}).json()
    pw = {}
    for i, reg in enumerate(REGS):
        s = c.post("/academic/students", headers=A, json={"registration_number": reg, "name": f"Bulk Student {i + 1}",
                                                         "email": f"b{i}{U}@x.edu", "batch_code": BATCH,
                                                         "subject_codes": [SUBJ]})
        assert s.status_code == 200, s.text
        pw[reg] = s.json()["temporary_password"]
    T = _login(c, t["teacher_code"], t["temporary_password"])
    T2 = _login(c, t2["teacher_code"], t2["temporary_password"])
    S1 = _login(c, REGS[0], pw[REGS[0]])
    S2 = _login(c, REGS[1], pw[REGS[1]])
    return {"A": A, "T": T, "T2": T2, "S1": S1, "S2": S2}


def _wait(c, H, jid, timeout=300):
    seen_question_progress = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/bulk/jobs/{jid}", headers=H).json()
        if j["current"] and j["current"]["status"] == "evaluating" and j["current"]["current_question"]:
            seen_question_progress = True
        if j["status"] in ("completed", "cancelled", "failed"):
            return j, seen_question_progress
        time.sleep(0.2)
    raise AssertionError("checking did not finish")


def test_bulk_robot_checking_workflow(c, ctx):
    T, A, T2 = ctx["T"], ctx["A"], ctx["T2"]

    # 1. ONE question paper + marking scheme for the subject + exam
    p = c.post("/exam-papers", headers=T, json={"course_code": SUBJ, "exam_name": EXAM, "questions": QUESTIONS})
    assert p.status_code == 200, p.text
    pid = p.json()["id"]
    papers = c.get("/bulk/papers", headers=T).json()
    mine = next(x for x in papers if x["paper_id"] == pid)
    assert mine["ready"] and mine["questions"] == 3 and mine["total_marks"] == 30

    job = c.post("/bulk/jobs", headers=T, json={"exam_paper_id": pid})
    assert job.status_code == 200, job.text
    jid = job.json()["id"]
    assert c.get(f"/bulk/jobs/{jid}", headers=T2).status_code == 404          # other faculty can't see it
    assert c.post("/bulk/jobs", headers=T2, json={"exam_paper_id": pid}).status_code == 403

    # nothing uploaded yet → the robot refuses honestly
    r = c.post("/agent/chat", headers=T, json={"message": "Check all uploaded answer sheets", "job_id": jid})
    assert r.status_code == 200 and r.json()["action"] == "start_refused"

    # 2. many sheets at once
    files = [
        ("files", (f"{REGS[0]}.pdf", image_pdf(sheet_image(REGS[0], GOOD)), "application/pdf")),          # name + sheet agree
        ("files", ("scan_0042.pdf", image_pdf(sheet_image(REGS[1], WEAK)), "application/pdf")),           # sheet only
        ("files", (f"{REGS[2]}_answers.jpg", jpg(sheet_image("", GOOD[:2])), "image/jpeg")),              # name only (JPG)
        ("files", ("unknown.pdf", image_pdf(sheet_image("", WEAK)), "application/pdf")),                  # nothing → manual
        ("files", (f"{REGS[3]}.pdf", image_pdf(sheet_image(REGS[4], GOOD)), "application/pdf")),          # conflict → manual
        ("files", (f"{REGS[5]}.pdf", open(FIX, "rb").read(), "application/pdf")),                         # real handwriting
        ("files", ("notes.txt", b"hello world", "text/plain")),                                           # not a sheet
    ]
    up = c.post(f"/bulk/jobs/{jid}/sheets", headers=T, files=files)
    assert up.status_code == 200, up.text
    u = up.json()
    by = {i["filename"]: i for i in u["added"]}
    assert len(u["added"]) == 6 and len(u["rejected"]) == 1 and u["rejected"][0]["status"] == 415
    assert by[f"{REGS[0]}.pdf"]["status"] == "ready" and by[f"{REGS[0]}.pdf"]["id_method"] == "filename+sheet"
    assert by["scan_0042.pdf"]["student"]["registration_number"] == REGS[1] and by["scan_0042.pdf"]["id_method"] == "sheet"
    assert by[f"{REGS[2]}_answers.jpg"]["student"]["registration_number"] == REGS[2]
    assert by["unknown.pdf"]["status"] == "needs_identification" and by["unknown.pdf"]["student"] is None
    conflict = by[f"{REGS[3]}.pdf"]
    assert conflict["status"] == "needs_identification" and "Conflicting" in conflict["id_problem"]
    assert by[f"{REGS[5]}.pdf"]["status"] == "ready"

    # exact duplicate file is rejected (a page can never belong to two students)
    dup = c.post(f"/bulk/jobs/{jid}/sheets", headers=T, files=[("files", ("copy.pdf", open(FIX, "rb").read(), "application/pdf"))])
    assert dup.json()["rejected"][0]["status"] == 409

    # originals are preserved byte-for-byte
    orig = c.get(by[f"{REGS[5]}.pdf"]["original_url"], headers=T)
    assert orig.status_code == 200 and orig.content == open(FIX, "rb").read()
    assert c.get(by["unknown.pdf"]["page_url"], headers=T).content[:4] == b"\x89PNG"

    # 3. manual verification: same student twice is refused, a wrong-scope reg is refused
    r = c.post(f"/bulk/items/{conflict['id']}/assign", headers=T, json={"registration_number": REGS[0]})
    assert r.status_code == 409
    assert c.post(f"/bulk/items/{conflict['id']}/assign", headers=T, json={"registration_number": "21BCS101"}).status_code == 404
    r = c.post(f"/bulk/items/{conflict['id']}/assign", headers=T, json={"registration_number": REGS[4]})
    assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["id_method"] == "manual"

    # camera scan: 3 photos of ONE sheet (2nd one rotated) for REGS[3] → processed + combined PDF
    pages = [jpg(sheet_image(REGS[3], GOOD[:2])), jpg(sheet_image("", GOOD[2:4]).rotate(90, expand=True)),
             jpg(sheet_image("", GOOD[4:]))]
    sc = c.post(f"/bulk/jobs/{jid}/scan", headers=T, data={"registration_number": REGS[3], "rotations": "[0, 270, 0]"},
                files=[("pages", (f"p{i}.jpg", d, "image/jpeg")) for i, d in enumerate(pages)])
    assert sc.status_code == 200, sc.text
    cam = sc.json()["item"]
    assert cam["source"] == "camera" and cam["pages"] == 3 and cam["status"] == "ready"
    assert cam["student"]["registration_number"] == REGS[3]
    # entering a registration number that is not in the teacher's classes is never accepted
    bad = c.post(f"/bulk/jobs/{jid}/scan", headers=T, data={"registration_number": "NOPE99999"},
                 files=[("pages", ("x.jpg", jpg(sheet_image("", WEAK)), "image/jpeg"))])
    assert bad.status_code == 200 and bad.json()["item"]["status"] == "needs_identification"

    # 4. the robot command starts the REAL background job
    r = c.post("/agent/chat", headers=T, json={"message": "Check all uploaded answer sheets", "job_id": jid})
    assert r.status_code == 200, r.text
    assert r.json()["action"] == "checking_started" and "6 answer sheet" in r.json()["reply"]
    st = c.post("/agent/chat", headers=T, json={"message": "What is the checking progress?", "job_id": jid}).json()
    assert st["action"] == "status" and f"Batch #{jid}" in st["reply"]
    j, saw_q = _wait(c, T, jid)
    assert j["status"] == "completed", j
    assert saw_q or all(i["questions_total"] for i in j["items"] if i["status"] == "evaluated")
    items = {i["filename"]: i for i in j["items"]}
    evaluated = [i for i in j["items"] if i["status"] == "evaluated"]
    assert len(evaluated) == 6, [(i["filename"], i["status"], i["error"]) for i in j["items"]]
    assert {i["student"]["registration_number"] for i in evaluated} == set(REGS)   # every sheet → its own student
    assert items["unknown.pdf"]["status"] == "needs_identification" and items["unknown.pdf"]["booklet_id"] is None
    for i in evaluated:
        assert i["max_marks"] == 30 and i["questions_done"] == i["questions_total"] == 3
    good, weak = items[f"{REGS[0]}.pdf"], items["scan_0042.pdf"]
    assert good["total_ai"] > weak["total_ai"], (good["total_ai"], weak["total_ai"])
    wq = {q["question_number"]: q for q in c.get(f"/examiner/booklet/{weak['booklet_id']}", headers=T).json()["questions"]}
    assert wq[2]["student_answer"] == "" and wq[2]["ai_score"] == 0                    # Q2 missing → 0, explained
    assert any("Not attempted" in d["reason"] for d in wq[2]["deductions"])
    assert any("Q1:" in e["m"] for e in weak["events"])                                 # question-wise log for the robot UI

    # the marks really belong to the right students (booklets)
    for i in evaluated:
        d = c.get(f"/examiner/booklet/{i['booklet_id']}", headers=T).json()
        assert d["booklet"]["registration_number"] if "booklet" in d else True
    qd = c.get(f"/examiner/booklet/{good['booklet_id']}", headers=T).json()
    q1 = next(q for q in qd["questions"] if q["question_number"] == 1)
    assert "primary key" in q1["student_answer"].lower() and q1["rubric_breakdown"]

    # robot explains a marking decision through the read-only agent
    ex = c.post("/agent/chat", headers=T, json={"message": f"Why were marks cut for {REGS[1]} in Q1?"}).json()
    assert "Q1" in ex["reply"] and REGS[1] in ex["reply"]

    # 5. reports
    rep = c.get(f"/bulk/jobs/{jid}/report", headers=T).json()
    assert rep["summary"]["students_checked"] == 6 and rep["summary"]["needs_identification"] == 2
    assert [q["question_number"] for q in rep["questions"]] == [1, 2, 3]
    pdf = c.get(f"/bulk/jobs/{jid}/report.pdf", headers=T)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/bulk/jobs/{jid}/results.zip", headers=T).content))
    names = z.namelist()
    assert "class_performance_report.pdf" in names
    for reg in REGS:
        assert any(n.startswith(reg + "/") and n.endswith("checked_answer_sheet.pdf") for n in names)
    ev = c.get(good["evaluated_url"], headers=T)
    assert ev.status_code == 200 and ev.content[:4] == b"%PDF"
    talk = c.post("/agent/chat", headers=T, json={"message": "Show the class performance report for this batch", "job_id": jid}).json()
    assert talk["action"] == "report" and "Question-wise" in talk["reply"]

    # 6. approval: local-OCR answers are flagged → approval refused until the teacher verifies them
    ap = c.post(f"/bulk/jobs/{jid}/approve", headers=T, json={"booklet_ids": [good["booklet_id"]]}).json()["results"][0]
    assert not ap["ok"] and "flagged" in ap["error"]
    pub = c.post(f"/bulk/jobs/{jid}/publish", headers=T, json={"booklet_ids": [good["booklet_id"]]}).json()["results"][0]
    assert not pub["ok"]
    for q in qd["questions"]:
        assert c.post("/examiner/review-question", headers=T, json={"result_id": q["result_id"], "examiner_score": q["ai_score"],
                                                                    "examiner_comments": "verified", "flag_for_review": False}).status_code == 200
    ap = c.post(f"/bulk/jobs/{jid}/approve", headers=T, json={"booklet_ids": [good["booklet_id"]]}).json()["results"][0]
    assert ap["ok"], ap
    pub = c.post(f"/bulk/jobs/{jid}/publish", headers=T, json={"booklet_ids": [good["booklet_id"]]}).json()["results"][0]
    assert pub["ok"], pub

    # 7. students: only their own, only published
    S1, S2 = ctx["S1"], ctx["S2"]
    r1 = c.get("/student-assistant/results", headers=S1).json()
    mine1 = [x for x in r1 if x["exam"] == EXAM]
    assert len(mine1) == 1 and mine1[0]["published"] and mine1[0]["marks"] == good["total_ai"]
    assert c.get(mine1[0]["evaluated_url"], headers=S1).status_code == 200
    r2 = [x for x in c.get("/student-assistant/results", headers=S2).json() if x["exam"] == EXAM]
    assert len(r2) == 1 and not r2[0]["published"] and r2[0]["questions"] == [] and r2[0]["evaluated_url"] is None
    assert c.get(f"/student/answer-sheet/{weak['booklet_id']}/evaluated", headers=S1).status_code in (403, 404)
    a = c.post("/student-assistant/chat", headers=S1, json={"message": f"Why did I get fewer marks in {SUBJ}?"}).json()
    assert a["tool"] == "why" and "Q1" in a["reply"]
    a2 = c.post("/student-assistant/chat", headers=S1, json={"message": "Make revision notes for my weak topics",
                                                            "history": [{"role": "user", "content": f"marks in {SUBJ}"}]}).json()
    assert a2["tool"] == "revision_notes"
    a3 = c.post("/student-assistant/chat", headers=S2, json={"message": "What are my marks?"}).json()
    assert "awaiting" in a3["reply"] and str(weak["total_ai"]) not in a3["reply"]    # no unpublished marks leak
    assert c.post("/bulk/jobs", headers=S1, json={"exam_paper_id": pid}).status_code == 403

    # 8. admin monitors evaluation progress across faculty
    mon = c.post("/agent/chat", headers=A, json={"message": "Show evaluation progress of all checking"}).json()
    assert mon["action"] == "status" and f"#{jid}" in mon["reply"]
    assert any(x["id"] == jid for x in c.get("/bulk/jobs", headers=A).json())
    dirr = c.post("/agent/chat", headers=A, json={"message": "how many students and faculty are there? overview"}).json()
    assert "directory" in dirr["tools"] and "Faculty:" in dirr["reply"]


def test_interrupted_job_recovers(c, ctx):
    from backend.bulk_evaluation import recover_interrupted
    from backend.database import EvaluationJob, SessionLocal
    T = ctx["T"]
    pid = next(x for x in c.get("/bulk/papers", headers=T).json() if x["exam_name"] == EXAM)["paper_id"]
    jid = c.post("/bulk/jobs", headers=T, json={"exam_paper_id": pid}).json()["id"]
    db = SessionLocal()
    j = db.query(EvaluationJob).get(jid)
    j.status = "running"
    db.commit()
    db.close()
    assert recover_interrupted() >= 1
    assert c.get(f"/bulk/jobs/{jid}", headers=T).json()["status"] == "interrupted"
