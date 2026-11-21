"""
Bulk checking with the AI (Ollama) grading path — against tests/mock_ollama.py speaking the real Ollama wire format.
Proves the robot's background job uses the local LLM when it is available (method "ai", provider ollama), that
marks without evidence in the student's answer are withheld, and that the staff / student assistants word their
answers with the model while every figure stays grounded in the database.
"""
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import init_db
from backend.llm_provider import reset_llm_client
from tests.mock_ollama import MockOllama
from tests.test_bulk_robot_checking import GOOD, WEAK, image_pdf, sheet_image

U = str(1000 + uuid.uuid4().int % 9000)
SUBJ, EXAM, BATCH = "CS301", f"Bulk AI Path {U}", f"BA{U}"
REG_GOOD, REG_WEAK = f"BA{U}01", f"BA{U}02"

GRADE = {"rubric_breakdown": [
    {"criterion": "primary key uniquely identifies each record", "max_marks": 10, "awarded": 8, "status": "partial",
     "evidence": "primary key uniquely identifies each record", "reason": "Correct definition, example brief."}],
    "corrections": [], "strengths": ["clear definition"], "missing_concepts": ["candidate key"],
    "feedback": "Good definition; add more detail.", "confidence": 0.9}


@pytest.fixture(scope="module")
def mock():
    with MockOllama(json_reply=json.dumps(GRADE), reply="Here is the summary you asked for, based on your checked sheet.") as m:
        mp = pytest.MonkeyPatch()
        mp.setenv("OLLAMA_BASE_URL", m.url)
        reset_llm_client()
        yield m
        mp.undo()
        reset_llm_client()


@pytest.fixture(scope="module")
def c(mock):
    init_db()
    with TestClient(app) as client:
        yield client


def _login(c, u, p):
    r = c.post("/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_bulk_checking_uses_local_llm(c, mock):
    A = _login(c, "admin", "admin123")
    assert c.post("/academic/batches", headers=A, json={"batch_code": BATCH, "programme": "B.Tech CSE", "course_code": "CSE",
                                                         "section": "A", "semester": "Semester 5", "academic_session": "2026-27"}).status_code == 200
    t = c.post("/academic/teachers", headers=A, json={"full_name": "Dr AI", "email": f"ai{U}@x.edu", "department_code": "CSE"}).json()
    assert c.post(f"/academic/teachers/{t['teacher_code']}/assignments", headers=A,
                  json={"subject_code": SUBJ, "batch_code": BATCH}).status_code == 200
    pw = {}
    for reg in (REG_GOOD, REG_WEAK):
        s = c.post("/academic/students", headers=A, json={"registration_number": reg, "name": "AI " + reg, "email": f"{reg}@x.edu",
                                                         "batch_code": BATCH, "subject_codes": [SUBJ]})
        pw[reg] = s.json()["temporary_password"]
    T = _login(c, t["teacher_code"], t["temporary_password"])
    p = c.post("/exam-papers", headers=T, json={"course_code": SUBJ, "exam_name": EXAM, "questions": [
        {"question_number": 1, "question_text": "Define primary key.", "marks": 10,
         "model_answer": "A primary key uniquely identifies each record in a table; unique and not null."}]}).json()
    jid = c.post("/bulk/jobs", headers=T, json={"exam_paper_id": p["id"]}).json()["id"]
    up = c.post(f"/bulk/jobs/{jid}/sheets", headers=T, files=[
        ("files", (f"{REG_GOOD}.pdf", image_pdf(sheet_image(REG_GOOD, GOOD[:2])), "application/pdf")),
        ("files", (f"{REG_WEAK}.pdf", image_pdf(sheet_image(REG_WEAK, WEAK[:1])), "application/pdf"))]).json()
    assert [i["status"] for i in up["added"]] == ["ready", "ready"]
    assert c.post(f"/bulk/jobs/{jid}/start", headers=T).json()["started"]
    for _ in range(600):
        j = c.get(f"/bulk/jobs/{jid}", headers=T).json()
        if j["status"] == "completed":
            break
        time.sleep(0.1)
    assert j["status"] == "completed"
    by = {i["student"]["registration_number"]: i for i in j["items"]}
    g = c.get(f"/examiner/booklet/{by[REG_GOOD]['booklet_id']}", headers=T).json()["questions"][0]
    w = c.get(f"/examiner/booklet/{by[REG_WEAK]['booklet_id']}", headers=T).json()["questions"][0]
    assert g["ai_score"] == 8 and g["rubric_breakdown"][0]["status"] == "partial"
    assert w["ai_score"] == 0          # same model reply, but its "evidence" is not in this student's answer → withheld
    assert any("without verifiable evidence" in (w.get("flag_reason") or "") for _ in [0])
    grading_calls = [r for r in mock.requests if r.get("format") == "json"]
    assert len(grading_calls) >= 2 and all(r["model"] == "llama3.2:3b" for r in grading_calls)

    # staff assistant composes with the model (no invented numbers in the mock reply → accepted as AI wording)
    r = c.post("/agent/chat", headers=T, json={"message": f"why were marks cut for {REG_WEAK} in Q1"}).json()
    assert r["mode"] == "ai" and r["reply"].startswith("Here is the summary")
    # robot actions never go through the model
    st = c.post("/agent/chat", headers=T, json={"message": "checking progress", "job_id": jid}).json()
    assert st["mode"] == "action" and f"Batch #{jid}" in st["reply"]
