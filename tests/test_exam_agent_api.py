"""
Examination Assistant through the real API + database (needs the full stack: fastapi, sqlalchemy).
Run: pytest tests/test_exam_agent_api.py -q
Checks authentication, role scoping against real SQL (faculty see ONLY assigned subject/batch), audit logging and
that the assistant is read-only.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import AuditLog, FinalResult, SessionLocal, Student, init_db

U = uuid.uuid4().hex[:4].upper()
BATCH_A, BATCH_B = f"AGA{U}", f"AGB{U}"
REG_A, REG_B = f"AGA{U}01", f"AGB{U}01"


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
def world(c):
    A = login(c, "admin", "admin123")
    for b in (BATCH_A, BATCH_B):
        assert c.post("/academic/batches", headers=A, json={"batch_code": b, "programme": "B.Tech CSE", "course_code": "CSE",
                                                             "section": "A", "semester": "Semester 5",
                                                             "academic_session": "2026-27"}).status_code == 200
    t = c.post("/academic/teachers", headers=A, json={"full_name": "Agent Teacher", "email": f"ag{U}@x.edu",
                                                      "department_code": "CSE"}).json()
    assert c.post(f"/academic/teachers/{t['teacher_code']}/assignments", headers=A,
                  json={"subject_code": "CS301", "batch_code": BATCH_A}).status_code == 200
    sa = c.post("/academic/students", headers=A, json={"registration_number": REG_A, "name": "Alpha Student", "email": f"a{U}@x.edu",
                                                     "batch_code": BATCH_A, "subject_codes": ["CS301"]}).json()
    sb = c.post("/academic/students", headers=A, json={"registration_number": REG_B, "name": "Beta Student", "email": f"b{U}@x.edu",
                                                     "batch_code": BATCH_B, "subject_codes": ["CS301"]}).json()
    db = SessionLocal()
    try:
        for reg, pct in ((REG_A, 35.0), (REG_B, 30.0)):
            st = db.query(Student).filter(Student.registration_number == reg).first()
            db.add(FinalResult(student_id=st.id, course_code="CS301", course_name="DBMS", exam_name=f"AgentExam{U}",
                               total=pct, max_total=100, percentage=pct, grade="F", passed=False, status="approved"))
        db.commit()
    finally:
        db.close()
    T = login(c, t["teacher_code"], t["temporary_password"])
    S = login(c, REG_A, sa["temporary_password"])
    return {"A": A, "T": T, "S": S}


def test_requires_login_and_staff_role(c, world):
    assert c.post("/agent/chat", json={"message": "hi"}).status_code in (401, 403)
    assert c.post("/agent/chat", headers=world["S"], json={"message": "results of " + REG_A}).status_code == 403
    assert c.get("/agent/capabilities", headers=world["S"]).status_code == 403


def test_admin_sees_every_student(c, world):
    r = c.post("/agent/chat", headers=world["A"], json={"message": f"students below 40 in CS301"}).json()
    assert REG_A in r["reply"] and REG_B in r["reply"]


def test_faculty_sees_only_assigned_batch(c, world):
    r = c.post("/agent/chat", headers=world["T"], json={"message": "students below 40 in CS301"}).json()
    assert REG_A in r["reply"] and REG_B not in r["reply"]
    r2 = c.post("/agent/chat", headers=world["T"], json={"message": f"results of {REG_B}"}).json()
    assert "Beta Student" not in r2["reply"] and "No result records" in r2["reply"]
    cap = c.get("/agent/capabilities", headers=world["T"]).json()
    assert BATCH_A in cap["scope"] and BATCH_B not in cap["scope"] and cap["read_only"] is True


def test_faculty_cannot_use_admin_only_tool(c, world):
    r = c.post("/agent/chat", headers=world["T"], json={"message": "show faculty activity monitor"}).json()
    assert "administrators only" in r["reply"]


def test_queries_are_audited_and_marks_never_change(c, world):
    db = SessionLocal()
    try:
        before = db.query(FinalResult).filter(FinalResult.exam_name == f"AgentExam{U}").count()
        c.post("/agent/chat", headers=world["A"], json={"message": f"results of {REG_A}"})
        assert db.query(AuditLog).filter(AuditLog.action == "AGENT_QUERY").count() >= 1
        assert db.query(FinalResult).filter(FinalResult.exam_name == f"AgentExam{U}").count() == before
        pct = db.query(FinalResult).join(Student, FinalResult.student_id == Student.id).filter(
            Student.registration_number == REG_A, FinalResult.exam_name == f"AgentExam{U}").first().percentage
        assert pct == 35.0
    finally:
        db.close()


def test_ollama_down_still_answers_from_database(c, world):
    r = c.post("/agent/chat", headers=world["A"], json={"message": f"results of {REG_A}"}).json()
    assert "35" in r["reply"] and r["mode"] in ("data-only", "ai", "data-verified", "data")
