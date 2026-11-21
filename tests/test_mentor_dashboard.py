"""
AI Mentor dashboard — API tests
===============================
Covers the ChatGPT-style mentor: chat send/follow-up, file upload validation
and extraction, file-grounded actions, privacy isolation, and that the LLM
prompt really contains the file content, conversation history and style.

Run on a throw-away database copy:
    DATABASE_URL=sqlite:///./test_copy.db pytest tests/test_mentor_dashboard.py -q
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend import mentor_service

NOTES = (
    "Week 1: Introduction to Cloud Computing\n"
    "Cloud computing is the on-demand delivery of computing resources such as servers, storage and networking over the internet.\n"
    "Virtualization is the key enabling technology of cloud computing. A hypervisor creates and runs virtual machines.\n"
    "Rapid elasticity means resources can be scaled out or scaled in automatically according to demand.\n"
    "Resource pooling means the provider serves multiple tenants from shared physical resources.\n"
    "Measured service means that resource usage is monitored, controlled and reported for pay-as-you-go billing.\n"
    "A Type 1 hypervisor runs directly on hardware, while a Type 2 hypervisor runs on a host operating system.\n"
    "Infrastructure as a Service provides virtual machines; Platform as a Service provides a managed runtime.\n"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _login(c, user):
    r = c.post("/auth/login", json={"username": user, "password": "student123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def h1(client):
    return _login(client, "student")


@pytest.fixture(scope="module")
def h2(client):
    return _login(client, "student2")


@pytest.fixture(scope="module")
def notes_id(client, h1):
    r = client.post("/mentor/files", headers=h1,
                    files={"file": ("cloud_week1.txt", io.BytesIO(NOTES.encode()), "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready" and body["chunks"] >= 1 and body["chars"] > 200
    return body["id"]


def test_typed_message_and_follow_up_same_session(client, h1):
    r = client.post("/mentor/chat", headers=h1, json={"message": "Why did I lose marks in Question 1?"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["reply"]
    r2 = client.post("/mentor/chat", headers=h1, json={"message": "Which topics should I revise?", "session_id": sid})
    assert r2.status_code == 200 and r2.json()["session_id"] == sid
    msgs = client.get(f"/mentor/session/{sid}/messages", headers=h1).json()
    assert [m["sender"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == "Which topics should I revise?"   # exact text stored


def test_empty_message_rejected(client, h1):
    assert client.post("/mentor/chat", headers=h1, json={"message": "   "}).status_code == 422


@pytest.mark.parametrize("name,data,code", [
    ("virus.exe", b"MZ\x90\x00", 415),
    ("fake.pdf", b"MZ not a pdf", 422),
    ("empty.txt", b"", 422),
    ("bin.txt", b"abc\x00def", 422),
    ("fake.docx", b"plain text", 422),
])
def test_invalid_uploads_rejected(client, h1, name, data, code):
    r = client.post("/mentor/files", headers=h1, files={"file": (name, io.BytesIO(data))})
    assert r.status_code == code, r.text


def test_file_summary_and_mcq_are_grounded(client, h1, notes_id):
    r = client.post("/mentor/chat", headers=h1, json={
        "message": "Summarize this file", "document_ids": [notes_id], "action": "summarize"})
    assert r.status_code == 200
    reply = r.json()["reply"]
    assert "hypervisor" in reply.lower() or "cloud computing" in reply.lower()
    assert r.json()["sources_cited"]

    r = client.post("/mentor/chat", headers=h1, json={
        "message": "Generate MCQs", "document_ids": [notes_id], "action": "mcq"})
    reply = r.json()["reply"]
    assert "**Answer:**" in reply and "A." in reply and "cloud_week1.txt" in reply


def test_question_not_in_file_is_refused(client, h1, notes_id):
    r = client.post("/mentor/chat", headers=h1, json={
        "message": "Explain Dijkstra shortest path algorithm", "document_ids": [notes_id]})
    assert r.json()["grounded"] is False
    assert "isn't covered" in r.json()["reply"]


def test_other_student_cannot_use_or_delete_my_file(client, h1, h2, notes_id):
    r = client.post("/mentor/chat", headers=h2, json={
        "message": "Summarize", "document_ids": [notes_id], "action": "summarize"})
    assert "cloud" not in r.json()["reply"].lower()
    assert client.delete(f"/mentor/files/{notes_id}", headers=h2).status_code == 404
    assert notes_id not in [f["id"] for f in client.get("/mentor/files", headers=h2).json()]


def test_other_student_cannot_read_or_delete_my_session(client, h1, h2):
    sid = client.post("/mentor/chat", headers=h1, json={"message": "hello"}).json()["session_id"]
    assert client.get(f"/mentor/session/{sid}/messages", headers=h2).status_code == 404
    assert client.delete(f"/mentor/session/{sid}", headers=h2).status_code == 404
    assert client.delete(f"/mentor/session/{sid}", headers=h1).status_code == 200


def test_context_endpoint(client, h1):
    c = client.get("/mentor/context", headers=h1).json()
    assert {"name", "llm_online", "rag_sources", "max_file_mb"} <= set(c)


def test_requires_auth(client):
    assert client.post("/mentor/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/mentor/files").status_code == 401


def test_llm_prompt_contains_file_history_and_style(client, h1, notes_id, monkeypatch):
    """With an LLM configured, the real file text + previous turns + style reach the model."""
    calls = []

    class FakeLLM:
        active_provider = "ollama(llama3.2:3b)"

        def is_available(self):
            return True

        def complete(self, system, user, max_tokens=1000):
            calls.append((system, user))
            return "Hypervisor (p.1) — answer from your file."

    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: FakeLLM())
    r1 = client.post("/mentor/chat", headers=h1, json={
        "message": "What is a hypervisor?", "document_ids": [notes_id], "response_style": "hinglish"})
    sid = r1.json()["session_id"]
    assert r1.json()["reply"].startswith("Hypervisor") and r1.json()["mode"] == "llm"
    system, user = calls[-1]
    assert "Hinglish" in system and "ONLY" in system
    assert "hypervisor creates and runs virtual machines" in user.lower()

    client.post("/mentor/chat", headers=h1, json={
        "message": "Explain it with an example", "document_ids": [notes_id], "session_id": sid})
    _, user2 = calls[-1]
    assert "Earlier in this conversation" in user2 and "What is a hypervisor?" in user2


# ── Phase 1 security fixes ───────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/submissions", "/stats", "/papers", "/booklets", "/metrics", "/result/1"])
def test_legacy_endpoints_require_staff_login(client, h1, path):
    assert client.get(path).status_code == 401                       # anonymous
    assert client.get(path, headers=h1).status_code == 403           # student


def test_legacy_endpoints_work_for_faculty(client):
    r = client.post("/auth/login", json={"username": "faculty", "password": "faculty123"})
    hf = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/papers", headers=hf).status_code == 200
    assert client.get("/health").status_code == 200                  # still public


def test_change_password_flow(client):
    import uuid
    from backend.database import SessionLocal, User
    from backend.auth import get_password_hash
    name = "pwtest_" + uuid.uuid4().hex[:6]
    db = SessionLocal()
    db.add(User(username=name, email=f"{name}@test.local", hashed_password=get_password_hash("OldPass123"),
                full_name="PW Test", role="student"))
    db.commit(); db.close()
    h = {"Authorization": "Bearer " + client.post("/auth/login", json={"username": name, "password": "OldPass123"}).json()["access_token"]}
    bad = client.post("/auth/change-password", headers=h, json={"current_password": "nope", "new_password": "NewPass456", "confirm_password": "NewPass456"})
    assert bad.status_code == 400
    mismatch = client.post("/auth/change-password", headers=h, json={"current_password": "OldPass123", "new_password": "NewPass456", "confirm_password": "NewPass457"})
    assert mismatch.status_code == 400
    weak = client.post("/auth/change-password", headers=h, json={"current_password": "OldPass123", "new_password": "onlyletters", "confirm_password": "onlyletters"})
    assert weak.status_code == 400
    ok = client.post("/auth/change-password", headers=h, json={"current_password": "OldPass123", "new_password": "NewPass456", "confirm_password": "NewPass456"})
    assert ok.status_code == 200
    assert client.post("/auth/login", json={"username": name, "password": "OldPass123"}).status_code == 401
    assert client.post("/auth/login", json={"username": name, "password": "NewPass456"}).status_code == 200


def test_login_rate_limit(client):
    codes = [client.post("/auth/login", json={"username": "ratelimit_nobody", "password": "x"}).status_code
             for _ in range(10)]
    assert codes[0] == 401 and codes[-1] == 429


def test_reply_language_reaches_llm(client, h1, notes_id, monkeypatch):
    calls = []

    class FakeLLM:
        active_provider = "ollama(llama3.2:3b)"
        def is_available(self): return True
        def complete(self, system, user, max_tokens=1000):
            calls.append(system); return "சுருக்கம் (p.1)"

    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: FakeLLM())
    r = client.post("/mentor/chat", headers=h1, json={"message": "Summarize", "document_ids": [notes_id],
                                                     "action": "summarize", "response_language": "Tamil"})
    assert r.status_code == 200 and "ENTIRE reply in Tamil" in calls[-1]
    assert "AI model (Ollama)" not in r.json()["reply"]


def test_reply_language_offline_is_honest(client, h1, notes_id):
    r = client.post("/mentor/chat", headers=h1, json={"message": "Summarize", "document_ids": [notes_id],
                                                     "action": "summarize", "response_language": "Spanish (Mexico)"})
    assert "Spanish (Mexico)" in r.json()["reply"] and "Ollama" in r.json()["reply"]


def test_roadmap_task_progress(client, h1):
    r = client.post("/mentor-ai/roadmap/generate", headers=h1, json={"daily_hours": 2})
    assert r.status_code == 200 and len(r.json()["tasks"]) == 7
    p = client.patch("/mentor-ai/roadmap/task/0", headers=h1, json={"completed": True})
    assert p.status_code == 200 and p.json()["progress_pct"] == pytest.approx(14.3, abs=0.1)
    assert client.get("/mentor-ai/roadmap", headers=h1).json()["tasks"][0]["completed"] is True


# ── Phase 3: advanced chat ───────────────────────────────────────────────────

def _sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((ev, json.loads(data or "{}")))
    return events


import json  # noqa: E402


def test_stream_offline_sends_done_only(client, h1, notes_id):
    r = client.post("/mentor/chat/stream", headers=h1, json={"message": "What is a hypervisor?", "document_ids": [notes_id]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _sse(r.text)
    assert [e for e, _ in ev] == ["done"]                      # no fake token streaming without an LLM
    assert "hypervisor" in ev[-1][1]["reply"].lower()


def test_stream_llm_tokens(client, h1, notes_id, monkeypatch):
    class StreamLLM:
        active_provider = "ollama(llama3.2:3b)"
        def is_available(self): return True
        def complete(self, s, u, max_tokens=1000): return "unused"
        def stream(self, s, u, max_tokens=1000, on_token=None):
            for t in ["A hyper", "visor runs ", "VMs (p.1)."]:
                on_token(t)
            return "A hypervisor runs VMs (p.1)."
    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: StreamLLM())
    r = client.post("/mentor/chat/stream", headers=h1, json={"message": "What is a hypervisor?", "document_ids": [notes_id]})
    ev = _sse(r.text)
    toks = "".join(d["t"] for e, d in ev if e == "token")
    assert toks == "A hypervisor runs VMs (p.1)." and ev[-1][0] == "done"
    assert ev[-1][1]["reply"] == "A hypervisor runs VMs (p.1)."


def test_stream_requires_auth(client):
    assert client.post("/mentor/chat/stream", json={"message": "hi"}).status_code == 401


def test_regenerate_replaces_last_answer(client, h1):
    sid = client.post("/mentor/chat", headers=h1, json={"message": "Explain my marks"}).json()["session_id"]
    client.post("/mentor/chat", headers=h1, json={"message": "Explain my marks", "session_id": sid, "regenerate": True})
    msgs = client.get(f"/mentor/session/{sid}/messages", headers=h1).json()
    assert [m["sender"] for m in msgs] == ["user", "assistant"]


def test_rename_and_search_sessions(client, h1, h2):
    sid = client.post("/mentor/chat", headers=h1, json={"message": "zebra-unique-question about virtualization"}).json()["session_id"]
    assert client.patch(f"/mentor/session/{sid}", headers=h1, json={"title": "My virtualization notes"}).json()["title"] == "My virtualization notes"
    assert client.patch(f"/mentor/session/{sid}", headers=h2, json={"title": "hack"}).status_code == 404
    by_title = [s["id"] for s in client.get("/mentor/sessions?q=virtualization notes", headers=h1).json()]
    by_body = [s["id"] for s in client.get("/mentor/sessions?q=zebra-unique", headers=h1).json()]
    assert sid in by_title and sid in by_body
    assert sid not in [s["id"] for s in client.get("/mentor/sessions?q=zebra-unique", headers=h2).json()]


def test_folders_and_preview(client, h1, h2):
    r = client.post("/mentor/files", headers=h1, data={"folder": "Semester 5/../CS408"},
                    files={"file": ("os.txt", io.BytesIO(NOTES.encode()), "text/plain")})
    fid = r.json()["id"]
    assert r.json()["folder"] == "Semester 5..CS408"               # path characters stripped
    assert client.patch(f"/mentor/files/{fid}", headers=h1, json={"folder": "Cloud"}).json()["folder"] == "Cloud"
    assert any(f["folder"] == "Cloud" for f in client.get("/mentor/folders", headers=h1).json())
    assert [f["id"] for f in client.get("/mentor/files?folder=Cloud", headers=h1).json()] == [fid] or \
           fid in [f["id"] for f in client.get("/mentor/files?folder=Cloud", headers=h1).json()]
    p = client.get(f"/mentor/files/{fid}/preview", headers=h1).json()
    assert "hypervisor" in p["preview_pages"][0]["text"].lower()
    assert client.get(f"/mentor/files/{fid}/preview", headers=h2).status_code == 404
    assert client.patch(f"/mentor/files/{fid}", headers=h2, json={"folder": "x"}).status_code == 404
