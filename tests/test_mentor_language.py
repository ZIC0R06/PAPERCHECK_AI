"""
AI Mentor — reply-language behaviour through the REAL API path (needs the full stack: fastapi, sqlalchemy).
A recording fake LLM captures the system prompt, so we can assert which language rule reached the model for
every scenario in the requirements: Auto+EN/HI/Hinglish, explicit EN/HI/Hinglish, follow-ups, regenerate,
documents in another language, streaming, summaries / notes / MCQs, offline fallback.

Run:  pytest tests/test_mentor_language.py -q
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend import mentor_service

HINDI_DOC = "हाइपरवाइज़र वर्चुअल मशीन बनाता और चलाता है। क्लाउड कंप्यूटिंग में संसाधन मांग पर मिलते हैं। " * 6
ENGLISH_DOC = ("A hypervisor creates and runs virtual machines. Cloud computing delivers computing resources on demand. "
               "Virtualization is the key enabling technology of cloud computing. " * 4)


class RecordingLLM:
    active_provider = "ollama(llama3.2:3b)"

    def __init__(self, reply="OK reply in English."):
        self.systems, self.users, self.reply = [], [], reply

    def is_available(self):
        return True

    def complete(self, system, user, max_tokens=1000, **kw):
        self.systems.append(system)
        self.users.append(user)
        return self.reply

    def stream(self, system, user, max_tokens=1000, on_token=None, **kw):
        self.systems.append(system)
        self.users.append(user)
        if on_token:
            on_token(self.reply)
        return self.reply


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def h(client):
    r = client.post("/auth/login", json={"username": "student", "password": "student123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _upload(client, h, name, text):
    r = client.post("/mentor/files", headers=h, files={"file": (name, io.BytesIO(text.encode()), "text/plain")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


@pytest.fixture
def llm(monkeypatch):
    fake = RecordingLLM()
    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: fake)
    return fake


def ask(client, h, message, lang="auto", **extra):
    r = client.post("/mentor/chat", headers=h, json={"message": message, "response_language": lang, **extra})
    assert r.status_code == 200, r.text
    return r.json()


EN_Q, HI_Q, HING_Q = "What is a hypervisor in cloud computing?", "क्लाउड कंप्यूटिंग में हाइपरवाइज़र क्या है?", \
    "Mujhe hypervisor samjhao, ye kya hota hai?"


@pytest.mark.parametrize("question,lang,marker", [
    (EN_Q, "auto", "English only"),
    (HI_Q, "auto", "Hindi using Devanagari"),
    (HING_Q, "auto", "ROMAN"),
    (HI_Q, "English", "English only"),
    (HING_Q, "English", "English only"),
    (EN_Q, "Hindi", "Hindi using Devanagari"),
    (EN_Q, "Hinglish", "ROMAN"),
    (HI_Q, "Hinglish", "ROMAN"),
])
def test_normal_chat_language_rule(client, h, llm, question, lang, marker):
    ask(client, h, question)            # make sure old history exists in the other language
    ask(client, h, question, lang)
    assert marker in llm.systems[-1], llm.systems[-1][-400:]


def test_english_question_with_marks_word_is_not_treated_as_hinglish(client, h, llm):
    import json as _json
    r = ask(client, h, "Why did I lose marks in question 1? Please help.")
    if llm.systems:      # answered by the model → the English rule must reach it
        assert "English only" in llm.systems[-1]
    else:                # answered from the official-result template (no model call) → English template, not Hinglish
        txt = _json.dumps(r, ensure_ascii=False).lower()
        assert not any(w in txt for w in ("kate", "kyu ", "samjhaunga", "notes se reference"))


def test_hindi_documents_do_not_override_explicit_english(client, h, llm):
    doc = _upload(client, h, "hindi_notes.txt", HINDI_DOC)
    ask(client, h, "What is a hypervisor?", "English", document_ids=[doc])
    assert "English only" in llm.systems[-1] and "must NOT change" in llm.systems[-1]


def test_english_documents_do_not_override_explicit_hindi(client, h, llm):
    doc = _upload(client, h, "english_notes.txt", ENGLISH_DOC)
    ask(client, h, "What is a hypervisor?", "Hindi", document_ids=[doc])
    assert "Devanagari" in llm.systems[-1]


def test_followup_uses_latest_message_not_history(client, h, llm):
    first = ask(client, h, HI_Q)                                    # Hindi turn
    ask(client, h, "Explain it again with an example", session_id=first["session_id"])   # English follow-up
    assert "English only" in llm.systems[-1]


def test_regenerate_keeps_language(client, h, llm):
    first = ask(client, h, HING_Q, "Hinglish")
    ask(client, h, HING_Q, "Hinglish", session_id=first["session_id"], regenerate=True)
    assert "ROMAN" in llm.systems[-1]


@pytest.mark.parametrize("action", ["summarize", "mcq", "important_questions", "short_questions"])
def test_document_actions_respect_language(client, h, llm, action):
    doc = _upload(client, h, f"notes_{action}.txt", ENGLISH_DOC)
    ask(client, h, "Do it", "English", document_ids=[doc], action=action)
    assert "English only" in llm.systems[-1]
    ask(client, h, "Do it", "Hindi", document_ids=[doc], action=action)
    assert "Devanagari" in llm.systems[-1]


def test_study_notes_request_in_english_auto(client, h, llm):
    ask(client, h, "Give me short notes on virtualization", response_style="short_notes")
    assert "English only" in llm.systems[-1] and "bullet" in llm.systems[-1].lower()


def test_streaming_respects_language(client, h, llm):
    r = client.post("/mentor/chat/stream", headers=h, json={"message": EN_Q, "response_language": "English"})
    assert r.status_code == 200 and "English only" in llm.systems[-1]


def test_wrong_language_reply_triggers_one_stricter_retry(client, h, monkeypatch):
    class Drifting(RecordingLLM):
        def complete(self, system, user, max_tokens=1000, **kw):
            super().complete(system, user, max_tokens)
            return ("हाइपरवाइज़र एक सॉफ़्टवेयर है जो वर्चुअल मशीन बनाता है और चलाता है।"
                    if len(self.systems) == 1 else "A hypervisor creates virtual machines.")
    fake = Drifting()
    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: fake)
    out = ask(client, h, EN_Q, "English")
    assert len(fake.systems) == 2 and "WRONG LANGUAGE" in fake.systems[1]
    assert out["reply"].startswith("A hypervisor")


class DownLLM:
    active_provider = "offline"

    def is_available(self):
        return False

    def complete(self, *a, **k):
        raise RuntimeError("down")


@pytest.mark.parametrize("lang,needle", [("English", "Ollama"), ("Hinglish", "nahi"), ("Hindi", "Ollama")])
def test_offline_fallback_messages_follow_language_and_are_honest(client, h, monkeypatch, lang, needle):
    monkeypatch.setattr(mentor_service, "get_llm_client", lambda: DownLLM())
    doc = _upload(client, h, f"off_{lang}.txt", ENGLISH_DOC)
    out = ask(client, h, "Summarize", lang, document_ids=[doc], action="summarize")
    assert needle in out["reply"] and out["mode"] == "offline"
