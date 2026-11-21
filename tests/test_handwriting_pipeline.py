"""
Handwriting reading + question-wise grading (backend.handwriting_reader, backend.answer_grader).

* Local OCR is exercised on a real image-only PDF written in a handwriting font (tests/fixtures).
* Vision-AI providers are exercised through a mocked HTTP layer: request shape, response parsing,
  answers continuing across pages, illegible words, and fallback to local OCR when the AI call fails.
  (Live calls need an API key; run with PAPERCHECK_TEST_LIVE_AI=1 and a key to try them.)
"""
import json
import os

import pytest
from PIL import Image

from backend import handwriting_reader as HR
from backend.answer_grader import grade_answer, derive_marking_points

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "handwritten_answer_sheet.pdf")
QUESTIONS = [{"question_number": 1, "question_text": "Define primary key with an example.", "marks": 10},
             {"question_number": 2, "question_text": "Explain normalization (1NF, 2NF, 3NF).", "marks": 10},
             {"question_number": 3, "question_text": "What is a deadlock? How can it be prevented?", "marks": 10}]


@pytest.fixture
def no_ai_keys(monkeypatch):
    for k in HR.PROVIDER_KEYS.values():
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("HANDWRITING_PROVIDER", "auto")


# ── local OCR ─────────────────────────────────────────────────────────────────

def test_local_ocr_reads_and_segments_handwritten_sheet(no_ai_keys):
    r = HR.read_answer_sheet(FIX, QUESTIONS)
    assert r.engine == "local-tesseract" and r.pages == 2
    got = {a.question_number: a for a in r.answers}
    assert set(got) == {1, 2, 3}
    assert "primary key" in got[1].text.lower() and "unique" in got[1].text.lower()
    assert "redundancy" in got[2].text.lower()
    assert "deadlock" in got[3].text.lower()
    assert got[3].page_refs[0]["page"] == 2 and got[1].page_refs[0]["page"] == 1
    assert 0 < got[1].page_refs[0]["y0"] < got[2].page_refs[0]["y0"] < 1
    assert any("local OCR" in w for w in r.warnings)          # always tells the teacher to verify


def test_misread_markers_are_matched_to_the_paper():
    S = HR.PageSegment
    segs = [S(1, 7, "Q7", "a", 0.1, 0.2, 0.8, "local-tesseract"),    # "Q1" read as "Q7"
            S(1, -1, "))", "b", 0.3, 0.4, 0.8, "local-tesseract"),   # unreadable marker
            S(2, None, "", "b contd", 0.0, 0.1, 0.8, "local-tesseract"),
            S(2, 3, "Q3", "c", 0.2, 0.3, 0.8, "local-tesseract")]
    HR.resolve_question_numbers(segs, [1, 2, 3])
    assert [s.question_number for s in segs] == [1, 2, None, 3]
    answers, unassigned = HR._merge(segs)
    assert [a.question_number for a in answers] == [1, 2, 3]
    assert answers[1].text == "b\nb contd" and [p["page"] for p in answers[1].page_refs] == [1, 2]


# ── vision AI (mocked HTTP) ───────────────────────────────────────────────────

def _pages(n=2):
    return [Image.new("RGB", (1240, 1754), "white") for _ in range(n)]


def test_vision_ai_reading_merges_pages_and_scores_legibility(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("HANDWRITING_PROVIDER", "anthropic")
    calls = []
    replies = [
        {"segments": [{"question_number": 1, "label": "Q1", "text": "Primary key identifies a row uniquely.",
                       "y_start": 0.05, "y_end": 0.3, "legibility": 0.95},
                      {"question_number": 2, "label": "Ans 2", "text": "Normalization reduces [illegible] redundancy",
                       "y_start": 0.35, "y_end": 0.9, "legibility": 0.8}]},
        {"segments": [{"question_number": None, "text": "and removes anomalies.", "y_start": 0.02, "y_end": 0.1,
                       "legibility": 0.9},
                      {"question_number": 3, "label": "Q3", "text": "Deadlock is a circular wait.", "y_start": 0.2,
                       "y_end": 0.4, "legibility": 0.9}]},
    ]

    def fake_post(url, headers, payload):
        calls.append((url, headers, payload))
        return {"content": [{"type": "text", "text": "```json\n" + json.dumps(replies[len(calls) - 1]) + "\n```"}]}

    monkeypatch.setattr(HR, "_http_post", fake_post)
    r = HR.read_answer_sheet("unused.pdf", QUESTIONS, pages=_pages())
    assert r.engine.startswith("anthropic:") and not r.used_local
    url, headers, payload = calls[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "test-key"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg"
    assert "Define primary key" in content[1]["text"] and "EXACTLY" in payload["system"]
    assert "ended inside the answer to Q2" in calls[1][2]["messages"][0]["content"][1]["text"]
    got = {a.question_number: a for a in r.answers}
    assert got[2].text.endswith("and removes anomalies.")
    assert [p["page"] for p in got[2].page_refs] == [1, 2]
    assert got[2].illegible_words == 1 and got[2].confidence < got[1].confidence


@pytest.mark.parametrize("provider,reply,url_part", [
    ("gemini", {"candidates": [{"content": {"parts": [{"text": json.dumps({"segments": [
        {"question_number": 1, "text": "abc", "y_start": 0.1, "y_end": 0.2, "legibility": 0.9}]})}]}}]},
     "generativelanguage.googleapis.com"),
    ("openai", {"choices": [{"message": {"content": json.dumps({"segments": [
        {"question_number": 1, "text": "abc", "y_start": 0.1, "y_end": 0.2, "legibility": 0.9}]})}}]},
     "api.openai.com"),
    ("groq", {"choices": [{"message": {"content": json.dumps({"segments": [
        {"question_number": "Q1", "text": "abc", "y_start": 0.1, "y_end": 0.2}]})}}]},
     "api.groq.com"),
])
def test_each_vision_provider_request_and_response(monkeypatch, provider, reply, url_part):
    for k in HR.PROVIDER_KEYS.values():
        monkeypatch.setenv(k, "")
    monkeypatch.setenv(HR.PROVIDER_KEYS[provider], "k")
    monkeypatch.setenv("HANDWRITING_PROVIDER", "auto")     # conftest forces "local" for every other test
    seen = {}

    def fake_post(url, headers, payload):
        seen["url"], seen["payload"] = url, payload
        return reply

    monkeypatch.setattr(HR, "_http_post", fake_post)
    r = HR.read_answer_sheet("x.pdf", QUESTIONS[:1], pages=_pages(1))
    assert url_part in seen["url"] and r.answers[0].question_number == 1 and r.answers[0].text == "abc"
    assert "image" in json.dumps(seen["payload"])[:5000] or "inline_data" in json.dumps(seen["payload"])


def test_failed_ai_call_falls_back_to_local_ocr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    monkeypatch.setenv("HANDWRITING_PROVIDER", "auto")

    def boom(url, headers, payload):
        raise RuntimeError("HTTP 401: invalid x-api-key")

    monkeypatch.setattr(HR, "_http_post", boom)
    r = HR.read_answer_sheet(FIX, QUESTIONS)
    assert r.used_local and {a.question_number for a in r.answers} == {1, 2, 3}
    assert any("vision reading failed" in w for w in r.warnings)


# ── grading ──────────────────────────────────────────────────────────────────

class FakeLLM:
    has_real_provider = True
    active_provider = "claude(test)"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user, max_tokens=0):
        self.prompts.append(user)
        return self.reply

    def generate_json(self, prompt, max_tokens=0, system=None, temperature=None):
        """Mirror of LLMClient.generate_json: parse or raise (never invent a result)."""
        from backend.llm_provider import parse_json_text
        self.prompts.append(prompt)
        return parse_json_text(self.reply)


Q = {"question_text": "What is a deadlock? How can it be prevented?", "marks": 10, "question_type": "open_ended",
     "model_answer": "Deadlock: transactions wait for each other's locks forever. Prevention: lock ordering, timeouts.",
     "rubric": [{"criterion": "Definition of deadlock", "marks": 4}, {"criterion": "Prevention methods", "marks": 6}]}


def test_ai_grading_is_criterion_wise_with_deductions_and_corrections():
    reply = json.dumps({"rubric_breakdown": [
        {"criterion": "Definition of deadlock", "max_marks": 4, "awarded": 4, "status": "met", "evidence": "wait for each other", "reason": "Correct."},
        {"criterion": "Prevention methods", "max_marks": 6, "awarded": 2.4, "status": "partial", "evidence": "timeouts", "reason": "Only one method."}],
        "corrections": [{"student_wrote": "deadlock is caused by slow CPU", "correct_version": "caused by circular wait on locks", "explanation": "x"}],
        "strengths": ["definition"], "missing_concepts": ["lock ordering"], "feedback": "Good start.", "confidence": 0.85})
    llm = FakeLLM(reply)
    g = grade_answer(Q, "Deadlock is when transactions wait for each other. Use timeouts.", llm_client=llm,
                     ocr_confidence=0.95, ocr_engine="anthropic:x")
    assert g.method == "ai" and g.score == 6.5          # 6.4 rounded to the 0.5 step
    assert g.deductions == [{"marks": 3.6, "criterion": "Prevention methods", "reason": "Only one method."}]
    assert g.corrections[0]["correct_version"].startswith("caused by circular")
    assert "Definition of deadlock" in llm.prompts[0] and "[4 marks]" in llm.prompts[0]
    assert not g.flags                                   # confident AI + confident reading → no flag


def test_ai_breakdown_is_rescaled_to_paper_marks_and_bad_reply_falls_back():
    llm = FakeLLM(json.dumps({"rubric_breakdown": [{"criterion": "a", "max_marks": 5, "awarded": 5, "evidence": "some answer"},
                                                   {"criterion": "b", "max_marks": 15, "awarded": 5, "evidence": "some answer"}], "confidence": 0.9}))
    g = grade_answer(Q, "some answer", llm_client=llm)
    assert g.score == 5.0 and sum(r["max_marks"] for r in g.rubric_breakdown) == pytest.approx(10)
    g2 = grade_answer(Q, "deadlock wait timeouts", llm_client=FakeLLM("not json at all"))
    assert g2.method == "keyword-matching" and any("AI model" in f for f in g2.flags)


def test_offline_grading_is_explained_and_always_flagged():
    q = {"question_text": "Define primary key", "marks": 10,
         "model_answer": "A primary key uniquely identifies each record in a table. It must be unique and not null. Example: roll number."}
    good = grade_answer(q, "A primary key is an attribute that uniquely identifies each row of a table. It cannot be null "
                           "and must be unique. Example: roll number.", llm_client=FakeLLM("") if False else None)
    bad = grade_answer(q, "A primary key is a key.")
    assert good.score >= 9 and bad.score <= 2 and good.method == "keyword-matching"
    assert all(b["status"] == "missing" for b in bad.rubric_breakdown[1:])
    assert sum(d["marks"] for d in bad.deductions) == pytest.approx(10 - bad.score, abs=0.5)
    assert good.flags and "AI model (Ollama)" in good.flags[0] and "verification REQUIRED" in good.flags[0]
    assert len(derive_marking_points(q["model_answer"], 10)) == 3


def test_unanswered_mcq_and_low_ocr_confidence():
    g = grade_answer(Q, "", llm_client=FakeLLM("{}"))
    assert g.score == 0 and g.deductions[0]["marks"] == 10 and "Not attempted" in g.deductions[0]["reason"]
    m = grade_answer({"question_type": "mcq", "marks": 2, "correct_option": "C"}, "Ans: (c)")
    assert m.score == 2 and m.method == "objective"
    w = grade_answer({"question_type": "mcq", "marks": 2, "correct_option": "C"}, "B")
    assert w.score == 0 and w.corrections[0]["correct_version"] == "C"
    reply = json.dumps({"rubric_breakdown": [{"criterion": "x", "max_marks": 10, "awarded": 7}], "confidence": 0.9})
    g = grade_answer(Q, "text", llm_client=FakeLLM(reply), ocr_confidence=0.5, ocr_engine="local-tesseract")
    assert any("local OCR" in f for f in g.flags) and any("Low handwriting" in f for f in g.flags)


def test_marks_without_evidence_in_the_student_answer_are_withheld():
    """A small local model may 'quote' words the student never wrote. Such marks must not be awarded."""
    reply = json.dumps({"rubric_breakdown": [
        {"criterion": "Definition of deadlock", "max_marks": 4, "awarded": 4, "status": "met",
         "evidence": "circular wait on exclusive locks", "reason": "Correct."},
        {"criterion": "Prevention methods", "max_marks": 6, "awarded": 3, "status": "partial",
         "evidence": "timeouts", "reason": "One method."}], "confidence": 0.9})
    g = grade_answer(Q, "Deadlock is bad. Use timeouts.", llm_client=FakeLLM(reply))
    assert g.score == 3.0                                  # the hallucinated 4 marks are withheld
    assert g.confidence <= 0.5 and any("without verifiable evidence" in f for f in g.flags)
    assert g.rubric_breakdown[0]["awarded"] == 0 and g.rubric_breakdown[0]["status"] == "missing"


def test_string_scores_from_small_models_are_parsed():
    reply = json.dumps({"rubric_breakdown": [
        {"criterion": "Definition of deadlock", "max_marks": "4 marks", "awarded": "4/4", "evidence": "wait for each other"},
        {"criterion": "Prevention methods", "max_marks": "6", "awarded": "2", "evidence": "timeouts"}], "confidence": "85"})
    g = grade_answer(Q, "Deadlock is when transactions wait for each other. Use timeouts.", llm_client=FakeLLM(reply))
    assert g.method == "ai" and g.score == 6.0 and g.confidence == pytest.approx(0.85)
