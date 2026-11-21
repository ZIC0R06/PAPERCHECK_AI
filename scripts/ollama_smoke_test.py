#!/usr/bin/env python3
"""
REAL-MODEL smoke test — run this on YOUR machine with Ollama running:

    python scripts/ollama_smoke_test.py

Checks against your actual local model (default llama3.2:3b):
  1 connection + model installed   2 plain generation   3 streaming   4 JSON output
  5-7 reply language: English / Hindi / Hinglish (explicit selection, using the Mentor's own language rules)
  8 AI marking of a sample answer (partial marks, evidence-checked)
Exit code 0 only if every check passes. Nothing here is mocked.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.language_policy import language_instruction, reply_matches, resolve_language  # noqa: E402
from backend.llm_provider import LLMClient, LLMError, get_llm_client  # noqa: E402

results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, f"{detail} ({time.time() - t0:.1f}s)"))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {str(e)[:300]}"))
    print(("PASS " if results[-1][1] else "FAIL ") + name + " — " + results[-1][2], flush=True)


c: LLMClient = get_llm_client()


def t_health():
    h = c.health(force=True)
    assert h["reachable"], h["error"]
    assert h["model_available"], h["error"]
    return f"{h['model']} @ {h['base_url']}, {h['latency_ms']} ms"


def t_generate():
    out = c.complete("You are concise.", "Reply with exactly one short sentence: what is a stack in computer science?",
                     max_tokens=80)
    assert len(out.strip()) > 10
    return out.strip()[:70]


def t_stream():
    parts = []
    full = c.stream("You are concise.", "Count from one to five in words.", max_tokens=60, on_token=parts.append)
    assert len(parts) > 1 and full.strip()
    return f"{len(parts)} chunks"


def t_json():
    d = c.generate_json('Return JSON {"capital": "<capital of France>", "confidence": <0..1>}', max_tokens=80)
    assert "paris" in str(d.get("capital", "")).lower(), d
    return str(d)


def lang_check(selected, question, must_be):
    def run():
        lang = resolve_language(selected, question)
        out = c.complete("You are a university study mentor. Answer in 3 short sentences. " + language_instruction(lang),
                         question, max_tokens=250)
        assert lang.code == must_be
        assert reply_matches(lang, out), f"wrong language: {out[:120]!r}"
        return out.strip()[:80].replace("\n", " ")
    return run


def t_grade():
    from backend.answer_grader import grade_answer
    q = {"question_text": "What is a deadlock? How can it be prevented?", "marks": 10, "question_type": "open_ended",
         "model_answer": "Deadlock: processes wait for each other's locks forever. Prevention: lock ordering, timeouts.",
         "rubric": [{"criterion": "Definition of deadlock", "marks": 4}, {"criterion": "Prevention methods", "marks": 6}]}
    g = grade_answer(q, "A deadlock is when two processes wait for each other forever. We can use timeouts.",
                     llm_client=c)
    assert g.method == "ai", f"AI grading was not used (method={g.method}); flags={g.flags}"
    assert 0 < g.score < 10, f"expected PARTIAL marks, got {g.score}"
    return f"score {g.score}/10, confidence {g.confidence}, flags={len(g.flags)}"


check("1 connection + model installed", t_health)
check("2 generation", t_generate)
check("3 streaming", t_stream)
check("4 JSON output", t_json)
check("5 English (auto, English question)", lang_check("auto", "What is a binary tree?", "en"))
check("5b English explicit, Hindi question", lang_check("English", "बाइनरी ट्री क्या है?", "en"))
check("6 Hindi explicit", lang_check("Hindi", "What is a binary tree?", "hi"))
check("6b Hindi auto, Hindi question", lang_check("auto", "बाइनरी ट्री क्या है?", "hi"))
check("7 Hinglish explicit", lang_check("Hinglish", "What is a binary tree?", "hinglish"))
check("7b Hinglish auto, Hinglish question", lang_check("auto", "Mujhe binary tree samjhao", "hinglish"))
check("8 AI marking with partial marks", t_grade)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
