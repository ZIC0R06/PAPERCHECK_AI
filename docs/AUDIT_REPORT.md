# PaperCheckAI — Audit Report (phase 1: AI engine, Mentor language, grading · phase 2: voice, OCR failure, assistants)

Scope covered so far: LLM provider/config, AI Mentor language, answer grading, status endpoints, startup/dependencies.
Scope NOT yet audited: OMR, scanner, multi-page processing, admin/faculty/student portal workflows, full UI click-through,
OCR accuracy, question-paper/answer-key parser quality. See "Remaining work".

| # | File | Root cause | Correction | Test performed | Remaining limitation |
|---|------|-----------|-----------|----------------|----------------------|
| 1 | backend/config.py | Default provider was Groq; `active_llm` only knew groq/claude, so Ollama could never be "active"; warned about missing cloud keys | Default `ollama`; Ollama settings added; `active_llm` recognises Ollama and falls back to it | Imported settings: active_llm=ollama, model llama3.2:3b, tokens 4096, temp 0.2 | Live reachability is reported by the provider, not config |
| 2 | backend/llm_provider.py | Groq/Claude only; missing keys meant no AI; on failure it returned an invented `score: 5.0` JSON, so marks could be fabricated | Rewritten: Ollama default (chat, history, system prompt, streaming, JSON mode, health, model check, timeout, retry, logging); cloud optional over HTTPS; raises `LLMError` instead of faking output | 25 provider tests against a mock Ollama HTTP server | Not run against a real Ollama in my sandbox: run `scripts/ollama_smoke_test.py` |
| 3 | Ollama context | Ollama's default context (2048 tokens) silently truncates long prompts | `num_ctx` defaults to 8192 (`OLLAMA_NUM_CTX`) | Asserted in request payload test | 3B model quality on very long documents is limited |
| 4 | backend/mentor_service.py | Language chosen by a keyword regex that treated English words ("marks", "help", "next time") as Hinglish; Auto instruction was vague; explicit Hinglish overwrote other styles; history/docs could steer language | New `language_policy.py`: explicit > latest message; firm per-language instructions; wrong-language reply retried once; static messages localised | 19 policy tests pass; `tests/test_mentor_language.py` written | API-level language tests could not be executed here (no FastAPI/SQLAlchemy) |
| 5 | mentor_service / learning_service `llm_available` | Checked for `_groq_client/_claude_client`, so Ollama was never seen as available | Use `client.is_available()` (live Ollama + model check) | Compiles; covered by updated fakes in tests | Not executed end to end |
| 6 | answer_grader.py | Marks from AI trusted without evidence; strings like "7/10" crashed parsing; offline text said "no AI configured" | JSON mode at temp 0; tolerant number parsing; marks withheld when quoted evidence is absent from the student answer; confidence capped and flag added | 6 grading tests pass (via pytest stand-in) | Evidence check is word-overlap based, not semantic |
| 7 | llm_evaluator.py | Offline heuristic reported confidence 0.55–0.92 | Capped at 0.45 so it is always flagged for review | Compiles | Not executed |
| 8 | parsers (answer key, question paper, student answers, classifier) | Asked for JSON in free text | Use JSON mode | Compiles | Real extraction quality with a 3B model unverified |
| 9 | api.py / system_routes.py | `/` hard-coded "groq"; `/health` hid AI state | Report real Ollama state; new `/system/llm-status` | Compiles | Not executed |
| 10 | handwriting_reader.py | No local vision option | Opt-in Ollama vision (needs a vision model; llama3.2:3b is text-only) | Compiles | Default stays local OCR with teacher flags |
| 11 | mentor_app.html | UI told users to add an LLM API key | Messages now point to Ollama | Visual diff only | Not clicked through in a browser |
| 12 | requirements.txt / .env.example / start.bat | Cloud SDKs required; no Ollama config; hard-coded Python paths | SDKs removed; Ollama defaults; new start.bat with Python/deps/Ollama/model/DB checks and DB backup | `scripts/check_ollama.py` exit codes verified | start.bat not executed (Linux sandbox) |
| 13 | tests/test_all.py etc. | Tests asserted the fake-score behaviour and Groq internals | Updated to the new contract | Runnable ones pass | Others need your environment |

## Phase 2 findings

| # | File | Root cause | Correction | Test performed | Remaining limitation |
|---|------|-----------|-----------|----------------|----------------------|
| 14 | frontend/mentor_app.html (voice) | Every sentence chunk was queued at once; voice picked before browser voices had loaded (Chrome loads them late); no retry, no watchdog; native pause flaky; Chrome drops speech started right after `cancel()`; errors swallowed so voice "sometimes" worked | New engine `frontend/voice_out.js` (sequential chunks, generation token = no overlap, waits for voices, settle delay, start-watchdog + 2 retries, soft pause/resume, honest error/unsupported states, per-answer status text). Embedded by `scripts/sync_voice.py` | 16 Node tests with a fake speech engine and fake clock; embed-sync test; page script passes `node --check` | NOT run in a real browser or with real audio — verify by pressing Speak in Chrome/Edge. Needs an installed Hindi voice for Hindi speech (reported if missing) |
| 15 | handwriting_reader.py | Tesseract located only via `TESSERACT_CMD`, not via the standard Windows folders | New `backend/tesseract_locator.py` (env → PATH → standard install dirs) used by every OCR module | 3 locator tests | — |
| 16 | handwriting_reader.py | If local OCR crashed, pages silently became blank and were graded "not attempted" (zero marks from unreadable text) | OCR failure on every page now raises with a precise reason; missing Tesseract is never swallowed | 2 tests | — |
| 17 | handwriting_reader.py | PDF rendering depended on PyMuPDF only | pypdfium2 fallback | fixture test (2 pages) | — |
| 18 | tests/conftest.py | Tests blanked cloud keys but not Ollama: with a real Ollama running, the e2e handwritten test would talk to the live model while asserting deterministic offline behaviour | Hermetic AI env (dead Ollama port, local handwriting) unless `PAPERCHECK_TEST_LIVE_AI=1` | Not runnable here | See "the failing e2e test" below |
| 19 | (new) backend/exam_agent.py, routes/agent_routes.py, portal_pages.py | No Admin/Faculty AI agent existed | Read-only, role-scoped assistant: deterministic tool routing → database facts → Ollama wording; answers with numbers not in the data are rejected; audited; Speak buttons | 21 logic tests (fake repo) | SQL repository + API tests written (`tests/test_exam_agent_api.py`) but NOT executed here |
| 20 | mentor_app.html robot | Only blink/float/talk | Added thinking (eye scan, antenna pulse), listening (ear pulse), speaking (sound rings), waving arm, voice-error tint; respects reduced-motion | CSS only; not viewed in a browser | Robot appears in the student Mentor page, not in the Streamlit dashboards |

### The "one failing handwritten OCR end-to-end test"
I could not run `tests/test_end_to_end_handwritten.py` (it needs FastAPI, SQLAlchemy and PyMuPDF, none installable offline here), so I cannot say which assertion failed for you. What I did establish by running the project's own reader on `tests/fixtures/handwritten_answer_sheet.pdf` with Tesseract: the logic is correct here — Tesseract reads the handwritten "Q1" as "Q7", the reconciler maps it to Q1, and Q1–Q3 land on the right pages. The code defects that could make it fail on a Windows machine are items 15–18 above. Regression tests for them are in `tests/test_ocr_regression.py`. If it still fails for you, send me the pytest output.

## Not verified / remaining work
Camera capture in a browser, multi-page ordering/crop/rotate UI, student-identity confirmation flow, duplicate-upload prevention, the existing re-evaluation module's audit trail ("never overwrite original marks"), database migrations, OMR — these exist in the project from before but I did NOT audit or test them in this phase. Handwriting accuracy: Tesseract is weak on real cursive; the optional local vision path (`ollama pull llama3.2-vision`, `HANDWRITING_PROVIDER=ollama`) is implemented but untested.

## Remaining work (phase 1 list)
Audit and test OMR/scanner/multi-page flows, admin/faculty/student workflows, UI click-through, and OCR/parsing accuracy on real papers.
