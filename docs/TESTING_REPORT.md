# Testing Report (honest status)

Environment of these runs: Linux sandbox, Python 3.12, no internet, no Ollama, no FastAPI/SQLAlchemy/pytest.

| Check | Result |
|---|---|
| Python compilation of backend, frontend, tests, scripts | PASS |
| tests/test_ollama_provider.py (25, mock Ollama HTTP server) | PASS |
| tests/test_language_policy.py (19) | PASS |
| tests/test_all.py::TestLLMProvider (5) | PASS |
| 6 grading tests in tests/test_handwriting_pipeline.py (run with a minimal pytest stand-in) | PASS |
| tests/test_ocr_regression.py (10, real Tesseract on the handwritten fixture) | PASS |
| tests/test_exam_agent.py (21, fake repository) | PASS |
| tests/test_voice_embed.py (3) | PASS |
| node --test tests/js/voice_out.test.js (16, fake speech engine) | PASS |
| scripts/check_ollama.py with Ollama absent | exit 2 as designed |
| scripts/ollama_smoke_test.py with Ollama absent | fails honestly, no fake output |
| intelligrade.db unchanged (md5 equal to original) | PASS |

## NOT executed (needs your machine)
real browser voice playback, tests/test_exam_agent_api.py, tests/test_end_to_end_handwritten.py (the previously failing test), SqlRepo queries, Streamlit pages, camera scanning,
Existing full suite (`pytest tests -q`), tests/test_mentor_language.py, backend and frontend startup, real llama3.2:3b generation/streaming/language checks (`python scripts/ollama_smoke_test.py`), PDF/image upload, OCR, paper evaluation end to end, OMR, admin/faculty/student workflows, DB persistence, start.bat.
