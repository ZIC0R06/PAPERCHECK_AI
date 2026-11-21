# AI Robot Examination System — upgrade notes & honest test status (Nov 2026)

## What was added
| Area | Files |
|---|---|
| Bulk checking engine: one paper reused, sheet storage (originals read-only), student identification (file name + OCR of page 1, OCR-confusion folding, conflicts → manual), duplicate checks, background worker with per-question progress, restart recovery, class report PDF, ZIP | `backend/bulk_evaluation.py`, models `EvaluationJob`/`EvaluationJobItem` in `backend/database.py` |
| API `/bulk/*` (create batch, upload, camera scan, assign, start/stop, progress, report, zip, approve, publish) | `backend/routes/bulk_routes.py` |
| Robot actions by chat/voice ("check all…", progress, stop, class report, unclear sheets, admin monitoring) | `backend/robot_agent.py`, `backend/routes/agent_routes.py` |
| Admin directory tool | `backend/exam_agent.py` |
| Student Examination Assistant (published results only; why/revision notes/improvement/summary/subjects; session memory) | `backend/student_exam_assistant.py`, `backend/routes/student_assistant_routes.py` |
| Robot UI: animated robot (idle/listening/thinking/processing/speaking/error, blink, head motion, mouth), chat bubbles + typing, voice in/out (shared tested engine), camera scanner, checking panel, student results | `frontend/exam_assistant_app.html`, embedded via `frontend/portal_pages.py` for admin, faculty **and students** |

Safety rules: never marks a sheet without a verified student; one sheet per student per exam; AI marks stay `in_review`; approval refused while flagged answers are unverified; publishing only after approval; students see only their own published results; faculty limited to assigned subject+batch; every action audited.

## Existing problems fixed
- Local OCR reader treated a header like "Answer Sheet" as a question marker and cut the first letters of answers ("Q3 Deadlock" → "dlock"). Fixed in `backend/handwriting_reader.py`.
- `/student-assistant` and `/agent` were not in the API's router list (students would be blocked by the legacy middleware).
- Double sign ("−-0.5") when a teacher restored marks.
- SQLite busy-timeout added for the background worker.
- 6 previously failing tests were stale (hermetic env forcing local OCR; offline-confidence cap is by design; template path) — updated to assert the intended behaviour.

## Tested here (Linux, Python 3.13, real Tesseract, NO real Ollama)
- `pytest tests` → **254 passed, 2 skipped, 0 failed** (was 245 passed / 6 failed).
- New `tests/test_bulk_robot_checking.py`: full workflow through the API (6 students, PDFs/JPG/real handwriting/camera scan, identification + conflicts + manual verification, duplicates, background job, reports, ZIP, approval rules, student isolation, admin monitoring, restart recovery).
- New `tests/test_bulk_ai_path.py`: bulk grading through the Ollama wire protocol (mock server) — method "ai", unsupported marks withheld.
- `node --test tests/js/voice_out.test.js` → 16/16.
- Headless Chromium against running backend: 25/25 UI checks (robot states follow real job status, upload, verification, fake-camera capture/rotate/delete/retake/save, chat command starts job, live card, completion message, ZIP download, approval refusal, student + admin pages, phone width, no JS errors).
- Streamlit portal: robot page loads inside the portal for admin, faculty and student.

## NOT tested here (needs your Windows PC)
- Real `llama3.2:3b` answers/quality (`python scripts/ollama_smoke_test.py`), real microphone speech recognition and audible speech output, a physical camera/phone, `start.bat` on Windows, accuracy on your real handwritten sheets (local OCR on handwriting is weak — that is why such answers are flagged for teacher verification).
