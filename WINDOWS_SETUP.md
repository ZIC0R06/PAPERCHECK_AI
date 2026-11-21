# PaperCheckAI — Windows installation & startup (simple)

## One-time install
1. **Python 3.10+** — https://www.python.org/downloads/ → tick **"Add python.exe to PATH"**.
2. **Ollama** — https://ollama.com/download → install, then open *Command Prompt* and run:
   `ollama pull llama3.2:3b`
3. **Tesseract OCR** (reads answer sheets) — https://github.com/UB-Mannheim/tesseract/wiki → install with defaults.
   (Found automatically in `C:\Program Files\Tesseract-OCR`.)
4. Unzip this project, e.g. to `C:\PaperCheckAI`.

## Every day
Double-click **`start.bat`**. It checks Python, installs packages on the first run (takes a few minutes),
checks Ollama + the model + Tesseract, backs up the database, then opens:
- Portal: http://localhost:8501  (use **Chrome or Edge** for voice and camera)
- API docs: http://localhost:8000/docs · AI status: http://localhost:8000/system/llm-status

Demo logins: `admin/admin123`, `faculty/faculty123`, `student/student123`.

## Robot bulk checking in 5 steps (Faculty / Admin)
1. **Question Papers & Model Answers** → create the paper for subject + examination ONCE (questions, marks,
   model answer and/or marking scheme `criterion | marks`), or import it from PDF + answer-key PDF.
2. Sidebar → **🤖 AI Examination Assistant** → *Checking* tab → choose the paper → **New checking batch**.
3. Upload all answer sheets (PDF/JPG/PNG, many at once) or **📷 Scan with camera** (page by page, rotate / retake /
   delete, saved as one PDF). Name files by registration number (e.g. `21BCS101.pdf`) or let the robot read the
   registration number on page 1. Unclear sheets show **Verify student** — choose the student manually.
4. Say or type **"Check all uploaded answer sheets"** (or press *Check all sheets*). Watch live progress; ask
   "why were marks cut for 21BCS101 in Q2?" any time.
5. Verify flagged answers in **Examiner Review & Moderation**, then **Approve verified sheets** → **Publish approved**.
   Download the class report PDF / ZIP of checked sheets.

Students: sidebar → **🤖 AI Examination Assistant** (results, downloads, "why fewer marks", revision notes).

## Troubleshooting
- "Ollama not ready" → open the Ollama app (or `ollama serve`) and `ollama pull llama3.2:3b`. Without it, answers
  still work from verified data and marking uses offline matching (every answer flagged for teacher check).
- Camera blocked → allow camera for `localhost` in the browser, or use *Add photos*.
- Microphone → allow microphone; voice input needs Chrome/Edge and internet (browser speech service).
