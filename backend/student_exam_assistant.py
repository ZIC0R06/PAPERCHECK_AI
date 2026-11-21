"""
AI Examination Assistant — Student mode
=======================================
Students ask about their marks, why marks were deducted, mistakes, subject-wise performance, and ask for
summaries, revision notes and improvement suggestions. Same design as the staff assistant:

  ROUTE (keywords) → FETCH the student's OWN *published* results only → COMPOSE with local Ollama using only
  those facts (numbers checked against the facts; if the model invents a figure the verified text is shown).

Unpublished sheets are never shown with marks — only "being checked / awaiting teacher approval".
Session memory: the client sends the recent conversation; the subject / question in focus is carried over.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from backend.exam_agent import numbers_grounded
from backend.language_policy import language_instruction, offline_notice, reply_matches, resolve_language

HELP = ("I'm your AI Examination Assistant. I can show your published marks, explain question by question why marks "
        "were deducted, point out mistakes, compare your subjects, and write a summary, revision notes or improvement "
        "plan from your checked answer sheets. Try: “Why did I get fewer marks in CS301?”, “Explain Q2”, "
        "“Make revision notes for my weak topics”, or “How can I improve?”.")

SYSTEM = ("You are the AI Examination Assistant for a university student. Use ONLY the VERIFIED RESULT DATA given. "
          "Never invent marks, questions or teacher comments. Be encouraging, specific and concise. When explaining lost "
          "marks, name the missing points. Use short headings and bullet points.")


def _has(m: str, *ws) -> bool:
    return any(w in m for w in ws)


def route(message: str) -> str:
    m = message.lower()
    if _has(m, "revision note", "notes", "revise", "revision", "study material", "padhai"):
        return "revision_notes"
    if _has(m, "improve", "suggestion", "tips", "how can i", "better marks", "next time", "plan", "sudhar"):
        return "improvement"
    if _has(m, "why", "lost", "lose", "deduct", "fewer", "less marks", "low marks", "cut", "kam", "kyon", "kyun",
            "mistake", "wrong", "galat", "explain", "q1", "q2", "q3", "q4", "q5", "question"):
        return "why"
    if _has(m, "summary", "summar", "overview", "overall"):
        return "summary"
    if _has(m, "subject", "compare", "performance", "best", "worst", "weak", "strong", "cgpa", "percentage"):
        return "subjects"
    if _has(m, "status", "checked yet", "being checked", "pending", "when will", "published"):
        return "status"
    if _has(m, "mark", "result", "score", "grade", "number", "nambar"):
        return "results"
    return "help"


class StudentRepo:
    def __init__(self, db, student):
        self.db, self.student = db, student

    def sheets(self, published_only: bool = True) -> List[Dict]:
        from backend.booklet_ocr import question_evaluations
        from backend.database import StudentBooklet
        q = self.db.query(StudentBooklet).filter(StudentBooklet.student_id == self.student.id)
        out = []
        for b in q.order_by(StudentBooklet.id).all():
            if published_only and not b.is_published:
                continue
            qe = [x for x in question_evaluations(b) if x["result_id"]] if b.is_published else []
            got = round(sum(x["awarded"] or 0 for x in qe), 2)
            mx = round(sum(x["max_marks"] for x in qe), 2)
            out.append({"booklet_id": b.id, "course_code": b.course_code, "course_name": b.course_name, "exam": b.exam_name,
                        "published": bool(b.is_published), "status": b.approval_status, "marks": got, "max_marks": mx,
                        "percentage": round(100 * got / mx, 1) if mx else None, "questions": qe})
        return out


def _course_in(text: str, codes: List[str]) -> Optional[str]:
    for t in re.findall(r"[A-Za-z]{2,5}\s?\d{2,4}", text or ""):
        c = t.replace(" ", "").upper()
        if c in codes:
            return c
    return None


def _q_in(text: str) -> Optional[int]:
    m = re.search(r"\bq(?:uestion)?\s*\.?\s*(\d{1,2})\b", text or "", re.I)
    return int(m.group(1)) if m else None


def _f(x) -> str:
    return "—" if x is None else f"{x:g}"


def _sheet_lines(sh: Dict, qn: Optional[int] = None, detail: bool = True) -> List[str]:
    out = [f"**{sh['course_code']} {sh['course_name'] or ''} — {sh['exam']}: {_f(sh['marks'])}/{_f(sh['max_marks'])}"
           f" ({_f(sh['percentage'])}%)**"]
    for q in sh["questions"]:
        if qn and q["question_number"] != qn:
            continue
        lost = (q["max_marks"] or 0) - (q["awarded"] or 0)
        out.append(f"- Q{q['question_number']}: {_f(q['awarded'])}/{_f(q['max_marks'])}"
                   + ("" if lost <= 0 else f" (−{lost:g})") + (" · teacher verified" if q["teacher_verified"] else ""))
        if not detail:
            continue
        if not (q["student_answer"] or "").strip():
            out.append("    • Not attempted.")
        for d in q["deductions"][:5]:
            try:
                v = float(d.get("marks") or 0)
                mk = (f"+{-v:g} restored: " if v < 0 else f"−{v:g}: ")    # negative deduction = teacher restored marks
            except Exception:
                mk = ""
            out.append(f"    • {mk}{d.get('criterion', '')} — {d.get('reason', '')}".rstrip(" —"))
        if q["missing_concepts"]:
            out.append("    • Missing: " + ", ".join(map(str, q["missing_concepts"][:6])))
        if q["strengths"]:
            out.append("    • Done well: " + ", ".join(map(str, q["strengths"][:4])))
        if q["teacher_comment"]:
            out.append(f"    • Teacher: {q['teacher_comment']}")
        if q["feedback"] and qn:
            out.append(f"    • Feedback: {q['feedback'][:400]}")
    return out


class StudentExamAssistant:
    def __init__(self, repo: StudentRepo, llm=None):
        self.repo, self.llm = repo, llm

    def _llm_ok(self) -> bool:
        try:
            return bool(self.llm and self.llm.is_available())
        except Exception:
            return False

    def chat(self, message: str, history: Optional[List[Dict]] = None, language: str = "auto") -> Dict:
        message = (message or "").strip()
        lang = resolve_language(language, message)
        kind = route(message)
        sheets = self.repo.sheets(published_only=True)
        codes = sorted({s["course_code"] for s in sheets})
        course, qn = _course_in(message, codes), _q_in(message)
        # session memory: carry the subject / question discussed earlier ("and Q3?", "why?")
        prev_user = [h.get("content", "") for h in (history or [])[-8:] if h.get("role") == "user"]
        if not course:
            course = next((c for c in (_course_in(x, codes) for x in reversed(prev_user)) if c), None)
        if not qn and kind == "why" and len(message.split()) <= 6 and prev_user:
            qn = _q_in(prev_user[-1])
        if kind == "help" and prev_user and len(message.split()) <= 4 and route(prev_user[-1]) != "help":
            kind = route(prev_user[-1])              # "and CS302?" → same kind of question as before
        if kind == "help":
            return {"reply": HELP, "mode": "help", "tool": "help"}
        if kind == "status" or not sheets:
            pend = [s for s in self.repo.sheets(published_only=False) if not s["published"]]
            lines = [f"- {s['course_code']} {s['exam']}: " + {"pending": "received, waiting to be checked",
                                                             "in_review": "checked by AI, awaiting your teacher's verification",
                                                             "approved": "approved, will be published soon"}.get(s["status"], s["status"])
                     for s in pend]
            txt = ("No published result yet." if not sheets else "") + ("\n\nAnswer sheets in progress:\n" + "\n".join(lines) if lines else "")
            if kind == "status" or not sheets:
                return {"reply": (txt.strip() or "You have no answer sheets yet.") +
                        "\n\nMarks are shown only after your teacher approves and publishes them.", "mode": "data", "tool": "status"}
        focus = [s for s in sheets if not course or s["course_code"] == course]
        if not focus:
            focus = sheets
        if kind in ("why",) and not course and len(sheets) > 1:
            focus = [min(sheets, key=lambda s: s["percentage"] if s["percentage"] is not None else 101)]
        title, lines, task = "", [], ""
        if kind == "results":
            title = "Your published results"
            for s in sheets:
                lines += _sheet_lines(s, detail=False)
        elif kind == "subjects":
            title = "Subject-wise performance"
            for s in sorted(sheets, key=lambda s: -(s["percentage"] or 0)):
                lines.append(f"- {s['course_code']} {s['course_name'] or ''} ({s['exam']}): {_f(s['percentage'])}% "
                             f"({_f(s['marks'])}/{_f(s['max_marks'])})")
            pc = [s["percentage"] for s in sheets if s["percentage"] is not None]
            if pc:
                lines.append(f"- Average across subjects: {round(sum(pc) / len(pc), 1):g}%")
            task = "Compare the subjects, name the strongest and weakest, and suggest where to focus."
        elif kind == "why":
            title = "Why marks were deducted"
            for s in focus[:2]:
                lines += _sheet_lines(s, qn=qn)
            task = ("Explain clearly, question by question, why marks were lost and what a full-marks answer needed. "
                    "Use only the deductions and missing points given.")
        else:   # summary / revision_notes / improvement
            weak = Counter()
            for s in focus:
                for q in s["questions"]:
                    if (q["awarded"] or 0) < (q["max_marks"] or 0):
                        weak.update(str(c).strip() for c in q["missing_concepts"] if str(c).strip())
            for s in focus[:3]:
                lines += _sheet_lines(s)
            if weak:
                lines.append("- Topics missed most: " + ", ".join(f"{t} ({n})" for t, n in weak.most_common(8)))
            title = {"summary": "Summary of your checked answer sheets", "revision_notes": "Revision notes from your mistakes",
                     "improvement": "How to improve"}[kind]
            task = {"summary": "Write a short performance summary: total, strongest and weakest questions, key mistakes.",
                    "revision_notes": ("Write compact revision notes, one section per missed topic: what the examiner "
                                       "expected (from the deductions / missing points), key points to remember, and a "
                                       "one-line exam tip. Do not invent facts beyond general explanations of the topic names."),
                    "improvement": ("Give a prioritised improvement plan (3-6 steps) based on the deductions: what to study, "
                                    "how to structure answers, and how to avoid the specific mistakes.")}[kind]
        verified = f"### {title}\n" + "\n".join(lines)
        base = {"tool": kind, "data": [{k: v for k, v in s.items() if k != "questions"} for s in focus]}
        if not self._llm_ok():
            extra = ""
            if kind == "revision_notes":
                extra = ("\n\n**Revise these points** (from your deductions):\n" +
                         "\n".join(f"- {l.split('Missing:')[1].strip()}" for l in lines if "Missing:" in l) or "")
            elif kind == "improvement":
                extra = ("\n\n**Plan:** 1) Re-learn each *missing* point above. 2) Write answers that cover every point of "
                         "the marking scheme with headings. 3) Attempt every question — unattempted answers score 0. "
                         "4) Practise one past question per weak topic and ask me to check your understanding.")
            return {**base, "reply": verified + extra + "\n\n_" + offline_notice(lang) + "_", "mode": "data-only"}
        prompt = f"STUDENT QUESTION: {message}\n\nTASK: {task or 'Answer the question.'}\n\nVERIFIED RESULT DATA:\n{verified[:6000]}"
        system = SYSTEM + "\n\n" + language_instruction(lang)
        hist = [{"role": h["role"], "content": h["content"]} for h in (history or [])[-6:] if h.get("content")]
        try:
            reply = (self.llm.complete(system, prompt, max_tokens=1000, history=hist) or "").strip()
        except Exception:
            reply = ""
        if reply and not reply_matches(lang, reply):
            try:
                reply = (self.llm.complete(system + " Rewrite strictly in the required language.", prompt, max_tokens=1000) or "").strip()
            except Exception:
                reply = ""
        if not reply:
            return {**base, "reply": verified + "\n\n_" + offline_notice(lang) + "_", "mode": "data-only"}
        if not numbers_grounded(reply, verified + " " + message):
            return {**base, "mode": "data-verified",
                    "reply": verified + "\n\n_The AI wording contained figures that are not in your results, so your verified "
                                        "results are shown instead._"}
        return {**base, "reply": reply, "mode": "ai"}
