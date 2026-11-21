"""
AI Examination Assistant (Admin + Faculty agent)
================================================
How it works (designed for a small local model such as llama3.2:3b):

  1. ROUTE   – deterministic intent routing picks which data tools to run (a 3B model is unreliable at choosing
               tools, and a wrong tool means wrong marks, so this is not left to the model).
  2. FETCH   – every tool reads the database through a role-scoped repository. Faculty can only ever see the
               subjects and batches assigned to them (same rules as the rest of the app: backend/access.py).
  3. COMPOSE – Ollama phrases the answer using ONLY the fetched facts. The reply is checked: any number that is
               not in the facts is treated as invented and the verified tool output is returned instead.
               If Ollama is unavailable the verified tool output is returned directly, and says so.

The agent is READ-ONLY. It never changes marks, approves, rejects or publishes; those stay in the audited
review / re-evaluation endpoints so a language model can never alter an examination result.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Protocol

from backend.language_policy import Lang, language_instruction, offline_notice, reply_matches, resolve_language

MAX_ROWS = 25
SUPPORT_DEFAULT_THRESHOLD = 40.0


# ── Repository interface (implemented by SqlRepo for the app, FakeRepo in tests) ─────────────────

class Repo(Protocol):
    is_admin: bool
    def scope_description(self) -> str: ...
    def known_course_codes(self) -> List[str]: ...
    def find_students(self, query: str) -> List[Dict]: ...
    def results(self, reg: Optional[str] = None, course: Optional[str] = None, exam: Optional[str] = None) -> List[Dict]: ...
    def question_wise(self, reg: str, course: Optional[str] = None) -> List[Dict]: ...
    def missing_concepts(self, course: Optional[str] = None) -> List[str]: ...
    def pending(self) -> Dict: ...
    def revaluations(self, status: Optional[str] = None) -> List[Dict]: ...
    def audit(self, limit: int = 15) -> List[Dict]: ...
    def faculty_activity(self) -> List[Dict]: ...


# ── Intent routing ─────────────────────────────────────────────────────────────────────────

def extract_entities(message: str, course_codes: List[str]) -> Dict[str, Any]:
    codes = {c.upper() for c in course_codes}
    tokens = re.findall(r"[A-Za-z0-9]+", message)
    course = next((t.upper() for t in tokens if t.upper() in codes), None)
    # registration number: a token with at least one digit and >= 5 characters that is not a known course code
    cands = [t for t in tokens if re.search(r"\d", t) and len(t) >= 5 and t.upper() not in codes]
    reg = cands[0].upper() if cands else None
    pct = re.search(r"(?:below|under|less than|<)\s*(\d{1,3})", message, re.I)
    qn = re.search(r"\bq(?:uestion)?\s*\.?\s*(\d{1,2})\b", message, re.I)
    return {"course": course, "reg": reg, "threshold": float(pct.group(1)) if pct else None,
            "question": int(qn.group(1)) if qn else None}


def route(message: str, ent: Dict[str, Any], is_admin: bool) -> List[str]:
    m = message.lower()
    has = lambda *ws: any(w in m for w in ws)          # noqa: E731
    tools: List[str] = []
    if has("revaluation", "re-evaluation", "reevaluation", "recheck", "re-check", "regrade", "dispute"):
        tools.append("revaluations")
    if has("pending", "unchecked", "not checked", "to be checked", "queue", "checking job", "in review", "backlog"):
        tools.append("pending")
    if has("audit", "checking history", "activity log", "who changed", "history"):
        tools.append("audit")
    if has("faculty") and has("activity", "performance", "workload", "checked by", "monitor"):
        tools.append("faculty_activity")
    if has("weak topic", "missing concept", "weak area", "common mistake", "revise", "topics"):
        tools.append("weak_topics")
    if has("support", "struggling", "at risk", "at-risk", "needs help", "need help", "failing", "failed", "fail ",
           "low marks", "weak student", "poor"):
        tools.append("support_needed")
    if ent["threshold"] is not None and "support_needed" not in tools:
        tools.append("support_needed")                      # "students below 40 in CS301"
    if ent["reg"] and has("deduct", "lost", "lose", "why", "question", "marks cut", "marks kat", "explain", "q1", "q2"):
        tools.append("question_wise")
    if has("report", "summary report", "full report"):
        tools.append("report")
    if has("average", "avg", "statistic", "pass rate", "pass percentage", "class", "subject", "highest", "lowest",
           "topper", "distribution", "performance", "summar", "how did"):
        if "student_results" not in tools and not ent["reg"]:
            tools.append("exam_stats")
    if ent["reg"] and "question_wise" not in tools:
        tools.append("student_results")
    if has("find", "search", "who is", "look up", "lookup") and not ent["reg"] and not tools:
        tools.append("find_student")
    if is_admin and has("how many student", "how many faculty", "how many teacher", "how many subject", "overview",
                        "directory", "manage students", "manage faculty", "subjects list", "list subjects", "list faculty"):
        tools.append("directory")
    if not tools and ent["course"]:
        tools.append("exam_stats")
    seen, out = set(), []
    for t in tools:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


# ── Tools: each returns {"title", "facts", "text"}; text is the verified, human-readable output ──────

def _pct(x) -> str:
    return f"{float(x):.1f}%"


def tool_find_student(repo: Repo, ent, message) -> Dict:
    q = ent["reg"] or re.sub(r"(?i)\b(find|search|who is|look up|lookup|student|named|name|for|me|the|show)\b", " ", message).strip(" ?.")
    rows = repo.find_students(q) if q else []
    if not rows:
        return {"title": "Student search", "facts": [], "text": f"No student matching “{q}” was found in your authorised scope."}
    lines = [f"- {r['reg']} — {r['name']} ({r.get('batch', '')}, {r.get('semester', '')})" for r in rows[:MAX_ROWS]]
    return {"title": "Student search", "facts": rows[:MAX_ROWS], "text": "Students found:\n" + "\n".join(lines)}


def tool_student_results(repo: Repo, ent, message) -> Dict:
    rows = repo.results(reg=ent["reg"], course=ent["course"])
    if not rows:
        return {"title": "Student results", "facts": [],
                "text": f"No result records for {ent['reg']} in your authorised scope (the student may not exist, may not be "
                        "in your assigned classes, or has no checked paper yet)."}
    name = rows[0]["name"]
    lines = [f"- {r['course_code']} {r['exam']}: {r['total']:g}/{r['max_total']:g} ({_pct(r['percentage'])}), grade "
             f"{r.get('grade') or '–'}, {'PASS' if r['passed'] else 'FAIL'}, status: {r['status']}" for r in rows[:MAX_ROWS]]
    return {"title": f"Results — {name} ({ent['reg']})", "facts": rows[:MAX_ROWS],
            "text": f"Results for {name} ({ent['reg']}):\n" + "\n".join(lines)}


def tool_question_wise(repo: Repo, ent, message) -> Dict:
    sheets = repo.question_wise(ent["reg"], ent["course"])
    if not sheets:
        return {"title": "Question-wise marks", "facts": [],
                "text": f"No checked answer sheet for {ent['reg']} in your authorised scope."}
    out, facts = [], []
    for sh in sheets[:3]:
        out.append(f"{sh['course_code']} {sh['exam']} — sheet status: {sh['status']}")
        for q in sh["questions"]:
            if ent["question"] and q["question_number"] != ent["question"]:
                continue
            aw = "not evaluated" if q["awarded"] is None else f"{q['awarded']:g}/{q['max_marks']:g}"
            line = f"- Q{q['question_number']}: {aw}"
            if q.get("flagged"):
                line += " (flagged for teacher verification)"
            out.append(line)
            for d in q.get("deductions", [])[:4]:
                try:
                    v = float(d.get("marks") or 0)
                except Exception:
                    v = 0.0
                sign = f"+{-v:g} restored" if v < 0 else f"−{v:g}"
                out.append(f"    • {sign}: {d.get('criterion', '')} — {d.get('reason', '')}".rstrip(" —"))
            if q.get("missing_concepts"):
                out.append("    • Missing concepts: " + ", ".join(map(str, q["missing_concepts"][:6])))
        facts.append(sh)
    return {"title": f"Question-wise marks — {ent['reg']}", "facts": facts, "text": "\n".join(out)}


def tool_exam_stats(repo: Repo, ent, message) -> Dict:
    rows = repo.results(course=ent["course"])
    if not rows:
        return {"title": "Examination statistics", "facts": [], "text": "No result records in your authorised scope" +
                (f" for {ent['course']}." if ent["course"] else ".")}
    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["course_code"], r["exam"]), []).append(r)
    lines, facts = [], []
    for (c, e), g in sorted(groups.items()):
        pcts = [x["percentage"] for x in g]
        passed = sum(1 for x in g if x["passed"])
        top = max(g, key=lambda x: x["percentage"]); low = min(g, key=lambda x: x["percentage"])
        f = {"course": c, "exam": e, "students": len(g), "average_pct": round(sum(pcts) / len(pcts), 1),
             "highest_pct": round(top["percentage"], 1), "highest_student": top["reg"], "lowest_pct": round(low["percentage"], 1),
             "lowest_student": low["reg"], "passed": passed, "failed": len(g) - passed,
             "pass_rate_pct": round(100 * passed / len(g), 1)}
        facts.append(f)
        lines.append(f"- {c} · {e}: {f['students']} students, average {f['average_pct']}%, highest {f['highest_pct']}% "
                     f"({top['reg']}), lowest {f['lowest_pct']}% ({low['reg']}), pass rate {f['pass_rate_pct']}% "
                     f"({passed} passed / {len(g) - passed} failed)")
    return {"title": "Examination statistics", "facts": facts, "text": "\n".join(lines[:MAX_ROWS])}


def tool_support_needed(repo: Repo, ent, message) -> Dict:
    thr = ent["threshold"] or SUPPORT_DEFAULT_THRESHOLD
    rows = [r for r in repo.results(course=ent["course"]) if r["percentage"] < thr or not r["passed"]]
    rows.sort(key=lambda r: r["percentage"])
    if not rows:
        return {"title": "Students needing support", "facts": [],
                "text": f"No student is below {thr:g}% or failing in your authorised scope."}
    lines = [f"- {r['reg']} {r['name']}: {r['course_code']} {r['exam']} — {_pct(r['percentage'])} "
             f"({'FAIL' if not r['passed'] else 'low'})" for r in rows[:MAX_ROWS]]
    return {"title": "Students needing academic support", "facts": rows[:MAX_ROWS],
            "text": f"{len(rows)} result(s) below {thr:g}% or failed:\n" + "\n".join(lines)}


def tool_weak_topics(repo: Repo, ent, message) -> Dict:
    topics = Counter(t.strip().lower() for t in repo.missing_concepts(ent["course"]) if str(t).strip())
    if not topics:
        return {"title": "Weak topics", "facts": [], "text": "No recorded missing concepts yet for your authorised scope."}
    top = topics.most_common(10)
    return {"title": "Most frequently missed concepts", "facts": [{"concept": t, "answers": n} for t, n in top],
            "text": "Most frequently missed concepts (number of answers where it was missing):\n" +
                    "\n".join(f"- {t}: {n}" for t, n in top)}


def tool_pending(repo: Repo, ent, message) -> Dict:
    p = repo.pending()
    by = p.get("by_status", {})
    lines = [f"- {k}: {v}" for k, v in sorted(by.items())] or ["- no answer sheets yet"]
    lines.append(f"- answers flagged for teacher verification: {p.get('flagged_answers', 0)}")
    for o in p.get("oldest", [])[:5]:
        lines.append(f"  • oldest waiting: sheet #{o['booklet_id']} {o.get('reg', '')} {o.get('course', '')} ({o['status']})")
    return {"title": "Checking queue", "facts": [p], "text": "Checking status of answer sheets in your scope:\n" + "\n".join(lines)}


def tool_revaluations(repo: Repo, ent, message) -> Dict:
    rows = repo.revaluations(None)
    if not rows:
        return {"title": "Re-evaluation requests", "facts": [], "text": "No re-evaluation requests in your authorised scope."}
    lines = [f"- #{r['id']} {r.get('reg', '')} sheet #{r['booklet_id']}" +
             (f" Q{r['question_number']}" if r.get("question_number") else " (whole sheet)") +
             f": {r['status']} — “{(r.get('reason') or '')[:80]}”" +
             (f" (marks {r['score_before']:g} → {r['score_after']:g})" if r.get("score_after") is not None else "")
             for r in rows[:MAX_ROWS]]
    return {"title": "Re-evaluation requests", "facts": rows[:MAX_ROWS],
            "text": f"{len(rows)} request(s):\n" + "\n".join(lines) +
                    "\n\nTo approve or reject, use the Re-evaluation page — the assistant is read-only so every mark "
                    "change stays audited."}


def tool_audit(repo: Repo, ent, message) -> Dict:
    rows = repo.audit(15)
    if not rows:
        return {"title": "Checking history", "facts": [], "text": "No audit entries yet."}
    lines = [f"- {r['when']}: {r['who']} ({r['role']}) {r['action']} {r['target']}" for r in rows]
    return {"title": "Recent checking history", "facts": rows, "text": "Recent audited actions:\n" + "\n".join(lines)}


def tool_faculty_activity(repo: Repo, ent, message) -> Dict:
    if not repo.is_admin:
        return {"title": "Faculty activity", "facts": [], "text": "Faculty-wide activity is available to administrators only."}
    rows = repo.faculty_activity()
    if not rows:
        return {"title": "Faculty activity", "facts": [], "text": "No faculty checking activity recorded yet."}
    return {"title": "Faculty checking activity", "facts": rows,
            "text": "\n".join(f"- {r['who']}: {r['actions']} audited actions ({', '.join(r['top'])})" for r in rows[:MAX_ROWS])}


def tool_directory(repo: Repo, ent, message) -> Dict:
    if not repo.is_admin or not hasattr(repo, "directory"):
        return {"title": "University directory", "facts": [], "text": "The university directory is available to administrators only."}
    d = repo.directory()
    subj = "\n".join(f"- {s['code']} {s['name']}: {s['students']} registered, faculty: {', '.join(s['faculty']) or '—'}"
                     for s in d["subjects"][:MAX_ROWS])
    return {"title": "University directory", "facts": [d],
            "text": (f"Students: {d['students']} active ({d['inactive_students']} inactive) · Faculty: {d['faculty']} · "
                     f"Subjects: {len(d['subjects'])} · Batches: {d['batches']}\n{subj}\n\nTo add, edit or deactivate "
                     "students, faculty and subjects use Student / Teacher Management (changes there are audited).")}


def tool_report(repo: Repo, ent, message) -> Dict:
    parts = [tool_exam_stats(repo, ent, message), tool_support_needed(repo, ent, message), tool_weak_topics(repo, ent, message),
             tool_pending(repo, ent, message)]
    text = "# Examination report\n\nScope: " + repo.scope_description() + "\n\n" + "\n\n".join(
        f"## {p['title']}\n{p['text']}" for p in parts)
    return {"title": "Examination report", "facts": [f for p in parts for f in p["facts"]], "text": text, "report": True}


TOOLS = {"find_student": tool_find_student, "student_results": tool_student_results, "question_wise": tool_question_wise,
         "exam_stats": tool_exam_stats, "support_needed": tool_support_needed, "weak_topics": tool_weak_topics,
         "pending": tool_pending, "revaluations": tool_revaluations, "audit": tool_audit,
         "faculty_activity": tool_faculty_activity, "report": tool_report, "directory": tool_directory}

HELP = ("I can look up students and their results, explain deductions question by question, summarise class and subject "
        "performance, list students who need academic support, show the most missed concepts, show the checking queue and "
        "re-evaluation requests, and generate an examination report. Try: “results of 2022CS001”, “why were marks cut for "
        "2022CS001 in CS301”, “students below 40% in CS301”, “pending checking”, or “generate a report”.")


# ── Grounding guard ──────────────────────────────────────────────────────────────────────────

def _numbers(s: str) -> set:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", s)}


def numbers_grounded(reply: str, facts_text: str) -> bool:
    """True when every number >9 in the reply appears in the verified facts (blocks invented marks/percentages)."""
    allowed = _numbers(facts_text)
    for n in _numbers(reply):
        if "." not in n and int(n) <= 9:
            continue
        if n not in allowed:
            return False
    return True


COMPOSE_SYSTEM = ("You are the Examination Assistant for a university. Answer the staff member's question using ONLY the "
                  "VERIFIED DATA provided. Never invent or estimate marks, names, percentages or dates. If the data does "
                  "not answer the question, say what is missing. Be concise and use short bullet points where helpful. "
                  "Refer to students by registration number. You are read-only: you cannot change marks or approve results.")


class ExamAgent:
    def __init__(self, repo: Repo, llm=None):
        self.repo, self.llm = repo, llm

    def _llm_ok(self) -> bool:
        try:
            return bool(self.llm and self.llm.is_available())
        except Exception:
            return False

    def chat(self, message: str, history: Optional[List[Dict]] = None, language: str = "auto") -> Dict[str, Any]:
        message = (message or "").strip()
        if not message:
            return {"reply": HELP, "tools": [], "mode": "help", "report": None}
        lang = resolve_language(language, message)
        ent = extract_entities(message, self.repo.known_course_codes())
        # a follow-up like "and for CS301?" inherits the student from the previous staff question
        if not ent["reg"] and history:
            for h in reversed(history[-6:]):
                if h.get("role") == "user":
                    prev = extract_entities(h.get("content", ""), self.repo.known_course_codes())
                    if prev["reg"] and len(message.split()) <= 8:
                        ent["reg"] = prev["reg"]
                    break
        names = route(message, ent, self.repo.is_admin)
        if not names:
            return {"reply": HELP, "tools": [], "mode": "help", "report": None}
        results = [TOOLS[n](self.repo, ent, message) for n in names]
        verified = "\n\n".join(f"### {r['title']}\n{r['text']}" for r in results)
        report = next((r["text"] for r in results if r.get("report")), None)
        has_data = any(r["facts"] for r in results)
        base = {"tools": names, "report": report, "data": [r["facts"] for r in results]}
        if not has_data or not self._llm_ok():
            note = "" if self._llm_ok() else "\n\n_" + offline_notice(lang) + "_"
            return {**base, "reply": verified + note, "mode": "data-only" if not self._llm_ok() else "data"}
        prompt = (f"STAFF QUESTION: {message}\n\nSCOPE: {self.repo.scope_description()}\n\nVERIFIED DATA:\n{verified[:6000]}\n\n"
                  "Write the answer now.")
        system = COMPOSE_SYSTEM + "\n\n" + language_instruction(lang)
        try:
            reply = (self.llm.complete(system, prompt, max_tokens=900, history=[
                {"role": h["role"], "content": h["content"]} for h in (history or [])[-4:] if h.get("content")]) or "").strip()
        except Exception:
            reply = ""
        if reply and not reply_matches(lang, reply):
            try:
                reply = (self.llm.complete(system + " YOUR PREVIOUS ANSWER USED THE WRONG LANGUAGE; rewrite strictly in "
                                           "the required language.", prompt, max_tokens=900) or "").strip()
            except Exception:
                reply = ""
        if not reply:
            return {**base, "reply": verified + "\n\n_" + offline_notice(lang) + "_", "mode": "data-only"}
        if not numbers_grounded(reply, verified + " " + message):
            return {**base, "mode": "data-verified",
                    "reply": verified + "\n\n_The AI wording contained figures that are not in the database, so the verified "
                                        "data is shown instead._"}
        return {**base, "reply": reply, "mode": "ai"}


# ── Real repository (SQLAlchemy, role-scoped via backend/access.py) ──────────────────────────────

class SqlRepo:
    def __init__(self, db, user):
        from backend import access as AC
        self.db, self.user, self.AC = db, user, AC
        self.is_admin = AC.is_admin(user)

    def scope_description(self) -> str:
        if self.is_admin:
            return "all students, subjects and examinations"
        pairs = self.AC.teacher_pairs(self.db, self.user)
        if not pairs:
            return "no assigned classes"
        from backend.database import Batch
        codes = {b.id: b.batch_code for b in self.db.query(Batch).filter(Batch.id.in_({p[1] for p in pairs})).all()}
        return "; ".join(f"{c} for {codes.get(b, b)}" for c, b in sorted(pairs))

    def known_course_codes(self) -> List[str]:
        from backend.database import Course
        return [c.code for c in self.db.query(Course.code).all()]

    def _final_q(self):
        from sqlalchemy import and_, false, func, or_
        from backend.database import FinalResult, Student
        q = self.db.query(FinalResult, Student).join(Student, FinalResult.student_id == Student.id)
        if self.is_admin:
            return q
        pairs = self.AC.teacher_pairs(self.db, self.user)
        if not pairs:
            return q.filter(false())
        return q.filter(or_(*[and_(func.upper(FinalResult.course_code) == c, Student.batch_id == b) for c, b in pairs]))

    def find_students(self, query: str) -> List[Dict]:
        from sqlalchemy import or_
        from backend.database import Student
        like = f"%{query}%"
        rows = (self.AC.scoped_students(self.db, self.user)
                .filter(or_(Student.registration_number.ilike(like), Student.name.ilike(like), Student.student_code.ilike(like)))
                .limit(10).all())
        return [{"reg": s.registration_number or s.student_code, "name": s.name, "batch": s.batch, "semester": s.semester}
                for s in rows]

    def results(self, reg=None, course=None, exam=None) -> List[Dict]:
        from sqlalchemy import func
        from backend.database import FinalResult, Student
        q = self._final_q()
        if reg:
            q = q.filter(func.upper(Student.registration_number) == reg.upper())
        if course:
            q = q.filter(func.upper(FinalResult.course_code) == course.upper())
        if exam:
            q = q.filter(FinalResult.exam_name == exam)
        out = []
        for fr, st in q.order_by(Student.registration_number, FinalResult.course_code).limit(500).all():
            out.append({"reg": st.registration_number or st.student_code, "name": st.name, "course_code": fr.course_code,
                        "course_name": fr.course_name, "exam": fr.exam_name, "total": fr.total or 0.0,
                        "max_total": fr.max_total or 0.0, "percentage": fr.percentage or 0.0, "grade": fr.grade,
                        "passed": bool(fr.passed), "status": fr.status})
        return out

    def _scoped_booklets_for(self, reg: Optional[str], course: Optional[str]):
        from sqlalchemy import func
        from backend.database import Student, StudentBooklet
        q = self.AC.scoped_booklets(self.db, self.user)
        if reg:
            ids = [s.id for s in self.AC.scoped_students(self.db, self.user).filter(
                func.upper(Student.registration_number) == reg.upper()).all()]
            q = q.filter(StudentBooklet.student_id.in_(ids or [-1]))
        if course:
            q = q.filter(func.upper(StudentBooklet.course_code) == course.upper())
        return q

    def question_wise(self, reg: str, course=None) -> List[Dict]:
        from backend.booklet_ocr import question_evaluations
        out = []
        for b in self._scoped_booklets_for(reg, course).all():
            out.append({"booklet_id": b.id, "course_code": b.course_code, "exam": b.exam_name,
                        "status": b.approval_status, "questions": question_evaluations(b)})
        return out

    def missing_concepts(self, course=None) -> List[str]:
        from backend.booklet_ocr import question_evaluations
        topics: List[str] = []
        for b in self._scoped_booklets_for(None, course).limit(300).all():
            for q in question_evaluations(b, include_unevaluated=False):
                if q["awarded"] is not None and q["max_marks"] and q["awarded"] < q["max_marks"]:
                    topics.extend(str(t) for t in q["missing_concepts"])
        return topics

    def pending(self) -> Dict:
        from backend.booklet_ocr import question_evaluations
        by: Counter = Counter()
        oldest, flagged = [], 0
        for b in self._scoped_booklets_for(None, None).limit(2000).all():
            by[b.approval_status or "pending"] += 1
            if (b.approval_status or "pending") != "published":
                flagged += sum(1 for q in question_evaluations(b, include_unevaluated=False) if q["flagged"])
                if (b.approval_status or "pending") in ("pending", "in_review"):
                    oldest.append({"booklet_id": b.id, "reg": (b.student.registration_number if b.student else ""),
                                   "course": b.course_code, "status": b.approval_status or "pending", "_t": b.parsed_at})
        oldest.sort(key=lambda o: o["_t"] or 0)
        for o in oldest:
            o.pop("_t", None)
        return {"by_status": dict(by), "flagged_answers": flagged, "oldest": oldest[:5]}

    def revaluations(self, status=None) -> List[Dict]:
        from backend.database import ReEvaluationRequest, StudentBooklet
        ids = [r[0] for r in self._scoped_booklets_for(None, None).with_entities(StudentBooklet.id).all()]
        if not ids:
            return []
        q = self.db.query(ReEvaluationRequest).filter(ReEvaluationRequest.booklet_id.in_(ids))
        if status:
            q = q.filter(ReEvaluationRequest.status == status)
        return [{"id": r.id, "booklet_id": r.booklet_id, "reg": (r.student.registration_number if r.student else ""),
                 "question_number": r.question_number, "reason": r.reason, "status": r.status,
                 "score_before": r.score_before, "score_after": r.score_after}
                for r in q.order_by(ReEvaluationRequest.id.desc()).limit(100).all()]

    def audit(self, limit=15) -> List[Dict]:
        from backend.database import AuditLog
        q = self.db.query(AuditLog)
        if not self.is_admin:
            q = q.filter(AuditLog.user_id == self.user.id)      # faculty see only their own actions
        return [{"when": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "", "who": a.username, "role": a.role,
                 "action": a.action, "target": f"{a.target_type} {a.target_id}".strip()}
                for a in q.order_by(AuditLog.id.desc()).limit(limit).all()]

    def directory(self) -> Dict:
        from backend.database import Batch, Course, CourseRegistration, Student, TeacherAssignment, TeacherProfile, User
        subjects = []
        for c in self.db.query(Course).order_by(Course.code).all():
            fac = {u.full_name or u.username for u in self.db.query(User).join(TeacherProfile, TeacherProfile.user_id == User.id)
                   .join(TeacherAssignment, TeacherAssignment.teacher_id == TeacherProfile.id)
                   .filter(TeacherAssignment.course_id == c.id).all()}
            subjects.append({"code": c.code, "name": c.name, "faculty": sorted(fac),
                             "students": self.db.query(CourseRegistration).filter(CourseRegistration.course_id == c.id).count()})
        return {"students": self.db.query(Student).filter(Student.is_active == True).count(),  # noqa: E712
                "inactive_students": self.db.query(Student).filter(Student.is_active == False).count(),  # noqa: E712
                "faculty": self.db.query(User).filter(User.role == "faculty").count(),
                "batches": self.db.query(Batch).count(), "subjects": subjects}

    def faculty_activity(self) -> List[Dict]:
        from backend.database import AuditLog
        acts: Dict[str, Counter] = {}
        for a in self.db.query(AuditLog).filter(AuditLog.role == "faculty").limit(5000).all():
            acts.setdefault(a.username, Counter())[a.action] += 1
        return sorted([{"who": w, "actions": sum(c.values()), "top": [k for k, _ in c.most_common(3)]}
                       for w, c in acts.items()], key=lambda r: -r["actions"])
