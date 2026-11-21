"""
Robot actions for the AI Examination Assistant (Faculty / Admin)
================================================================
The examination assistant (backend/exam_agent.py) is read-only. This adds the ONE action staff can trigger by
chat or voice: starting the real bulk checking of uploaded answer sheets ("Check all uploaded answer sheets"),
plus live status / stop / report questions about that checking. Starting checking only creates AI marks in
"in_review" — approval and publishing remain manual, audited teacher actions.

Intent detection is deterministic (keywords), never left to a 3B model.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from sqlalchemy.orm import Session

from backend import bulk_evaluation as BE
from backend.database import EvaluationJob, User

_START = re.compile(r"\b(check|evaluate|grade|mark|correct|start|begin|run)\b.*\b(all|uploaded|answer ?sheets?|sheets|papers|copies|batch|checking|evaluation)\b"
                    r"|\bstart (the )?(checking|evaluation|robot)\b|\bjaanch\b|\bcheck karo\b|\bchalu karo\b", re.I)
_STATUS = re.compile(r"\b(progress|status|how far|how many (are )?(left|pending|done|checked)|which (student|paper|sheet)"
                     r"|currently|being checked|pending papers|completed papers|kitne)\b", re.I)
_STOP = re.compile(r"\b(stop|cancel|pause|halt)\b.*\b(check|checking|evaluation|robot|batch)\b|\bstop checking\b", re.I)
_REPORT = re.compile(r"\b(class|batch)\b.*\b(report|performance|summary|result)s?\b|\bclass performance\b", re.I)
_UNCLEAR = re.compile(r"\b(unidentified|not identified|manual verification|needs? (verification|identification)|unclear)\b", re.I)


def intent(message: str) -> Optional[str]:
    m = message or ""
    if _STOP.search(m):
        return "stop"
    if _UNCLEAR.search(m):
        return "unclear"
    # "why did 21BCS101 lose marks in Q2" is a marking question, not a start command
    if _START.search(m) and not re.search(r"\b(why|explain|kyon|how many marks)\b", m, re.I):
        return "start"
    if _REPORT.search(m) and re.search(r"\b(check|batch|robot|class)\b", m, re.I) and not re.search(r"\d{5,}", m):
        return "report"
    if _STATUS.search(m) and re.search(r"\b(check|checking|sheet|paper|evaluat|robot|batch|copies|student)\w*", m, re.I):
        return "status_all" if re.search(r"\b(all|every|faculty|overall)\b", m, re.I) else "status"
    return None


def _latest_job(db: Session, user: User, job_id: Optional[int]) -> Optional[EvaluationJob]:
    if job_id:
        try:
            return BE.job_or_404(db, user, job_id)
        except Exception:
            return None
    q = db.query(EvaluationJob)
    if user.role != "admin":
        q = q.filter(EvaluationJob.created_by_id == user.id)
    for j in q.order_by(EvaluationJob.id.desc()).limit(20).all():
        if j.status in BE.ACTIVE or any(i.status == "ready" for i in j.items):
            return j
    return q.order_by(EvaluationJob.id.desc()).first()


def _status_text(j: EvaluationJob) -> str:
    d = BE.job_dict(j)
    lines = [f"**Batch #{j.id} — {j.course_code} · {j.exam_name}** · status: **{j.status}**",
             f"- Sheets: {d['sheets']} · checked: {d['completed']} · pending: {d['pending']} · "
             f"need student verification: {d['needs_identification']}"]
    if d["current"]:
        c = d["current"]
        who = c["student"]["registration_number"] + " " + (c["student"]["name"] or "") if c["student"] else c["filename"]
        q = f"question {c['current_question']} ({c['questions_done']}/{c['questions_total']})" if c["current_question"] else c["status"]
        lines.append(f"- Now checking: **{who}** — {q}")
    for it in d["items"]:
        if it["status"] == "evaluated":
            lines.append(f"  - ✅ {it['student']['registration_number']}: {it['total_ai']:g}/{it['max_marks']:g}"
                         + (f" · {it['flagged']} flagged" if it["flagged"] else ""))
        elif it["status"] in ("needs_review", "failed"):
            lines.append(f"  - ⚠️ {it['filename']}: {it['status']} — {it['error'][:140]}")
    if j.message:
        lines.append(f"\n_{j.message}_")
    return "\n".join(lines)


def handle(db: Session, user: User, kind: str, job_id: Optional[int] = None) -> Dict:
    base = {"tools": [f"robot_{kind.replace('_all', '')}"], "report": None, "data": [], "mode": "action"}
    if kind == "status_all" and user.role != "admin":
        kind = "status"
    if kind == "status_all" or (kind == "status" and user.role == "admin" and not job_id):   # admin: every faculty
        jobs = db.query(EvaluationJob).order_by(EvaluationJob.id.desc()).limit(15).all()
        if jobs:
            rows = []
            for x in jobs:
                d = BE.job_dict(x, items=False)
                rows.append(f"- #{x.id} {x.course_code} · {x.exam_name} by {d['created_by']}: **{x.status}** — "
                            f"{d['completed']}/{d['workable']} checked, {d['needs_identification']} need verification"
                            + (f", now on {d['current']['student']['registration_number']}" if d["current"] and d["current"]["student"] else ""))
            return {**base, "action": "status", "job_id": None,
                    "reply": "**Evaluation progress across faculty** (latest batches):\n" + "\n".join(rows)}
    j = _latest_job(db, user, job_id)
    if not j:
        return {**base, "reply": "There is no checking batch yet. Open **Checking** → choose the question paper → upload or "
                                 "scan the answer sheets, then say “Check all uploaded answer sheets”.", "job_id": None}
    if kind == "start":
        from fastapi import HTTPException
        try:
            r = BE.start_job(db, user, j)
        except HTTPException as e:
            return {**base, "reply": f"I can't start checking batch #{j.id} yet: {e.detail}", "job_id": j.id,
                    "action": "start_refused"}
        if r.get("already_running"):
            return {**base, "reply": "Checking is already running.\n\n" + _status_text(j), "job_id": j.id, "action": "checking_running"}
        rd = r["readiness"]
        note = "" if rd["ready"] else (f"\n\n⚠️ Only {rd['ready_questions']} of {rd['questions']} questions have a model answer / "
                                        "marking scheme; the others will be skipped and listed for you.")
        waiting = sum(1 for i in j.items if i.status == "needs_identification")
        if waiting:
            note += f"\n\n⚠️ {waiting} sheet(s) are not checked because the student must be verified manually first."
        return {**base, "action": "checking_started", "job_id": j.id,
                "reply": (f"Starting the checking of **{r['sheets']} answer sheet(s)** for **{j.course_code} · {j.exam_name}** "
                          f"using the same question paper and marking scheme for every student. I'll read each sheet, mark "
                          f"every question, explain deductions and total the marks. You'll see the live progress here. "
                          f"All marks stay **awaiting your approval** before anything is published." + note)}
    if kind == "stop":
        if j.status in BE.ACTIVE:
            j.status, j.message = "cancelling", "Stopping after the current question…"
            from backend import access as AC
            AC.audit(db, user, "BULK_CHECKING_CANCELLED", "evaluation_job", j.id, {"via": "assistant"})
            db.commit()
            return {**base, "action": "checking_stopping", "job_id": j.id,
                    "reply": "Okay — I'll stop after the current question. Sheets not finished stay ready to check later."}
        return {**base, "job_id": j.id, "reply": f"Batch #{j.id} is not running (status: {j.status})."}
    if kind == "unclear":
        rows = [i for i in j.items if i.status == "needs_identification"]
        if not rows:
            return {**base, "job_id": j.id, "reply": f"Every sheet in batch #{j.id} is identified."}
        import json as _j
        txt = "\n".join(f"- Sheet #{i.id} **{i.filename}** — {(_j.loads(i.id_detail or '{}').get('problem') or 'not identified')}"
                        for i in rows[:20])
        return {**base, "job_id": j.id, "reply": f"These sheets need manual student verification (I never guess a student):\n{txt}\n\n"
                                                   "Open each one in the Checking panel and choose the student."}
    if kind == "report":
        rep = BE.class_report(db, j)
        s = rep["summary"]
        if not s["students_checked"]:
            return {**base, "job_id": j.id, "reply": f"No sheet of batch #{j.id} has been checked yet.\n\n" + _status_text(j)}
        f = lambda v, suf="%": "—" if v is None else f"{v:g}{suf}"   # noqa: E731
        q = "\n".join(f"- Q{x['question_number']}: average {x['average']:g}/{x['max_marks']:g} ({f(x['average_pct'])}), "
                      f"attempted by {f(x['attempt_rate_pct'])}" for x in rep["questions"])
        mc = ", ".join(c["concept"] for c in rep["missed_concepts"][:6]) or "—"
        return {**base, "job_id": j.id, "action": "report", "data": [rep],
                "reply": (f"**Class performance — {j.course_code} · {j.exam_name}** (AI marks, awaiting approval)\n"
                          f"- Checked: {s['students_checked']} · average {f(s['average_pct'])} · highest {f(s['highest_pct'])} · "
                          f"lowest {f(s['lowest_pct'])}\n- Passed {s['passed']}, failed {s['failed']} (pass rate {f(s['pass_rate_pct'])}, "
                          f"pass mark {s['pass_mark_pct']:g}%)\n- Flagged answers to verify: {s['flagged_answers']}\n\n"
                          f"**Question-wise**\n{q}\n\n**Most missed concepts:** {mc}\n\nDownload the PDF report from the Checking panel.")}
    return {**base, "job_id": j.id, "action": "status", "reply": _status_text(j)}
