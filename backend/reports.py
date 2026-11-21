"""
Printable PDF reports (ReportLab)
=================================
* evaluation_report_pdf  — per answer sheet: the 20 items listed in spec §14
* final_result_pdf       — a student's consolidated, officially approved results
* student_report_pdf     — admin "complete examination report" for one registration number
* class_result_pdf       — class result sheet for a batch / section / subject filter

Every figure comes from the database; AI marks and teacher-approved marks are labelled separately.
"""

import io
import json
import os
from datetime import datetime
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from backend.results_service import booklet_status_label, booklet_totals, grade_for

UNIVERSITY = os.getenv("UNIVERSITY_NAME", "University Examination Department")
NAVY = colors.HexColor("#1e2a6e")
LIGHT = colors.HexColor("#eef2ff")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Title"], fontSize=15, textColor=NAVY, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=_ss["Normal"], fontSize=9, alignment=TA_CENTER, textColor=colors.grey)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=11.5, textColor=NAVY, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("b", parent=_ss["Normal"], fontSize=8.8, leading=11.5)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.8, leading=10, textColor=colors.HexColor("#333333"))


def _p(text, style=BODY):
    t = "" if text is None else str(text)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    return Paragraph(t, style)


def _grid(data, widths, header=True, zebra=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cde8")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def _kv(rows, widths=(45 * mm, 135 * mm)):
    data = [[_p(f"<b>{k}</b>".replace("&lt;b&gt;", ""), BODY), _p(v)] for k, v in rows]
    for r, (k, _) in zip(data, rows):
        r[0] = Paragraph(f"<b>{k}</b>", BODY)
    return _grid(data, list(widths), header=False)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(15 * mm, 10 * mm, f"{UNIVERSITY} · generated {datetime.now().strftime('%d %b %Y %H:%M')}")
    canvas.drawRightString(doc.pagesize[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _doc(buf, title, pagesize=A4):
    return SimpleDocTemplate(buf, pagesize=pagesize, title=title, leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=14 * mm, bottomMargin=16 * mm)


def _header(title):
    return [Paragraph(UNIVERSITY, H1), Paragraph(title, SUB), Spacer(1, 6)]


def _lst(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw) if str(raw).startswith("[") else [raw]
    except Exception:
        return [raw]


# ── 1. Evaluation report (spec §14) ───────────────────────────────────────────

def evaluation_report_pdf(booklet, final_result=None) -> bytes:
    st = booklet.student
    batch = st.batch_rel if st else None
    t = booklet_totals(booklet)
    approved = booklet.approval_status in ("approved", "published")
    buf = io.BytesIO()
    doc = _doc(buf, f"Evaluation report {booklet.register_number} {booklet.course_code}")
    el = _header("Answer-sheet Evaluation Report")
    el.append(_kv([
        ("Examination", f"{booklet.exam_name}  ·  Date: {booklet.date_of_exam or '—'}"),
        ("Student name", st.name if st else "—"),
        ("Registration number", booklet.register_number or (st.registration_number if st else "")),
        ("Course / Programme · Section", f"{booklet.programme or (st.programme if st else '')} · Section {(st.section if st else '') or (batch.section if batch else '—')}"),
        ("Batch ID · Semester", f"{batch.batch_code if batch else '—'} · {booklet.semester or (st.semester if st else '')}"),
        ("Subject", f"{booklet.course_code} — {booklet.course_name}"),
        ("AI evaluation status", booklet_status_label(booklet)),
        ("Teacher verification", "Verified & approved by teacher" if approved else "NOT yet verified — marks are provisional"),
        ("Total — AI marks", f"{t['ai']:g} / {t['max']:g}"),
        ("Total — final approved marks", f"{t['final']:g} / {t['max']:g}" if approved else "Pending teacher approval"),
    ]))
    el.append(Paragraph("Question-wise evaluation", H2))
    for a in sorted(booklet.answers, key=lambda x: (x.question_number or 0)):
        r = a.submission.result if a.submission else None
        eq = a.exam_question
        mx = (r.max_marks if r else (eq.marks if eq else None)) or 0
        ai = r.final_score if r else None
        fin = (r.examiner_score if r and r.examiner_approved and r.examiner_score is not None else ai)
        rubric = "; ".join(f"{x.criterion} ({x.marks:g})" for x in eq.rubrics) if eq and eq.rubrics else ""
        rows = [
            [Paragraph(f"<b>Q{a.question_number}</b>  {(eq.question_text if eq else '') or ''}"[:900], BODY), ""],
            [Paragraph("<b>Student's answer</b>", SMALL), Paragraph("<b>Expected answer / marking criteria</b>", SMALL)],
            [_p((a.answer_text or "— no answer detected —")[:1800], SMALL),
             _p(((eq.teacher_answer or "") if eq else "")[:1500] + (f"\nCriteria: {rubric}" if rubric else ""), SMALL)],
            [Paragraph(f"<b>AI marks:</b> {ai if ai is not None else '—'} / {mx:g}   "
                       f"<b>Final:</b> {fin if fin is not None else '—'} / {mx:g}   "
                       f"<b>Deducted:</b> {round(mx - (fin or 0), 2) if fin is not None else '—'}", BODY), ""],
            [_p("Reason for deduction / feedback: " + ((r.examiner_comments or r.feedback) if r else "Not evaluated"), SMALL), ""],
            [_p("Missing concepts: " + (", ".join(_lst(r.missing_concepts)) if r else "—"), SMALL),
             _p("Strengths: " + (", ".join(_lst(r.strengths)) if r else "—"), SMALL)],
            [_p("Suggestion: cover the missing concepts above with a definition, the key steps and an example."
                if r and _lst(r.missing_concepts) else "Suggestion: —", SMALL),
             _p(("Teacher verified" if r and r.examiner_approved else "AI only (not verified)") +
                (f" · Flag: {r.flag_reason}" if r and r.flagged_for_review and r.flag_reason else ""), SMALL)],
        ]
        tb = Table(rows, colWidths=[90 * mm, 90 * mm])
        tb.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("SPAN", (0, 3), (1, 3)), ("SPAN", (0, 4), (1, 4)),
                                ("BACKGROUND", (0, 0), (1, 0), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cde8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el += [KeepTogether([tb]), Spacer(1, 6)]
    if final_result is not None:
        el.append(Paragraph("Consolidated result", H2))
        el.append(_kv([("OMR marks", f"{final_result.omr_score:g} / {final_result.omr_max:g}" if final_result.omr_score is not None else "—"),
                       ("Subjective marks", f"{final_result.subjective_score:g} / {final_result.subjective_max:g}" if final_result.subjective_score is not None else "—"),
                       ("Total", f"{final_result.total:g} / {final_result.max_total:g} ({final_result.percentage:g}%) · Grade {final_result.grade}"),
                       ("Status", final_result.status)]))
    el.append(Spacer(1, 8))
    el.append(Paragraph("AI marks are provisional. Only marks approved by an authorised teacher and published by the "
                        "examination department are official.", SMALL))
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── 2. Final result (student) ─────────────────────────────────────────────────

def final_result_pdf(student, rows: List[Dict], cgpa_value=None) -> bytes:
    buf = io.BytesIO()
    doc = _doc(buf, f"Final result {student.registration_number}")
    batch = student.batch_rel
    el = _header("Statement of Published Examination Results")
    el.append(_kv([("Student", student.name), ("Registration number", student.registration_number),
                   ("Programme · Semester", f"{student.programme} · {student.semester}"),
                   ("Batch · Section · Session", f"{batch.batch_code if batch else '—'} · {student.section or '—'} · {student.academic_session or '—'}")]))
    el.append(Spacer(1, 8))
    data = [["Subject", "Examination", "OMR", "Written", "Total", "%", "Grade", "Result"]]
    for r in rows:
        data.append([_p(f"{r['subject_code']} {r['subject_name'] or ''}", SMALL), _p(r["exam_name"], SMALL),
                     "—" if r.get("omr_marks") is None else f"{r['omr_marks']:g}",
                     "—" if r.get("subjective_marks") is None else f"{r['subjective_marks']:g}",
                     f"{r['marks']:g}/{r['max_marks']:g}" if r.get("marks") is not None else "—",
                     "—" if r.get("percentage") is None else f"{r['percentage']:g}", r.get("grade") or "—",
                     "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "—")])
    el.append(_grid(data, [42 * mm, 38 * mm, 14 * mm, 16 * mm, 22 * mm, 13 * mm, 14 * mm, 17 * mm]))
    if cgpa_value is not None:
        el += [Spacer(1, 6), Paragraph(f"<b>CGPA (credit-weighted, published results):</b> {cgpa_value}", BODY)]
    el += [Spacer(1, 10), Paragraph("Only results officially published by the examination department are listed.", SMALL)]
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── 3. Student complete report (admin) ───────────────────────────────────────

def student_report_pdf(profile: Dict) -> bytes:
    buf = io.BytesIO()
    doc = _doc(buf, f"Student report {profile['registration_number']}")
    el = _header("Complete Student Examination Report")
    el.append(_kv([("Student", profile["name"]), ("Registration number", profile["registration_number"]),
                   ("Programme · Department", f"{profile.get('programme')} · {profile.get('department')}"),
                   ("Batch · Section · Semester", f"{profile.get('batch_code') or '—'} · {profile.get('section') or '—'} · {profile.get('semester')}"),
                   ("Academic session", profile.get("academic_session") or "—"),
                   ("Subjects", ", ".join(f"{s['code']}" for s in profile.get("subjects", [])) or "—"),
                   ("Account status", "Active" if profile.get("is_active") else "Inactive"),
                   ("Overall", f"{profile['summary'].get('average_percentage') or '—'}% average · "
                               f"{profile['summary'].get('passed', 0)} passed / {profile['summary'].get('failed', 0)} failed · CGPA {profile.get('cgpa') or '—'}")]))
    el.append(Paragraph("Examination history", H2))
    data = [["Subject", "Exam", "AI marks", "Approved marks", "Max", "%", "Grade", "Status"]]
    for r in profile.get("results", []):
        data.append([r["subject_code"], _p(r["exam_name"], SMALL),
                     "—" if r.get("ai_marks") is None else f"{r['ai_marks']:g}",
                     "—" if r.get("final_approved_marks") is None else f"{r['final_approved_marks']:g}",
                     "—" if not r.get("max_marks") else f"{r['max_marks']:g}",
                     "—" if r.get("percentage") is None else f"{r['percentage']:g}", r.get("grade") or "—",
                     _p(r.get("ai_evaluation_status") or r.get("status"), SMALL)])
    el.append(_grid(data, [18 * mm, 40 * mm, 17 * mm, 23 * mm, 13 * mm, 13 * mm, 14 * mm, 42 * mm]))
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── 4. Class result sheet ────────────────────────────────────────────────────

def class_result_pdf(rows: List[Dict], filters: Dict, summary: Dict) -> bytes:
    buf = io.BytesIO()
    doc = _doc(buf, "Class result", pagesize=landscape(A4))
    el = _header("Class Result Sheet")
    f = " · ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in filters.items() if v) or "All records"
    el += [Paragraph(f, BODY), Spacer(1, 4),
           Paragraph(f"Students: {summary['students']} · Graded: {summary['graded']} · Passed: {summary['passed']} · "
                     f"Failed: {summary['failed']} · Pass %: {summary['pass_percentage'] if summary['pass_percentage'] is not None else '—'} · "
                     f"Average %: {summary['average_percentage'] if summary['average_percentage'] is not None else '—'}", BODY),
           Spacer(1, 6)]
    data = [["Reg. No.", "Name", "Batch", "Sec", "Sem", "Subject", "Exam", "AI marks", "Final marks", "Max", "%", "Grade", "Result", "AI evaluation status"]]
    for r in rows:
        data.append([r["registration_number"], _p(r["student_name"], SMALL), r["batch_code"], r["section"],
                     _p(r["semester"], SMALL), r["subject_code"], _p(r["exam_name"], SMALL),
                     "—" if r.get("ai_marks") is None else f"{r['ai_marks']:g}",
                     "—" if r.get("final_approved_marks") is None else f"{r['final_approved_marks']:g}",
                     "—" if not r.get("max_marks") else f"{r['max_marks']:g}",
                     "—" if r.get("percentage") is None else f"{r['percentage']:g}", r.get("grade") or "—",
                     "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "—"),
                     _p(r.get("ai_evaluation_status"), SMALL)])
    el.append(_grid(data, [22 * mm, 32 * mm, 20 * mm, 9 * mm, 17 * mm, 16 * mm, 30 * mm, 15 * mm, 17 * mm, 11 * mm,
                           11 * mm, 12 * mm, 14 * mm, 40 * mm]))
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
