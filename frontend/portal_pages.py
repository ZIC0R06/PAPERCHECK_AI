"""
Portal pages for the three roles (admin / teacher / student).
Every number shown comes from the backend APIs; nothing here invents data.
dashboard.py calls init(...) once, then the page functions.
"""

import io
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

_C = {}


def init(api_base, headers_fn, theme_mod):
    _C.update(api=api_base, headers=headers_fn, theme=theme_mod)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(method, path, **kw):
    try:
        return requests.request(method, _C["api"] + path, headers=_C["headers"](), timeout=kw.pop("timeout", 120), **kw)
    except Exception:
        return None


def _detail(r, fallback="Request failed"):
    if r is None:
        return "Cannot reach the backend. Is the API running?"
    try:
        d = r.json().get("detail")
        if isinstance(d, list):
            d = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg', '')}" for e in d)
        return d or f"{fallback} ({r.status_code})"
    except Exception:
        return f"{fallback} ({r.status_code})"


def get(path, **params):
    r = _req("GET", path, params={k: v for k, v in params.items() if v not in (None, "", "All")})
    if r is not None and r.status_code == 200:
        return r.json()
    st.session_state["_last_err"] = _detail(r)
    return None


def send(method, path, ok_msg=None, **kw):
    """POST/PATCH/PUT/DELETE; shows the real backend error. Returns JSON or None."""
    r = _req(method, path, **kw)
    if r is not None and 200 <= r.status_code < 300:
        if ok_msg:
            st.success(ok_msg)
        try:
            return r.json()
        except Exception:
            return {}
    st.error(_detail(r))
    return None


def get_bytes(path, **params):
    r = _req("GET", path, params={k: v for k, v in params.items() if v not in (None, "", "All")})
    if r is not None and r.status_code == 200:
        return r.content
    st.error(_detail(r, "Download failed"))
    return None


def _hdr(title, sub, crumb=""):
    _C["theme"].page_header(title, sub, crumb=crumb)


def _kpis(items):
    _C["theme"].kpi_row(items)


def _fmt(v, suffix=""):
    return "—" if v is None else f"{v:g}{suffix}" if isinstance(v, (int, float)) else str(v)


def _df(rows, cols=None, rename=None):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if "passed" in df.columns:
        df["passed"] = df["passed"].map({True: "PASS", False: "FAIL"}).fillna("—")
    df = df.astype(object).where(df.notna(), "—")
    return df.rename(columns=rename or {c: c.replace("_", " ").title() for c in df.columns})


def _chart(ch, **_ignored):
    """Altair chart styled for the dark portal theme."""
    ch = (ch.configure(background="transparent")
            .configure_view(strokeOpacity=0)
            .configure_axis(labelColor="#cbd5f5", titleColor="#9aa6d6", gridColor="#24305e", domainColor="#3a4680",
                            tickColor="#3a4680", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor="#cbd5f5", titleColor="#9aa6d6")
            .configure_range(category=["#5b7cff", "#22c55e", "#f59e0b", "#a855f7", "#38bdf8", "#ef4444", "#14b8a6", "#f472b6"]))
    st.altair_chart(ch, use_container_width=True, theme=None)


def _subjects():
    return get("/academic/subjects") or []


def _batches():
    return get("/academic/batches") or []


def _my_subjects():
    """Teachers: only the subjects assigned to them. Admin: every subject."""
    if st.session_state.current_user.get("role") == "faculty":
        d = get("/results/teacher/dashboard") or {}
        return sorted({a["subject_code"] for a in ((d.get("profile") or {}).get("assignments") or [])})
    return [s["code"] for s in _subjects()]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: class results & analytics (admin = all, teacher = assigned only)
# ═══════════════════════════════════════════════════════════════════════════════

RESULT_COLS = ["registration_number", "student_name", "batch_code", "section", "semester", "subject_code", "exam_name",
               "omr_marks", "subjective_marks", "ai_marks", "final_approved_marks", "max_marks", "percentage", "grade",
               "passed", "ai_evaluation_status"]


def _filters(prefix, with_exam=True):
    batches = _batches()
    subs = _subjects()
    c1, c2, c3, c4 = st.columns(4)
    batch = c1.selectbox("Batch ID", ["All"] + [b["batch_code"] for b in batches], key=f"{prefix}_b")
    course = c2.selectbox("Course code", ["All"] + sorted({b["course_code"] for b in batches if b["course_code"]}), key=f"{prefix}_c")
    section = c3.selectbox("Section", ["All"] + sorted({b["section"] for b in batches}), key=f"{prefix}_s")
    sem = c4.selectbox("Semester", ["All"] + sorted({b["semester"] for b in batches}), key=f"{prefix}_sem")
    c5, c6, c7 = st.columns([1, 1, 2])
    subj = c5.selectbox("Subject code", ["All"] + [s["code"] for s in subs], key=f"{prefix}_subj")
    sess = c6.selectbox("Academic session", ["All"] + sorted({b["academic_session"] for b in batches}), key=f"{prefix}_sess")
    exam = c7.text_input("Examination (exact name, optional)", key=f"{prefix}_exam") if with_exam else ""
    return {"batch_code": batch, "course_code": course, "section": section, "semester": sem, "subject_code": subj,
            "session": sess, "exam_name": exam}


def page_class_results(role):
    _hdr("Class Results", "Search by Batch ID, course code, section, semester and subject code. "
         + ("You see only the classes assigned to you." if role == "faculty" else "All batches."), crumb="Results")
    f = _filters("cr")
    data = get("/results/class", **f)
    if data is None:
        st.error(st.session_state.get("_last_err"))
        return
    s = data["summary"]
    _kpis([{"label": "Students", "value": s["students"], "hint": f"{s['records']} records"},
           {"label": "Passed", "value": s["passed"], "hint": f"{_fmt(s['pass_percentage'], '%')} pass", "tone": "green"},
           {"label": "Failed", "value": s["failed"], "hint": f"{_fmt(s['fail_percentage'], '%')} fail", "tone": "amber"},
           {"label": "Average", "value": _fmt(s["average_percentage"], "%"),
            "hint": f"high {_fmt(s['highest_percentage'], '%')} · low {_fmt(s['lowest_percentage'], '%')}", "tone": "cyan"},
           {"label": "Pending", "value": s["pending"], "hint": "not yet approved", "tone": "violet"}])
    if not data["rows"]:
        _C["theme"].empty_state("No results for this selection", "Change the filters, or evaluate and approve answer sheets first.", "📑")
        return
    st.dataframe(_df(data["rows"], RESULT_COLS), use_container_width=True, hide_index=True)
    st.caption("AI marks are provisional; “Final Approved Marks” are teacher-approved. Pass mark and grades follow the configured scale.")
    c1, c2, c3 = st.columns(3)
    for col, fmt, mime, label in ((c1, "csv", "text/csv", "⬇️ CSV"),
                                  (c2, "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "⬇️ Excel"),
                                  (c3, "pdf", "application/pdf", "⬇️ PDF")):
        if col.button(f"Prepare {fmt.upper()}", key=f"prep_{fmt}"):
            b = get_bytes("/results/class", **{**f, "format": fmt})
            if b:
                col.download_button(label, b, file_name=f"class_result.{fmt}", mime=mime, use_container_width=True, key=f"dl_{fmt}")


def page_analytics(role):
    _hdr("Performance Analytics", "Compare batches, sections, semesters, subjects, teachers and sessions. "
         "Only teacher-approved marks are counted.", crumb="Analytics")
    gb = st.radio("Compare by", ["subject", "batch", "section", "semester", "teacher", "session", "exam"], horizontal=True)
    f = _filters("an", with_exam=False)
    f.pop("exam_name", None)
    a = get("/results/analytics", group_by=gb, **f)
    if a is None:
        st.error(st.session_state.get("_last_err"))
        return
    o = a["overall"]
    _kpis([{"label": "Students", "value": o["students"], "hint": f"{o['graded']} graded results"},
           {"label": "Pass %", "value": _fmt(o["pass_percentage"], "%"), "hint": f"{o['passed']} passed", "tone": "green"},
           {"label": "Fail %", "value": _fmt(o["fail_percentage"], "%"), "hint": f"{o['failed']} failed", "tone": "amber"},
           {"label": "Average", "value": _fmt(o["average_percentage"], "%"), "hint": "of approved marks", "tone": "cyan"},
           {"label": "Highest / lowest", "value": f"{_fmt(o['highest_percentage'])} / {_fmt(o['lowest_percentage'])}", "hint": "%", "tone": "violet"}])
    g = pd.DataFrame([x for x in a["groups"] if x.get("graded")])
    if g.empty:
        _C["theme"].empty_state("No approved results to compare yet", "Approve and publish results first.", "📈")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pass % by {gb}**")
        _chart(alt.Chart(g).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=46).encode(
            x=alt.X("group:N", title=None, sort=None), y=alt.Y("pass_percentage:Q", title="Pass %", scale=alt.Scale(domain=[0, 100])),
            color=alt.value("#5b7cff"), tooltip=["group", "pass_percentage", "passed", "failed"]).properties(height=260))
    with c2:
        st.markdown(f"**Average / highest / lowest % by {gb}**")
        m = g.melt(id_vars="group", value_vars=["average_percentage", "highest_percentage", "lowest_percentage"])
        _chart(alt.Chart(m).mark_line(point=True).encode(
            x=alt.X("group:N", title=None, sort=None), y=alt.Y("value:Q", title="%"), color=alt.Color("variable:N", title=None),
            tooltip=["group", "variable", "value"]).properties(height=260))
    c3, c4 = st.columns([1, 1.3])
    with c3:
        st.markdown("**Grade distribution**")
        gd = pd.DataFrame(a["grade_distribution"])
        if not gd.empty:
            _chart(alt.Chart(gd).mark_arc(innerRadius=50).encode(
                theta="count:Q", color=alt.Color("grade:N", sort=["O", "A+", "A", "B+", "B", "C", "P", "F"]),
                tooltip=["grade", "count"]).properties(height=260))
    with c4:
        st.markdown("**Comparison table**")
        st.dataframe(_df(a["groups"], ["group", "students", "graded", "passed", "failed", "pass_percentage",
                                        "average_percentage", "highest_percentage", "lowest_percentage", "average_marks"]),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export comparison (CSV)", g.to_csv(index=False).encode(), file_name=f"analytics_{gb}.csv")
    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown("**Topic-wise weaknesses** (most missed concepts)")
        st.dataframe(_df(a["weak_topics"]), use_container_width=True, hide_index=True)
    with c6:
        st.markdown("**Most difficult questions**")
        st.dataframe(_df(a["difficult_questions"]), use_container_width=True, hide_index=True)
    with c7:
        st.markdown("**Subjects with low pass % (<60%)**")
        st.dataframe(_df(a["low_pass_subjects"], ["subject", "pass_percentage", "graded"]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

def page_admin_dashboard():
    _hdr("Examination Department Dashboard", "Live figures from the central database.", crumb="Admin · Overview")
    o = get("/academic/overview")
    if not o:
        st.error(st.session_state.get("_last_err"))
        return
    _kpis([{"label": "Students", "value": o["students"], "hint": "registered"},
           {"label": "Teachers", "value": o["teachers"], "hint": "with teacher IDs", "tone": "cyan"},
           {"label": "Courses / subjects", "value": o["subjects"], "hint": f"{o['programmes']} programmes", "tone": "violet"},
           {"label": "Batches", "value": o["batches"], "hint": f"{o['sections']} sections", "tone": "green"}])
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    r = o["results"]
    _kpis([{"label": "Examinations", "value": o["examinations"], "hint": "question papers"},
           {"label": "Evaluated sheets", "value": o["evaluated"], "hint": f"of {o['answer_sheets']} uploaded", "tone": "cyan"},
           {"label": "Pending evaluation", "value": o["pending_evaluation"], "hint": f"{o['awaiting_approval']} awaiting approval", "tone": "amber"},
           {"label": "Published results", "value": o["published"], "hint": "visible to students", "tone": "green"},
           {"label": "Overall pass %", "value": _fmt(r["pass_percentage"], "%"), "hint": f"{r['graded']} graded", "tone": "violet"}])
    dq = get("/academic/data-quality") or {}
    if dq.get("duplicate_answer_sheets"):
        with st.expander(f"⚠️ {len(dq['duplicate_answer_sheets'])} duplicate answer-sheet group(s) detected — same student, subject and exam uploaded more than once"):
            st.dataframe(_df(dq["duplicate_answer_sheets"]), use_container_width=True, hide_index=True)
            st.caption("Review these booklets; duplicates inflate result counts. New uploads of an existing student/subject/exam are now blocked.")
    if dq.get("students_without_batch"):
        st.info(f"{dq['students_without_batch']} student(s) are not assigned to a batch yet (Student Management).")
    if o["omr_needing_verification"]:
        st.warning(f"{o['omr_needing_verification']} OMR sheet(s) have unclear readings waiting for manual verification.")
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    a = get("/results/analytics", group_by="subject")
    b = get("/results/analytics", group_by="batch")
    c1, c2 = st.columns(2)
    for col, data, title in ((c1, a, "Pass % by subject"), (c2, b, "Average % by batch")):
        with col:
            st.markdown(f"**{title}**")
            df = pd.DataFrame([x for x in (data or {}).get("groups", []) if x.get("graded")])
            if df.empty:
                st.caption("No approved results yet.")
            else:
                y = "pass_percentage" if "subject" in title else "average_percentage"
                _chart(alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=46).encode(
                    x=alt.X("group:N", title=None), y=alt.Y(f"{y}:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["group", y, "graded"]).properties(height=240))


def page_reg_search():
    _hdr("Registration Number Search", "Complete profile, subjects, examination history, marks, answer sheets and "
         "reports for one student.", crumb="Admin · Search")
    reg = st.text_input("Registration number", placeholder="e.g. 21BCS101").strip()
    if not reg:
        matches = None
        q = st.text_input("…or search by name / partial number", key="rs_q")
        if q:
            matches = get("/academic/students", q=q)
            if matches:
                st.dataframe(_df(matches, ["registration_number", "name", "batch_code", "section", "semester", "programme"]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No matching students.")
        return
    p = get(f"/academic/students/{reg}/profile360")
    if not p:
        st.error(st.session_state.get("_last_err"))
        return
    st.markdown(f"""<div class="card"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
      <div><div style="font-size:1.25rem;font-weight:700">{p['name']}</div>
      <div style="color:var(--muted);font-size:.85rem">{p['registration_number']} · {p['programme']} · {p['department']}</div>
      <div style="color:var(--muted);font-size:.85rem">Batch <b>{p['batch_code'] or '—'}</b> · Section <b>{p['section'] or '—'}</b> ·
      {p['semester']} · Session {p['academic_session'] or '—'} · {p['email']}</div></div>
      <div>{_C['theme'].status_pill('ACTIVE' if p['is_active'] else 'INACTIVE', 'ok' if p['is_active'] else 'bad')}</div></div></div>""",
                unsafe_allow_html=True)
    s = p["summary"]
    _kpis([{"label": "Subjects", "value": len(p["subjects"]), "hint": ", ".join(x["code"] for x in p["subjects"])[:40]},
           {"label": "Average", "value": _fmt(s["average_percentage"], "%"), "hint": f"{s['graded']} approved results", "tone": "cyan"},
           {"label": "Passed / failed", "value": f"{s['passed']} / {s['failed']}", "hint": "approved results", "tone": "green"},
           {"label": "CGPA", "value": _fmt(p["cgpa"]), "hint": "published results", "tone": "violet"}])
    st.markdown("#### Subject-wise marks & examination history")
    if p["results"]:
        st.dataframe(_df(p["results"], ["subject_code", "subject_name", "exam_name", "semester", "omr_marks", "subjective_marks",
                                         "ai_marks", "final_approved_marks", "max_marks", "percentage", "grade", "passed",
                                         "ai_evaluation_status"]), use_container_width=True, hide_index=True)
    else:
        st.info("No examination records yet.")
    if p["semester_wise"]:
        st.markdown("#### Semester-wise results")
        st.dataframe(_df([{"semester": k, **v} for k, v in p["semester_wise"].items()],
                         ["semester", "graded", "passed", "failed", "average_percentage", "highest_percentage", "lowest_percentage"]),
                     use_container_width=True, hide_index=True)
    st.markdown("#### Answer sheets & reports")
    for b in p["answer_sheets"]:
        c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
        c1.markdown(f"**{b['subject_code']}** · {b['exam_name']} · {b['pages'] or '?'} page(s)  \n<span style='color:var(--muted);font-size:.8rem'>{b['status']}</span>", unsafe_allow_html=True)
        if c2.button("Original", key=f"o{b['booklet_id']}"):
            d = get_bytes(b["original_url"])
            if d:
                c2.download_button("⬇️ Save", d, file_name=f"original_{reg}_{b['subject_code']}.pdf", key=f"od{b['booklet_id']}")
        if c3.button("AI report", key=f"r{b['booklet_id']}"):
            d = get_bytes(b["report_url"])
            if d:
                c3.download_button("⬇️ Save", d, file_name=f"evaluation_{reg}_{b['subject_code']}.pdf", key=f"rd{b['booklet_id']}")
        if b["evaluated_url"] and c4.button("Evaluated", key=f"e{b['booklet_id']}"):
            d = get_bytes(b["evaluated_url"])
            if d:
                c4.download_button("⬇️ Save", d, file_name=f"evaluated_{reg}_{b['subject_code']}.pdf", key=f"ed{b['booklet_id']}")
    if not p["answer_sheets"]:
        st.caption("No answer sheets uploaded.")
    if p["omr"]:
        st.markdown("#### OMR sheets")
        st.dataframe(_df(p["omr"]), use_container_width=True, hide_index=True)
    if st.button("📄 Prepare complete examination report (PDF)"):
        d = get_bytes(f"/academic/students/{reg}/report.pdf")
        if d:
            st.download_button("⬇️ Download report", d, file_name=f"report_{reg}.pdf", mime="application/pdf")


def page_students():
    _hdr("Student Management", "Register, edit, assign batch / section / semester / subjects, reset passwords, "
         "activate or deactivate.", crumb="Admin · Students")
    t_list, t_new = st.tabs(["🔎 Search & edit", "➕ Register student"])
    batches = _batches()
    subs = _subjects()
    with t_list:
        c1, c2, c3 = st.columns([2, 1, 1])
        q = c1.text_input("Search (registration no., name, email)")
        bf = c2.selectbox("Batch", ["All"] + [b["batch_code"] for b in batches])
        act = c3.selectbox("Status", ["All", "Active", "Inactive"])
        rows = get("/academic/students", q=q, batch_code=bf,
                   active={"Active": "true", "Inactive": "false"}.get(act)) or []
        st.dataframe(_df(rows, ["registration_number", "name", "batch_code", "section", "semester", "programme", "email", "is_active"]),
                     use_container_width=True, hide_index=True)
        if not rows:
            return
        sel = st.selectbox("Edit student", [r["registration_number"] for r in rows],
                           format_func=lambda x: f"{x} — {next(r['name'] for r in rows if r['registration_number'] == x)}")
        cur = next(r for r in rows if r["registration_number"] == sel)
        with st.form(f"edit_{sel}"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name", cur["name"])
            email = c2.text_input("Email", cur["email"])
            prog = c1.text_input("Programme / course", cur["programme"])
            codes = [b["batch_code"] for b in batches]
            bsel = c2.selectbox("Batch (sets section, semester & session)", ["(none)"] + codes,
                                index=(codes.index(cur["batch_code"]) + 1) if cur["batch_code"] in codes else 0)
            subj = st.multiselect("Subjects", [s["code"] for s in subs], default=[s["code"] for s in cur["subjects"]])
            active = st.checkbox("Account active", value=cur["is_active"])
            if st.form_submit_button("💾 Save changes"):
                body = {"name": name, "email": email, "programme": prog, "subject_codes": subj, "is_active": active}
                if bsel != "(none)":
                    body["batch_code"] = bsel
                if send("PATCH", f"/academic/students/{sel}", json=body, ok_msg="Saved."):
                    st.rerun()
        if st.button(f"🔑 Reset password for {sel}"):
            r = send("POST", f"/academic/users/{sel}/reset-password")
            if r:
                st.success(f"Temporary password for **{r['username']}**: `{r['temporary_password']}` — shown once; share it securely.")
    with t_new:
        with st.form("new_student", clear_on_submit=False):
            c1, c2 = st.columns(2)
            reg = c1.text_input("Registration number *")
            name = c2.text_input("Full name *")
            email = c1.text_input("Email *")
            prog = c2.text_input("Programme / course", "B.Tech Computer Science")
            dept = c1.text_input("Department", "Computer Science & Engineering")
            b = c2.selectbox("Batch", ["(none)"] + [x["batch_code"] for x in batches])
            subj = st.multiselect("Subjects", [s["code"] for s in subs])
            if st.form_submit_button("➕ Register student"):
                r = send("POST", "/academic/students", json={"registration_number": reg, "name": name, "email": email,
                                                              "programme": prog, "department": dept,
                                                              "batch_code": None if b == "(none)" else b, "subject_codes": subj})
                if r:
                    st.success(f"Registered **{r['registration_number']}**. Login: `{r['username']}` · temporary password "
                               f"`{r['temporary_password']}` — shown once.")


def page_teachers():
    _hdr("Teacher Management", "Register teachers (unique teacher IDs), assign subjects to batches/sections, "
         "view results, reset passwords, activate or deactivate.", crumb="Admin · Teachers")
    t_list, t_new = st.tabs(["🔎 Search teachers", "➕ Register teacher"])
    with t_list:
        q = st.text_input("Search by teacher ID, name, email or department")
        rows = get("/academic/teachers", q=q) or []
        st.dataframe(_df([{**t, "assignments": ", ".join(f"{a['subject_code']}@{a['batch_code']}" for a in t["assignments"])}
                          for t in rows], ["teacher_code", "full_name", "department", "designation", "email", "assignments", "is_active"]),
                     use_container_width=True, hide_index=True)
        if not rows:
            return
        code = st.selectbox("Open teacher", [t["teacher_code"] for t in rows],
                            format_func=lambda c: f"{c} — {next(t['full_name'] for t in rows if t['teacher_code'] == c)}")
        t = get(f"/academic/teachers/{code}")
        if not t:
            return
        s = t["stats"]
        o = s["overall"]
        _kpis([{"label": "Batches", "value": s["total_batches"], "hint": "assigned"},
               {"label": "Subjects", "value": s["total_subjects"], "hint": "assigned", "tone": "violet"},
               {"label": "Students", "value": s["total_students"], "hint": "taught", "tone": "cyan"},
               {"label": "Pass %", "value": _fmt(o["pass_percentage"], "%"), "hint": f"fail {_fmt(o['fail_percentage'], '%')}", "tone": "green"},
               {"label": "Class average", "value": _fmt(o["average_percentage"], "%"), "hint": "approved marks", "tone": "amber"}])
        c1, c2 = st.columns(2)
        c1.markdown("**Subject-wise results**")
        c1.dataframe(_df([{"subject": k, **v} for k, v in s["subject_wise"].items()],
                         ["subject", "students", "graded", "pass_percentage", "fail_percentage", "average_percentage"]),
                     use_container_width=True, hide_index=True)
        c2.markdown("**Batch-wise results**")
        c2.dataframe(_df([{"batch": k, **v} for k, v in s["batch_wise"].items()],
                         ["batch", "students", "graded", "pass_percentage", "fail_percentage", "average_percentage"]),
                     use_container_width=True, hide_index=True)
        st.markdown("**Assignments**")
        for a in t["assignments"]:
            c1, c2 = st.columns([5, 1])
            c1.write(f"{a['subject_code']} {a['subject_name']} → batch **{a['batch_code']}** (section {a['section']}, "
                     f"{a['semester']}, {a['students']} students)")
            if c2.button("Remove", key=f"rm{a['id']}"):
                if send("DELETE", f"/academic/teachers/{code}/assignments/{a['id']}", ok_msg="Removed."):
                    st.rerun()
        with st.form(f"assign_{code}"):
            c1, c2 = st.columns(2)
            sc = c1.selectbox("Subject", [x["code"] for x in _subjects()])
            bc = c2.selectbox("Batch", [b["batch_code"] for b in _batches()])
            if st.form_submit_button("➕ Assign"):
                if send("POST", f"/academic/teachers/{code}/assignments", json={"subject_code": sc, "batch_code": bc}, ok_msg="Assigned."):
                    st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("🔑 Reset password"):
            r = send("POST", f"/academic/users/{code}/reset-password")
            if r:
                st.success(f"Temporary password for **{code}**: `{r['temporary_password']}` — shown once.")
        lbl = "🚫 Deactivate account" if t["is_active"] else "✅ Activate account"
        if c2.button(lbl):
            if send("PATCH", f"/academic/teachers/{code}", json={"is_active": not t["is_active"]}, ok_msg="Updated."):
                st.rerun()
    with t_new:
        with st.form("new_teacher"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Full name *")
            email = c2.text_input("Email *")
            dept = c1.text_input("Department", "Computer Science & Engineering")
            dcode = c2.text_input("Department code (for the teacher ID)", "CSE")
            desig = c1.text_input("Designation", "Assistant Professor")
            phone = c2.text_input("Phone")
            if st.form_submit_button("➕ Register teacher"):
                r = send("POST", "/academic/teachers", json={"full_name": name, "email": email, "department": dept,
                                                             "department_code": dcode, "designation": desig, "phone": phone})
                if r:
                    st.success(f"Teacher ID **{r['teacher_code']}** created. Temporary password `{r['temporary_password']}` "
                               "— shown once. Now assign subjects and batches in the Search tab.")


def page_batches():
    _hdr("Batches & Sections", "Every batch has a unique Batch ID (e.g. CSE2026A = course, section, semester).",
         crumb="Admin · Batches")
    rows = _batches()
    st.dataframe(_df([{**b, "teachers": ", ".join(f"{a['teacher_code']}:{a['subject_code']}" for a in b["assignments"])} for b in rows],
                     ["batch_code", "programme", "course_code", "section", "semester", "academic_session", "total_students", "teachers", "is_active"]),
                 use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**➕ Create batch**")
        with st.form("new_batch"):
            code = st.text_input("Batch ID *", placeholder="CSE2026A")
            prog = st.text_input("Programme", "B.Tech Computer Science")
            cc = st.text_input("Course code", "CSE")
            sec = st.text_input("Section", "A")
            sem = st.selectbox("Semester", [f"Semester {i}" for i in range(1, 11)], index=4)
            sess = st.text_input("Academic session", "2026-27")
            if st.form_submit_button("Create"):
                if send("POST", "/academic/batches", json={"batch_code": code, "programme": prog, "course_code": cc,
                                                           "section": sec, "semester": sem, "academic_session": sess},
                        ok_msg="Batch created."):
                    st.rerun()
    with c2:
        if rows:
            st.markdown("**✏️ Edit batch**")
            code = st.selectbox("Batch", [b["batch_code"] for b in rows])
            b = next(x for x in rows if x["batch_code"] == code)
            with st.form(f"edit_b_{code}"):
                sec = st.text_input("Section", b["section"])
                sem = st.text_input("Semester", b["semester"])
                sess = st.text_input("Academic session", b["academic_session"])
                active = st.checkbox("Active", b["is_active"])
                if st.form_submit_button("Save (updates the batch's students too)"):
                    if send("PATCH", f"/academic/batches/{code}", json={"section": sec, "semester": sem,
                                                                        "academic_session": sess, "is_active": active}, ok_msg="Saved."):
                        st.rerun()
            if st.button(f"🗑️ Delete {code} (only if it has no students)"):
                if send("DELETE", f"/academic/batches/{code}", ok_msg="Deleted."):
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TEACHER
# ═══════════════════════════════════════════════════════════════════════════════

def page_teacher_dashboard():
    d = get("/results/teacher/dashboard")
    if d is None:
        st.error(st.session_state.get("_last_err"))
        return
    if not d.get("profile"):
        _hdr("Teacher Dashboard", "")
        st.warning(d.get("message"))
        return
    p, s, a = d["profile"], d["stats"], d["answer_sheets"]
    _hdr(f"Welcome, {p['full_name']}", f"Teacher ID {p['teacher_code']} · {p['designation']} · {p['department']}",
         crumb="Teacher · Dashboard")
    o = s["overall"]
    _kpis([{"label": "Batches", "value": s["total_batches"], "hint": "assigned"},
           {"label": "Subjects", "value": s["total_subjects"], "hint": "assigned", "tone": "violet"},
           {"label": "Students", "value": s["total_students"], "hint": "you teach", "tone": "cyan"},
           {"label": "Pass %", "value": _fmt(o["pass_percentage"], "%"), "hint": f"avg {_fmt(o['average_percentage'], '%')}", "tone": "green"}])
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    _kpis([{"label": "Pending answer sheets", "value": a["pending_evaluation"], "hint": "need AI evaluation", "tone": "amber"},
           {"label": "Awaiting your review", "value": a["awaiting_review"], "hint": "AI evaluated", "tone": "violet"},
           {"label": "Approved, unpublished", "value": a["approved_unpublished"], "hint": "ready to publish", "tone": "cyan"},
           {"label": "Results awaiting approval", "value": d["results_awaiting_approval"], "hint": "OMR + written", "tone": "amber"},
           {"label": "OMR to verify", "value": d["omr_needing_verification"], "hint": "unclear bubbles", "tone": "amber"}])
    st.markdown("#### Your assignments")
    st.dataframe(_df(p["assignments"], ["subject_code", "subject_name", "batch_code", "section", "semester", "programme", "students"]),
                 use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    sw = pd.DataFrame([{"subject": k, **v} for k, v in s["subject_wise"].items() if v["graded"]])
    bw = pd.DataFrame([{"batch": k, **v} for k, v in s["batch_wise"].items() if v["graded"]])
    with c1:
        st.markdown("**Subject-wise performance**")
        if sw.empty:
            st.caption("No approved results yet.")
        else:
            _chart(alt.Chart(sw).mark_bar(size=46, color="#5b7cff").encode(x=alt.X("subject:N", title=None), y=alt.Y("average_percentage:Q", title="Average %"),
                                                            tooltip=["subject", "average_percentage", "pass_percentage", "highest_percentage", "lowest_percentage"]).properties(height=220))
    with c2:
        st.markdown("**Batch-wise performance**")
        if bw.empty:
            st.caption("No approved results yet.")
        else:
            st.dataframe(_df(bw.to_dict("records"), ["batch", "graded", "pass_percentage", "fail_percentage", "average_percentage",
                                                     "highest_percentage", "lowest_percentage"]), use_container_width=True, hide_index=True)
    an = get("/results/analytics", group_by="subject")
    if an and an["weak_topics"]:
        st.markdown("**Topic-wise student weaknesses** (concepts most often missing in your students' answers)")
        st.dataframe(_df(an["weak_topics"]), use_container_width=True, hide_index=True)


def page_my_classes():
    _hdr("My Classes & Students", "Only the batches and subjects assigned to you.", crumb="Teacher · Classes")
    d = get("/results/teacher/dashboard") or {}
    asg = (d.get("profile") or {}).get("assignments", [])
    if not asg:
        st.info("You have no assignments yet.")
        return
    i = st.selectbox("Class", range(len(asg)), format_func=lambda k: f"{asg[k]['subject_code']} · {asg[k]['batch_code']} "
                                                                       f"(section {asg[k]['section']}, {asg[k]['students']} students)")
    a = asg[i]
    q = st.text_input("Search student")
    studs = get("/academic/students", batch_code=a["batch_code"], subject_code=a["subject_code"], q=q) or []
    res = get("/results/class", batch_code=a["batch_code"], subject_code=a["subject_code"]) or {"rows": []}
    by = {}
    for r in res["rows"]:
        by.setdefault(r["registration_number"], []).append(f"{r['exam_name']}: {_fmt(r['final_approved_marks'])}/{_fmt(r['max_marks'])} ({r['ai_evaluation_status']})")
    st.dataframe(pd.DataFrame([{"Registration No.": s["registration_number"], "Name": s["name"], "Section": s["section"],
                                "Results": " | ".join(by.get(s["registration_number"], [])) or "—"} for s in studs]),
                 use_container_width=True, hide_index=True)


def page_scanner():
    _hdr("Answer-Sheet Scanner", "Scan all pages of ONE student's answer sheet, review, then submit as one PDF saved "
         "under the registration number. Then start the next student.", crumb="Teacher · Scanner")
    done = st.session_state.get("scan_done")  # result of the last submission — shown first
    if done:
        st.success(done.get("message", "Submitted."))
        if st.button("🤖 Run OCR + AI evaluation for this sheet now"):
            with st.spinner("Reading the handwriting and scoring against the marking scheme…"):
                r = send("POST", f"/examiner/evaluate-booklet/{done['booklet_id']}", timeout=900)
            if r:
                st.success(f"AI score {r['total_ai_score']} / {r['max_marks']} — review it in Examiner Review.")
                if r.get("skipped_questions"):
                    st.warning(f"Not evaluated (no model answer): questions {r['skipped_questions']}")
            st.session_state.scan_done = None
    d = get("/results/teacher/dashboard") if st.session_state.current_user["role"] == "faculty" else None
    open_sessions = get("/scanner/sessions", status="open") or []
    sid = st.session_state.get("scan_sid")
    if sid and not any(s["id"] == sid for s in open_sessions):
        sid = None
    with st.expander("▶️ Start a new student", expanded=not sid):
        asg = ((d or {}).get("profile") or {}).get("assignments", [])
        c1, c2, c3 = st.columns(3)
        if asg:
            k = c1.selectbox("Class", range(len(asg)), format_func=lambda i: f"{asg[i]['subject_code']} · {asg[i]['batch_code']}")
            subj, batch = asg[k]["subject_code"], asg[k]["batch_code"]
            studs = get("/academic/students", batch_code=batch, subject_code=subj) or []
            reg = c2.selectbox("Student", [s["registration_number"] for s in studs],
                               format_func=lambda r: f"{r} — {next(s['name'] for s in studs if s['registration_number'] == r)}") if studs else None
        else:
            subj = c1.text_input("Subject code")
            reg = c2.text_input("Registration number")
        exam = c3.text_input("Examination name", st.session_state.get("scan_exam", "End Semester Examination 2026"))
        if st.button("Start scanning this student", type="primary", disabled=not (reg and subj and exam)):
            r = send("POST", "/scanner/sessions", json={"registration_number": reg, "subject_code": subj, "exam_name": exam})
            if r:
                st.session_state.scan_sid, st.session_state.scan_exam = r["id"], exam
                st.rerun()
    if open_sessions:
        labels = {s["id"]: f"#{s['id']} · {s['registration_number']} {s['student_name']} · {s['subject_code']} · "
                           f"{s['exam_name']} · {len(s['pages'])} page(s)" for s in open_sessions}
        ids = list(labels)
        sid = st.selectbox("Open session", ids, index=ids.index(sid) if sid in ids else 0, format_func=labels.get)
        st.session_state.scan_sid = sid
    if not sid:
        st.info("Start a session above. Each session belongs to exactly one student, so pages can never be mixed.")
        return
    s = get(f"/scanner/sessions/{sid}")
    if not s:
        return
    st.markdown(f"""<div class="card"><b>Scanning:</b> {s['student_name']} · <b>{s['registration_number']}</b> · batch
        {s['batch_code']} · {s['subject_code']} · {s['exam_name']} — <b>{len(s['pages'])}</b> page(s)</div>""", unsafe_allow_html=True)
    t_cam, t_up = st.tabs(["📷 Camera", "📁 Upload images"])
    with t_cam:
        st.caption("Place the page on a dark surface, all four corners visible. The page is auto-cropped, flattened, "
                   "rotated and enhanced.")
        shot = st.camera_input("Capture page", key=f"cam_{sid}_{len(s['pages'])}")
        if shot is not None:
            r = _req("POST", f"/scanner/sessions/{sid}/pages", files={"file": ("camera.jpg", shot.getvalue(), "image/jpeg")})
            if r is not None and r.status_code < 300:
                st.rerun()
            st.error(_detail(r))
    with t_up:
        ups = st.file_uploader("Page images (JPG/PNG) — added in the order selected", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key=f"up_{sid}_{st.session_state.get('up_round', 0)}")
        if ups and st.button(f"Add {len(ups)} page(s)"):
            errs = []
            for f in ups:
                r = _req("POST", f"/scanner/sessions/{sid}/pages", files={"file": (f.name, f.getvalue(), f.type or "image/jpeg")})
                if r is None or r.status_code >= 300:
                    errs.append(f"{f.name}: {_detail(r)}")
            st.session_state.scan_errors = errs
            st.session_state.up_round = st.session_state.get("up_round", 0) + 1
            st.rerun()
    for e in st.session_state.pop("scan_errors", []) or []:
        st.error(e)
    pages = s["pages"]
    if pages:
        st.markdown("#### Review pages")
        cols = st.columns(4)
        for i, p in enumerate(pages):
            with cols[i % 4]:
                img = get_bytes(p["image_url"])
                if img:
                    st.image(img, caption=f"Page {p['position']} · quality {p['quality'].get('score', '?')}/100",
                             use_container_width=True)
                for w in p["quality"].get("warnings", []):
                    st.caption(f"⚠️ {w}")
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("↑", key=f"up{p['id']}", disabled=i == 0, help="Move earlier"):
                    order = [x["id"] for x in pages]
                    order[i - 1], order[i] = order[i], order[i - 1]
                    send("PUT", f"/scanner/sessions/{sid}/order", json={"page_ids": order})
                    st.rerun()
                if b2.button("↓", key=f"dn{p['id']}", disabled=i == len(pages) - 1, help="Move later"):
                    order = [x["id"] for x in pages]
                    order[i + 1], order[i] = order[i], order[i + 1]
                    send("PUT", f"/scanner/sessions/{sid}/order", json={"page_ids": order})
                    st.rerun()
                if b3.button("⟳", key=f"rot{p['id']}", help="Rotate 90°"):
                    send("POST", f"/scanner/sessions/{sid}/pages/{p['id']}/rotate", json={"degrees": 90})
                    st.rerun()
                if b4.button("🗑", key=f"del{p['id']}", help="Delete page"):
                    send("DELETE", f"/scanner/sessions/{sid}/pages/{p['id']}")
                    st.rerun()
        c1, c2, c3 = st.columns(3)
        if c1.button("👁️ Build preview PDF", use_container_width=True):
            pdf = get_bytes(f"/scanner/sessions/{sid}/preview.pdf")
            if pdf:
                c1.download_button("⬇️ Open preview", pdf, file_name=f"preview_{s['registration_number']}.pdf", mime="application/pdf")
        if c2.button(f"✅ Submit {len(pages)} page(s) for {s['registration_number']}", type="primary", use_container_width=True):
            r = send("POST", f"/scanner/sessions/{sid}/submit")
            if r:
                st.session_state.scan_done = r
                st.session_state.scan_sid = None
                st.rerun()
        if c3.button("Discard session", use_container_width=True):
            if send("POST", f"/scanner/sessions/{sid}/cancel"):
                st.session_state.scan_sid = None
                st.rerun()


def page_omr():
    _hdr("OMR Evaluation", "Answer keys → printable sheets → scan → automatic marking → verify unclear bubbles.",
         crumb="Teacher · OMR")
    keys = get("/omr/keys") or []
    t_key, t_print, t_scan, t_verify, t_res = st.tabs(["🔑 Answer keys", "🖨️ Print sheets", "📷 Scan sheets",
                                                      "⚠️ Verify flagged", "📊 Results"])
    with t_key:
        if keys:
            st.dataframe(_df(keys, ["id", "subject_code", "exam_name", "num_questions", "num_options", "answers",
                                    "marks_correct", "marks_wrong", "multiple_policy", "allow_negative_total", "max_score"]),
                         use_container_width=True, hide_index=True)
        with st.form("omr_key"):
            c1, c2 = st.columns(2)
            subj = c1.selectbox("Subject", _my_subjects())
            exam = c2.text_input("Examination name", "End Semester Examination 2026")
            ans = st.text_area("Correct answers in order (e.g. ABCDA BDCAB …)", height=80)
            c3, c4, c5 = st.columns(3)
            nopt = c3.selectbox("Options per question", [4, 5, 3, 2])
            mc = c4.number_input("Marks per correct", 0.25, 100.0, 1.0, 0.25)
            mw = c5.number_input("Negative marks per wrong", 0.0, 100.0, 0.0, 0.25)
            c6, c7 = st.columns(2)
            pol_lbl = {"flag": "Flag for me to decide", "wrong": "Count as wrong (negative marking applies)",
                       "zero": "Zero marks, no penalty"}
            pol = c6.selectbox("Two or more bubbles filled in one question", list(pol_lbl), format_func=pol_lbl.get)
            neg = c7.checkbox("Allow a negative total score", value=False,
                              help="Off: a sheet's total never goes below 0.")
            if st.form_submit_button("💾 Save answer key"):
                if send("POST", "/omr/keys", json={"subject_code": subj, "exam_name": exam, "answers": ans, "num_options": nopt,
                                                   "marks_correct": mc, "marks_wrong": mw, "multiple_policy": pol,
                                                   "allow_negative_total": neg}, ok_msg="Answer key saved."):
                    st.rerun()
    if not keys:
        for t in (t_print, t_scan, t_verify, t_res):
            with t:
                st.info("No answer key yet. Create one in the 🔑 Answer keys tab first — printing, scanning and "
                        "results all work per answer key.")
        return
    klabel = {k["id"]: f"#{k['id']} {k['subject_code']} · {k['exam_name']} ({k['num_questions']} Q)" for k in keys}
    with t_print:
        kid = st.selectbox("Answer key", list(klabel), format_func=klabel.get, key="pk")
        c1, c2 = st.columns(2)
        batch = c1.selectbox("Personalised sheets for batch", ["(none)"] + [b["batch_code"] for b in _batches()])
        blank = c2.number_input("Extra blank sheets", 0, 200, 0)
        if st.button("Generate printable PDF"):
            pdf = get_bytes(f"/omr/keys/{kid}/sheets.pdf", batch_code=None if batch == "(none)" else batch, blank=blank)
            if pdf:
                st.download_button("⬇️ Download OMR sheets", pdf, file_name=f"omr_sheets_{kid}.pdf", mime="application/pdf")
        st.caption("Personalised sheets carry a QR code with the registration number, so the right student is identified "
                   "automatically when scanned.")
    with t_scan:
        kid = st.selectbox("Answer key", list(klabel), format_func=klabel.get, key="sk")
        st.caption("Use a sheet printed from 🖨️ Print sheets for this key. Photograph the WHOLE sheet with all 4 black "
                   "corner squares visible — any angle or orientation works. Phone photos and scans work best.")
        c1, c2 = st.columns([2, 1])
        reg = c1.text_input("Registration number (only if the sheet is blank or its QR code can't be read)")
        replace = c2.checkbox("Replace an existing scan for this student")
        src = st.radio("Source", ["Upload photo / scan (JPG, PNG, PDF)", "Camera"], horizontal=True)
        files = []
        if src == "Camera":
            st.caption("Laptop webcams are often too low-resolution for OMR; if reading fails, upload a phone photo instead.")
            shot = st.camera_input("Photograph the whole OMR sheet", key=f"omrcam_{st.session_state.get('omr_round', 0)}")
            if shot:
                files = [("camera.jpg", shot.getvalue(), "image/jpeg")]
        else:
            ups = st.file_uploader("OMR sheets (you can select several)", type=["jpg", "jpeg", "png", "pdf"],
                                   accept_multiple_files=True, key=f"omrup_{st.session_state.get('omr_round', 0)}")
            files = [(f.name, f.getvalue(), f.type) for f in ups or []]
        if files and st.button(f"Read {len(files)} sheet(s)", type="primary"):
            out = []
            with st.spinner("Reading bubbles…"):
                for name, data, typ in files:
                    if data[:4] == b"%PDF":     # a PDF may hold a whole stack of sheets → one result per page
                        r = _req("POST", f"/omr/keys/{kid}/scan-batch", files=[("files", (name, data, "application/pdf"))],
                                 data={"replace": str(replace).lower()}, timeout=900)
                        if r is not None and r.status_code == 200:
                            for it in r.json()["results"]:
                                if it.get("ok"):
                                    msg = f"**{it['source']}** → **{it['registration_number']}**: {it['score']:g}/{it['max_score']:g}"
                                    out.append(("warning", msg + f" · ⚠️ {it['flagged']} unclear → ⚠️ Verify flagged")
                                               if it["flagged"] else ("success", msg))
                                else:
                                    out.append(("error", f"**{it['source']}**: {it['error']}"))
                        else:
                            out.append(("error", f"**{name}**: {_detail(r, 'Could not read the PDF')}"))
                        continue
                    r = _req("POST", f"/omr/keys/{kid}/scan", files={"file": (name, data, typ or "image/jpeg")},
                             data={"registration_number": reg.strip(), "replace": str(replace).lower()}, timeout=180)
                    if r is not None and r.status_code == 200:
                        d = r.json()
                        msg = (f"**{name}** → **{d['registration_number']}** {d['student_name']}: "
                               f"{d['score']:g}/{d['max_score']:g} — {d['correct']} correct, {d['incorrect']} wrong, "
                               f"{d['unanswered']} blank")
                        out.append(("warning", msg + f" · ⚠️ {d['flagged']} unclear bubble(s) → open ⚠️ Verify flagged")
                                   if d["needs_verification"] else ("success", msg))
                    else:
                        out.append(("error", f"**{name}**: {_detail(r, 'Could not read the sheet')}"))
            st.session_state.omr_last = out
            st.session_state.omr_round = st.session_state.get("omr_round", 0) + 1
            st.rerun()
        for kind, msg in st.session_state.get("omr_last", []):
            getattr(st, kind)(msg)
        if st.session_state.get("omr_last") and st.button("Clear messages"):
            st.session_state.omr_last = []
            st.rerun()
    with t_verify:
        flagged = get("/omr/responses", needs_verification="true") or []
        if not flagged:
            st.success("No OMR sheets need verification.")
        for r in flagged:
            det = get(f"/omr/responses/{r['id']}")
            if not det:
                continue
            with st.expander(f"{r['registration_number']} {r['student_name']} · {r['subject_code']} · {r['flagged']} unclear", expanded=True):
                img = get_bytes(det["image_url"])
                c1, c2 = st.columns([1, 1.2])
                if img and img[:4] != b"%PDF":
                    c1.image(img, use_container_width=True)
                with c2.form(f"verify_{r['id']}"):
                    choices = {}
                    for q in det["questions"]:
                        if q["result"] == "flagged":
                            fills = ", ".join(f"{'ABCDE'[i]}:{v:.0%}" for i, v in enumerate(q["fill"]))
                            choices[q["q"]] = st.selectbox(
                                f"Q{q['q']} — {q['reading']} (marked {q['marked'] or '-'}, light {q['light'] or '-'}; fill {fills})",
                                ["", *"ABCDE"[:len(q["fill"])]], format_func=lambda x: x or "No valid answer (blank / invalid)",
                                key=f"v{r['id']}_{q['q']}")
                    note = st.text_input("Note", key=f"vn{r['id']}")
                    if st.form_submit_button("Save verification"):
                        if send("POST", f"/omr/responses/{r['id']}/verify", json={"answers": {str(k): v for k, v in choices.items()},
                                                                                     "note": note}, ok_msg="Verified."):
                            st.rerun()
    with t_res:
        kid = st.selectbox("Answer key", list(klabel), format_func=klabel.get, key="rk")
        rows = get("/omr/responses", key_id=kid) or []
        st.dataframe(_df(rows, ["registration_number", "student_name", "score", "max_score", "correct", "incorrect",
                                "unanswered", "flagged", "needs_verification", "verified"]), use_container_width=True, hide_index=True)
        if rows:
            sel = st.selectbox("Per-question report", [r["id"] for r in rows],
                               format_func=lambda i: next(f"{r['registration_number']} {r['student_name']}" for r in rows if r["id"] == i))
            det = get(f"/omr/responses/{sel}")
            if det:
                st.dataframe(pd.DataFrame([{"Q": q["q"], "Marked": ",".join(q["marked"]) or "—", "Final": q["final_answer"] if q["final_answer"] is not None else "?",
                                            "Key": q["key"], "Result": q["result"], "Reading": q["reading"]} for q in det["questions"]]),
                             use_container_width=True, hide_index=True)


def page_results_publish():
    _hdr("Results Compilation & Publishing", "Combine verified OMR marks and teacher-approved written marks into one "
         "result per student, approve, then publish to students.", crumb="Results")
    subs = _my_subjects()
    mode = st.radio("Compile for", ["Whole batch", "One student"], horizontal=True)
    with st.form("compile"):
        c1, c2, c3 = st.columns(3)
        subj = c1.selectbox("Subject", subs)
        exam = c2.text_input("Examination name", "End Semester Examination 2026")
        target = c3.selectbox("Batch", [b["batch_code"] for b in _batches()]) if mode == "Whole batch" else c3.text_input("Registration number")
        if st.form_submit_button("🧮 Compile results"):
            body = {"subject_code": subj, "exam_name": exam}
            body["batch_code" if mode == "Whole batch" else "registration_number"] = target
            r = send("POST", "/results/compile", json=body)
            if r is not None:
                st.success(f"Compiled {len(r['compiled'])} result(s).")
                for p in r["not_compiled"]:
                    st.warning(f"{p['registration_number']}: {p['error']}")
    status = st.radio("Show", ["draft", "approved", "published", "all"], horizontal=True)
    rows = get("/results/final", status=None if status == "all" else status) or []
    if not rows:
        st.info("No consolidated results with this status.")
        return
    st.dataframe(_df(rows, ["id", "registration_number", "student_name", "course_code", "exam_name", "omr_score", "omr_max",
                            "subjective_score", "subjective_max", "total", "max_total", "percentage", "grade", "passed", "status"]),
                 use_container_width=True, hide_index=True)
    ids = st.multiselect("Select results", [r["id"] for r in rows],
                         format_func=lambda i: next(f"#{i} {r['registration_number']} {r['course_code']} {r['total']:g}/{r['max_total']:g}" for r in rows if r["id"] == i))
    c1, c2 = st.columns(2)
    if c1.button(f"✅ Approve {len(ids)} selected", disabled=not ids):
        for i in ids:
            send("POST", f"/results/final/{i}/approve")
        st.rerun()
    if c2.button(f"🚀 Publish {len(ids)} selected", disabled=not ids):
        for i in ids:
            send("POST", f"/results/final/{i}/publish")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT
# ═══════════════════════════════════════════════════════════════════════════════

def page_student_dashboard():
    d = get("/results/my/dashboard")
    if d is None:
        st.error(st.session_state.get("_last_err"))
        return
    p = d["profile"]
    _hdr(f"Hello, {p['name'].split(' ')[0]}", f"{p['registration_number']} · {p['programme']} · Batch {p['batch_code'] or '—'} · "
         f"Section {p['section'] or '—'} · {p['semester']} · {p['academic_session'] or ''}", crumb="Student · Dashboard")
    s = d["summary"]
    _kpis([{"label": "Subjects", "value": d["total_subjects"], "hint": "registered"},
           {"label": "Overall %", "value": _fmt(s["average_percentage"], "%"), "hint": f"{s['graded']} published results", "tone": "cyan"},
           {"label": "Passed / failed", "value": f"{s['passed']} / {s['failed']}", "hint": f"pass mark {d['pass_mark_percent']:g}%", "tone": "green"},
           {"label": "CGPA", "value": _fmt(d["cgpa"]), "hint": "credit-weighted", "tone": "violet"},
           {"label": "Quiz average", "value": _fmt(d["quiz"]["average_percentage"], "%"), "hint": f"{d['quiz']['attempts']} attempts", "tone": "amber"}])
    st.markdown("#### Subject-wise results (published)")
    if d["results"]:
        st.dataframe(_df(d["results"], ["subject_code", "subject_name", "exam_name", "semester", "omr_marks", "subjective_marks",
                                         "marks", "max_marks", "percentage", "grade", "passed"]), use_container_width=True, hide_index=True)
        df = pd.DataFrame([r for r in d["results"] if r.get("percentage") is not None])
        if not df.empty:
            _chart(alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=46).encode(
                x=alt.X("subject_code:N", title=None), y=alt.Y("percentage:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                color=alt.condition(alt.datum.passed, alt.value("#22c55e"), alt.value("#ef4444")),
                tooltip=["subject_code", "exam_name", "marks", "max_marks", "grade"]).properties(height=220))
        if st.button("📄 Prepare final result PDF"):
            pdf = get_bytes("/results/my/final.pdf")
            if pdf:
                st.download_button("⬇️ Download final approved result", pdf, file_name="final_result.pdf", mime="application/pdf")
    else:
        st.info("No published results yet.")
    if d["semester_wise"]:
        st.markdown("#### Semester-wise")
        st.dataframe(_df(d["semester_wise"], ["semester", "graded", "passed", "failed", "average_percentage"]), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Evaluation in progress")
        if d["evaluation_in_progress"]:
            st.dataframe(_df(d["evaluation_in_progress"]), use_container_width=True, hide_index=True)
            st.caption("Marks stay hidden until a teacher approves and the result is published.")
        else:
            st.caption("Nothing pending.")
    with c2:
        st.markdown("#### Weak topics & study roadmap")
        for w in d["weak_topics"]:
            st.markdown(f"- {w['topic']} — quiz accuracy {w['accuracy']:g}%")
        if not d["weak_topics"]:
            st.caption("No weak topics recorded yet.")
        if d["roadmap"]["exists"]:
            st.progress(min(1.0, (d["roadmap"]["progress_pct"] or 0) / 100), text=f"Roadmap progress {d['roadmap']['progress_pct'] or 0:g}%")


# ═══════════════════════════════════════════════════════════════════════════════
# ALL ROLES
# ═══════════════════════════════════════════════════════════════════════════════

def page_notifications():
    _hdr("Notifications", "Results, evaluations, feedback, roadmaps and quiz results.", crumb="Notifications")
    n = get("/notifications") or {"items": [], "unread": 0}
    if n["unread"] and st.button(f"Mark all {n['unread']} as read"):
        send("POST", "/notifications/read-all")
        st.rerun()
    if not n["items"]:
        _C["theme"].empty_state("No notifications yet", "", "🔔")
    icons = {"result_published": "📢", "sheet_evaluated": "📝", "feedback": "💬", "roadmap": "🗺️", "quiz": "🎯"}
    for it in n["items"]:
        st.markdown(f"""<div class="card" style="{'border-color:#5b7cff' if not it['is_read'] else 'opacity:.75'}">
          <b>{icons.get(it['kind'], '🔔')} {it['title']}</b>{' <span class="pill pill-b">NEW</span>' if not it['is_read'] else ''}
          <div style="color:var(--muted);font-size:.85rem">{it['body']}</div>
          <div style="color:var(--muted);font-size:.72rem">{it['created_at'][:16].replace('T', ' ')} UTC</div></div>""",
                    unsafe_allow_html=True)
    if any(not it["is_read"] for it in n["items"]):
        send("POST", "/notifications/read-all")


def unread_count():
    n = get("/notifications", unread_only="true")
    return n["unread"] if n else 0



# ═══════════════════════════════════════════════════════════════════════════════
# Teacher / admin: official question paper, model answers and marking scheme
# ═══════════════════════════════════════════════════════════════════════════════

QTYPE_LABELS = {"open_ended": "Long answer", "short_answer": "Short answer", "numerical": "Numerical",
                "diagram": "Diagram", "fill_blank": "Fill in the blank", "mcq": "MCQ", "true_false": "True / False"}


def _rubric_text(rubric):
    return "\n".join(f"{c['criterion']} | {c['marks']:g}" for c in rubric or [])


def _parse_rubric(text):
    out, bad = [], []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        if "|" not in line:
            bad.append(line); continue
        crit, marks = line.rsplit("|", 1)
        try:
            out.append({"criterion": crit.strip(), "marks": float(marks.strip())})
        except ValueError:
            bad.append(line)
    return out, bad


def page_exam_papers():
    _hdr("Question Papers, Model Answers & Marking Schemes",
         "The AI marks every answer against what you enter here: the question, its marks, the model answer and the "
         "marking scheme (criteria with marks). One paper per subject + examination.", crumb="Teacher · Question papers")
    subs = _my_subjects()
    if not subs:
        st.info("You have no subjects assigned yet.")
        return
    c1, c2 = st.columns([1, 2])
    subj = c1.selectbox("Subject", subs, key="qp_subj")
    papers = get("/exam-papers", course_code=subj) or []
    opts = {p["id"]: f"{p['exam_name']} · {p['num_questions']} Q · {p['total_marks']:g} marks"
            + ("" if p["ready_for_evaluation"] else " · ⚠️ incomplete") for p in papers}
    goto = st.session_state.pop("qp_goto", None)          # set after creating/importing a paper
    if goto in opts:
        st.session_state.qp_sel = goto
    sel = c2.selectbox("Examination", [None] + list(opts), format_func=lambda i: "➕ New examination paper" if i is None else opts[i],
                       key="qp_sel")
    if sel is None:
        t_manual, t_pdf = st.tabs(["✍️ Type the paper", "📤 Import from PDF"])
        with t_manual:
            with st.form("qp_new"):
                exam = st.text_input("Examination name (must match the name used when scanning answer sheets)",
                                     "End Semester Examination 2026")
                n = st.number_input("Number of questions", 1, 60, 3)
                m = st.number_input("Marks per question (you can change each question afterwards)", 0.5, 100.0, 10.0, 0.5)
                if st.form_submit_button("Create paper", type="primary"):
                    d = send("POST", "/exam-papers", json={"course_code": subj, "exam_name": exam, "questions": [
                        {"question_number": i + 1, "question_text": "", "marks": m} for i in range(int(n))]},
                        ok_msg="Paper created — now fill in each question below.")
                    if d:
                        st.session_state.qp_goto = d["id"]
                        st.rerun()
        with t_pdf:
            with st.form("qp_pdf"):
                exam = st.text_input("Examination name", "End Semester Examination 2026", key="qp_pdf_exam")
                f = st.file_uploader("Question paper PDF (typed or scanned)", type=["pdf"])
                k = st.file_uploader("Answer key / model answers PDF (optional)", type=["pdf"])
                if st.form_submit_button("Import", type="primary") and f:
                    with st.spinner("Reading the question paper…"):
                        d = send("POST", "/exam-papers/import-pdf", files={"file": (f.name, f.getvalue(), "application/pdf")},
                                 data={"course_code": subj, "exam_name": exam}, timeout=300)
                    if d and k:
                        with st.spinner("Reading the answer key…"):
                            send("POST", f"/exam-papers/{d['id']}/answer-key-pdf",
                                 files={"file": (k.name, k.getvalue(), "application/pdf")}, timeout=300)
                    if d:
                        st.success(f"Imported {d['num_questions']} question(s). Check them below.")
                        st.session_state.qp_goto = d["id"]
                        st.rerun()
        return

    p = get(f"/exam-papers/{sel}")
    if not p:
        st.error(st.session_state.get("_last_err", "Could not load the paper"))
        return
    _kpis([{"label": "Questions", "value": p["num_questions"], "hint": p["exam_name"], "tone": ""},
           {"label": "Total marks", "value": p["total_marks"], "hint": "sum of questions", "tone": "cyan"},
           {"label": "Answer sheets linked", "value": p["answer_sheets_linked"], "hint": "same subject + exam", "tone": "violet"},
           {"label": "Ready for AI marking", "value": "Yes" if p["ready_for_evaluation"] else "No",
            "hint": f"{len(p['warnings'])} issue(s)", "tone": "green" if p["ready_for_evaluation"] else "amber"}])
    for w in p["warnings"]:
        st.warning(w)
    with st.expander("📤 Attach model answers from an answer-key PDF"):
        k = st.file_uploader("Answer key PDF", type=["pdf"], key=f"ak_{sel}")
        if k and st.button("Read answer key", key=f"akb_{sel}"):
            with st.spinner("Reading…"):
                if send("POST", f"/exam-papers/{sel}/answer-key-pdf", files={"file": (k.name, k.getvalue(), "application/pdf")},
                        ok_msg="Model answers attached.", timeout=300):
                    st.rerun()
    st.caption("Marking scheme: one criterion per line as `criterion | marks`, e.g. `Defines deadlock correctly | 4`. "
               "Without a scheme the AI uses the key points of the model answer.")
    for q in p["questions"]:
        qn = q["question_number"]
        title = (f"Q{qn} · {q['marks']:g} marks · {QTYPE_LABELS.get(q['question_type'], q['question_type'])}"
                 + (f" — {q['question_text'][:70]}" if q["question_text"] else " — (question text missing)"))
        with st.expander(title, expanded=not (q["model_answer"] or q["rubric"] or q["correct_option"])):
            with st.form(f"q_{sel}_{qn}"):
                text = st.text_area("Question", q["question_text"], height=80)
                c1, c2, c3 = st.columns(3)
                marks = c1.number_input("Marks", 0.5, 200.0, float(q["marks"] or 1), 0.5)
                qt = c2.selectbox("Type", list(QTYPE_LABELS), index=list(QTYPE_LABELS).index(q["question_type"])
                                  if q["question_type"] in QTYPE_LABELS else 0, format_func=QTYPE_LABELS.get)
                opt = c3.text_input("Correct option (MCQ: A–E, True/False: T or F)", q["correct_option"])
                model = st.text_area("Model answer", q["model_answer"], height=140)
                rub = st.text_area("Marking scheme (criterion | marks)", _rubric_text(q["rubric"]), height=110)
                b1, b2 = st.columns([1, 1])
                save = b1.form_submit_button("💾 Save question", type="primary")
                delete = b2.form_submit_button("🗑️ Remove question")
            if save:
                rubric, bad = _parse_rubric(rub)
                if bad:
                    st.error(f"These lines need the form `criterion | marks`: {bad}")
                elif send("PUT", f"/exam-papers/{sel}/questions/{qn}", json={
                        "question_number": qn, "question_text": text, "marks": marks, "question_type": qt,
                        "model_answer": model, "correct_option": opt, "rubric": rubric}, ok_msg=f"Q{qn} saved."):
                    st.rerun()
            if delete and send("DELETE", f"/exam-papers/{sel}/questions/{qn}", ok_msg=f"Q{qn} removed."):
                st.rerun()
    with st.form(f"addq_{sel}"):
        nxt = max([q["question_number"] for q in p["questions"]] or [0]) + 1
        c1, c2 = st.columns([1, 3])
        qn = c1.number_input("Add question number", 1, 200, nxt)
        m = c2.number_input("Marks", 0.5, 200.0, 10.0, 0.5, key=f"addm_{sel}")
        if st.form_submit_button("➕ Add question"):
            if send("PUT", f"/exam-papers/{sel}/questions/{int(qn)}", json={"question_number": int(qn), "question_text": "",
                                                                             "marks": m}, ok_msg="Added."):
                st.rerun()


def ai_status_banner():
    """Tell teachers honestly which engines are reading and marking."""
    s = get("/system/ai-status")
    if not s:
        return
    line = f"Handwriting reader: **{s['handwriting_reader']}** · Marking: **{s['grader']}**"
    if s["vision_ai"] and s["semantic_grading"]:
        st.caption("🤖 " + line)
    else:
        st.warning("⚠️ " + line + "\n\n" + " ".join(s["notes"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher: review AI evaluation → correct → approve → publish
# ═══════════════════════════════════════════════════════════════════════════════

_STATUS_ICON = {"met": "✅", "partial": "🟡", "missing": "❌", "incorrect": "⛔"}


def _breakdown_table(rb):
    if not rb:
        return
    st.dataframe(pd.DataFrame([{"": _STATUS_ICON.get(r.get("status"), ""), "Marking point": r.get("criterion", ""),
                                "Avail.": r.get("max_marks"), "Given": r.get("awarded"),
                                "Why": (r.get("reason") or "") + (f" — “{r['evidence']}”" if r.get("evidence") else "")}
                               for r in rb]), use_container_width=True, hide_index=True)


def page_examiner_review():
    _hdr("Examiner Review & Moderation",
         "Read the sheet + AI evaluation → check each answer against the original handwriting → adjust marks with a "
         "reason → approve → publish. You only see students you are assigned to.", crumb="Review")
    ai_status_banner()
    f1, f2 = st.columns([1, 1])
    status = f1.selectbox("Status", ["all", "pending", "in_review", "approved", "published"])
    course = f2.text_input("Subject code filter")
    booklets = get("/examiner/booklets", approval_status=None if status == "all" else status,
                   course_code=course.strip().upper() or None) or []
    if not booklets:
        _C["theme"].empty_state("No answer sheets in your queue",
                                "Scan sheets in “Answer Sheet Scanner” or upload them below.", "📝")
    else:
        labels = {b["id"]: f"#{b['id']} · {b['student_name']} ({b['registration_number']}) — {b['course_code']} · "
                           f"{b['exam_name']} [{b['approval_status'].upper()}]" for b in booklets}
        bid = st.selectbox("Answer sheet", list(labels), format_func=labels.get)
        d = get(f"/examiner/booklet/{bid}")
        if d:
            _review_booklet(bid, d)
    with st.expander("📤 Upload a scanned answer sheet (PDF / image)"):
        with st.form("fac_upload"):
            u1, u2, u3 = st.columns(3)
            reg = u1.text_input("Registration number")
            sub = u2.text_input("Subject code")
            exam = u3.text_input("Examination", "End Semester Examination 2026")
            f = st.file_uploader("Answer sheet", type=["pdf", "jpg", "jpeg", "png"])
            if st.form_submit_button("Upload") and f:
                r = send("POST", "/examiner/booklet/upload", files={"file": (f.name, f.getvalue(), f.type)},
                         data={"student_registration": reg, "course_code": sub, "exam_name": exam})
                if r:
                    st.success(f"Uploaded as answer sheet #{r['booklet_id']}. Select it above and run the AI evaluation.")


def _review_booklet(bid, d):
    published = d["is_published"]
    flagged = [q for q in d["questions"] if q["flagged_for_review"] and not q["examiner_approved"]]
    _kpis([{"label": "Student", "value": d["student"]["registration_number"], "hint": d["student"]["name"], "tone": ""},
           {"label": "AI total", "value": d["total_ai_score"], "hint": f"of {d['max_total_marks']:g}", "tone": "cyan"},
           {"label": "Final total", "value": d["total_final_score"], "hint": "teacher marks where set", "tone": "violet"},
           {"label": "Needs your check", "value": len(flagged), "hint": d["approval_status"].replace("_", " "),
            "tone": "amber" if flagged else "green"}])
    if not d.get("exam_paper_id"):
        st.warning(f"No question paper is linked for {d['course_code']} · {d['exam_name']}. Create it in "
                   "“Question Papers & Model Answers” (same examination name) before evaluating.")
    rep = d.get("ocr_report") or {}
    if d.get("ocr_engine"):
        st.caption(f"Read with: {d['ocr_engine']} · {d.get('total_pages') or rep.get('pages') or '?'} page(s)")
    for w in rep.get("warnings", [])[:4]:
        st.caption(f"⚠️ {w}")
    if rep.get("unassigned_text"):
        with st.expander("Text that could not be matched to a question"):
            st.text(rep["unassigned_text"])
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("🤖 Read + AI evaluate", use_container_width=True, disabled=published,
                 help="Reads the handwriting (if not read yet) and marks every answer."):
        with st.spinner("Reading the handwriting and marking against the question paper…"):
            r = send("POST", f"/examiner/evaluate-booklet/{bid}", timeout=1800)
        if r:
            st.success(f"AI total {r['total_ai_score']:g} / {r['max_marks']:g} · read by {r['reader']} · marked by {r['grader']}")
            if r.get("skipped_questions"):
                st.warning(f"Not evaluated (no model answer / marking scheme): questions {r['skipped_questions']}")
            st.rerun()
    if c2.button("🔁 Re-read sheet", use_container_width=True, disabled=published,
                 help="Discards the current transcription and marks, and reads the sheet again (e.g. after adding an AI key)."):
        with st.spinner("Re-reading…"):
            if send("POST", f"/examiner/evaluate-booklet/{bid}", params={"reread": "true"}, timeout=1800, ok_msg="Re-read and re-evaluated."):
                st.rerun()
    if c3.button("✅ Approve marks", use_container_width=True, disabled=published):
        if send("POST", f"/examiner/approve-booklet/{bid}", ok_msg="Approved."):
            st.rerun()
    if c4.button("🚀 Publish", use_container_width=True, disabled=published or d["approval_status"] != "approved"):
        if send("POST", f"/examiner/publish-booklet/{bid}", ok_msg="Published to the student portal."):
            st.rerun()
    if c5.button("📄 Evaluated PDF", use_container_width=True):
        pdf = get_bytes(f"/examiner/booklet/{bid}/evaluated.pdf")
        if pdf:
            c5.download_button("⬇️ Save", pdf, file_name=f"evaluated_{bid}.pdf", mime="application/pdf")
    o1, _ = st.columns([1, 4])
    if o1.button("📄 Original sheet"):
        pdf = get_bytes(f"/student/answer-sheet/{bid}/original")
        if pdf:
            o1.download_button("⬇️ Save original", pdf, file_name=f"original_{bid}.pdf", mime="application/pdf")
    if not d["questions"]:
        st.info("No answers read yet — click “Read + AI evaluate”.")
    for q in d["questions"]:
        _review_question(bid, q, published)


def _review_question(bid, q, published):
    qn = q["question_number"]
    final = q["awarded"]
    head = (f"Q{qn} — AI {_fmt(q['ai_score'])} · final {_fmt(final)} / {q['max_marks']:g}"
            + (" · ⚠️ CHECK" if q["flagged_for_review"] and not q["examiner_approved"] else "")
            + (" · ✔ verified" if q["examiner_approved"] else "")
            + ("" if (q["student_answer"] or "").strip() else " · not attempted"))
    with st.expander(head, expanded=bool(q["flagged_for_review"] and not q["examiner_approved"])):
        if q.get("question_text"):
            st.markdown(f"**Question:** {q['question_text']}")
        if q.get("flag_reason"):
            for reason in q["flag_reason"].split(" | "):
                st.warning(reason)
        refs = q.get("page_refs") or []
        ca, cb = st.columns([1.1, 1])
        with ca:
            st.markdown("**Original handwriting**")
            if refs:
                for ref in refs[:3]:
                    img = get_bytes(f"/examiner/booklet/{bid}/page/{ref['page']}.png", y0=ref["y0"], y1=ref["y1"])
                    if img:
                        st.image(img, caption=f"page {ref['page']}", use_container_width=True)
            else:
                st.caption("Location on the sheet unknown — open the original sheet.")
        with cb:
            conf = q.get("ocr_confidence")
            st.markdown("**What was read (and marked)**" + (f" · reading confidence {conf:.0%}" if conf is not None else ""))
            st.text_area("read", q["student_answer"] or "(no answer found on the sheet)", height=170, disabled=True,
                         key=f"sa_{bid}_{qn}", label_visibility="collapsed")
            ref_txt = (q.get("expected_answer") or "") or ("Correct option: " + q["correct_option"] if q.get("correct_option") else "")
            if ref_txt or q.get("marking_criteria"):
                with st.popover("Model answer & marking scheme"):
                    st.write(ref_txt or "(no model answer)")
                    for c in q.get("marking_criteria") or []:
                        st.write(f"• {c['criterion']} ({c['marks']:g})")
        _breakdown_table(q.get("rubric_breakdown"))
        for dd in q.get("deductions") or []:
            st.markdown(f"<span class='pill pill-r'>−{_fmt(dd.get('marks'))} {dd.get('criterion') or ''}</span> {dd.get('reason') or ''}",
                        unsafe_allow_html=True)
        for cr in q.get("corrections") or []:
            st.markdown(f"✏️ *“{cr.get('student_wrote', '')}”* → **{cr.get('correct_version', '')}** {cr.get('explanation', '')}")
        if q.get("feedback"):
            st.caption(f"AI feedback ({q.get('provider') or q.get('method') or 'AI'}): {q['feedback']}")
        if published:
            return
        if q.get("result_id"):
            m1, m2 = st.columns([1, 2])
            new = m1.number_input(f"Final marks (Q{qn})", 0.0, float(q["max_marks"]), float(final or 0), 0.5, key=f"score_{bid}_{qn}")
            note = m2.text_input("Reason / comment for the student (required if you change the AI marks)",
                                 q["examiner_comments"] or "", key=f"notes_{bid}_{qn}")
            fl = st.checkbox("Keep flagged for a second examiner", value=False, key=f"flag_{bid}_{qn}")
            if st.button(f"💾 Save Q{qn} (marks verified)", key=f"btn_{bid}_{qn}"):
                if send("POST", "/examiner/review-question", json={"result_id": q["result_id"], "examiner_score": new,
                                                                  "examiner_comments": note, "flag_for_review": fl,
                                                                  "flag_reason": "Second examiner requested" if fl else ""},
                        ok_msg=f"Q{qn} saved."):
                    st.rerun()
        with st.popover("✏️ Correct the transcription & re-mark"):
            fixed = st.text_area("Type exactly what the student wrote", q["student_answer"] or "", height=160, key=f"ocr_{bid}_{qn}")
            if st.button("Save & re-mark", key=f"ocrb_{bid}_{qn}"):
                if send("PATCH", f"/examiner/booklet/{bid}/answer/{q['answer_id']}", json={"corrected_text": fixed},
                        ok_msg="Transcription corrected and re-marked."):
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# AI Examination Assistant (Admin + Faculty) — answers come from /agent/chat (database + local Ollama)
# ═══════════════════════════════════════════════════════════════════════════════

_VOICE_JS = None
_VOICE_TEMPLATE = """<div style="font:13px system-ui,sans-serif;color:#cbd5e1;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
<button id="p" style="cursor:pointer">🔊 Speak</button><button id="pa" hidden style="cursor:pointer">⏸ Pause</button>
<button id="s" hidden style="cursor:pointer">⏹ Stop</button><span id="st" role="status" aria-live="polite"></span></div>
<script>
__JS__
(function(){
  var TEXT = __TEXT__, RL = __RL__;
  var $ = function(i){ return document.getElementById(i); };
  var synth = null; try { synth = window.speechSynthesis; } catch(e) {}
  var vo = PCVoiceOut.create({ synth: synth, Utterance: window.SpeechSynthesisUtterance, onState: function(s){
    var act = ["loading","speaking","retrying","paused"].indexOf(s.state) >= 0;
    $("p").hidden = act; $("pa").hidden = !act; $("s").hidden = !act;
    $("pa").textContent = s.state === "paused" ? "▶ Resume" : "⏸ Pause";
    $("p").textContent = s.state === "error" ? "🔁 Retry voice" : "🔊 Speak";
    $("st").style.color = s.state === "error" ? "#fca5a5" : s.state === "speaking" ? "#86efac" : "#fcd34d";
    $("st").textContent = s.state === "speaking" ? "Speaking" + (s.total > 1 ? " " + s.chunk + "/" + s.total : "")
      : s.state === "paused" ? "Paused" : s.state === "idle" ? "" : (s.message || s.state);
  }});
  $("p").onclick = function(){
    // one voice at a time across every answer on the page (each answer lives in its own iframe)
    try { window.parent.speechSynthesis.cancel(); } catch(e) {}
    var o = {}; if (RL) o.replyLang = RL;
    if (!vo.supported) { $("st").textContent = "Voice is not supported in this browser (use Chrome or Edge). The text answer is unaffected."; return; }
    vo.speak(TEXT, o);
  };
  $("pa").onclick = function(){ vo.state.state === "paused" ? vo.resume() : vo.pause(); };
  $("s").onclick = function(){ vo.stop(); };
  window.addEventListener("beforeunload", function(){ try { vo.stop(); } catch(e) {} });
})();
</script>"""


def _plain_for_speech(md: str) -> str:
    t = re.sub(r"```.*?```", " code omitted. ", md or "", flags=re.S)
    t = re.sub(r"[#*_`>|~\[\]]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def voice_widget(text: str, reply_lang: str = ""):
    """Speak / Pause / Stop for one answer (same tested engine as the student Mentor). Text is never affected."""
    global _VOICE_JS
    import streamlit.components.v1 as components
    if _VOICE_JS is None:
        with open(os.path.join(os.path.dirname(__file__), "voice_out.js"), encoding="utf-8") as f:
            _VOICE_JS = f.read()
    rl = {"English": "english", "Hindi": "hindi", "Hinglish": "hinglish"}.get(reply_lang, "")
    payload = json.dumps(_plain_for_speech(text)).replace("</", "<\\/")
    html = (_VOICE_TEMPLATE.replace("__JS__", _VOICE_JS).replace("__TEXT__", payload).replace("__RL__", json.dumps(rl)))
    components.html(html, height=58)


_ASSISTANT_PROMPTS = {
    "admin": ["Show examination statistics", "Which students need academic support?", "Show pending checking jobs",
              "List re-evaluation requests", "Show faculty checking activity", "Generate an examination report"],
    "faculty": ["How did my classes perform?", "Which students are below 40%?", "What are the weak topics?",
                "Show pending checking", "List re-evaluation requests", "Generate an examination report"],
}


def page_exam_assistant(role):
    """Robot Examination Assistant (frontend/exam_assistant_app.html) for admin, faculty and students.
    The page talks to the FastAPI backend directly with the user's own JWT: chat (/agent, /student-assistant),
    bulk checking (/bulk), camera scanning and downloads are all real backend calls."""
    import streamlit.components.v1 as components
    classic = st.toggle("Classic view", key=f"agent_classic_{role}", value=False,
                        help="Simple Streamlit chat (no robot, camera or bulk checking panel).") if role != "student" else False
    if classic:
        return page_exam_assistant_classic(role)
    st.markdown("<style>.block-container{padding-top:.8rem!important;padding-bottom:0!important;max-width:100%!important;"
                "padding-left:1rem!important;padding-right:1rem!important}iframe{border-radius:14px}</style>",
                unsafe_allow_html=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exam_assistant_app.html")
    with open(path, encoding="utf-8") as f:
        tpl = f.read()
    u = st.session_state.get("current_user") or {}
    cfg = {"apiBase": os.getenv("PUBLIC_API_BASE_URL", _C["api"]), "token": st.session_state.get("auth_token"),
           "role": role, "reduceMotion": bool(st.session_state.get("reduce_motion")),
           "user": {"name": u.get("full_name") or u.get("username") or "", "department": u.get("department") or ""}}
    components.html(tpl.replace("__EXAM_CONFIG__", json.dumps(cfg).replace("</", "<\\/")),
                    height=int(os.getenv("EXAM_ASSISTANT_UI_HEIGHT", "920")), scrolling=False)


def page_exam_assistant_classic(role):
    _hdr("🤖 AI Examination Assistant",
         "Ask about students, marks, deductions, class performance, the checking queue and re-evaluations. "
         "Answers come from the examination database; local Ollama only words them.", "Assistant")
    cap = get("/agent/capabilities")
    if not cap:
        st.error("The assistant is unavailable: " + st.session_state.get("_last_err", "backend not reachable"))
        return
    ai = cap.get("ai", {})
    if ai.get("ready"):
        st.caption(f"🟢 AI model: **{ai.get('model')}** (local Ollama) · Scope: {cap.get('scope')} · read-only")
    else:
        st.warning("🟠 Ollama is not ready — answers will show verified database results without AI wording. "
                   + (ai.get("error") or ""))
    key = f"agent_msgs_{role}"
    st.session_state.setdefault(key, [])
    lang = st.selectbox("Answer language", ["Auto", "English", "Hindi", "Hinglish"], key=f"agent_lang_{role}")
    cols = st.columns(3)
    picked = None
    for i, q in enumerate(_ASSISTANT_PROMPTS.get(role, _ASSISTANT_PROMPTS["faculty"])):
        if cols[i % 3].button(q, key=f"agent_q_{role}_{i}", use_container_width=True):
            picked = q
    if st.button("🧹 Clear conversation", key=f"agent_clear_{role}"):
        st.session_state[key] = []
        st.rerun()
    for i, m in enumerate(st.session_state[key]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                mode = m.get("mode")
                st.caption({"ai": "✨ AI-worded, numbers verified against the database",
                            "data-verified": "✅ Verified database result (AI wording rejected: unverified figures)",
                            "data-only": "📋 Verified database result (AI model not used)",
                            "data": "📋 Verified database result", "help": "ℹ️ Help"}.get(mode, ""))
                voice_widget(m["content"], lang)
                if m.get("report"):
                    st.download_button("⬇️ Download report (.md)", m["report"], file_name="examination_report.md",
                                       mime="text/markdown", key=f"agent_dl_{role}_{i}")
    typed = st.chat_input("Ask about students, marks, classes, checking or re-evaluation…")
    question = picked or typed
    if question:
        hist = [{"role": x["role"], "content": x["content"]} for x in st.session_state[key]][-6:]
        st.session_state[key].append({"role": "user", "content": question})
        with st.spinner("Looking up the examination records…"):
            r = _req("POST", "/agent/chat", json={"message": question, "history": hist, "response_language": lang}, timeout=180)
        if r is not None and r.status_code == 200:
            d = r.json()
            st.session_state[key].append({"role": "assistant", "content": d["reply"], "mode": d.get("mode"),
                                          "report": d.get("report")})
        else:
            st.session_state[key].append({"role": "assistant", "content": "⚠️ " + _detail(r, "The assistant could not answer"),
                                          "mode": "error"})
        st.rerun()
