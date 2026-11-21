"""
AI Paper Checking — University AI Examination & Student Mentoring Platform
===========================================================================
Enterprise-grade multi-role dashboard supporting:
  • University Admin: Departments, Courses, Student Roster, CSV/Excel Bulk Import, Audit Logs
  • Faculty & Examiner: Model Answers, Marking Schemes, Marks Moderation, Flagging, Approval & Publication
  • Student Portal: Strictly authorized view of published results, breakdown, evaluated answer sheet download
  • AI Academic Mentor: Interactive RAG-based chatbot explaining marks deductions, missing points & practice quizzes
"""

import streamlit as st
import os, json, time, re, requests
import pandas as pd
from datetime import datetime

try:
    from frontend import theme
    from frontend import portal_pages as PP
except ImportError:  # launched from inside frontend/
    import theme
    import portal_pages as PP
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Paper Checking — University Exam System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State for Authentication ───────────────────────────────────────────
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None   # set only after a real /auth/login
if "reduce_motion" not in st.session_state:
    st.session_state.reduce_motion = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None

# ── Design system ──────────────────────────────────────────────────────────────
theme.inject(reduce_motion=st.session_state.reduce_motion)


# ── API Helpers ────────────────────────────────────────────────────────────────
def api_headers():
    """Only a token issued by the backend's /auth/login is ever sent."""
    if st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}


def _logout(message=None):
    st.session_state.auth_token = None
    st.session_state.current_user = None
    if message:
        st.session_state.login_notice = message
    st.rerun()


def _handle_401(r):
    if r is not None and r.status_code == 401 and st.session_state.get("auth_token"):
        _logout("Your session expired. Please sign in again.")


def api_get(endpoint):
    """GET JSON from the backend. Returns None on failure (error kept in st.session_state.api_error)."""
    st.session_state.api_error = None
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=api_headers(), timeout=12)
        _handle_401(r)
        if r.status_code == 200:
            return r.json()
        st.session_state.api_error = f"{endpoint} → HTTP {r.status_code}"
    except Exception as e:
        st.session_state.api_error = f"{endpoint} → {type(e).__name__}"
    return None


def api_post(endpoint, data=None, json_data=None, files=None):
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            data=data,
            json=json_data,
            files=files,
            headers=api_headers(),
            timeout=120
        )
        _handle_401(r)
        return r
    except Exception:
        return None


def _api_detail(r, fallback):
    """Human-readable error from a backend response."""
    if r is None:
        return "Cannot reach the backend. Is the API running?"
    try:
        d = r.json().get("detail")
        if isinstance(d, list):
            d = "; ".join(e.get("msg", "") for e in d)
        return d or fallback
    except Exception:
        return fallback


def _material_picker(prefix):
    """Subject → chapter → document selectors built from the student's REAL uploads."""
    subs = api_get("/mentor-ai/subjects") or []
    if not subs:
        st.info("You haven't uploaded any study material yet. Add a book or notes in "
                "“📚 My Books & Study Documents” (or attach a file in the AI Mentor) first.")
        return None
    names = ["All Subjects"] + [x["subject"] for x in subs]
    sub = st.selectbox("Subject", names, key=f"{prefix}_sub")
    pool = subs if sub == "All Subjects" else [x for x in subs if x["subject"] == sub]
    docs = [d for x in pool for d in x["documents"]]
    chapters = sorted({c for x in pool for c in x["chapters"]})
    c1, c2 = st.columns(2)
    doc_labels = ["All documents in this subject"] + [f"{d['title']}" + ("" if d["mine"] else " (department)") for d in docs]
    di = c1.selectbox("Document", range(len(doc_labels)), format_func=lambda i: doc_labels[i], key=f"{prefix}_doc")
    chap = c2.selectbox("Chapter / topic", ["Whole document"] + chapters + ["Type a topic…"], key=f"{prefix}_chap")
    if chap == "Type a topic…":
        chap = st.text_input("Topic to focus on", key=f"{prefix}_chap_txt")
    return {"subject": sub, "chapter": "" if chap == "Whole document" else (chap or ""),
            "document_id": docs[di - 1]["id"] if di > 0 else None}


PP.init(API_BASE, api_headers, theme)


# ── Authentication gate (real backend login — role comes from the account, never from a button) ──
def _login_screen():
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        theme.landing_hero()
    with right:
        st.markdown('<div style="height:2.2rem"></div><div class="login-title">Sign in to your portal</div>'
                    '<div class="login-sub">Use the credentials issued by your institution. '
                    'Your role is determined by your account.</div>', unsafe_allow_html=True)
        if st.session_state.get("login_notice"):
            st.warning(st.session_state.pop("login_notice"))
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username or registration number", placeholder="e.g. 21BCS101")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter portal →", use_container_width=True)
        if submitted:
            if not username.strip() or not password:
                st.error("Please enter both username and password.")
            else:
                with st.spinner("Verifying credentials…"):
                    try:
                        r = requests.post(f"{API_BASE}/auth/login",
                                          json={"username": username.strip(), "password": password}, timeout=15)
                    except Exception:
                        r = None
                if r is None:
                    st.error(f"Cannot reach the backend at {API_BASE}. Start the API (python run.py api) and retry.")
                elif r.status_code == 200:
                    body = r.json()
                    st.session_state.auth_token = body["access_token"]
                    st.session_state.current_user = body["user"]
                    st.rerun()
                elif r.status_code in (401, 403):
                    try:
                        detail = r.json().get("detail", "Sign-in failed.")
                    except Exception:
                        detail = "Sign-in failed."
                    st.error(detail)
                else:
                    st.error(f"Server error ({r.status_code}). Please try again.")
        st.checkbox("Reduce motion", key="reduce_motion",
                    help="Turns off animations. Your operating-system setting is also respected.")
        st.markdown('<div class="legal">Forgot your password? Ask the examination office to reset it — '
                    'self-service reset is not available yet.<br/>AI scores are provisional until '
                    'reviewed and approved by an authorised examiner.</div>', unsafe_allow_html=True)


if not st.session_state.current_user or not st.session_state.auth_token:
    _login_screen()
    st.stop()


# ── Sidebar & Portal Navigation ────────────────────────────────────────────────
with st.sidebar:
    theme.sidebar_brand()

    # API Online Indicator
    try:
        ok = requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        ok = False
    dot = 'dot-on' if ok else 'dot-off'
    label = 'Backend API Online' if ok else 'Backend API Offline'
    st.markdown(f'<div class="status-row"><span class="status-dot"><span class="{dot}"></span>{label}</span></div>', unsafe_allow_html=True)

    # Current User Profile Block
    user = st.session_state.current_user
    role = user["role"]
    role_css = f"role-{role}"
    _name = user.get("full_name") or user.get("username", "")
    _dept = user.get("department") or ""
    st.markdown(f"""
    <div style="padding: 0.5rem 1.25rem 0.75rem; border-bottom: 1px solid var(--line); margin-bottom: 0.75rem;">
        <div style="font-size:0.88rem; font-weight:600;">{_name}</div>
        <div style="font-size:0.75rem; color:var(--muted);">{_dept}</div>
        <div class="role-badge {role_css}">{role.upper()} PORTAL</div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Role-based Menus (role comes from the authenticated account)
    if role == "admin":
        nav_options = [
            "🏠  Admin Dashboard",
            "🔎  Registration No. Search",
            "👩‍🎓  Student Management",
            "👨‍🏫  Teacher Management",
            "🏫  Batches & Sections",
            "📑  Class Results & Export",
            "📈  Performance Analytics",
            "🧾  Results Compilation & Publishing",
            "🔍  Examiner Review & Moderation",
            "🤖  AI Examination Assistant",
            "🏛️  University Administration",
            "📥  Bulk Student Import",
            "📦  Batch Answer Sheet Upload",
            "📝  Question Papers & Marking Schemes",
            "📄  Question Paper Manager",
            "📜  Audit Logs & Institutional Reports",
            "⚙️  Settings"
        ]
    elif role == "faculty":
        nav_options = [
            "🏠  Teacher Dashboard",
            "👥  My Classes & Students",
            "📷  Answer Sheet Scanner",
            "🔍  Examiner Review & Moderation",
            "🟢  OMR Evaluation",
            "🧾  Results Compilation & Publishing",
            "📑  Class Results & Export",
            "📈  Performance Analytics",
            "📄  Question Papers & Model Answers",
            "⚖️  Re-Evaluation Requests",
            "🤖  AI Examination Assistant",
            "📊  AI Evaluation Accuracy Metrics",
            "⚙️  Settings"
        ]
    else:  # student
        nav_options = [
            "🏠  My Dashboard",
            "🎓  My Academic Profile",
            "📋  Published Examination Results",
            "📥  Original & Evaluated Answer Sheet",
            "⚖️  Submit Re-Evaluation Dispute",
            "🤖  AI Academic Mentor",
            "🤖  AI Examination Assistant",
            "📚  My Books & Study Documents",
            "🎯  Quiz Center & Practice",
            "🗺️  Adaptive Study Roadmap",
            "⚙️  Settings"
        ]

    _unread = PP.unread_count()
    nav_options.append(f"🔔  Notifications" + (f" ({_unread})" if _unread else ""))
    page = st.radio("navigation", nav_options, label_visibility="collapsed")
    if page.startswith("🔔  Notifications"):
        page = "🔔  Notifications"

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    _c1, _c2 = st.columns(2)
    if _c1.button("Sign out", use_container_width=True):
        _logout()
    _c2.checkbox("Reduce motion", key="reduce_motion")


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PORTAL
# ═══════════════════════════════════════════════════════════════════════════════

_NEW_PAGES = {
    "🏠  Admin Dashboard": PP.page_admin_dashboard,
    "🔎  Registration No. Search": PP.page_reg_search,
    "👩‍🎓  Student Management": PP.page_students,
    "👨‍🏫  Teacher Management": PP.page_teachers,
    "🏫  Batches & Sections": PP.page_batches,
    "📑  Class Results & Export": lambda: PP.page_class_results(role),
    "📈  Performance Analytics": lambda: PP.page_analytics(role),
    "🧾  Results Compilation & Publishing": PP.page_results_publish,
    "🏠  Teacher Dashboard": PP.page_teacher_dashboard,
    "👥  My Classes & Students": PP.page_my_classes,
    "📷  Answer Sheet Scanner": PP.page_scanner,
    "🟢  OMR Evaluation": PP.page_omr,
    "📄  Question Papers & Model Answers": PP.page_exam_papers,
    "📝  Question Papers & Marking Schemes": PP.page_exam_papers,
    "🔑  Marking Rubrics": PP.page_exam_papers,
    "🔍  Examiner Review & Moderation": PP.page_examiner_review,
    "🏠  My Dashboard": PP.page_student_dashboard,
    "🔔  Notifications": PP.page_notifications,
    "🤖  AI Examination Assistant": lambda: PP.page_exam_assistant(role),
}

if page in _NEW_PAGES:
    _NEW_PAGES[page]()

elif page == "🏛️  University Administration":
    theme.page_header("Administration Overview", "Live figures from the examination database. "
                      "Nothing on this page is sample data.", crumb="Admin · Overview")

    _students = api_get("/admin/students")
    _students_err = st.session_state.api_error
    _depts = api_get("/admin/departments")
    _courses = api_get("/admin/courses")
    _users = api_get("/admin/users")
    _booklets = api_get("/examiner/booklets")
    _booklets_err = st.session_state.api_error

    theme.kpi_row([
        {"label": "Students", "value": _students.get("total") if _students else None, "hint": "registered records", "tone": ""},
        {"label": "Departments", "value": len(_depts) if isinstance(_depts, list) else None, "hint": "academic units", "tone": "cyan"},
        {"label": "Courses", "value": len(_courses) if isinstance(_courses, list) else None, "hint": "active catalogue", "tone": "violet"},
        {"label": "User accounts", "value": len(_users.get("users", [])) if isinstance(_users, dict) else None, "hint": "admin · faculty · student", "tone": "green"},
    ])
    st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)

    if isinstance(_booklets, list):
        _cnt = {}
        for _b in _booklets:
            _k = (_b.get("approval_status") or "unknown").lower()
            _cnt[_k] = _cnt.get(_k, 0) + 1
        theme.kpi_row([
            {"label": "Answer booklets", "value": len(_booklets), "hint": "uploaded", "tone": ""},
            {"label": "Awaiting review", "value": _cnt.get("pending", 0) + _cnt.get("in_review", 0), "hint": "pending + in review", "tone": "amber"},
            {"label": "Approved", "value": _cnt.get("approved", 0), "hint": "not yet published", "tone": "cyan"},
            {"label": "Published", "value": _cnt.get("published", 0), "hint": "visible to students", "tone": "green"},
        ])
        st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
        if _booklets:
            st.markdown('<div class="card"><div class="card-label">Evaluation pipeline status</div>', unsafe_allow_html=True)
            st.bar_chart(pd.Series(_cnt, name="booklets"), height=220)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            theme.empty_state("No answer booklets yet", "Upload a batch from “Batch Answer Sheet Upload” to start evaluation.", "📦")
    else:
        st.warning(f"Booklet statistics are unavailable ({_booklets_err or 'backend not reachable'}).")

    _logs = api_get("/admin/audit-logs?limit=8")
    if isinstance(_logs, list) and _logs:
        st.markdown('<div class="card"><div class="card-label">Recent activity</div>', unsafe_allow_html=True)
        _df = pd.DataFrame(_logs)
        _cols = [c for c in ("created_at", "username", "role", "action", "target_type", "target_id") if c in _df.columns]
        st.dataframe(_df[_cols], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👥 Student Roster", "📚 Courses & Departments", "👤 System Users"])

    with tab1:
        st.markdown('<div class="card"><div class="card-label">Registered Students</div>', unsafe_allow_html=True)
        search = st.text_input("🔍 Search Student by Name or Registration Number", placeholder="e.g. Arjun, 21BCS101")
        data = api_get(f"/admin/students?search={search}" if search else "/admin/students")
        if data and data.get("students"):
            df = pd.DataFrame(data["students"])
            st.dataframe(df[["registration_number", "name", "department", "programme", "semester", "batch", "booklets_count"]], use_container_width=True)
        else:
            st.info("No students found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="card-label">Departments</div>', unsafe_allow_html=True)
            depts = api_get("/admin/departments") or []
            if depts:
                st.dataframe(pd.DataFrame(depts), use_container_width=True)
            with st.form("add_dept"):
                st.caption("Add New Department")
                d_code = st.text_input("Code (e.g. CSE)", max_chars=10)
                d_name = st.text_input("Department Name")
                if st.form_submit_button("Add Department") and d_code and d_name:
                    res = api_post("/admin/departments", json_data={"code": d_code, "name": d_name})
                    if res and res.status_code == 200:
                        st.success(f"Department {d_code} created.")
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-label">Courses</div>', unsafe_allow_html=True)
            courses = api_get("/admin/courses") or []
            if courses:
                st.dataframe(pd.DataFrame(courses)[["code", "name", "department", "credits", "semester"]], use_container_width=True)
            with st.form("add_course"):
                st.caption("Add New Course")
                c_code = st.text_input("Course Code (e.g. CS305)")
                c_name = st.text_input("Course Title")
                c_creds = st.number_input("Credits", min_value=1, max_value=6, value=4)
                c_sem = st.selectbox("Semester", ["Semester 1", "Semester 2", "Semester 3", "Semester 4", "Semester 5", "Semester 6", "Semester 7", "Semester 8"], index=4)
                if st.form_submit_button("Add Course") and c_code and c_name:
                    res = api_post("/admin/courses", json_data={"code": c_code, "name": c_name, "credits": c_creds, "semester": c_sem})
                    if res and res.status_code == 200:
                        st.success(f"Course {c_code} created.")
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card"><div class="card-label">System User Accounts</div>', unsafe_allow_html=True)
        users_data = api_get("/admin/users") or {}
        if users_data.get("users"):
            st.dataframe(pd.DataFrame(users_data["users"])[["username", "full_name", "role", "email", "department", "is_active"]], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "📥  Bulk Student Import":
    st.markdown('<div class="pg-title">📥 Bulk Student Record Import</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload CSV or Excel spreadsheets containing institutional student cohorts. Automatically creates student profiles and authentication accounts with duplicate detection.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Upload Student Roster Spreadsheet</div>', unsafe_allow_html=True)
    st.info("Expected columns: `student_name`, `registration_number`, `email`, `department`, `programme`, `semester`, `batch`")

    uploaded_file = st.file_uploader("Upload CSV or Excel (.xlsx) file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if st.button("🚀 Process & Import Students", use_container_width=True):
            with st.spinner("Parsing records and validating against database..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = api_post("/admin/students/bulk-import", files=files)
                if res and res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ Successfully processed {data.get('total_rows_processed', 0)} rows! Imported: {data.get('imported_count', 0)} new students.")
                    if data.get("duplicate_count", 0) > 0:
                        st.warning(f"⚠️ Detected {data.get('duplicate_count')} duplicate records (skipped to protect data integrity).")
                    _creds = data.get("temporary_credentials") or []
                    if _creds:
                        st.warning("Temporary passwords are shown only once. Download them now and share them securely.")
                        st.download_button("⬇️ Download temporary credentials (CSV)",
                                           pd.DataFrame(_creds).to_csv(index=False).encode("utf-8"),
                                           file_name="temporary_credentials.csv", mime="text/csv")
                else:
                    st.error(f"Failed to import: {res.text if res else 'Connection error'}")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "📦  Batch Answer Sheet Upload":
    st.markdown('<div class="pg-title">📦 Batch Answer Sheet Upload & Mapping</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload entire batches of student answer booklets. System maps each file to the student record, detects duplicate uploads, and flags unmatched scripts.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Batch Booklet Upload</div>', unsafe_allow_html=True)
    course_code = st.text_input("Course Code (e.g. CS301)", value="CS301")
    batch_files = st.file_uploader("Upload Answer Sheets (PDF / Images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    if batch_files and st.button("📤 Upload and Map Batch"):
        with st.spinner("Mapping files to student records..."):
            multi_files = [("files", (f.name, f.getvalue(), f.type)) for f in batch_files]
            res = requests.post(
                f"{API_BASE}/admin/booklets/batch-upload",
                data={"course_code": course_code},
                files=multi_files,
                headers=api_headers(),
                timeout=120
            )
            if res and res.status_code == 200:
                out = res.json()
                st.success(f"✅ Mapped {out.get('mapped_count', 0)} booklets to registered students!")
                if out.get("duplicate_count", 0) > 0:
                    st.warning(f"⚠️ {out.get('duplicate_count')} duplicate uploads detected and prevented.")
                if out.get("unmatched_count", 0) > 0:
                    st.error(f"❌ {out.get('unmatched_count')} unmatched files (filename did not match any registered student).")
            else:
                st.error("Upload error. Please check backend connection.")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "📜  Audit Logs & Institutional Reports":
    st.markdown('<div class="pg-title">📜 Audit Logs & Quality Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Immutable audit trail of examiner mark modifications, result publication, and institutional performance summaries.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Audit Trail", "📊 Examination Summary Report"])

    with tab1:
        st.markdown('<div class="card"><div class="card-label">Recent Administrative & Examiner Actions</div>', unsafe_allow_html=True)
        logs = api_get("/admin/audit-logs?limit=50") or []
        if logs:
            st.dataframe(pd.DataFrame(logs)[["created_at", "username", "role", "action", "target_type", "details"]], use_container_width=True)
        else:
            st.info("No audit logs recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card"><div class="card-label">Downloadable Quality Audit Report</div>', unsafe_allow_html=True)
        rep = api_get("/admin/reports/examination-summary")
        if rep and "institutional_report" in rep:
            r_data = rep["institutional_report"]
            st.json(r_data)
            st.download_button(
                "📥 Download Institutional Report (JSON)",
                data=json.dumps(r_data, indent=2),
                file_name="university_exam_summary_report.json",
                mime="application/json",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "📄  Question Paper Manager":
    st.markdown('<div class="pg-title">📄 Institutional Question Paper & Bank Manager</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Create, search, filter, and organize questions in the Question Bank, assemble complete examination papers, or upload PDFs.</div>', unsafe_allow_html=True)

    tab_bank, tab_create, tab_pdf, tab_papers = st.tabs(["📚 Question Bank", "➕ Add Question", "📤 Upload Paper PDF", "📋 Assembled Papers"])

    with tab_bank:
        st.markdown('<div class="card"><div class="card-label">Curated Questions & Rubrics</div>', unsafe_allow_html=True)
        q_search = st.text_input("🔍 Search Question Bank", placeholder="e.g. QuickSort, Dynamic Programming, ACID...")
        q_items = api_get("/mentor-ai/questions/bank") or []
        if q_items:
            if q_search:
                q_items = [q for q in q_items if q_search.lower() in q.get("question_text", "").lower() or q_search.lower() in q.get("teacher_answer", "").lower()]
            for item in q_items:
                with st.expander(f"Q#{item['id']} · [{item.get('question_type', 'open_ended').upper()}] {item['question_text'][:70]}... ({item.get('max_marks', 5.0)} marks)"):
                    st.markdown(f"**Full Question:** {item['question_text']}")
                    st.markdown(f"**Model / Teacher Answer:** {item.get('teacher_answer') or '(No model answer provided)'}")
                    c_del1, c_del2 = st.columns([8, 2])
                    with c_del2:
                        if st.button("🗑️ Delete", key=f"del_q_{item['id']}"):
                            d_res = requests.delete(f"{API_BASE}/mentor-ai/questions/bank/{item['id']}", headers=api_headers())
                            if d_res.status_code == 200:
                                st.success("Question deleted.")
                                st.rerun()
        else:
            st.info("Question Bank is currently empty. Add your first question in the 'Add Question' tab!")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_create:
        st.markdown('<div class="card"><div class="card-label">Create New Question Entry</div>', unsafe_allow_html=True)
        with st.form("new_question_form"):
            q_text = st.text_area("Question Text", placeholder="Explain the time and space complexity of MergeSort...")
            q_type = st.selectbox("Question Type", ["open_ended", "mcq", "true_false", "numerical"])
            t_ans = st.text_area("Model Answer / Rubric Key", placeholder="Full reference explanation...")
            max_m = st.number_input("Maximum Marks", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
            corr_opt = st.text_input("Correct Option (For MCQ/TF only, e.g. A or True)", value="")

            if st.form_submit_button("💾 Save to Question Bank"):
                if not q_text.strip():
                    st.error("Question text is required.")
                else:
                    payload = {
                        "question_text": q_text,
                        "question_type": q_type,
                        "teacher_answer": t_ans,
                        "max_marks": max_m,
                        "correct_option": corr_opt or None
                    }
                    r = api_post("/mentor-ai/questions/bank", json_data=payload)
                    if r and r.status_code == 200:
                        st.success("✅ Question added successfully to the Question Bank!")
                        st.rerun()
                    else:
                        st.error("Failed to add question.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_pdf:
        st.markdown('<div class="card"><div class="card-label">Upload Exam Paper PDF with Automated OCR</div>', unsafe_allow_html=True)
        p_file = st.file_uploader("Upload Question Paper PDF", type=["pdf"])
        if p_file and st.button("🚀 Parse & Extract Questions"):
            with st.spinner("Analyzing PDF layout and segmenting questions..."):
                r = api_post("/paper/upload", files={"file": (p_file.name, p_file.getvalue(), "application/pdf")})
                if r and r.status_code == 200:
                    st.success("Exam paper parsed and registered!")
                    st.rerun()
                else:
                    st.error("Failed to parse PDF paper.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_papers:
        st.markdown('<div class="card"><div class="card-label">Registered Examination Papers</div>', unsafe_allow_html=True)
        papers = api_get("/papers") or []
        if papers:
            _pdf = pd.DataFrame(papers)
            st.dataframe(_pdf[[c for c in ["paper_id", "course_code", "course_name", "exam_name", "total_marks",
                                           "num_questions", "exam_date"] if c in _pdf.columns]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No exam papers found.")
        st.markdown('</div>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# FACULTY & EXAMINER PORTAL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚖️  Re-Evaluation Requests":
    st.markdown('<div class="pg-title">⚖️ Student Grade Disputes & Re-Evaluations</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Address appeals and re-checking requests submitted by students. Adjust marks if justified and provide official examiner remarks.</div>', unsafe_allow_html=True)

    reqs = api_get("/examiner/re-evaluations") or []
    if not reqs:
        st.info("No re-evaluation requests submitted by students.")
    else:
        for r in reqs:
            with st.expander(f"Dispute #{r['id']} · {r['student_name']} ({r['registration_number']}) — {r['course_code']} [Status: {r['status'].upper()}]", expanded=(r['status'] == 'pending')):
                st.markdown(f"**Disputed Item:** Question {r['question_number']}")
                st.markdown(f"**Student's Reason for Dispute:** {r['reason']}")
                st.markdown(f"**Score Prior to Appeal:** {r['score_before']}")

                if r['status'] == "pending":
                    st.divider()
                    st.markdown("**Examiner Resolution:**")
                    col1, col2 = st.columns(2)
                    action = col1.selectbox("Decision", ["approved", "rejected"], key=f"dec_{r['id']}")
                    new_marks = col2.number_input("New marks for this question (if approved)", min_value=0.0,
                                                  value=float(r['score_before'] or 0.0), step=0.5, key=f"new_m_{r['id']}")
                    remarks = st.text_area("Examiner Remarks", placeholder="Official reason for accepting/rejecting dispute...", key=f"rem_{r['id']}")

                    if st.button("Submit Resolution", key=f"res_{r['id']}"):
                        if PP.send("POST", "/examiner/resolve-reevaluation", json={
                            "request_id": r['id'],
                            "status": action,
                            "examiner_remarks": remarks,
                            "score_after": new_marks if action == "approved" else None
                        }, ok_msg=f"Dispute #{r['id']} resolved as {action}. The student's evaluated sheet now shows the decision."):
                            st.rerun()
                else:
                    st.success(f"Resolved as **{r['status'].upper()}** on {r.get('updated_at')}. Remarks: {r.get('examiner_remarks')}")


elif page == "📊  AI Evaluation Accuracy Metrics":
    st.markdown('<div class="pg-title">📊 ML/NLP Evaluation Accuracy & Agreement Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Track statistical correlation, Mean Absolute Error (MAE), and agreement between AI scoring and examiner evaluations.</div>', unsafe_allow_html=True)

    metrics_data = api_get("/metrics") or {}
    oe = metrics_data.get("open_ended", {})

    def _fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"

    if not any(isinstance(v, (int, float)) for v in oe.values()):
        st.info("No agreement metrics yet. They appear once AI-scored answers have matching examiner-reviewed scores.")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="mtile"><div class="mtile-val">{_fmt(oe.get("mae"))}</div><div class="mtile-label">MAE (Marks Error)</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="mtile"><div class="mtile-val">{_fmt(oe.get("rmse"))}</div><div class="mtile-label">RMSE</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="mtile"><div class="mtile-val">{_fmt(oe.get("pearson_r"))}</div><div class="mtile-label">Pearson r Correlation</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="mtile"><div class="mtile-val">{_fmt(oe.get("cohen_kappa"))}</div><div class="mtile-label">Quadratic Cohen κ</div></div>', unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.info("💡 **Accuracy Note:** In this University Examination System, AI scores serve as objective preliminary suggestions. Official final marks are subject to authorized examiner review and approval.")


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT PORTAL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎓  My Academic Profile":
    st.markdown('<div class="pg-title">🎓 Student Academic Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Your registered university credentials, academic degree programme, and registered courses.</div>', unsafe_allow_html=True)

    profile = api_get("/student/profile")
    if profile:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:1.3rem; font-weight:700; color:#fff;">{profile['name']}</div>
                <div style="font-size:0.85rem; color:#93c5fd; margin-top:2px;">Reg No: {profile['registration_number']}</div>
                <div style="font-size:0.8rem; color:var(--muted); margin-top:10px;">
                    <div>🏛️ {profile['department']}</div>
                    <div>🎓 {profile['programme']}</div>
                    <div>📅 {profile['semester']} · Batch {profile.get('batch_code') or '—'} · Section {profile.get('section') or '—'}</div>
                    <div>🗓️ Academic session {profile.get('academic_session') or '—'}</div>
                    <div>✉️ {profile['email']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-label">Registered Semester Courses</div>', unsafe_allow_html=True)
            courses = profile.get("registered_courses", [])
            if courses:
                st.dataframe(pd.DataFrame(courses), use_container_width=True)
            else:
                st.info("No courses registered for this semester.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Could not load student profile. Please verify login.")


elif page == "📋  Published Examination Results":
    theme.page_header("Published Examination Results",
                      "Question-wise marks, deductions and feedback. Only results approved by a teacher and published "
                      "by the examination department appear here.", crumb="Student · Results")
    results = api_get("/student/results") or []
    if not results:
        theme.empty_state("No published results yet", "Results appear after teacher approval and publication.", "📋")
    for r in results:
        b_id = r["booklet_id"]
        detail = api_get(f"/student/result/{b_id}")
        if not detail:
            continue
        st.markdown(f"""
        <div class="score-hero">
            <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; color:#93c5fd;">{detail['course_code']} · {detail['course_name']} · {detail['exam_name']}</div>
            <div class="score-num">{detail['total_awarded']:g}<span class="score-denom"> / {detail['total_max']:g}</span></div>
            <div style="font-size:0.95rem; color:#86efac; font-weight:600; margin-top:0.4rem;">{detail['percentage']:.1f}% · Teacher-approved, officially published (written answers)</div>
        </div>""", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        for col, path, label, fname in ((d1, f"/student/answer-sheet/{b_id}/original", "Original answer sheet", "original"),
                                        (d2, f"/student/answer-sheet/{b_id}/evaluated", "Evaluated answer sheet (marks on your pages)", "evaluated"),
                                        (d3, f"/results/booklet/{b_id}/report.pdf", "Detailed marks analysis", "marks_analysis")):
            if col.button(f"📄 {label}", key=f"{fname}_{b_id}", use_container_width=True):
                data = PP.get_bytes(path)
                if data:
                    col.download_button("⬇️ Save PDF", data, file_name=f"{fname}_{detail['course_code']}.pdf", mime="application/pdf", key=f"dl{fname}{b_id}")
        for q in detail.get("questions", []):
            with st.expander(f"Q{q['question_number']} — {q['marks_awarded']:g} / {q['max_marks']:g} marks"
                             f" · deducted {q.get('marks_deducted', 0):g}"
                             + (" · teacher verified" if q["is_examiner_certified"] else " · AI evaluated"), expanded=False):
                if q.get("question_text"):
                    st.markdown(f"**Question:** {q['question_text']}")
                c1, c2 = st.columns(2)
                c1.markdown("**Your answer**")
                c1.text_area("yours", q.get("student_answer") or "(no answer detected)", height=150, disabled=True,
                             label_visibility="collapsed", key=f"ya{b_id}_{q['question_number']}")
                c2.markdown("**Expected answer / marking criteria**")
                c2.text_area("exp", q.get("expected_answer") or "(not provided)", height=150, disabled=True,
                             label_visibility="collapsed", key=f"xa{b_id}_{q['question_number']}")
                if q.get("ai_score") is not None and q["is_examiner_certified"] and q["ai_score"] != q["marks_awarded"]:
                    st.caption(f"AI suggested {q['ai_score']:g}; your teacher awarded {q['marks_awarded']:g} (official).")
                if q.get("rubric_breakdown"):
                    PP._breakdown_table(q["rubric_breakdown"])
                for _d in q.get("deductions") or []:
                    st.markdown(f"<span class='pill pill-r'>−{_d.get('marks')} {_d.get('criterion') or ''}</span> {_d.get('reason') or ''}",
                                unsafe_allow_html=True)
                for _c in q.get("corrections") or []:
                    st.markdown(f"✏️ You wrote *“{_c.get('student_wrote', '')}”* → correct: **{_c.get('correct_version', '')}** {_c.get('explanation', '')}")
                if q.get("strengths"):
                    st.markdown("**What you did well:** " + " ".join(f'<span class="pill pill-g">✓ {x}</span>' for x in q["strengths"]), unsafe_allow_html=True)
                if q.get("missing_concepts"):
                    st.markdown("**Why marks were deducted (missing concepts):** " + " ".join(f'<span class="pill pill-r">✗ {x}</span>' for x in q["missing_concepts"]), unsafe_allow_html=True)
                    st.caption("Suggestion: add a clear definition, the key steps and an example for each missing concept. "
                               "Ask the AI Mentor “Why did I lose marks in Q" + str(q["question_number"]) + "?” for a full explanation.")
                if q.get("examiner_comments"):
                    st.info(f"👨‍🏫 Teacher: {q['examiner_comments']}")
                elif q.get("feedback"):
                    st.caption(f"Feedback: {q['feedback']}")


elif page == "📥  Original & Evaluated Answer Sheet":
    st.markdown('<div class="pg-title">📥 Original & Evaluated Answer Sheet Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Inspect or download your original scanned answer script and the officially evaluated copy with examiner annotations and stamps.</div>', unsafe_allow_html=True)

    results = api_get("/student/results") or []
    if not results:
        st.info("No answer sheets ready for viewing. Only published examination scripts can be downloaded.")
    else:
        b_opts = [f"#{r['booklet_id']} — {r['course_code']} ({r['course_name']})" for r in results]
        sel_idx = st.selectbox("Select Examination Script", range(len(results)), format_func=lambda i: b_opts[i])
        b_id = results[sel_idx]["booklet_id"]

        st.markdown('<div class="card"><div class="card-label">Secure Script Downloads</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Original Uploaded Script")
            st.caption("The authentic, unmodified handwritten/typed booklet uploaded for evaluation.")
            if st.button("📥 Download Original Script", key="dl_orig", use_container_width=True):
                r_orig = requests.get(f"{API_BASE}/student/answer-sheet/{b_id}/original", headers=api_headers())
                if r_orig.status_code == 200:
                    st.download_button("Click to Save Original PDF", data=r_orig.content, file_name=f"original_script_{b_id}.pdf", mime="application/pdf", use_container_width=True)
                else:
                    st.error("Could not retrieve original file.")

        with col2:
            st.markdown("### 📑 Evaluated Script with Annotations")
            st.caption("Official copy featuring question-wise marks breakdown, examiner seal, strengths, and missing concept annotations.")
            if st.button("📥 Generate & Download Evaluated PDF", key="dl_eval", use_container_width=True):
                with st.spinner("Compiling evaluated script with official stamps..."):
                    r_eval = requests.get(f"{API_BASE}/student/answer-sheet/{b_id}/evaluated", headers=api_headers())
                    if r_eval.status_code == 200:
                        st.download_button("Click to Save Evaluated PDF", data=r_eval.content, file_name=f"evaluated_script_{b_id}.pdf", mime="application/pdf", use_container_width=True)
                    else:
                        st.error("Failed to generate evaluated sheet.")
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "⚖️  Submit Re-Evaluation Dispute":
    st.markdown('<div class="pg-title">⚖️ Grade Appeal & Re-Evaluation Request</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">If you believe an answer was evaluated unfairly or missing concepts were in fact addressed, submit a formal re-evaluation request to the Chief Examiner.</div>', unsafe_allow_html=True)

    results = api_get("/student/results") or []
    if not results:
        st.info("You do not have any published results to dispute at this time.")
    else:
        with st.form("reeval_form"):
            b_opts = [f"#{r['booklet_id']} — {r['course_code']} ({r['course_name']})" for r in results]
            sel_b_idx = st.selectbox("Select Examination Result", range(len(results)), format_func=lambda i: b_opts[i])
            chosen_booklet_id = results[sel_b_idx]["booklet_id"]

            q_num = st.number_input("Question Number (Enter 0 to request full booklet review)", min_value=0, max_value=50, value=1)
            reason = st.text_area("Detailed Reason for Re-Evaluation", placeholder="Explain specifically where valid arguments, formulas, or concepts were presented...")

            if st.form_submit_button("Submit Re-Evaluation Request") and reason:
                res = api_post("/student/re-evaluation", json_data={
                    "booklet_id": chosen_booklet_id,
                    "question_number": q_num if q_num > 0 else None,
                    "reason": reason
                })
                if res and res.status_code == 200:
                    st.success("Re-evaluation request submitted! Track status below.")
                    st.rerun()
                else:
                    st.error(f"Error: {res.text if res else 'Could not submit'}")

        st.markdown('<div class="card"><div class="card-label">Submitted Re-Evaluation History</div>', unsafe_allow_html=True)
        my_reqs = api_get("/student/re-evaluations") or []
        if my_reqs:
            st.dataframe(pd.DataFrame(my_reqs)[["submitted_at", "course_code", "question_number", "status", "score_before", "score_after", "examiner_remarks"]], use_container_width=True)
        else:
            st.caption("No re-evaluation requests submitted yet.")
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "🤖  AI Academic Mentor":
    # Full ChatGPT-style Mentor AI dashboard (frontend/mentor_app.html).
    # The browser talks to the existing FastAPI /mentor/* endpoints directly with the
    # student's own JWT, so typed AND spoken questions really reach the backend.
    import streamlit.components.v1 as components

    st.markdown(
        "<style>.block-container{padding-top:.6rem!important;padding-bottom:0!important;max-width:100%!important;"
        "padding-left:1rem!important;padding-right:1rem!important}"
        "iframe[title='st.iframe'],iframe{border-radius:14px}"
        ".st-key-toggle_portal_nav button{background:#0f1a47!important;color:#dbe3ff!important;"
        "border:1px solid rgba(99,125,255,.35)!important;border-radius:10px!important;min-height:34px!important;padding:2px 14px!important}"
        ".st-key-toggle_portal_nav button:hover{border-color:rgba(140,160,255,.7)!important}"
        ".st-key-toggle_portal_nav{margin-bottom:-.4rem}</style>",
        unsafe_allow_html=True,
    )
    # Portal sidebar is hidden by default on this page so the mentor gets the full width.
    # It is only hidden with CSS — navigation, sign-out and every portal page stay intact.
    if "portal_nav_open_on_mentor" not in st.session_state:
        st.session_state.portal_nav_open_on_mentor = False
    _nav_open = st.session_state.portal_nav_open_on_mentor
    if st.button("✕  Hide portal menu" if _nav_open else "☰  Portal menu",
                 key="toggle_portal_nav",
                 help="Show or hide the student portal navigation (results, answer sheets, quizzes, settings…)"):
        st.session_state.portal_nav_open_on_mentor = not _nav_open
        st.rerun()
    if not _nav_open:
        st.markdown(
            "<style>section[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
            "[data-testid='stExpandSidebarButton']{display:none!important}"
            # keep the toggle reachable above Streamlit's top bar, even after scrolling
            ".st-key-toggle_portal_nav{position:fixed!important;top:10px;left:14px;z-index:1000002;width:auto!important;margin:0!important}"
            "</style>",
            unsafe_allow_html=True,
        )
    _tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mentor_app.html")
    with open(_tpl_path, "r", encoding="utf-8") as _f:
        _tpl = _f.read()
    _u = st.session_state.current_user or {}
    _cfg = {
        # PUBLIC_API_BASE_URL = backend URL as seen from the student's BROWSER
        # (differs from API_BASE inside Docker, e.g. http://localhost:8000)
        "apiBase": os.getenv("PUBLIC_API_BASE_URL", API_BASE),
        "token": st.session_state.auth_token,
        "autoSpeak": False,  # off by default; student toggles "Auto Speak" (remembered per browser)
        "maxFileMb": int(os.getenv("MENTOR_MAX_FILE_MB", "15")),
        "user": {
            "name": _u.get("full_name") or _u.get("username") or "Student",
            "department": _u.get("department") or "",
        },
    }
    _html = _tpl.replace("__MENTOR_CONFIG__", json.dumps(_cfg).replace("</", "<\\/"))
    components.html(_html, height=int(os.getenv("MENTOR_UI_HEIGHT", "900")), scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MENTOR AI: STUDY BOOKS, QUIZ CENTER & ADAPTIVE ROADMAP
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📚  My Books & Study Documents":
    st.markdown('<div class="pg-title">📚 My Books & Knowledge Base (RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload textbooks, lecture notes, and syllabus modules (PDF, DOCX, TXT). Mentor AI parses, extracts, and generates real vector indexes for source-grounded answers.</div>', unsafe_allow_html=True)

    tab_lib, tab_upload, tab_explain = st.tabs(["📖 Document Library", "📤 Upload New Book / Notes", "🔍 Explain My Book"])

    with tab_lib:
        st.markdown('<div class="card"><div class="card-label">Indexed Textbooks & Study Materials</div>', unsafe_allow_html=True)
        docs = api_get("/mentor-ai/documents") or []
        if docs:
            for d in docs:
                with st.expander(f"📘 {d['title']} · [{d.get('subject','General')}] {d.get('chapter','')} ({d.get('chunks_count',0)} Chunks Indexed)"):
                    st.markdown(f"**Filename:** `{d['filename']}` ({d.get('file_type','').upper()})")
                    st.markdown(f"**Indexed Status:** <span class='pill pill-g'>{d.get('status','ready').upper()}</span> · **Total Pages:** {d.get('total_pages',1)}", unsafe_allow_html=True)
                    st.caption(f"Added on: {d.get('created_at','Recently')}")

                    c_ask1, c_ask2 = st.columns([8, 2])
                    with c_ask2:
                        if st.button("🗑️ Remove", key=f"del_doc_{d['id']}"):
                            d_del = requests.delete(f"{API_BASE}/mentor-ai/documents/{d['id']}", headers=api_headers())
                            if d_del.status_code == 200:
                                st.success("Document deleted.")
                                st.rerun()
        else:
            st.info("No study books or notes uploaded yet. Upload your syllabus or textbook in the 'Upload New Book' tab!")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_upload:
        st.markdown('<div class="card"><div class="card-label">Upload Textbook / Notes File</div>', unsafe_allow_html=True)
        with st.form("book_upload_form"):
            b_title = st.text_input("Book / Notes Title", placeholder="e.g. Introduction to Algorithms (CLRS)")
            b_sub = st.selectbox("Subject Domain", ["Computer Science & Engineering", "Design & Analysis of Algorithms", "Database Management Systems", "Artificial Intelligence", "General"])
            b_chap = st.text_input("Chapter / Unit Name", value="Chapter 1: Foundations & Complexities")
            b_file = st.file_uploader("Select Document (PDF, DOCX, TXT)", type=["pdf", "docx", "doc", "txt", "md"])

            if st.form_submit_button("🚀 Upload, Extract & Index"):
                if not b_title.strip() or not b_file:
                    st.error("Please provide both title and document file.")
                else:
                    with st.spinner("Extracting text pages, running semantic chunking, and creating embeddings..."):
                        files = {"file": (b_file.name, b_file.getvalue(), b_file.type)}
                        data = {"title": b_title, "subject": b_sub, "chapter": b_chap}
                        r = api_post("/mentor-ai/documents/upload", data=data, files=files)
                        if r and r.status_code == 200:
                            out = r.json()
                            st.success(f"✅ Successfully indexed '{out.get('title')}' with {out.get('chunks_indexed', 0)} chunks ready for RAG!")
                            st.rerun()
                        else:
                            st.error(f"Failed to index: {r.text if r else 'Connection error'}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_explain:
        st.markdown('<div class="card"><div class="card-label">Generate Source-Grounded Chapter Explanation</div>', unsafe_allow_html=True)
        _sel = _material_picker("ex")
        c_style = st.selectbox("Explanation Style", ["Detailed Academic", "Easy Hinglish Guide", "Formulas & Exam Points", "Simple English Summary"])
        if _sel and st.button("✨ Explain Chapter Now", use_container_width=True):
            with st.spinner("Reading your material and preparing the explanation..."):
                r = api_post("/mentor-ai/explain/chapter", json_data={**_sel, "style": c_style})
                if r is not None and r.status_code == 200:
                    ex_data = r.json()
                    st.markdown(ex_data.get("explanation", ""))
                    if ex_data.get("citations"):
                        st.caption("📖 Sources: " + ", ".join(ex_data["citations"]))
                else:
                    st.error(_api_detail(r, "Failed to generate explanation."))
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "🎯  Quiz Center & Practice":
    st.markdown('<div class="pg-title">🎯 AI Quiz Center & Weak-Topic Practice</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Test your understanding against uploaded textbook chapters. Submissions update your Weak Topics and trigger adaptive study recommendations.</div>', unsafe_allow_html=True)

    tab_take, tab_weak, tab_hist = st.tabs(["📝 Take Quiz", "🔍 My Weak Topics", "🕘 Previous Attempts"])

    with tab_take:
        st.markdown('<div class="card"><div class="card-label">Configure & Generate Practice Quiz</div>', unsafe_allow_html=True)
        _sel = _material_picker("qz")
        c1, c2 = st.columns(2)
        q_diff = c1.selectbox("Difficulty Level", ["Medium", "Easy", "Hard", "Mixed"])
        num_q = c2.slider("Number of Questions", min_value=2, max_value=15, value=5)

        if _sel and st.button("🎲 Generate Quiz From My Material"):
            with st.spinner("Building questions from your material..."):
                res = api_post("/mentor-ai/quiz/generate", json_data={**_sel, "difficulty": q_diff, "num_questions": num_q})
                if res is not None and res.status_code == 200:
                    st.session_state.current_quiz = res.json()
                    st.session_state.quiz_result = None
                else:
                    st.error(_api_detail(res, "Could not generate quiz."))

        q_data = st.session_state.get("current_quiz")
        if q_data:
            st.divider()
            st.markdown(f"### 📋 Quiz: {q_data.get('chapter') or q_data.get('subject')} · {len(q_data['questions'])} questions")
            if q_data.get("note"):
                st.caption("ℹ️ " + q_data["note"])
            if q_data.get("citations"):
                st.caption("📖 From: " + ", ".join(q_data["citations"]))
            with st.form("quiz_submission_form"):
                picks = []
                for q in q_data.get("questions", []):
                    st.markdown(f"**Q{q['id']}. {q['question']}**")
                    choice = st.radio(f"Answer for Q{q['id']}", q["options"], index=None, key=f"q_choice_{q_data['quiz_id']}_{q['id']}")
                    picks.append({"question_id": q["id"], "selected_answer": choice or ""})
                    st.markdown("---")
                submitted = st.form_submit_button("🚀 Submit Quiz & Grade Answers")
            if submitted:
                unanswered = sum(1 for p_ in picks if not p_["selected_answer"])
                with st.spinner("Grading on the server and updating your topic performance..."):
                    sub_res = api_post("/mentor-ai/quiz/submit", json_data={"quiz_id": q_data["quiz_id"], "answers": picks})
                if sub_res is not None and sub_res.status_code == 200:
                    st.session_state.quiz_result = sub_res.json()
                    st.session_state.current_quiz = None
                    st.rerun()
                else:
                    st.error(_api_detail(sub_res, "Error submitting quiz attempt."))
        out = st.session_state.get("quiz_result")
        if out:
            st.success(f"🏆 Score: {out['score']:g} / {out['max_score']:g} ({out['percentage']}%)")
            for a in out.get("answers", []):
                if a["is_correct"]:
                    st.markdown(f"✅ **Q{a['question_id']}:** Correct — {a['correct_answer']}")
                else:
                    st.markdown(f"❌ **Q{a['question_id']}:** Your answer: *{a['selected_answer'] or 'not answered'}* · Correct: **{a['correct_answer']}**  \n"
                                f"   *Why:* {a['explanation']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_hist:
        _att = api_get("/mentor-ai/quiz/attempts") or []
        if _att:
            st.dataframe(pd.DataFrame(_att)[["created_at", "subject", "topic", "score", "max_score", "percentage"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No quiz attempts yet.")

    with tab_weak:
        st.markdown('<div class="card"><div class="card-label">Identified Weak Topics from Real Attempts</div>', unsafe_allow_html=True)
        weak_list = api_get("/mentor-ai/weak-topics") or []
        if weak_list:
            for w in weak_list:
                status_color = "pill-r" if w["status"]=="Needs Practice" else ("pill-b" if w["status"]=="Improving" else "pill-g")
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1f2937; border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff; font-size:0.95rem;">{w['topic_name']}</div>
                        <div style="font-size:0.75rem; color:#9ca3af;">{w['subject']} · {w['attempts_count']} Attempts Recorded · {w.get('remedy_notes','')}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="pill {status_color}">{w['status'].upper()}</span>
                        <div style="font-size:1.1rem; font-weight:700; color:#388bfd; margin-top:2px;">{w['accuracy']}% Acc</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No weak topics identified yet. Complete a quiz to analyze conceptual strengths and weaknesses!")
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "🗺️  Adaptive Study Roadmap":
    st.markdown('<div class="pg-title">🗺️ Adaptive Personalized Study Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Built from your published exam results and quiz history. Regenerate after new quizzes or results to adapt it.</div>', unsafe_allow_html=True)

    existing_roadmap = api_get("/mentor-ai/roadmap")

    with st.expander("⚙️ Roadmap goals & study hours", expanded=(not existing_roadmap)):
        target_exam = st.text_input("Target Examination", value=existing_roadmap.get("target_exam", "End Semester Examination") if existing_roadmap else "End Semester Examination")
        target_date = st.date_input("Target Examination Date", value=datetime(2026, 12, 15)).strftime("%Y-%m-%d")
        daily_hrs = st.slider("Daily Study Time (Hours)", min_value=1.0, max_value=8.0, value=float(existing_roadmap.get("daily_hours", 2.5)) if existing_roadmap else 2.5, step=0.5)
        if st.button("🔄 Generate / Adapt Roadmap Now", use_container_width=True):
            with st.spinner("Analysing your results and quiz history..."):
                res = api_post("/mentor-ai/roadmap/generate", json_data={
                    "target_exam": target_exam, "target_date": target_date, "daily_hours": daily_hrs, "subjects": []})
            if res is not None and res.status_code == 200:
                st.success("Roadmap updated from your latest performance.")
                st.rerun()
            else:
                st.warning(_api_detail(res, "Could not generate a roadmap."))

    if existing_roadmap and existing_roadmap.get("tasks"):
        st.markdown(f"### 🎯 Plan for **{existing_roadmap.get('target_exam')}** (target {existing_roadmap.get('target_date')})")
        st.progress(min(1.0, (existing_roadmap.get("progress_pct") or 0) / 100.0),
                    text=f"Progress: {existing_roadmap.get('progress_pct') or 0:g}% of tasks completed")
        wk, sg = existing_roadmap.get("weak_topics") or [], existing_roadmap.get("strong_topics") or []
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔴 Weak topics (evidence)**")
            for w in wk:
                st.markdown(f"- {w['topic']} — _{w['evidence']}_")
            if not wk:
                st.caption("—")
        with c2:
            st.markdown("**🟢 Strong topics**")
            for t in sg:
                st.markdown(f"- {t}")
            if not sg:
                st.caption("None mastered yet (≥80% quiz accuracy).")
        st.divider()
        for i, t in enumerate(existing_roadmap.get("tasks", [])):
            cc1, cc2 = st.columns([0.07, 0.93])
            done = cc1.checkbox("done", value=bool(t.get("completed")), key=f"rm_task_{i}", label_visibility="collapsed")
            if done != bool(t.get("completed")):
                rr = requests.patch(f"{API_BASE}/mentor-ai/roadmap/task/{i}", json={"completed": done},
                                    headers=api_headers(), timeout=15)
                if rr.status_code == 200:
                    st.rerun()
                else:
                    st.error(_api_detail(rr, "Could not update the task."))
            cc2.markdown(f"""
            <div style="background:#111827; border:1px solid #1f2937; border-radius:10px; padding:.8rem 1rem; margin-bottom:.6rem; {'opacity:.55;' if t.get('completed') else ''}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; color:#60a5fa;">{t['day']} · {t['task_type']}</div>
                    <span class="pill pill-b">{t['allocated_time']}</span>
                </div>
                <div style="color:#fff; font-weight:600; margin-top:4px;">📌 {t['focus_topic']}</div>
                <div style="font-size:.82rem; color:#cbd5e1; margin-top:3px;">{t.get('details', '')}</div>
                <div style="font-size:.78rem; color:#9ca3af; margin-top:3px;">💡 {t['reason']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No roadmap yet. Take a quiz (or wait for a published result), then click ‘Generate / Adapt Roadmap Now’.")


elif page == "⚙️  Settings":
    st.markdown('<div class="pg-title">⚙️ System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Configure LLM providers, model weights, and backend connectivity.</div>', unsafe_allow_html=True)

    # ── Change my password (every role) ─────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">🔒 Change Password</div>', unsafe_allow_html=True)
    with st.form("change_password_form", clear_on_submit=True):
        _cur = st.text_input("Current password", type="password")
        _new = st.text_input("New password", type="password",
                             help="At least 8 characters, with letters and numbers.")
        _conf = st.text_input("Confirm new password", type="password")
        _ok = st.form_submit_button("Save new password")
    if _ok:
        if not _cur or not _new or not _conf:
            st.error("Please fill in all three fields.")
        elif _new != _conf:
            st.error("New password and confirmation do not match.")
        else:
            _r = api_post("/auth/change-password", json_data={
                "current_password": _cur, "new_password": _new, "confirm_password": _conf})
            if _r is not None and _r.status_code == 200:
                st.success(_r.json().get("message", "Password changed."))
            else:
                try:
                    _d = _r.json().get("detail") if _r is not None else None
                    if isinstance(_d, list):  # pydantic validation errors
                        _d = "; ".join(e.get("msg", "") for e in _d)
                except Exception:
                    _d = None
                st.error(_d or "Could not change the password — is the backend running?")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Backend Connection</div>', unsafe_allow_html=True)
    st.write(f"Connected to backend: `{API_BASE}`")
    st.markdown('</div>', unsafe_allow_html=True)
