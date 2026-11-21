"""
Academic structure & administration (prefix /academic)
=====================================================
ADMIN: batches, sections, teachers (unique teacher IDs) and their subject/batch assignments,
student registration & editing, password resets, activation, 360° registration-number search,
teacher-ID search, institution overview.
"""

import re
import secrets
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import access as AC
from backend import results_service as RS
from backend.auth import get_password_hash, require_role
from backend.database import (
    Batch, Course, CourseRegistration, Department, ExamPaper, FinalResult, OmrResponse, Student, StudentBooklet,
    TeacherAssignment, TeacherProfile, User, get_db,
)

router = APIRouter()
ADMIN = Depends(require_role(["admin"]))
STAFF = Depends(require_role(["admin", "faculty"]))

CODE_RE = r"^[A-Za-z0-9][A-Za-z0-9\-_]{1,29}$"


def _temp_password() -> str:
    return secrets.token_urlsafe(8) + "9a"  # letters + digits guaranteed


# ── schemas ───────────────────────────────────────────────────────────────────

class BatchIn(BaseModel):
    batch_code: str = Field(..., pattern=CODE_RE)
    programme: str = Field("B.Tech", max_length=100)
    course_code: str = Field("", max_length=20)
    department: str = Field("Computer Science & Engineering", max_length=100)
    section: str = Field("A", min_length=1, max_length=10)
    semester: str = Field("Semester 1", max_length=20)
    academic_session: str = Field("2026-27", max_length=20)


class BatchPatch(BaseModel):
    programme: Optional[str] = None
    course_code: Optional[str] = None
    department: Optional[str] = None
    section: Optional[str] = None
    semester: Optional[str] = None
    academic_session: Optional[str] = None
    is_active: Optional[bool] = None


class TeacherIn(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    department: str = Field("Computer Science & Engineering", max_length=100)
    department_code: str = Field("CSE", pattern=r"^[A-Za-z]{2,8}$")
    designation: str = Field("Assistant Professor", max_length=100)
    phone: str = Field("", max_length=30)


class TeacherPatch(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class AssignmentIn(BaseModel):
    subject_code: str
    batch_code: str


class StudentIn(BaseModel):
    registration_number: str = Field(..., pattern=CODE_RE)
    name: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    programme: str = Field("B.Tech", max_length=100)
    department: str = Field("Computer Science & Engineering", max_length=100)
    batch_code: Optional[str] = None
    semester: Optional[str] = None
    subject_codes: List[str] = []


class StudentPatch(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    programme: Optional[str] = None
    department: Optional[str] = None
    batch_code: Optional[str] = None
    semester: Optional[str] = None
    subject_codes: Optional[List[str]] = None
    is_active: Optional[bool] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _batch_dict(db: Session, b: Batch) -> dict:
    teachers = (db.query(TeacherAssignment).filter(TeacherAssignment.batch_id == b.id).all())
    return {"id": b.id, "batch_code": b.batch_code, "programme": b.programme, "course_code": b.course_code,
            "department": b.department, "section": b.section, "semester": b.semester,
            "academic_session": b.academic_session, "is_active": b.is_active,
            "total_students": db.query(Student).filter(Student.batch_id == b.id).count(),
            "assignments": [{"teacher_code": a.teacher.teacher_code, "teacher_name": a.teacher.user.full_name,
                             "subject_code": a.course.code, "subject_name": a.course.name} for a in teachers]}


def _teacher_dict(db: Session, t: TeacherProfile, with_stats: bool = False) -> dict:
    d = {"id": t.id, "teacher_code": t.teacher_code, "username": t.user.username, "full_name": t.user.full_name,
         "email": t.user.email, "department": t.department, "designation": t.designation, "phone": t.phone,
         "is_active": t.user.is_active,
         "assignments": [{"id": a.id, "subject_code": a.course.code, "subject_name": a.course.name,
                          "batch_code": a.batch.batch_code, "section": a.batch.section, "semester": a.batch.semester,
                          "programme": a.batch.programme,
                          "students": db.query(Student).filter(Student.batch_id == a.batch_id).count()}
                         for a in t.assignments]}
    if with_stats:
        d["stats"] = teacher_stats(db, t)
    return d


def teacher_stats(db: Session, t: TeacherProfile) -> dict:
    pairs = {(a.course.code, a.batch_id) for a in t.assignments}
    batch_ids = {b for _, b in pairs}
    students = db.query(Student).filter(Student.batch_id.in_(batch_ids)).all() if batch_ids else []
    rows = RS.subject_rows(db, students, allowed_pairs=pairs)
    by_subject, by_batch = {}, {}
    for r in rows:
        by_subject.setdefault(r["subject_code"], []).append(r)
        by_batch.setdefault(r["batch_code"], []).append(r)
    return {"total_batches": len(batch_ids), "total_subjects": len({c for c, _ in pairs}),
            "total_students": len(students), "overall": RS.summarize(rows),
            "subject_wise": {k: RS.summarize(v) for k, v in by_subject.items()},
            "batch_wise": {k: RS.summarize(v) for k, v in by_batch.items()}}


def _get_batch(db: Session, code: Optional[str]) -> Optional[Batch]:
    if not code:
        return None
    b = db.query(Batch).filter(func.upper(Batch.batch_code) == code.strip().upper()).first()
    if not b:
        raise HTTPException(status_code=404, detail=f"Batch {code} not found")
    return b


def _set_subjects(db: Session, st: Student, codes: List[str]):
    wanted = {c.strip().upper() for c in codes if c.strip()}
    courses = db.query(Course).filter(Course.code.in_(wanted)).all() if wanted else []
    missing = wanted - {c.code for c in courses}
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown subject code(s): {', '.join(sorted(missing))}")
    keep = {c.id for c in courses}
    for reg in list(st.course_registrations):
        if reg.course_id not in keep:
            db.delete(reg)
    have = {r.course_id for r in st.course_registrations}
    for c in courses:
        if c.id not in have:
            db.add(CourseRegistration(student_id=st.id, course_id=c.id, semester=st.semester,
                                      academic_year=st.academic_session or ""))


def _find_student(db: Session, reg: str) -> Student:
    st = db.query(Student).filter((func.upper(Student.registration_number) == reg.strip().upper()) |
                                  (func.upper(Student.student_code) == reg.strip().upper())).first()
    if not st:
        raise HTTPException(status_code=404, detail=f"No student with registration number {reg}")
    return st


# ── overview ──────────────────────────────────────────────────────────────────

@router.get("/overview", summary="Institution overview counts", dependencies=[ADMIN])
def overview(db: Session = Depends(get_db)):
    booklets = db.query(StudentBooklet).all()
    status = {}
    for b in booklets:
        status[b.approval_status or "pending"] = status.get(b.approval_status or "pending", 0) + 1
    rows = RS.subject_rows(db, db.query(Student).all())
    return {
        "students": db.query(Student).count(), "teachers": db.query(TeacherProfile).count(),
        "courses": db.query(Course).count(), "programmes": len({b.programme for b in db.query(Batch).all()}),
        "batches": db.query(Batch).count(), "sections": db.query(Batch.section).distinct().count(),
        "subjects": db.query(Course).count(), "departments": db.query(Department).count(),
        "examinations": db.query(ExamPaper).count(),
        "answer_sheets": len(booklets), "evaluated": sum(v for k, v in status.items() if k != "pending"),
        "pending_evaluation": status.get("pending", 0), "awaiting_approval": status.get("in_review", 0),
        "published": status.get("published", 0) + db.query(FinalResult).filter(FinalResult.status == "published").count(),
        "omr_sheets": db.query(OmrResponse).count(),
        "omr_needing_verification": db.query(OmrResponse).filter(OmrResponse.needs_verification == True).count(),  # noqa: E712
        "results": RS.summarize(rows),
    }


# ── batches ───────────────────────────────────────────────────────────────────

@router.get("/batches", summary="List / search batches", dependencies=[STAFF])
def list_batches(batch_code: Optional[str] = None, course_code: Optional[str] = None, section: Optional[str] = None,
                 semester: Optional[str] = None, session: Optional[str] = None, db: Session = Depends(get_db),
                 user: User = STAFF):
    q = db.query(Batch)
    if user.role == "faculty":
        ids = AC.teacher_batch_ids(db, user)
        q = q.filter(Batch.id.in_(ids)) if ids else q.filter(False)
    if batch_code:
        q = q.filter(Batch.batch_code.ilike(f"%{batch_code.strip()}%"))
    if course_code:
        q = q.filter(Batch.course_code.ilike(course_code.strip()))
    if section:
        q = q.filter(Batch.section.ilike(section.strip()))
    if semester:
        q = q.filter(Batch.semester.ilike(f"%{semester.strip()}%"))
    if session:
        q = q.filter(Batch.academic_session == session.strip())
    return [_batch_dict(db, b) for b in q.order_by(Batch.batch_code).all()]


@router.post("/batches", summary="Create batch", dependencies=[ADMIN])
def create_batch(data: BatchIn, db: Session = Depends(get_db), user: User = ADMIN):
    code = data.batch_code.strip().upper()
    if db.query(Batch).filter(func.upper(Batch.batch_code) == code).first():
        raise HTTPException(status_code=409, detail=f"Batch ID {code} already exists.")
    b = Batch(**{**data.model_dump(), "batch_code": code, "course_code": data.course_code.strip().upper(),
                 "section": data.section.strip().upper()})
    db.add(b)
    db.flush()
    AC.audit(db, user, "BATCH_CREATE", "batch", b.id, data.model_dump())
    db.commit()
    return _batch_dict(db, b)


@router.patch("/batches/{batch_code}", summary="Edit batch", dependencies=[ADMIN])
def update_batch(batch_code: str, data: BatchPatch, db: Session = Depends(get_db), user: User = ADMIN):
    b = _get_batch(db, batch_code)
    changes = {k: v for k, v in data.model_dump().items() if v is not None}
    for k, v in changes.items():
        setattr(b, k, v.strip().upper() if k in ("section", "course_code") and isinstance(v, str) else v)
    for st in db.query(Student).filter(Student.batch_id == b.id).all():  # keep student rows consistent
        st.section, st.semester, st.academic_session = b.section, b.semester, b.academic_session
    AC.audit(db, user, "BATCH_UPDATE", "batch", b.id, changes)
    db.commit()
    return _batch_dict(db, b)


@router.delete("/batches/{batch_code}", summary="Delete an empty batch", dependencies=[ADMIN])
def delete_batch(batch_code: str, db: Session = Depends(get_db), user: User = ADMIN):
    b = _get_batch(db, batch_code)
    if db.query(Student).filter(Student.batch_id == b.id).count():
        raise HTTPException(status_code=409, detail="Move the batch's students to another batch first.")
    AC.audit(db, user, "BATCH_DELETE", "batch", b.id, {"batch_code": b.batch_code})
    db.query(TeacherAssignment).filter(TeacherAssignment.batch_id == b.id).delete()
    db.delete(b)
    db.commit()
    return {"status": "deleted"}


# ── teachers ──────────────────────────────────────────────────────────────────

@router.get("/teachers", summary="List / search teachers", dependencies=[ADMIN])
def list_teachers(q: Optional[str] = None, db: Session = Depends(get_db)):
    rows = db.query(TeacherProfile).join(User, TeacherProfile.user_id == User.id)
    if q:
        like = f"%{q.strip()}%"
        rows = rows.filter((TeacherProfile.teacher_code.ilike(like)) | (User.full_name.ilike(like)) |
                           (User.email.ilike(like)) | (TeacherProfile.department.ilike(like)))
    return [_teacher_dict(db, t) for t in rows.order_by(TeacherProfile.teacher_code).all()]


@router.post("/teachers", summary="Register teacher (generates unique teacher ID)", dependencies=[ADMIN])
def create_teacher(data: TeacherIn, db: Session = Depends(get_db), user: User = ADMIN):
    if db.query(User).filter(func.lower(User.email) == data.email.lower()).first():
        raise HTTPException(status_code=409, detail="A user with this email already exists.")
    prefix = "T" + data.department_code.upper()
    existing = [t.teacher_code for t in db.query(TeacherProfile).filter(TeacherProfile.teacher_code.like(f"{prefix}%")).all()]
    nums = [int(m.group(1)) for c in existing if (m := re.match(rf"^{prefix}(\d+)$", c))]
    code = f"{prefix}{(max(nums) + 1 if nums else 1):04d}"
    pw = _temp_password()
    u = User(username=code, email=data.email, full_name=data.full_name.strip(), role="faculty",
             department=data.department, hashed_password=get_password_hash(pw), is_active=True)
    db.add(u)
    db.flush()
    t = TeacherProfile(user_id=u.id, teacher_code=code, designation=data.designation, department=data.department,
                       phone=data.phone)
    db.add(t)
    db.flush()
    AC.audit(db, user, "TEACHER_CREATE", "teacher", t.id, {"teacher_code": code, "email": data.email})
    db.commit()
    return {**_teacher_dict(db, t), "temporary_password": pw,
            "note": "Share the teacher ID and temporary password securely; it is shown only once."}


@router.get("/teachers/{teacher_code}", summary="Teacher profile with assignments and results", dependencies=[ADMIN])
def get_teacher(teacher_code: str, db: Session = Depends(get_db)):
    t = db.query(TeacherProfile).filter(func.upper(TeacherProfile.teacher_code) == teacher_code.strip().upper()).first()
    if not t:
        raise HTTPException(status_code=404, detail=f"No teacher with ID {teacher_code}")
    return _teacher_dict(db, t, with_stats=True)


@router.patch("/teachers/{teacher_code}", summary="Edit / activate / deactivate teacher", dependencies=[ADMIN])
def update_teacher(teacher_code: str, data: TeacherPatch, db: Session = Depends(get_db), user: User = ADMIN):
    t = db.query(TeacherProfile).filter(func.upper(TeacherProfile.teacher_code) == teacher_code.strip().upper()).first()
    if not t:
        raise HTTPException(status_code=404, detail="Teacher not found")
    ch = {k: v for k, v in data.model_dump().items() if v is not None}
    if "full_name" in ch: t.user.full_name = ch["full_name"]
    if "email" in ch: t.user.email = ch["email"]
    if "department" in ch: t.department = t.user.department = ch["department"]
    if "designation" in ch: t.designation = ch["designation"]
    if "phone" in ch: t.phone = ch["phone"]
    if "is_active" in ch: t.user.is_active = ch["is_active"]
    AC.audit(db, user, "TEACHER_UPDATE", "teacher", t.id, ch)
    db.commit()
    return _teacher_dict(db, t)


@router.post("/teachers/{teacher_code}/assignments", summary="Assign subject + batch", dependencies=[ADMIN])
def add_assignment(teacher_code: str, data: AssignmentIn, db: Session = Depends(get_db), user: User = ADMIN):
    t = db.query(TeacherProfile).filter(func.upper(TeacherProfile.teacher_code) == teacher_code.strip().upper()).first()
    if not t:
        raise HTTPException(status_code=404, detail="Teacher not found")
    c = db.query(Course).filter(func.upper(Course.code) == data.subject_code.strip().upper()).first()
    if not c:
        raise HTTPException(status_code=404, detail=f"Subject {data.subject_code} not found")
    b = _get_batch(db, data.batch_code)
    if db.query(TeacherAssignment).filter_by(teacher_id=t.id, course_id=c.id, batch_id=b.id).first():
        raise HTTPException(status_code=409, detail="This assignment already exists.")
    a = TeacherAssignment(teacher_id=t.id, course_id=c.id, batch_id=b.id)
    db.add(a)
    AC.audit(db, user, "TEACHER_ASSIGN", "teacher", t.id, {"subject": c.code, "batch": b.batch_code})
    db.commit()
    return _teacher_dict(db, t)


@router.delete("/teachers/{teacher_code}/assignments/{assignment_id}", summary="Remove assignment", dependencies=[ADMIN])
def remove_assignment(teacher_code: str, assignment_id: int, db: Session = Depends(get_db), user: User = ADMIN):
    t = db.query(TeacherProfile).filter(func.upper(TeacherProfile.teacher_code) == teacher_code.strip().upper()).first()
    a = db.query(TeacherAssignment).filter(TeacherAssignment.id == assignment_id).first()
    if not t or not a or a.teacher_id != t.id:
        raise HTTPException(status_code=404, detail="Assignment not found")
    AC.audit(db, user, "TEACHER_UNASSIGN", "teacher", t.id, {"subject": a.course.code, "batch": a.batch.batch_code})
    db.delete(a)
    db.commit()
    return _teacher_dict(db, t)


@router.post("/users/{username}/reset-password", summary="Reset any user's password (temporary password)", dependencies=[ADMIN])
def reset_password(username: str, db: Session = Depends(get_db), user: User = ADMIN):
    u = db.query(User).filter(func.upper(User.username) == username.strip().upper()).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    pw = _temp_password()
    u.hashed_password = get_password_hash(pw)
    AC.audit(db, user, "PASSWORD_RESET", "user", u.id, {"username": u.username})
    db.commit()
    return {"username": u.username, "temporary_password": pw, "note": "Shown once — share it securely."}


# ── students ──────────────────────────────────────────────────────────────────

@router.post("/students", summary="Register a student", dependencies=[ADMIN])
def create_student(data: StudentIn, db: Session = Depends(get_db), user: User = ADMIN):
    reg = data.registration_number.strip().upper()
    if db.query(Student).filter((func.upper(Student.registration_number) == reg) |
                                (func.upper(Student.student_code) == reg)).first() or \
            db.query(User).filter(func.upper(User.username) == reg).first():
        raise HTTPException(status_code=409, detail=f"Registration number {reg} is already in use.")
    if db.query(User).filter(func.lower(User.email) == data.email.lower()).first():
        raise HTTPException(status_code=409, detail="A user with this email already exists.")
    b = _get_batch(db, data.batch_code)
    pw = _temp_password()
    u = User(username=reg, email=data.email, full_name=data.name.strip(), role="student", department=data.department,
             registration_number=reg, hashed_password=get_password_hash(pw), is_active=True)
    db.add(u)
    db.flush()
    st = Student(user_id=u.id, student_code=reg, registration_number=reg, name=data.name.strip(), email=data.email,
                 department=data.department, programme=data.programme,
                 semester=(b.semester if b else (data.semester or "Semester 1")),
                 batch=(b.academic_session if b else ""), batch_id=b.id if b else None,
                 section=b.section if b else "", academic_session=b.academic_session if b else "", is_active=True)
    db.add(st)
    db.flush()
    _set_subjects(db, st, data.subject_codes)
    AC.audit(db, user, "STUDENT_CREATE", "student", st.id, {"registration_number": reg, "batch": data.batch_code})
    db.commit()
    return {**student_profile_dict(db, st), "temporary_password": pw,
            "note": "Share the temporary password securely; it is shown only once."}


@router.patch("/students/{reg}", summary="Edit student / assign batch, section, semester, subjects", dependencies=[ADMIN])
def update_student(reg: str, data: StudentPatch, db: Session = Depends(get_db), user: User = ADMIN):
    st = _find_student(db, reg)
    ch = {k: v for k, v in data.model_dump().items() if v is not None}
    if "name" in ch:
        st.name = ch["name"]
        if st.user: st.user.full_name = ch["name"]
    if "email" in ch:
        st.email = ch["email"]
        if st.user: st.user.email = ch["email"]
    if "programme" in ch: st.programme = ch["programme"]
    if "department" in ch:
        st.department = ch["department"]
        if st.user: st.user.department = ch["department"]
    if "batch_code" in ch:
        b = _get_batch(db, ch["batch_code"])
        st.batch_id, st.section, st.semester, st.academic_session = b.id, b.section, b.semester, b.academic_session
    elif "semester" in ch:
        st.semester = ch["semester"]
    if "subject_codes" in ch:
        _set_subjects(db, st, ch["subject_codes"])
    if "is_active" in ch:
        st.is_active = ch["is_active"]
        if st.user: st.user.is_active = ch["is_active"]
    AC.audit(db, user, "STUDENT_UPDATE", "student", st.id, ch)
    db.commit()
    db.refresh(st)
    return student_profile_dict(db, st)


def student_profile_dict(db: Session, st: Student) -> dict:
    b = st.batch_rel
    return {"id": st.id, "registration_number": st.registration_number or st.student_code, "name": st.name,
            "email": st.email, "programme": st.programme, "department": st.department, "semester": st.semester,
            "batch_code": b.batch_code if b else "", "section": st.section or (b.section if b else ""),
            "academic_session": st.academic_session or (b.academic_session if b else ""),
            "course_code": b.course_code if b else "", "is_active": st.is_active and (st.user.is_active if st.user else True),
            "username": st.user.username if st.user else None,
            "subjects": [{"code": r.course.code, "name": r.course.name, "credits": r.course.credits}
                         for r in st.course_registrations if r.course]}


@router.get("/students", summary="Search students (reg no, name, batch, section, semester)", dependencies=[STAFF])
def search_students(q: Optional[str] = None, batch_code: Optional[str] = None, section: Optional[str] = None,
                    semester: Optional[str] = None, subject_code: Optional[str] = None, active: Optional[bool] = None,
                    limit: int = Query(100, le=500), db: Session = Depends(get_db), user: User = STAFF):
    qq = AC.scoped_students(db, user, subject_code)
    if q:
        like = f"%{q.strip()}%"
        qq = qq.filter((Student.registration_number.ilike(like)) | (Student.name.ilike(like)) |
                       (Student.student_code.ilike(like)) | (Student.email.ilike(like)))
    if batch_code:
        b = _get_batch(db, batch_code)
        qq = qq.filter(Student.batch_id == b.id)
    if section:
        qq = qq.filter(Student.section.ilike(section.strip()))
    if semester:
        qq = qq.filter(Student.semester.ilike(f"%{semester.strip()}%"))
    if subject_code and user.role == "admin":
        c = db.query(Course).filter(func.upper(Course.code) == subject_code.strip().upper()).first()
        ids = [r.student_id for r in db.query(CourseRegistration).filter(CourseRegistration.course_id == (c.id if c else -1))]
        qq = qq.filter(Student.id.in_(ids))
    if active is not None:
        qq = qq.filter(Student.is_active == active)
    return [student_profile_dict(db, s) for s in qq.order_by(Student.registration_number).limit(limit).all()]


@router.get("/students/{reg}/profile360", summary="Everything about one registration number", dependencies=[STAFF])
def profile360(reg: str, db: Session = Depends(get_db), user: User = STAFF):
    st = _find_student(db, reg)
    allowed = None
    if user.role == "faculty":
        pairs = set(AC.teacher_pairs(db, user))
        if st.batch_id not in {b for _, b in pairs}:
            raise HTTPException(status_code=404, detail=f"No student with registration number {reg}")
        allowed = pairs
    rows = RS.subject_rows(db, [st], allowed_pairs=allowed)
    credits = {r.course.code: r.course.credits for r in st.course_registrations if r.course}
    booklets = [{"booklet_id": b.id, "subject_code": b.course_code, "exam_name": b.exam_name,
                 "status": RS.booklet_status_label(b), "pages": b.total_pages,
                 "original_url": f"/student/answer-sheet/{b.id}/original",
                 "evaluated_url": f"/student/answer-sheet/{b.id}/evaluated" if b.is_published else None,
                 "report_url": f"/results/booklet/{b.id}/report.pdf"}
                for b in db.query(StudentBooklet).filter(StudentBooklet.student_id == st.id).order_by(StudentBooklet.id).all()
                if allowed is None or (b.course_code, st.batch_id) in allowed]
    sem = {}
    for r in rows:
        sem.setdefault(r["semester"] or "—", []).append(r)
    return {**student_profile_dict(db, st), "results": rows, "summary": RS.summarize(rows),
            "semester_wise": {k: RS.summarize(v) for k, v in sem.items()},
            "cgpa": RS.cgpa([r for r in rows if r.get("status") == "published"], credits),
            "answer_sheets": booklets,
            "omr": [{"subject_code": o.key.course_code, "exam_name": o.key.exam_name, "score": o.score,
                     "max": o.max_score, "needs_verification": o.needs_verification}
                    for o in db.query(OmrResponse).filter(OmrResponse.student_id == st.id).all()]}


@router.get("/students/{reg}/report.pdf", summary="Download complete examination report", dependencies=[STAFF])
def student_report(reg: str, db: Session = Depends(get_db), user: User = STAFF):
    from backend.reports import student_report_pdf
    data = profile360(reg, db, user)
    pdf = student_report_pdf(data)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="report_{data["registration_number"]}.pdf"'})


@router.get("/subjects", summary="Subjects (courses) list", dependencies=[STAFF])
def subjects(db: Session = Depends(get_db)):
    return [{"code": c.code, "name": c.name, "credits": c.credits, "semester": c.semester} for c in
            db.query(Course).order_by(Course.code).all()]


@router.get("/data-quality", summary="Duplicate answer sheets and other data problems", dependencies=[ADMIN])
def data_quality(db: Session = Depends(get_db)):
    groups = {}
    for b in db.query(StudentBooklet).all():
        groups.setdefault((b.student_id, (b.course_code or "").upper(), (b.exam_name or "").strip().lower()), []).append(b)
    dups = []
    for (sid, cc, _), items in groups.items():
        if len(items) > 1 and sid:
            st = items[0].student
            dups.append({"registration_number": st.registration_number if st else "", "student_name": st.name if st else "",
                         "subject_code": cc, "exam_name": items[0].exam_name, "count": len(items),
                         "booklet_ids": [b.id for b in items],
                         "published": sum(1 for b in items if b.is_published)})
    unlinked = db.query(StudentBooklet).filter(StudentBooklet.student_id == None).count()  # noqa: E711
    no_batch = db.query(Student).filter(Student.batch_id == None).count()  # noqa: E711
    return {"duplicate_answer_sheets": dups, "booklets_without_student": unlinked, "students_without_batch": no_batch}
