"""
AI Paper Checking — University Admin Routes
============================================
Comprehensive university administrative dashboard endpoints:
  • Student, Faculty, Department, and Course management
  • Bulk import of student records via CSV/Excel
  • Batch upload of answer sheets with student mapping and duplicate detection
  • Audit logs & downloadable institutional analytics reports
"""

import os
import re
import secrets
import csv
import io
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import (
    get_db, User, Student, Department, Course, CourseRegistration,
    ExamPaper, StudentBooklet, StudentAnswerText, AuditLog, Result, Submission
)
from backend.auth import require_role, get_password_hash
from backend.schemas import (
    UserCreateSchema, DepartmentCreateSchema, CourseCreateSchema
)

router = APIRouter(dependencies=[Depends(require_role(["admin"]))])


# ─────────────────────────────────────────────────────────────
# 1. User Management (Admins, Faculty, Students)
# ─────────────────────────────────────────────────────────────

@router.get("/users", summary="List system users")
def list_users(
    role: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(User)
    if role:
        query = query.filter(User.role == role)
    if search:
        s = f"%{search.strip()}%"
        query = query.filter((User.username.ilike(s)) | (User.full_name.ilike(s)) | (User.email.ilike(s)))

    total = query.count()
    users = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "department": u.department,
                "registration_number": u.registration_number,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat() if u.created_at else None
            }
            for u in users
        ]
    }


@router.post("/users", summary="Create user account")
def create_user(
    data: UserCreateSchema,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_role(["admin"]))
):
    existing = db.query(User).filter((User.username == data.username) | (User.email == data.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or Email already registered.")

    new_user = User(
        username=data.username.strip(),
        email=data.email.strip(),
        full_name=data.full_name.strip(),
        hashed_password=get_password_hash(data.password),
        role=data.role,
        department=data.department,
        registration_number=data.registration_number,
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # If student, link or create student record
    if data.role == "student":
        student = db.query(Student).filter(Student.student_code == data.username).first()
        if not student and data.registration_number:
            student = db.query(Student).filter(Student.registration_number == data.registration_number).first()
        if not student:
            student = Student(
                user_id=new_user.id,
                student_code=data.registration_number or data.username,
                name=data.full_name,
                email=data.email,
                registration_number=data.registration_number or data.username,
                department=data.department,
                is_active=True
            )
            db.add(student)
            db.commit()
        else:
            student.user_id = new_user.id
            db.commit()

    # Audit log
    audit = AuditLog(
        user_id=admin_user.id,
        username=admin_user.username,
        role=admin_user.role,
        action="USER_CREATE",
        target_type="user",
        target_id=str(new_user.id),
        details=json.dumps({"created_user": new_user.username, "role": new_user.role})
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "user_id": new_user.id, "username": new_user.username}


@router.delete("/users/{user_id}", summary="Deactivate user")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_role(["admin"]))
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.username == "admin":
        raise HTTPException(status_code=400, detail="Cannot deactivate root admin account")

    user.is_active = False
    audit = AuditLog(
        user_id=admin_user.id,
        username=admin_user.username,
        role=admin_user.role,
        action="USER_DEACTIVATE",
        target_type="user",
        target_id=str(user.id),
        details=json.dumps({"deactivated_user": user.username})
    )
    db.add(audit)
    db.commit()
    return {"status": "success", "message": f"User {user.username} deactivated."}


# ─────────────────────────────────────────────────────────────
# 2. Departments & Courses Management
# ─────────────────────────────────────────────────────────────

@router.get("/departments", summary="List academic departments")
def list_departments(db: Session = Depends(get_db)):
    depts = db.query(Department).all()
    return [
        {"id": d.id, "code": d.code, "name": d.name, "courses_count": len(d.courses)}
        for d in depts
    ]


@router.post("/departments", summary="Create academic department")
def create_department(data: DepartmentCreateSchema, db: Session = Depends(get_db)):
    existing = db.query(Department).filter(Department.code == data.code.strip().upper()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Department code already exists.")
    dept = Department(code=data.code.strip().upper(), name=data.name.strip())
    db.add(dept)
    db.commit()
    return {"status": "success", "id": dept.id, "code": dept.code}


@router.get("/courses", summary="List courses")
def list_courses(db: Session = Depends(get_db)):
    courses = db.query(Course).all()
    return [
        {
            "id": c.id,
            "code": c.code,
            "name": c.name,
            "department": c.department_rel.name if c.department_rel else "General",
            "credits": c.credits,
            "semester": c.semester,
            "has_syllabus_notes": bool(c.syllabus_notes)
        }
        for c in courses
    ]


@router.post("/courses", summary="Create course")
def create_course(data: CourseCreateSchema, db: Session = Depends(get_db)):
    existing = db.query(Course).filter(Course.code == data.code.strip().upper()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Course code already exists.")
    course = Course(
        code=data.code.strip().upper(),
        name=data.name.strip(),
        department_id=data.department_id,
        credits=data.credits,
        semester=data.semester,
        syllabus_notes=data.syllabus_notes or ""
    )
    db.add(course)
    db.commit()
    return {"status": "success", "id": course.id, "code": course.code}


# ─────────────────────────────────────────────────────────────
# 3. Students Management & Bulk CSV / Excel Import
# ─────────────────────────────────────────────────────────────

@router.get("/students", summary="List registered students")
def list_students(
    search: Optional[str] = None,
    department: Optional[str] = None,
    semester: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(Student)
    if department:
        query = query.filter(Student.department == department)
    if semester:
        query = query.filter(Student.semester == semester)
    if search:
        s = f"%{search.strip()}%"
        query = query.filter((Student.name.ilike(s)) | (Student.student_code.ilike(s)) | (Student.registration_number.ilike(s)))

    total = query.count()
    students = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "students": [
            {
                "id": s.id,
                "name": s.name,
                "student_code": s.student_code,
                "registration_number": s.registration_number or s.student_code,
                "email": s.email,
                "department": s.department,
                "programme": s.programme,
                "semester": s.semester,
                "batch": s.batch,
                "registered_courses": len(s.course_registrations),
                "booklets_count": len(s.booklets)
            }
            for s in students
        ]
    }


@router.post("/students/bulk-import", summary="Bulk import students via CSV/Excel")
async def bulk_import_students(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_role(["admin"]))
):
    """
    Imports student records from CSV or Excel file.
    Expected headers: student_name, registration_number, email, department, programme, semester, batch
    """
    content = await file.read()
    filename = file.filename.lower()

    records = []
    if filename.endswith(".csv"):
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        records = list(reader)
    elif filename.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(content))
            records = df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")

    imported_count = 0
    duplicate_count = 0
    duplicates = []
    temp_credentials = []  # shown once to the admin; never stored in plaintext

    for row in records:
        # Normalize keys
        row_norm = {str(k).strip().lower().replace(" ", "_"): str(v).strip() for k, v in row.items() if k}
        reg_no = row_norm.get("registration_number") or row_norm.get("reg_no") or row_norm.get("student_code") or row_norm.get("roll_number")
        name = row_norm.get("student_name") or row_norm.get("name") or "Student"
        email = row_norm.get("email") or f"{reg_no}@university.edu" if reg_no else ""
        dept = row_norm.get("department") or "Computer Science & Engineering"
        prog = row_norm.get("programme") or "B.Tech"
        sem = row_norm.get("semester") or "Semester 5"
        batch = row_norm.get("batch") or "2022-2026"

        if not reg_no or reg_no == "nan":
            continue

        existing = db.query(Student).filter(
            (Student.student_code == reg_no) | (Student.registration_number == reg_no)
        ).first()

        if existing:
            duplicate_count += 1
            duplicates.append(reg_no)
            continue

        # Create User account for student
        user_acct = db.query(User).filter(User.username == reg_no).first()
        if not user_acct:
            temp_pw = secrets.token_urlsafe(9)
            temp_credentials.append({"registration_number": reg_no, "temporary_password": temp_pw})
            user_acct = User(
                username=reg_no,
                email=email,
                full_name=name,
                hashed_password=get_password_hash(temp_pw),
                role="student",
                department=dept,
                registration_number=reg_no,
                is_active=True
            )
            db.add(user_acct)
            db.commit()
            db.refresh(user_acct)

        student = Student(
            user_id=user_acct.id,
            student_code=reg_no,
            name=name,
            email=email,
            registration_number=reg_no,
            department=dept,
            programme=prog,
            semester=sem,
            batch=batch,
            is_active=True
        )
        db.add(student)
        imported_count += 1

    db.commit()

    # Audit log
    audit = AuditLog(
        user_id=admin_user.id,
        username=admin_user.username,
        role=admin_user.role,
        action="STUDENT_BULK_IMPORT",
        target_type="student",
        target_id=f"count:{imported_count}",
        details=json.dumps({"imported": imported_count, "duplicates": duplicate_count, "filename": file.filename})
    )
    db.add(audit)
    db.commit()

    return {
        "status": "success",
        "total_rows_processed": len(records),
        "imported_count": imported_count,
        "duplicate_count": duplicate_count,
        "duplicates": duplicates[:10],
        "temporary_credentials": temp_credentials,
    }


# ─────────────────────────────────────────────────────────────
# 4. Batch Upload & Student Mapping of Answer Sheets
# ─────────────────────────────────────────────────────────────

@router.post("/booklets/batch-upload", summary="Batch upload answer sheets with student mapping")
async def batch_upload_answer_sheets(
    files: List[UploadFile] = File(...),
    course_code: str = Form(""),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_role(["admin"]))
):
    """
    Accepts batch upload of student answer sheets (PDFs/images).
    Maps files to students by parsing registration number from filename or booklet metadata.
    Detects duplicate uploads and unmatched students.
    """
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    mapped_count = 0
    unmatched = []
    duplicates = []

    for file in files:
        filename = file.filename
        content = await file.read()
        save_path = upload_dir / filename

        # Pattern match registration number from filename: e.g. 21BCS101_CS301.pdf or 21BCS101.pdf
        match = re.search(r'([0-9]{2}[A-Za-z]{2,4}[0-9]{2,5})', filename, re.IGNORECASE)
        reg_no = match.group(1).upper() if match else None

        student = None
        if reg_no:
            student = db.query(Student).filter(
                (Student.registration_number.ilike(reg_no)) | (Student.student_code.ilike(reg_no))
            ).first()

        if not student:
            unmatched.append({"filename": filename, "reason": "No matching student registration number detected in filename"})
            continue

        # Check duplicate upload for this student
        existing_booklet = db.query(StudentBooklet).filter(
            StudentBooklet.student_id == student.id,
            StudentBooklet.course_code == course_code if course_code else True,
            StudentBooklet.pdf_path.ilike(f"%{filename}%")
        ).first()

        if existing_booklet:
            duplicates.append({"filename": filename, "student": student.name, "reg_no": student.registration_number})
            continue

        with open(save_path, "wb") as f:
            f.write(content)

        booklet = StudentBooklet(
            student_id=student.id,
            pdf_path=str(save_path),
            course_code=course_code,
            roll_number=student.student_code,
            register_number=student.registration_number,
            department=student.department,
            programme=student.programme,
            semester=student.semester,
            approval_status="pending",
            is_published=False
        )
        db.add(booklet)
        db.flush()

        ans = StudentAnswerText(
            booklet_id=booklet.id,
            question_number=1,
            part_name="Part-A",
            answer_text=f"Batch uploaded answer sheet ({filename}) assigned to {student.name} ({student.registration_number}).",
            raw_ocr_text=f"Uploaded file: {filename}",
            evaluated=False
        )
        db.add(ans)
        mapped_count += 1

    db.commit()

    return {
        "status": "success",
        "total_files": len(files),
        "mapped_count": mapped_count,
        "duplicate_count": len(duplicates),
        "unmatched_count": len(unmatched),
        "duplicates": duplicates,
        "unmatched": unmatched
    }


# ─────────────────────────────────────────────────────────────
# 5. Audit Trail & Examination Reports
# ─────────────────────────────────────────────────────────────

@router.get("/audit-logs", summary="System audit logs")
def get_audit_logs(
    action: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action)
    logs = query.order_by(AuditLog.id.desc()).offset(offset).limit(limit).all()

    return [
        {
            "id": l.id,
            "username": l.username,
            "role": l.role,
            "action": l.action,
            "target_type": l.target_type,
            "target_id": l.target_id,
            "details": json.loads(l.details) if l.details and l.details.startswith("{") else l.details,
            "created_at": l.created_at.isoformat() if l.created_at else None
        }
        for l in logs
    ]


@router.get("/reports/examination-summary", summary="Institutional Examination Summary Report")
def get_examination_summary_report(db: Session = Depends(get_db)):
    """
    Generates downloadable analytical summary of institutional examination evaluation.
    """
    total_students = db.query(Student).count()
    total_courses = db.query(Course).count()
    total_booklets = db.query(StudentBooklet).count()
    published_booklets = db.query(StudentBooklet).filter(StudentBooklet.is_published == True).count()
    pending_approval = db.query(StudentBooklet).filter(StudentBooklet.approval_status == "pending").count()

    results = db.query(Result).all()
    scores = [r.final_score for r in results if r.final_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "institutional_report": {
            "title": "University Examination Evaluation & Quality Audit Report",
            "generated_at": datetime.utcnow().isoformat(),
            "total_registered_students": total_students,
            "active_courses": total_courses,
            "total_booklets_uploaded": total_booklets,
            "published_results_count": published_booklets,
            "pending_examiner_reviews": pending_approval,
            "mean_score": round(avg_score, 2),
            "evaluation_engine_status": "Active (Hybrid AI + Examiner Verification)"
        }
    }
