"""
Role-based data scoping (server-side)
=====================================
* ADMIN   — every academic record.
* TEACHER — (role "faculty") only students in batches they are assigned to, and only for the
            subjects (course codes) assigned for that batch. A teacher with no assignments sees nothing.
* STUDENT — only their own records (enforced in student routes).

Every teacher-facing endpoint must go through these helpers; the frontend is never trusted.
"""

import json
from typing import Iterable, List, Optional, Set, Tuple

from fastapi import HTTPException
from sqlalchemy import and_, false, or_
from sqlalchemy.orm import Query, Session

from backend.database import (
    AuditLog, Batch, Course, Notification, Student, StudentBooklet, TeacherAssignment, TeacherProfile, User,
)

Pair = Tuple[str, int]  # (course_code, batch_id)


def is_admin(user: User) -> bool:
    return user is not None and user.role == "admin"


def teacher_profile(db: Session, user: User) -> Optional[TeacherProfile]:
    return db.query(TeacherProfile).filter(TeacherProfile.user_id == user.id).first()


def teacher_pairs(db: Session, user: User) -> List[Pair]:
    prof = teacher_profile(db, user)
    if not prof:
        return []
    rows = (db.query(Course.code, TeacherAssignment.batch_id)
            .join(TeacherAssignment, TeacherAssignment.course_id == Course.id)
            .filter(TeacherAssignment.teacher_id == prof.id).all())
    return [(c.upper(), b) for c, b in rows]


def teacher_batch_ids(db: Session, user: User) -> Set[int]:
    return {b for _, b in teacher_pairs(db, user)}


def _pair_filter(pairs: Iterable[Pair]):
    conds = [and_(StudentBooklet.course_code == c, Student.batch_id == b) for c, b in pairs]
    return or_(*conds) if conds else false()


def scoped_booklets(db: Session, user: User) -> Query:
    """StudentBooklet query limited to what this user may see."""
    q = db.query(StudentBooklet)
    if is_admin(user):
        return q
    if user.role != "faculty":
        return q.filter(false())
    return q.join(Student, StudentBooklet.student_id == Student.id).filter(_pair_filter(teacher_pairs(db, user)))


def can_access_booklet(db: Session, user: User, booklet: StudentBooklet) -> bool:
    if is_admin(user):
        return True
    if user.role != "faculty" or not booklet or not booklet.student:
        return False
    return ((booklet.course_code or "").upper(), booklet.student.batch_id) in set(teacher_pairs(db, user))


def booklet_or_404(db: Session, user: User, booklet_id: int) -> StudentBooklet:
    b = db.query(StudentBooklet).filter(StudentBooklet.id == booklet_id).first()
    # same 404 for "missing" and "not yours" — don't leak existence
    if not b or not can_access_booklet(db, user, b):
        raise HTTPException(status_code=404, detail="Booklet not found")
    return b


def scoped_students(db: Session, user: User, course_code: Optional[str] = None) -> Query:
    q = db.query(Student)
    if is_admin(user):
        return q
    if user.role != "faculty":
        return q.filter(false())
    pairs = teacher_pairs(db, user)
    if course_code:
        pairs = [p for p in pairs if p[0] == course_code.upper()]
    ids = {b for _, b in pairs}
    return q.filter(Student.batch_id.in_(ids)) if ids else q.filter(false())


def can_teach(db: Session, user: User, student: Student, course_code: str) -> bool:
    if is_admin(user):
        return True
    return user.role == "faculty" and student is not None and \
        ((course_code or "").upper(), student.batch_id) in set(teacher_pairs(db, user))


def require_can_teach(db: Session, user: User, student: Student, course_code: str):
    if not can_teach(db, user, student, course_code):
        raise HTTPException(status_code=403,
                            detail=f"You are not assigned to teach {course_code.upper()} to this student's batch.")


# ── audit + notifications ─────────────────────────────────────────────────────

def audit(db: Session, user: Optional[User], action: str, target_type: str, target_id, details: dict):
    db.add(AuditLog(user_id=user.id if user else None, username=user.username if user else "system",
                    role=user.role if user else "system", action=action, target_type=target_type,
                    target_id=str(target_id), details=json.dumps(details, default=str)))


def notify(db: Session, user_id: Optional[int], kind: str, title: str, body: str = ""):
    if user_id:
        db.add(Notification(user_id=user_id, kind=kind, title=title[:200], body=body))


def notify_student(db: Session, student: Optional[Student], kind: str, title: str, body: str = ""):
    if student is not None:
        notify(db, student.user_id, kind, title, body)
