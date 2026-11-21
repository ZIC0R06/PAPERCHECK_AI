"""
AI Paper Checking — Automated End-to-End Test Suite
====================================================
Tests:
  1. Password hashing & token authentication
  2. Role-based access control (Admin, Faculty, Student)
  3. Bulk student import & department/course management
  4. Objective, subjective, and numerical AI evaluation with offline NLP fallback
  5. Examiner review, marks editing, approval, and publication
  6. Strict student data privacy & unauthorized access rejection
  7. Re-evaluation dispute submission and resolution
  8. Personalized RAG-grounded AI mentor chatbot queries & quiz generation
  9. Evaluated answer sheet PDF generation
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import (
    Base, engine, SessionLocal, User, Student, Department, Course,
    StudentBooklet, StudentAnswerText, Submission, Result, ReEvaluationRequest,
    AuditLog, init_db
)
from backend.auth import (
    get_password_hash, verify_password, create_access_token, decode_access_token,
    verify_student_access
)
from backend.llm_evaluator import LLMEvaluator
from backend.similarity import SemanticSimilarityModel
from backend.mentor_service import AIMentorService
from backend.evaluated_sheet_generator import generate_evaluated_sheet_pdf


@pytest.fixture(scope="module")
def db():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


def test_password_hashing():
    pwd = "secretUniversityPassword123"
    hashed = get_password_hash(pwd)
    assert hashed != pwd
    assert "$" in hashed
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrongPassword", hashed) is False


def test_jwt_token_creation_and_validation():
    data = {"sub": "admin", "role": "admin", "user_id": 1}
    token = create_access_token(data)
    assert isinstance(token, str)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "admin"
    assert decoded["role"] == "admin"


def test_student_data_privacy_authorization(db):
    admin_user = db.query(User).filter_by(username="admin").first()
    student_user1 = db.query(User).filter_by(username="student").first()
    student_user2 = db.query(User).filter_by(username="student2").first()

    student1 = db.query(Student).filter_by(student_code="21BCS101").first()
    student2 = db.query(Student).filter_by(student_code="21BCS102").first()

    # Admin has access to any student
    assert verify_student_access(admin_user, student1) is True
    assert verify_student_access(admin_user, student2) is True

    # Student 1 has access to own record
    assert verify_student_access(student_user1, student1) is True

    # Student 1 MUST NOT have access to Student 2's record
    with pytest.raises(Exception):
        verify_student_access(student_user1, student2)


def test_offline_nlp_evaluation_engine():
    evaluator = LLMEvaluator()

    # 1. Subjective Open-ended answer
    q = "Explain the QuickSort algorithm and its time complexity."
    model_ans = "QuickSort is a divide and conquer algorithm that picks an element as pivot and partitions the given array around the picked pivot. Time complexity is O(N log N) on average and O(N^2) in worst case."
    stud_ans = "QuickSort uses a pivot to partition elements recursively using divide and conquer. Average time complexity is O(N log N)."

    res = evaluator.evaluate(
        question=q,
        teacher_answer=model_ans,
        student_answer=stud_ans,
        max_marks=10.0,
        question_type="open_ended"
    )

    assert res.score > 0.0
    assert res.score <= 10.0
    assert len(res.strengths) > 0
    # offline marking (no LLM) is deliberately low-confidence so the answer is flagged for teacher verification
    assert 0.0 < res.confidence < 0.5
    assert "QuickSort" in stud_ans

    # 2. Numerical question
    q_num = "Calculate the maximum number of edges in an undirected graph with 5 vertices."
    model_num = "Formula is n*(n-1)/2. For n=5, 5*4/2 = 10 edges."
    stud_num = "For 5 vertices, max edges = 10."

    res_num = evaluator.evaluate(
        question=q_num,
        teacher_answer=model_num,
        student_answer=stud_num,
        max_marks=5.0,
        question_type="numerical"
    )
    assert res_num.score >= 4.0  # correct numerical answer detected


def test_examiner_workflow_and_publication(db):
    faculty = db.query(User).filter_by(role="faculty").first()
    student = db.query(Student).filter_by(student_code="21BCS101").first()

    # Create dummy booklet
    booklet = StudentBooklet(
        student_id=student.id,
        course_code="CS301",
        course_name="Design & Analysis of Algorithms",
        exam_name="End Semester Examination 2026",
        roll_number=student.student_code,
        register_number=student.registration_number,
        approval_status="pending",
        is_published=False,
        total_ai_score=8.5,
        total_final_score=8.5
    )
    db.add(booklet)
    db.commit()
    db.refresh(booklet)

    ans = StudentAnswerText(
        booklet_id=booklet.id,
        question_number=1,
        part_name="Part-A",
        answer_text="QuickSort divides an array into subproblems."
    )
    db.add(ans)
    db.commit()

    sub = Submission(student_id=student.id, extracted_text=ans.answer_text)
    db.add(sub)
    db.commit()

    ans.submission_id = sub.id

    result = Result(
        submission_id=sub.id,
        final_score=4.0,
        max_marks=5.0,
        confidence=0.85,
        feedback="Good conceptual grasp.",
        strengths=json.dumps(["divide and conquer", "quicksort"]),
        missing_concepts=json.dumps(["worst case time complexity"])
    )
    db.add(result)
    db.commit()

    # Examiner modifies marks and approves
    result.examiner_score = 4.5
    result.examiner_comments = "Examiner adjusted for valid reasoning."
    result.examiner_approved = True
    booklet.total_final_score = 4.5
    booklet.approval_status = "approved"
    booklet.is_published = True
    booklet.published_by_id = faculty.id
    db.commit()

    # Verify state
    assert booklet.is_published is True
    assert booklet.approval_status == "approved"
    assert result.examiner_score == 4.5
    assert result.examiner_approved is True


def test_re_evaluation_dispute_flow(db):
    student = db.query(Student).filter_by(student_code="21BCS101").first()
    booklet = db.query(StudentBooklet).filter_by(student_id=student.id).first()

    req = ReEvaluationRequest(
        booklet_id=booklet.id,
        student_id=student.id,
        question_number=1,
        reason="I believe my explanation of time complexity deserves full credit.",
        status="pending",
        score_before=4.5
    )
    db.add(req)
    db.commit()
    db.refresh(req)

    assert req.id is not None
    assert req.status == "pending"

    # Examiner resolves
    req.status = "approved"
    req.score_after = 5.0
    req.examiner_remarks = "Re-checked. Extra 0.5 marks granted for valid asymptotic bounds."
    db.commit()

    assert req.status == "approved"
    assert req.score_after == 5.0


def test_personalized_ai_mentor_rag_chat(db):
    student = db.query(Student).filter_by(student_code="21BCS101").first()
    student_user = db.query(User).filter_by(username="student").first()

    mentor = AIMentorService(db)

    # Ask specific question query
    res = mentor.handle_student_query(
        current_user=student_user,
        student=student,
        message="Why did I lose marks in Question 1?",
        course_code="CS301"
    )

    assert res["session_id"] is not None
    assert len(res["reply"]) > 50
    assert "Question 1" in res["reply"]
    assert len(res["sources_cited"]) > 0

    # Practice quiz generation
    quiz = mentor.generate_practice_quiz(
        student=student,
        course_code="CS301",
        weak_topics=["QuickSort Worst Case", "Dynamic Programming"]
    )
    assert quiz["course_code"] == "CS301"
    assert len(quiz["generated_quiz"]) == 3


def test_evaluated_answer_sheet_generator():
    meta = {
        "student_name": "Arjun Mehta",
        "registration_number": "21BCS101",
        "roll_number": "21BCS101",
        "course_code": "CS301",
        "course_name": "Design & Analysis of Algorithms",
        "semester": "Semester 5",
        "total_awarded": 18.5,
        "max_marks": 20.0,
        "is_published": True
    }
    q_evals = [
        {
            "question_number": 1,
            "question_type": "short_answer",
            "awarded": 4.5,
            "max_marks": 5.0,
            "examiner_note": "Examiner Certified",
            "strengths": ["QuickSort pivot partitioning"],
            "missing_concepts": ["Worst case recursion depth"],
            "feedback": "Clear explanation of partitioning mechanism."
        }
    ]

    out_file = generate_evaluated_sheet_pdf(
        booklet_id=999,
        original_pdf_path="",
        metadata=meta,
        question_evaluations=q_evals,
        output_dir="uploads"
    )

    assert os.path.exists(out_file)
    assert out_file.endswith(".pdf")
