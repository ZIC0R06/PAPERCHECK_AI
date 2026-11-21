"""
AI Paper Checking — Database Models & Migrations
=================================================
SQLAlchemy ORM models for the University AI Examination System.

Tables:
  • User               — authentication and role-based permissions (admin, faculty, student)
  • Department         — academic departments (CSE, ECE, Mech, etc.)
  • Course             — university courses and subjects
  • CourseRegistration — maps students to registered courses
  • ExamPaper          — question papers parsed from PDFs
  • ExamPart           — sections/parts (Part-A, Part-B)
  • ExamQuestion       — individual questions with marks
  • Student            — student profiles and academic metadata
  • Rubric             — marking scheme criteria
  • Submission         — question-level student answers
  • Result             — AI grading, examiner edits, approval flags
  • MetricsSnapshot    — accuracy & agreement metrics (MAE, RMSE, kappa)
  • StudentBooklet     — uploaded exam booklets (PDF/images)
  • StudentAnswerText  — segmented student answers
  • ReEvaluationRequest— student grade dispute & re-checking requests
  • ChatSession        — student AI mentor chat conversations
  • ChatMessage        — messages with cited rubrics & course notes
  • CourseMaterial     — course syllabus notes for RAG
  • AuditLog           — immutable audit trail for grade modifications and publication
"""

import json
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'intelligrade.db'}")

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────
# Authentication & Role Models
# ─────────────────────────────────────────────────────────

class User(Base):
    """
    Users table supporting three primary roles:
      - 'admin'   : Full university administration, student bulk import, course setup
      - 'faculty' : Upload question papers, answer keys, review/edit AI marks, approve & publish
      - 'student' : View only own published results, evaluated answer sheets, dispute grades, mentor chat
    """
    __tablename__ = "users"

    id                  = Column(Integer, primary_key=True, index=True)
    username            = Column(String(100), unique=True, index=True, nullable=False)
    email               = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password     = Column(String(255), nullable=False)
    full_name           = Column(String(200), default="")
    role                = Column(String(20), default="student", index=True)  # 'admin', 'faculty', 'student'
    department          = Column(String(100), default="Computer Science & Engineering")
    registration_number = Column(String(50), index=True, default="")
    is_active           = Column(Boolean, default=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    student_profile     = relationship("Student", back_populates="user", uselist=False)
    audit_logs          = relationship("AuditLog", back_populates="user")


class Department(Base):
    __tablename__ = "departments"

    id         = Column(Integer, primary_key=True, index=True)
    code       = Column(String(20), unique=True, index=True)   # e.g. "CSE"
    name       = Column(String(150), nullable=False)           # e.g. "Computer Science & Engineering"
    created_at = Column(DateTime, default=datetime.utcnow)

    courses = relationship("Course", back_populates="department_rel", cascade="all, delete-orphan")


class Course(Base):
    __tablename__ = "courses"

    id            = Column(Integer, primary_key=True, index=True)
    code          = Column(String(50), unique=True, index=True)   # e.g. "CS301"
    name          = Column(String(200), nullable=False)           # e.g. "Design & Analysis of Algorithms"
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    credits       = Column(Integer, default=4)
    semester      = Column(String(20), default="Semester 5")
    syllabus_notes= Column(Text, default="")
    created_at    = Column(DateTime, default=datetime.utcnow)

    department_rel = relationship("Department", back_populates="courses")
    registrations  = relationship("CourseRegistration", back_populates="course", cascade="all, delete-orphan")


class CourseRegistration(Base):
    __tablename__ = "course_registrations"

    id            = Column(Integer, primary_key=True, index=True)
    student_id    = Column(Integer, ForeignKey("students.id"), index=True)
    course_id     = Column(Integer, ForeignKey("courses.id"), index=True)
    semester      = Column(String(20), default="Semester 5")
    academic_year = Column(String(20), default="2025-2026")
    registered_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="course_registrations")
    course  = relationship("Course", back_populates="registrations")


# ─────────────────────────────────────────────────────────
# Dynamic Exam Paper Structure
# ─────────────────────────────────────────────────────────

class ExamPaper(Base):
    __tablename__ = "exam_papers"

    id              = Column(Integer, primary_key=True, index=True)
    paper_id        = Column(String(200), unique=True, index=True)
    pdf_path        = Column(String(500), nullable=True)
    course_code     = Column(String(50),  index=True)
    course_name     = Column(String(200))
    exam_name       = Column(String(200))
    total_marks     = Column(Float, default=0.0)
    duration_hours  = Column(Float, default=2.0)
    exam_date       = Column(String(30))
    batch           = Column(String(30))
    programme       = Column(String(100))
    semester        = Column(String(20))
    set_name        = Column(String(20), default="")
    raw_text        = Column(Text, nullable=True)
    parsed_at       = Column(DateTime, default=datetime.utcnow)

    parts       = relationship("ExamPart",     back_populates="paper",   cascade="all, delete-orphan")
    questions   = relationship("ExamQuestion", back_populates="paper",   cascade="all, delete-orphan")
    submissions = relationship("Submission",   back_populates="exam_paper")


class ExamPart(Base):
    __tablename__ = "exam_parts"

    id                  = Column(Integer, primary_key=True, index=True)
    paper_id_fk         = Column(Integer, ForeignKey("exam_papers.id"), index=True)
    part_name           = Column(String(50))
    marks_per_question  = Column(Float, default=2.0)
    num_questions       = Column(Integer, default=0)
    instructions        = Column(String(200), default="Answer ALL questions")

    paper = relationship("ExamPaper", back_populates="parts")


class ExamQuestion(Base):
    __tablename__ = "exam_questions"

    id              = Column(Integer, primary_key=True, index=True)
    paper_id_fk     = Column(Integer, ForeignKey("exam_papers.id"), index=True)
    question_number = Column(Integer)
    question_text   = Column(Text)
    marks           = Column(Float)
    part_name       = Column(String(50))
    question_type   = Column(String(30), default="open_ended")
    is_or_option    = Column(Boolean, default=False)
    teacher_answer  = Column(Text, nullable=True)
    correct_option  = Column(String(5), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    paper   = relationship("ExamPaper",  back_populates="questions")
    rubrics = relationship("Rubric",     back_populates="exam_question")


# ─────────────────────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    student_code        = Column(String(50), unique=True, index=True)
    name                = Column(String(200), nullable=True, default="")
    email               = Column(String(150), default="")
    registration_number = Column(String(50), index=True, default="")
    department          = Column(String(100), default="Computer Science & Engineering")
    programme           = Column(String(100), default="B.Tech")
    semester            = Column(String(20), default="Semester 5")
    batch               = Column(String(30), default="2022-2026")
    batch_id            = Column(Integer, ForeignKey("batches.id"), nullable=True, index=True)
    section             = Column(String(10), default="")
    academic_session    = Column(String(20), default="")
    is_active           = Column(Boolean, default=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    user                 = relationship("User", back_populates="student_profile")
    batch_rel            = relationship("Batch")
    submissions          = relationship("Submission", back_populates="student")
    course_registrations = relationship("CourseRegistration", back_populates="student", cascade="all, delete-orphan")
    re_evaluations       = relationship("ReEvaluationRequest", back_populates="student")
    chat_sessions        = relationship("ChatSession", back_populates="student")


class Question(Base):
    __tablename__ = "questions"

    id              = Column(Integer, primary_key=True, index=True)
    exam_name       = Column(String(200))
    question_number = Column(Integer)
    question_text   = Column(Text)
    question_type   = Column(String(20), default="open_ended")
    teacher_answer  = Column(Text)
    correct_option  = Column(String(1))
    mcq_options     = Column(Text)
    max_marks       = Column(Float, default=10.0)
    created_at      = Column(DateTime, default=datetime.utcnow)

    rubrics     = relationship("Rubric",      back_populates="question")
    submissions = relationship("Submission",  back_populates="question")


class Rubric(Base):
    __tablename__ = "rubrics"

    id               = Column(Integer, primary_key=True, index=True)
    question_id      = Column(Integer, ForeignKey("questions.id"),      nullable=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=True)
    criterion        = Column(String(200))
    marks            = Column(Float, default=1.0)
    is_required      = Column(Boolean, default=False)

    question      = relationship("Question",     back_populates="rubrics")
    exam_question = relationship("ExamQuestion", back_populates="rubrics")


class Submission(Base):
    __tablename__ = "submissions"

    id               = Column(Integer, primary_key=True, index=True)
    student_id       = Column(Integer, ForeignKey("students.id"))
    question_id      = Column(Integer, ForeignKey("questions.id"),      nullable=True)
    exam_paper_id    = Column(Integer, ForeignKey("exam_papers.id"),    nullable=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=True)
    image_path       = Column(String(500))
    extracted_text   = Column(Text)
    ocr_confidence   = Column(Float)
    submitted_at     = Column(DateTime, default=datetime.utcnow)

    student    = relationship("Student",      back_populates="submissions")
    question   = relationship("Question",     back_populates="submissions")
    exam_paper = relationship("ExamPaper",    back_populates="submissions")
    result     = relationship("Result",       back_populates="submission", uselist=False,
                              cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id                  = Column(Integer, primary_key=True, index=True)
    submission_id       = Column(Integer, ForeignKey("submissions.id"), unique=True)
    question_type       = Column(String(20), default="open_ended")
    final_score         = Column(Float)
    max_marks           = Column(Float, default=10.0)
    confidence          = Column(Float)
    evaluation_time_sec = Column(Float)
    feedback            = Column(Text)
    created_at          = Column(DateTime, default=datetime.utcnow)

    llm_score        = Column(Float, default=0.0)
    similarity_score = Column(Float, default=0.0)
    strengths        = Column(Text)
    missing_concepts = Column(Text)

    mcq_correct         = Column(Boolean,  nullable=True)
    mcq_detected_answer = Column(String(1), nullable=True)
    mcq_correct_answer  = Column(String(1), nullable=True)

    # Examiner Review & Approval Columns
    examiner_approved   = Column(Boolean, default=False)
    examiner_score      = Column(Float, nullable=True)
    examiner_comments   = Column(Text, default="")
    flagged_for_review  = Column(Boolean, default=False)
    flag_reason         = Column(String(255), default="")
    approved_by_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at         = Column(DateTime, nullable=True)
    # Question-wise justification: {"provider", "rubric_breakdown": [...], "deductions": [...],
    # "corrections": [...], "method"} — what the AI (or teacher) based the marks on.
    evaluation_json     = Column(Text, default="")

    submission = relationship("Submission", back_populates="result")


class MetricsSnapshot(Base):
    __tablename__ = "metrics_snapshots"

    id              = Column(Integer, primary_key=True, default=1)
    open_ended_json = Column(Text, nullable=True)
    mcq_json        = Column(Text, nullable=True)
    total_evaluated = Column(Integer, default=0)
    computed_at     = Column(DateTime, default=datetime.utcnow)

    @property
    def open_ended(self):
        return json.loads(self.open_ended_json) if self.open_ended_json else None

    @property
    def mcq(self):
        return json.loads(self.mcq_json) if self.mcq_json else None

    @classmethod
    def upsert(cls, db, open_ended_dict, mcq_dict, total_evaluated: int):
        snapshot = db.query(cls).filter_by(id=1).first()
        if snapshot is None:
            snapshot = cls(id=1)
            db.add(snapshot)
        snapshot.open_ended_json = json.dumps(open_ended_dict) if open_ended_dict else None
        snapshot.mcq_json        = json.dumps(mcq_dict)        if mcq_dict        else None
        snapshot.total_evaluated = total_evaluated
        snapshot.computed_at     = datetime.utcnow()
        db.commit()
        return snapshot

    @classmethod
    def load(cls, db):
        return db.query(cls).filter_by(id=1).first()


# ─────────────────────────────────────────────────────────
# Student Booklet Tables & Publication Workflow
# ─────────────────────────────────────────────────────────

class StudentBooklet(Base):
    __tablename__ = "student_booklets"

    id            = Column(Integer, primary_key=True, index=True)
    student_id    = Column(Integer, ForeignKey("students.id"),     nullable=True)
    exam_paper_id = Column(Integer, ForeignKey("exam_papers.id"),  nullable=True)

    pdf_path    = Column(String(500))
    total_pages = Column(Integer, default=0)

    # Cover metadata
    roll_number     = Column(String(50),  default="")
    register_number = Column(String(50),  default="")
    set_name        = Column(String(20),  default="")
    course_code     = Column(String(20),  default="")
    course_name     = Column(String(200), default="")
    exam_name       = Column(String(200), default="")
    programme       = Column(String(100), default="")
    semester        = Column(String(10),  default="")
    date_of_exam    = Column(String(20),  default="")
    department      = Column(String(100), default="Computer Science & Engineering")

    raw_full_ocr = Column(Text,     default="")
    parsed_at    = Column(DateTime, default=datetime.utcnow)

    # Formal Approval & Publication Status
    is_published       = Column(Boolean, default=False)
    published_at       = Column(DateTime, nullable=True)
    published_by_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    total_ai_score     = Column(Float, default=0.0)
    total_final_score  = Column(Float, default=0.0)
    approval_status    = Column(String(30), default="pending")  # pending, in_review, approved, published
    ocr_engine         = Column(String(80), default="")      # how the handwriting was read
    ocr_report         = Column(Text, default="")            # JSON: pages, warnings, unassigned text

    student        = relationship("Student",           backref="booklets")
    exam_paper     = relationship("ExamPaper",         backref="student_booklets")
    answers        = relationship("StudentAnswerText", back_populates="booklet",
                                  cascade="all, delete-orphan")
    re_evaluations = relationship("ReEvaluationRequest", back_populates="booklet")


class StudentAnswerText(Base):
    __tablename__ = "student_answer_texts"

    id               = Column(Integer, primary_key=True, index=True)
    booklet_id       = Column(Integer, ForeignKey("student_booklets.id"), index=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"),   nullable=True)

    question_number = Column(Integer)
    part_name       = Column(String(20),  default="")
    is_or_option    = Column(Boolean,     default=False)
    answer_text     = Column(Text,        default="")
    raw_ocr_text    = Column(Text,        default="")

    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)
    evaluated     = Column(Boolean, default=False)
    # where the answer is on the scanned sheet: [{"page": 1, "y0": 0.1, "y1": 0.45}] (fractions of page height)
    page_refs      = Column(Text, default="[]")
    ocr_confidence = Column(Float, nullable=True)
    ocr_engine     = Column(String(80), default="")

    booklet       = relationship("StudentBooklet",  back_populates="answers")
    exam_question = relationship("ExamQuestion",    backref="student_answers")
    submission    = relationship("Submission",      backref="student_answer_text")


# ─────────────────────────────────────────────────────────
# Re-Evaluation / Dispute Model
# ─────────────────────────────────────────────────────────

class ReEvaluationRequest(Base):
    """
    Tracks student grade re-evaluation requests with status,
    examiner notes, and marks before/after audit trail.
    """
    __tablename__ = "re_evaluation_requests"

    id               = Column(Integer, primary_key=True, index=True)
    booklet_id       = Column(Integer, ForeignKey("student_booklets.id"), index=True)
    student_id       = Column(Integer, ForeignKey("students.id"), index=True)
    question_number  = Column(Integer, nullable=True)  # None = entire booklet
    reason           = Column(Text, nullable=False)
    status           = Column(String(30), default="pending")  # pending, under_review, approved, rejected
    examiner_remarks = Column(Text, default="")
    score_before     = Column(Float, default=0.0)
    score_after      = Column(Float, nullable=True)
    resolved_by_id   = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    booklet = relationship("StudentBooklet", back_populates="re_evaluations")
    student = relationship("Student", back_populates="re_evaluations")


# ─────────────────────────────────────────────────────────
# AI Mentor Chatbot Models (RAG)
# ─────────────────────────────────────────────────────────

class ChatSession(Base):
    """
    Individual chat session for a student discussing their exam performance.
    """
    __tablename__ = "chat_sessions"

    id          = Column(Integer, primary_key=True, index=True)
    student_id  = Column(Integer, ForeignKey("students.id"), index=True)
    course_code = Column(String(50), nullable=True)
    booklet_id  = Column(Integer, ForeignKey("student_booklets.id"), nullable=True)
    title       = Column(String(200), default="Exam Feedback Consultation")
    created_at  = Column(DateTime, default=datetime.utcnow)

    student  = relationship("Student", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id            = Column(Integer, primary_key=True, index=True)
    session_id    = Column(Integer, ForeignKey("chat_sessions.id"), index=True)
    sender        = Column(String(20), default="user")  # 'user', 'assistant', 'system'
    content       = Column(Text, nullable=False)
    sources_cited = Column(Text, default="[]")          # JSON array of reference rubrics/notes
    created_at    = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class CourseMaterial(Base):
    """
    Approved syllabus reference materials used by RAG chatbot to explain concepts.
    """
    __tablename__ = "course_materials"

    id           = Column(Integer, primary_key=True, index=True)
    course_code  = Column(String(50), index=True)
    topic        = Column(String(200), nullable=False)
    content_text = Column(Text, nullable=False)
    keywords     = Column(String(255), default="")
    created_at   = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────
# Audit Trail Model
# ─────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Immutable audit log recording critical examiner and admin actions:
    e.g. marks edits, publication, user management, re-evaluation decisions.
    """
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    username    = Column(String(100), default="system")
    role        = Column(String(20), default="system")
    action      = Column(String(50), index=True, nullable=False)  # MARKS_OVERRIDE, PUBLISH, REEVAL_RESOLVE, etc.
    target_type = Column(String(50), default="")                  # booklet, result, user
    target_id   = Column(String(100), default="")
    details     = Column(Text, default="{}")                      # JSON payload of changes
    ip_address  = Column(String(50), default="")
    created_at  = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")


# ─────────────────────────────────────────────────────────
# Mentor AI: Study Books, Vectors, Quizzes & Adaptive Roadmaps
# ─────────────────────────────────────────────────────────

class StudyDocument(Base):
    """
    Uploaded textbooks, lecture notes, syllabus modules for Book-based RAG.
    """
    __tablename__ = "study_documents"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    title        = Column(String(255), nullable=False)
    subject      = Column(String(100), default="General")
    chapter      = Column(String(150), default="Chapter 1")
    filename     = Column(String(255), nullable=False)
    file_path    = Column(String(500), nullable=False)
    file_type    = Column(String(20), default="pdf")
    status       = Column(String(30), default="ready") # uploading, extracting, indexing, ready, failed
    total_pages  = Column(Integer, default=1)
    chunk_count  = Column(Integer, default=0)
    folder       = Column(String(100), default="")     # student-defined folder ("" = unfiled)
    created_at   = Column(DateTime, default=datetime.utcnow)

    chunks   = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    user     = relationship("User")


class DocumentChunk(Base):
    """
    Chunked passages with metadata and vector embeddings for real RAG retrieval.
    """
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("study_documents.id"), index=True)
    chunk_index = Column(Integer, default=0)
    page_number = Column(Integer, default=1)
    chapter     = Column(String(150), default="")
    content     = Column(Text, nullable=False)
    embedding   = Column(Text, default="[]") # JSON-serialized float vector or word-frequency vector
    keywords    = Column(String(255), default="")
    created_at  = Column(DateTime, default=datetime.utcnow)

    document = relationship("StudyDocument", back_populates="chunks")


class QuizAttempt(Base):
    """
    Saved student quiz attempts, questions, chosen answers, and scores.
    """
    __tablename__ = "quiz_attempts"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), index=True)
    subject        = Column(String(100), default="Computer Science")
    topic          = Column(String(150), default="Algorithms")
    total_score    = Column(Float, default=0.0)
    max_score      = Column(Float, default=10.0)
    percentage     = Column(Float, default=0.0)
    quiz_data      = Column(Text, default="[]") # JSON list of questions, student answers, explanations
    created_at     = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class GeneratedQuiz(Base):
    """
    A quiz generated from the student's study material. Correct answers are kept
    server-side so submissions are graded by the backend, never by the browser.
    """
    __tablename__ = "generated_quizzes"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    subject        = Column(String(100), default="General")
    chapter        = Column(String(150), default="")
    document_id    = Column(Integer, ForeignKey("study_documents.id", ondelete="SET NULL"), nullable=True)
    difficulty     = Column(String(20), default="Medium")
    mode           = Column(String(20), default="offline")    # llm | offline
    questions_json = Column(Text, nullable=False)              # includes correct answers + explanations
    sources_json   = Column(Text, default="[]")
    attempt_id     = Column(Integer, ForeignKey("quiz_attempts.id"), nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


class WeakTopic(Base):
    """
    Student weak topics identified dynamically from quiz mistakes and exam score deductions.
    """
    __tablename__ = "weak_topics"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True)
    subject         = Column(String(100), default="Computer Science")
    topic_name      = Column(String(150), nullable=False)
    accuracy        = Column(Float, default=0.0)
    attempts_count  = Column(Integer, default=1)
    status          = Column(String(30), default="Needs Practice") # Needs Practice, Improving, Mastered
    remedy_notes    = Column(Text, default="")
    updated_at      = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class StudyRoadmap(Base):
    """
    Adaptive personalized daily and weekly study roadmap.
    """
    __tablename__ = "study_roadmaps"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True)
    target_exam     = Column(String(150), default="Semester Finals")
    target_date     = Column(String(50), default="2026-12-15")
    daily_hours     = Column(Float, default=2.0)
    plan_data       = Column(Text, default="[]") # JSON list of daily/weekly tasks
    progress_pct    = Column(Float, default=0.0)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")



# ─────────────────────────────────────────────────────────
# Academic Structure: Batches, Sections, Teachers, Assignments
# ─────────────────────────────────────────────────────────

class Batch(Base):
    """A class group, e.g. CSE2026A = B.Tech CSE, section A, semester 5, session 2026-27."""
    __tablename__ = "batches"

    id               = Column(Integer, primary_key=True, index=True)
    batch_code       = Column(String(30), unique=True, index=True, nullable=False)
    programme        = Column(String(100), default="B.Tech")              # course / programme name
    course_code      = Column(String(20), default="", index=True)          # programme code, e.g. CSE
    department       = Column(String(100), default="Computer Science & Engineering")
    section          = Column(String(10), default="A", index=True)
    semester         = Column(String(20), default="Semester 5", index=True)
    academic_session = Column(String(20), default="2026-27", index=True)
    is_active        = Column(Boolean, default=True)
    created_at       = Column(DateTime, default=datetime.utcnow)


class TeacherProfile(Base):
    __tablename__ = "teacher_profiles"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    teacher_code = Column(String(30), unique=True, index=True, nullable=False)   # e.g. TCSE0001
    designation  = Column(String(100), default="Assistant Professor")
    department   = Column(String(100), default="Computer Science & Engineering")
    phone        = Column(String(30), default="")
    created_at   = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User")
    assignments = relationship("TeacherAssignment", back_populates="teacher", cascade="all, delete-orphan")


class TeacherAssignment(Base):
    """Teacher teaches subject (Course) to batch (which fixes section & semester)."""
    __tablename__ = "teacher_assignments"
    __table_args__ = (UniqueConstraint("teacher_id", "course_id", "batch_id", name="uq_teacher_course_batch"),)

    id         = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"), index=True, nullable=False)
    course_id  = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    batch_id   = Column(Integer, ForeignKey("batches.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    teacher = relationship("TeacherProfile", back_populates="assignments")
    course  = relationship("Course")
    batch   = relationship("Batch")


class Notification(Base):
    __tablename__ = "notifications"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    kind       = Column(String(40), default="info")     # result_published, sheet_evaluated, feedback, roadmap, quiz
    title      = Column(String(200), nullable=False)
    body       = Column(Text, default="")
    is_read    = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────
# Answer-sheet Scanner (multi-page, one student per session)
# ─────────────────────────────────────────────────────────

class ScanSession(Base):
    __tablename__ = "scan_sessions"

    id            = Column(Integer, primary_key=True, index=True)
    created_by_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    student_id    = Column(Integer, ForeignKey("students.id"), index=True, nullable=False)
    exam_paper_id = Column(Integer, ForeignKey("exam_papers.id"), nullable=True)
    course_code   = Column(String(20), default="")
    exam_name     = Column(String(200), default="")
    status        = Column(String(20), default="open")     # open | submitted | cancelled
    pdf_path      = Column(String(500), nullable=True)
    booklet_id    = Column(Integer, ForeignKey("student_booklets.id"), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    submitted_at  = Column(DateTime, nullable=True)

    student = relationship("Student")
    pages   = relationship("ScanPage", back_populates="session", cascade="all, delete-orphan",
                           order_by="ScanPage.position")


class ScanPage(Base):
    __tablename__ = "scan_pages"

    id             = Column(Integer, primary_key=True, index=True)
    session_id     = Column(Integer, ForeignKey("scan_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    position       = Column(Integer, default=0)
    original_path  = Column(String(500), nullable=False)
    processed_path = Column(String(500), nullable=False)
    sha256         = Column(String(64), index=True)
    quality_json   = Column(Text, default="{}")    # blur, brightness, boundary_found, warnings
    created_at     = Column(DateTime, default=datetime.utcnow)

    session = relationship("ScanSession", back_populates="pages")


# ─────────────────────────────────────────────────────────
# OMR evaluation
# ─────────────────────────────────────────────────────────

class OmrKey(Base):
    __tablename__ = "omr_keys"
    __table_args__ = (UniqueConstraint("course_code", "exam_name", name="uq_omr_key_exam"),)

    id             = Column(Integer, primary_key=True, index=True)
    course_code    = Column(String(20), index=True, nullable=False)
    exam_name      = Column(String(200), nullable=False)
    num_questions  = Column(Integer, nullable=False)
    num_options    = Column(Integer, default=4)
    answers_json   = Column(Text, nullable=False)           # ["A","C",...]
    marks_correct  = Column(Float, default=1.0)
    marks_wrong    = Column(Float, default=0.0)             # negative marking as a positive number, e.g. 0.25
    # what a question with two or more filled bubbles counts as: "flag" (teacher decides), "wrong" (negative
    # marking applies), "zero" (no marks, no penalty)
    multiple_policy = Column(String(10), default="flag")
    allow_negative_total = Column(Boolean, default=False)   # False → total never goes below 0
    created_by_id  = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


class OmrResponse(Base):
    __tablename__ = "omr_responses"
    __table_args__ = (UniqueConstraint("key_id", "student_id", name="uq_omr_key_student"),)

    id                 = Column(Integer, primary_key=True, index=True)
    key_id             = Column(Integer, ForeignKey("omr_keys.id", ondelete="CASCADE"), index=True, nullable=False)
    student_id         = Column(Integer, ForeignKey("students.id"), index=True, nullable=False)
    image_path         = Column(String(500), default="")
    detected_json      = Column(Text, default="[]")     # per question: marked, fill ratios, status
    reg_detected       = Column(String(30), default="")
    correct            = Column(Integer, default=0)
    incorrect          = Column(Integer, default=0)
    unanswered         = Column(Integer, default=0)
    flagged            = Column(Integer, default=0)
    score              = Column(Float, default=0.0)
    max_score          = Column(Float, default=0.0)
    needs_verification = Column(Boolean, default=False)
    verified_by_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    verified_at        = Column(DateTime, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)

    key     = relationship("OmrKey")
    student = relationship("Student")


# ─────────────────────────────────────────────────────────
# Consolidated examination result (OMR + subjective)
# ─────────────────────────────────────────────────────────

class FinalResult(Base):
    __tablename__ = "final_results"
    __table_args__ = (UniqueConstraint("student_id", "course_code", "exam_name", name="uq_final_result"),)

    id               = Column(Integer, primary_key=True, index=True)
    student_id       = Column(Integer, ForeignKey("students.id"), index=True, nullable=False)
    course_code      = Column(String(20), index=True, nullable=False)
    course_name      = Column(String(200), default="")
    exam_name        = Column(String(200), nullable=False)
    booklet_id       = Column(Integer, ForeignKey("student_booklets.id"), nullable=True)
    omr_response_id  = Column(Integer, ForeignKey("omr_responses.id"), nullable=True)
    omr_score        = Column(Float, nullable=True)
    omr_max          = Column(Float, nullable=True)
    subjective_score = Column(Float, nullable=True)
    subjective_max   = Column(Float, nullable=True)
    total            = Column(Float, default=0.0)
    max_total        = Column(Float, default=0.0)
    percentage       = Column(Float, default=0.0)
    grade            = Column(String(4), default="")
    grade_point      = Column(Float, default=0.0)
    passed           = Column(Boolean, default=False)
    status           = Column(String(20), default="draft")   # draft | approved | published
    approved_by_id   = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at      = Column(DateTime, nullable=True)
    published_at     = Column(DateTime, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student")


# ─────────────────────────────────────────────────────────
# AI Robot bulk examination checking (one question paper → many answer sheets)
# ─────────────────────────────────────────────────────────

class EvaluationJob(Base):
    """One checking batch: ONE official question paper (+ answer key / marking scheme) reused for every sheet."""
    __tablename__ = "evaluation_jobs"

    id             = Column(Integer, primary_key=True, index=True)
    created_by_id  = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    exam_paper_id  = Column(Integer, ForeignKey("exam_papers.id"), index=True, nullable=False)
    course_code    = Column(String(20), index=True, nullable=False)
    exam_name      = Column(String(200), nullable=False)
    title          = Column(String(200), default="")
    # draft (collecting sheets) | queued | running | completed | cancelling | cancelled | interrupted | failed
    status         = Column(String(20), default="draft", index=True)
    current_item_id = Column(Integer, nullable=True)
    message        = Column(String(500), default="")
    created_at     = Column(DateTime, default=datetime.utcnow)
    started_at     = Column(DateTime, nullable=True)
    finished_at    = Column(DateTime, nullable=True)

    paper = relationship("ExamPaper")
    created_by = relationship("User")
    items = relationship("EvaluationJobItem", back_populates="job", cascade="all, delete-orphan",
                         order_by="EvaluationJobItem.id")


class EvaluationJobItem(Base):
    """One uploaded / scanned answer sheet inside a checking batch."""
    __tablename__ = "evaluation_job_items"

    id               = Column(Integer, primary_key=True, index=True)
    job_id           = Column(Integer, ForeignKey("evaluation_jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    source           = Column(String(20), default="upload")     # upload | camera
    filename         = Column(String(300), default="")
    original_path    = Column(String(500), nullable=False)      # untouched copy of what was uploaded
    pdf_path         = Column(String(500), nullable=False)      # PDF used for reading / annotation
    sha256           = Column(String(64), index=True)
    pages            = Column(Integer, default=0)
    student_id       = Column(Integer, ForeignKey("students.id"), nullable=True, index=True)
    id_method        = Column(String(40), default="")           # filename | sheet | filename+sheet | entered | manual
    id_detail        = Column(Text, default="{}")               # JSON evidence: candidates per source, conflicts
    # needs_identification | ready | reading | evaluating | evaluated | needs_review | failed | skipped
    status           = Column(String(30), default="needs_identification", index=True)
    booklet_id       = Column(Integer, ForeignKey("student_booklets.id"), nullable=True)
    current_question = Column(Integer, nullable=True)
    questions_done   = Column(Integer, default=0)
    questions_total  = Column(Integer, default=0)
    total_ai         = Column(Float, nullable=True)
    max_marks        = Column(Float, nullable=True)
    flagged          = Column(Integer, default=0)
    error            = Column(Text, default="")
    events           = Column(Text, default="[]")               # short JSON log shown in the robot UI
    created_at       = Column(DateTime, default=datetime.utcnow)
    finished_at      = Column(DateTime, nullable=True)

    job = relationship("EvaluationJob", back_populates="items")
    student = relationship("Student")
    booklet = relationship("StudentBooklet")


# ─────────────────────────────────────────────────────────
# Database Helpers & Seeder
# ─────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_db():
    """
    Apply idempotent ALTER TABLE migrations for SQLite / PostgreSQL.
    """
    migrations = [
        # Results table extensions
        "ALTER TABLE results ADD COLUMN max_marks REAL DEFAULT 10.0",
        "ALTER TABLE results ADD COLUMN llm_score REAL DEFAULT 0.0",
        "ALTER TABLE results ADD COLUMN similarity_score REAL DEFAULT 0.0",
        "ALTER TABLE results ADD COLUMN question_type TEXT DEFAULT 'open_ended'",
        "ALTER TABLE results ADD COLUMN mcq_correct INTEGER",
        "ALTER TABLE results ADD COLUMN mcq_detected_answer TEXT",
        "ALTER TABLE results ADD COLUMN mcq_correct_answer TEXT",
        "ALTER TABLE results ADD COLUMN examiner_approved INTEGER DEFAULT 0",
        "ALTER TABLE results ADD COLUMN examiner_score REAL",
        "ALTER TABLE results ADD COLUMN examiner_comments TEXT DEFAULT ''",
        "ALTER TABLE results ADD COLUMN flagged_for_review INTEGER DEFAULT 0",
        "ALTER TABLE results ADD COLUMN flag_reason TEXT DEFAULT ''",
        "ALTER TABLE results ADD COLUMN approved_by_id INTEGER REFERENCES users(id)",
        "ALTER TABLE results ADD COLUMN approved_at TIMESTAMP",

        # Submissions extensions
        "ALTER TABLE submissions ADD COLUMN exam_paper_id INTEGER REFERENCES exam_papers(id)",
        "ALTER TABLE submissions ADD COLUMN exam_question_id INTEGER REFERENCES exam_questions(id)",
        "ALTER TABLE submissions ADD COLUMN ocr_confidence REAL",
        "ALTER TABLE submissions ADD COLUMN extracted_text TEXT",

        # Rubrics extensions
        "ALTER TABLE rubrics ADD COLUMN exam_question_id INTEGER REFERENCES exam_questions(id)",

        # Students extensions
        "ALTER TABLE students ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE students ADD COLUMN name TEXT DEFAULT ''",
        "ALTER TABLE students ADD COLUMN email TEXT DEFAULT ''",
        "ALTER TABLE students ADD COLUMN registration_number TEXT DEFAULT ''",
        "ALTER TABLE students ADD COLUMN department TEXT DEFAULT 'Computer Science & Engineering'",
        "ALTER TABLE students ADD COLUMN programme TEXT DEFAULT 'B.Tech'",
        "ALTER TABLE students ADD COLUMN semester TEXT DEFAULT 'Semester 5'",
        "ALTER TABLE students ADD COLUMN batch TEXT DEFAULT '2022-2026'",
        "ALTER TABLE students ADD COLUMN is_active INTEGER DEFAULT 1",

        # Booklets extensions
        "ALTER TABLE student_booklets ADD COLUMN is_published INTEGER DEFAULT 0",
        "ALTER TABLE student_booklets ADD COLUMN published_at TIMESTAMP",
        "ALTER TABLE student_booklets ADD COLUMN published_by_id INTEGER REFERENCES users(id)",
        "ALTER TABLE student_booklets ADD COLUMN total_ai_score REAL DEFAULT 0.0",
        "ALTER TABLE student_booklets ADD COLUMN total_final_score REAL DEFAULT 0.0",
        "ALTER TABLE student_booklets ADD COLUMN approval_status TEXT DEFAULT 'pending'",
        "ALTER TABLE student_booklets ADD COLUMN department TEXT DEFAULT 'Computer Science & Engineering'",

        # Student Answer Texts
        "ALTER TABLE student_answer_texts ADD COLUMN submission_id INTEGER REFERENCES submissions(id)",
        "ALTER TABLE student_answer_texts ADD COLUMN evaluated INTEGER DEFAULT 0",
        # AI Mentor: study-material folders
        "ALTER TABLE study_documents ADD COLUMN folder VARCHAR(100) DEFAULT ''",
        # Academic structure
        "ALTER TABLE students ADD COLUMN batch_id INTEGER REFERENCES batches(id)",
        # handwriting reading + justified evaluation (Oct 2026)
        "ALTER TABLE results ADD COLUMN evaluation_json TEXT DEFAULT ''",
        "ALTER TABLE student_booklets ADD COLUMN ocr_engine VARCHAR(80) DEFAULT ''",
        "ALTER TABLE student_booklets ADD COLUMN ocr_report TEXT DEFAULT ''",
        "ALTER TABLE student_answer_texts ADD COLUMN page_refs TEXT DEFAULT '[]'",
        "ALTER TABLE student_answer_texts ADD COLUMN ocr_confidence REAL",
        "ALTER TABLE student_answer_texts ADD COLUMN ocr_engine VARCHAR(80) DEFAULT ''",
        "ALTER TABLE omr_keys ADD COLUMN multiple_policy VARCHAR(10) DEFAULT 'flag'",
        "ALTER TABLE omr_keys ADD COLUMN allow_negative_total INTEGER DEFAULT 0",
        "ALTER TABLE students ADD COLUMN section VARCHAR(10) DEFAULT ''",
        "ALTER TABLE students ADD COLUMN academic_session VARCHAR(20) DEFAULT ''",
    ]

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        for sql in migrations:
            try:
                cursor.execute(sql)
                raw_conn.commit()
            except Exception:
                raw_conn.rollback()
    finally:
        raw_conn.close()


def seed_demo_data(db):
    """
    Seeds essential initial records:
    1. Demo Users: admin, faculty, student
    2. Departments: CSE, ECE, IT
    3. Courses: CS301, CS302, AI201
    4. Course Materials for RAG
    """
    # Import password hasher lazily to avoid circular imports
    from backend.auth import get_password_hash

    # Check if admin already exists
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        admin_user = User(
            username="admin",
            email="admin@university.edu",
            full_name="University Examination Controller",
            role="admin",
            department="Office of Examinations",
            hashed_password=get_password_hash("admin123"),
            is_active=True,
        )
        db.add(admin_user)

    faculty_user = db.query(User).filter(User.username == "faculty").first()
    if not faculty_user:
        faculty_user = User(
            username="faculty",
            email="faculty@university.edu",
            full_name="Dr. Anand Sharma (Chief Examiner)",
            role="faculty",
            department="Computer Science & Engineering",
            hashed_password=get_password_hash("faculty123"),
            is_active=True,
        )
        db.add(faculty_user)

    student_user = db.query(User).filter(User.username == "student").first()
    if not student_user:
        student_user = User(
            username="student",
            email="arjun.mehta@university.edu",
            full_name="Arjun Mehta",
            role="student",
            department="Computer Science & Engineering",
            registration_number="21BCS101",
            hashed_password=get_password_hash("student123"),
            is_active=True,
        )
        db.add(student_user)
        db.commit()
        db.refresh(student_user)

    student_user2 = db.query(User).filter(User.username == "student2").first()
    if not student_user2:
        student_user2 = User(
            username="student2",
            email="priya.sharma@university.edu",
            full_name="Priya Sharma",
            role="student",
            department="Computer Science & Engineering",
            registration_number="21BCS102",
            hashed_password=get_password_hash("student123"),
            is_active=True,
        )
        db.add(student_user2)
        db.commit()
        db.refresh(student_user2)

    db.commit()

    # Link Student profile if missing
    student_profile = db.query(Student).filter(Student.student_code == "21BCS101").first()
    if not student_profile:
        student_profile = Student(
            user_id=student_user.id,
            student_code="21BCS101",
            name="Arjun Mehta",
            email="arjun.mehta@university.edu",
            registration_number="21BCS101",
            department="Computer Science & Engineering",
            programme="B.Tech Computer Science",
            semester="Semester 5",
            batch="2022-2026",
            is_active=True
        )
        db.add(student_profile)

    student_profile2 = db.query(Student).filter(Student.student_code == "21BCS102").first()
    if not student_profile2:
        student_profile2 = Student(
            user_id=student_user2.id,
            student_code="21BCS102",
            name="Priya Sharma",
            email="priya.sharma@university.edu",
            registration_number="21BCS102",
            department="Computer Science & Engineering",
            programme="B.Tech Computer Science",
            semester="Semester 5",
            batch="2022-2026",
            is_active=True
        )
        db.add(student_profile2)

    # Departments
    departments_data = [
        ("CSE", "Computer Science & Engineering"),
        ("ECE", "Electronics & Communication Engineering"),
        ("IT", "Information Technology"),
    ]
    dept_map = {}
    for code, name in departments_data:
        d = db.query(Department).filter(Department.code == code).first()
        if not d:
            d = Department(code=code, name=name)
            db.add(d)
            db.commit()
            db.refresh(d)
        dept_map[code] = d

    # Courses
    courses_data = [
        ("CS301", "Design & Analysis of Algorithms", "CSE", 4, "Semester 5"),
        ("CS302", "Database Management Systems", "CSE", 4, "Semester 5"),
        ("AI201", "Artificial Intelligence & Expert Systems", "CSE", 3, "Semester 5"),
    ]
    for code, name, dept_code, creds, sem in courses_data:
        c = db.query(Course).filter(Course.code == code).first()
        if not c:
            dept_id = dept_map.get(dept_code).id if dept_map.get(dept_code) else None
            c = Course(code=code, name=name, department_id=dept_id, credits=creds, semester=sem)
            db.add(c)
            db.commit()
            db.refresh(c)

        # Register students to course if not registered
        if student_profile:
            reg = db.query(CourseRegistration).filter_by(student_id=student_profile.id, course_id=c.id).first()
            if not reg:
                db.add(CourseRegistration(student_id=student_profile.id, course_id=c.id, semester=sem))

    # Course Materials for RAG Chatbot
    materials_data = [
        ("CS301", "Divide and Conquer: QuickSort & MergeSort",
         "MergeSort divides the array into two halves, recursively sorts them, and merges them in O(N log N) time. "
         "QuickSort chooses a pivot element and partitions the array such that elements smaller than the pivot precede elements larger. "
         "Average case for QuickSort is O(N log N), worst case is O(N^2) when the array is already sorted and worst pivot is picked.",
         "quicksort, mergesort, divide and conquer, time complexity, worst case"),
        ("CS301", "Dynamic Programming vs Greedy Approach",
         "Dynamic Programming solves optimization problems by breaking them down into overlapping subproblems and storing subproblem solutions (memoization or tabulation). "
         "Greedy algorithms make the locally optimal choice at each step without reconsidering previous choices. DP guarantees global optimum when principle of optimality holds.",
         "dynamic programming, greedy, memoization, optimal substructure, overlapping subproblems"),
        ("CS302", "ACID Properties in Transactions",
         "ACID stands for Atomicity, Consistency, Isolation, and Durability. "
         "Atomicity guarantees all-or-nothing execution. "
         "Consistency ensures database transitions from one valid state to another. "
         "Isolation guarantees concurrent transactions execute as if sequential. "
         "Durability ensures committed changes survive system crashes.",
         "acid, atomicity, consistency, isolation, durability, database transaction")
    ]
    for c_code, topic, content, kws in materials_data:
        mat = db.query(CourseMaterial).filter_by(course_code=c_code, topic=topic).first()
        if not mat:
            db.add(CourseMaterial(course_code=c_code, topic=topic, content_text=content, keywords=kws))

    db.commit()
    _seed_demo_academic_structure(db)


def _seed_demo_academic_structure(db):
    """DEVELOPMENT DEMO DATA ONLY (SEED_DEMO_DATA=0 disables): one batch, the demo teacher's
    profile + assignments, and the two demo students placed in that batch."""
    batch = db.query(Batch).filter(Batch.batch_code == "CSE2026A").first()
    if not batch:
        batch = Batch(batch_code="CSE2026A", programme="B.Tech Computer Science", course_code="CSE",
                      department="Computer Science & Engineering", section="A", semester="Semester 5",
                      academic_session="2026-27")
        db.add(batch)
        db.commit()
        db.refresh(batch)
    for code in ("21BCS101", "21BCS102"):
        st = db.query(Student).filter(Student.student_code == code).first()
        if st and not st.batch_id:
            st.batch_id, st.section, st.academic_session = batch.id, batch.section, batch.academic_session
    fac = db.query(User).filter(User.username == "faculty").first()
    if fac:
        prof = db.query(TeacherProfile).filter(TeacherProfile.user_id == fac.id).first()
        if not prof:
            prof = TeacherProfile(user_id=fac.id, teacher_code="TCSE0001", designation="Professor",
                                  department="Computer Science & Engineering")
            db.add(prof)
            db.commit()
            db.refresh(prof)
        for code in ("CS301", "CS302"):
            c = db.query(Course).filter(Course.code == code).first()
            if c and not db.query(TeacherAssignment).filter_by(teacher_id=prof.id, course_id=c.id, batch_id=batch.id).first():
                db.add(TeacherAssignment(teacher_id=prof.id, course_id=c.id, batch_id=batch.id))
    db.commit()


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    if os.getenv("SEED_DEMO_DATA", "1").strip().lower() not in ("0", "false", "no"):
        db = SessionLocal()
        try:
            seed_demo_data(db)
        finally:
            db.close()
    try:
        print("[OK] Database tables created, migrated, and initial seed data verified.")
    except Exception:
        pass


import logging as _logging
logger = _logging.getLogger(__name__)
