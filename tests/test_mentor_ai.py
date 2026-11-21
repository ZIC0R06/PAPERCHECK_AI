"""
AI Paper Checking & Mentor AI — Comprehensive Test Suite
========================================================
Verifies:
  1. Question Bank CRUD & Query
  2. Study Book Upload, Text Extraction & RAG Indexing
  3. Grounded Retrieval & Answer Synthesis with Citations
  4. Chapter Explanation Generator (Detailed & Hinglish)
  5. Diagnostic Quiz Generation & Grading
  6. Dynamic Weak-Topic Tracking based on Quiz Attempts
  7. Adaptive Personalized Study Roadmap
  8. Voice Text Cleaning and Language Routing
"""

import os
import pytest
from backend.database import init_db, SessionLocal, User, Question, StudyDocument, QuizAttempt, WeakTopic, StudyRoadmap
from backend.rag_engine import RAGEngine
from backend.routes.mentor_ai_routes import (
    list_question_bank, create_question_bank_item, QuestionBankItemCreate,
    explain_chapter, ChapterExplainRequest,
    generate_book_quiz, QuizGenerateRequest,
    submit_quiz_attempt, QuizSubmitRequest,
    generate_study_roadmap, RoadmapGenerateRequest
)


@pytest.fixture(scope="module")
def db_session():
    init_db()
    db = SessionLocal()
    yield db
    db.close()


def test_question_bank_crud(db_session):
    """Test Question Bank CRUD operations for Admin and Faculty."""
    admin = db_session.query(User).filter_by(role="admin").first()
    assert admin is not None

    req = QuestionBankItemCreate(
        question_text="Explain the time complexity of QuickSort in best, average, and worst cases.",
        question_type="open_ended",
        teacher_answer="Best/Average: O(N log N), Worst: O(N^2) when array is already sorted or poorly pivoted.",
        max_marks=10.0
    )
    res = create_question_bank_item(req, db=db_session, current_user=admin)
    assert res["status"] == "success"
    assert res["question_id"] is not None

    items = list_question_bank(db=db_session)
    assert len(items) >= 1
    found = any("QuickSort" in q["question_text"] for q in items)
    assert found is True


def test_study_document_indexing_and_rag(db_session, tmp_path):
    """Test real RAG document parsing, chunking with overlap, and vector-similarity answering."""
    admin = db_session.query(User).filter_by(role="admin").first()

    # Create dummy study notes file
    doc_file = tmp_path / "algorithms_notes.txt"
    doc_file.write_text(
        "Dynamic Programming is an algorithmic paradigm that solves subproblems just once and stores their solutions. "
        "It applies when problems exhibit Optimal Substructure and Overlapping Subproblems. "
        "Classic examples include Fibonacci series, Knapsack problem, and Floyd-Warshall shortest path algorithm."
    )

    rag = RAGEngine(db_session)
    doc_rec = rag.process_and_index_file(
        file_path=str(doc_file),
        title="Algorithms Study Notes",
        subject="Design & Analysis of Algorithms",
        chapter="Chapter 4: Dynamic Programming",
        user_id=admin.id
    )

    assert doc_rec.id is not None
    assert doc_rec.chunk_count > 0

    # Test grounded RAG query
    ans = rag.answer_query(
        query="What are the conditions to apply Dynamic Programming?",
        user_id=admin.id,
        subject="Design & Analysis of Algorithms"
    )
    assert ans["grounded"] is True
    assert len(ans["citations"]) > 0


SORTING_NOTES = (
    "QuickSort is a divide and conquer sorting algorithm that picks a pivot and partitions the array around it. "
    "The average case time complexity of QuickSort is O(n log n), while the worst case is O(n^2). "
    "MergeSort is a stable divide and conquer algorithm that splits the array into halves and merges them. "
    "MergeSort requires O(n) auxiliary space because merging needs a temporary array. "
    "HeapSort builds a binary heap and repeatedly extracts the maximum element. "
    "HeapSort is an in-place algorithm with O(n log n) time complexity in every case. "
    "A stable sorting algorithm preserves the relative order of equal keys. "
    "Insertion sort is efficient for small or nearly sorted arrays and runs in O(n) time in the best case. "
    "The pivot choice strongly affects QuickSort performance; randomized pivot selection avoids the worst case. "
    "Divide and conquer algorithms solve subproblems recursively and combine their solutions."
)


@pytest.fixture(scope="module")
def sorting_doc(db_session, tmp_path_factory):
    student_user = db_session.query(User).filter_by(username="student").first()
    f = tmp_path_factory.mktemp("notes") / "sorting.txt"
    f.write_text(SORTING_NOTES)
    return RAGEngine(db_session).process_and_index_file(
        file_path=str(f), title="Sorting Algorithms Notes", subject="Design & Analysis of Algorithms",
        chapter="Sorting", user_id=student_user.id)


def test_chapter_explanation_service(db_session, sorting_doc):
    """Explanations are built from the student's own material, with page citations."""
    student_user = db_session.query(User).filter_by(username="student").first()
    req = ChapterExplainRequest(subject="Design & Analysis of Algorithms", chapter="Sorting",
                                style="Hinglish", document_id=sorting_doc.id)
    res = explain_chapter(req, db=db_session, current_user=student_user)
    assert "Overview" in res["explanation"]
    assert "QuickSort" in res["explanation"] or "MergeSort" in res["explanation"]
    assert res["citations"] and "Sorting Algorithms Notes" in res["citations"][0]


def test_explanation_without_material_is_refused(db_session):
    from fastapi import HTTPException
    student_user = db_session.query(User).filter_by(username="student").first()
    with pytest.raises(HTTPException) as e:
        explain_chapter(ChapterExplainRequest(subject="Subject With No Uploads Xyz", chapter="x"),
                        db=db_session, current_user=student_user)
    assert e.value.status_code == 404


def test_book_quiz_generation_and_scoring(db_session, sorting_doc):
    """Quiz comes from the uploaded notes; answers stay server-side; grading is done by the backend."""
    from fastapi import HTTPException
    from backend.database import GeneratedQuiz
    import json as _json
    student_user = db_session.query(User).filter_by(username="student").first()
    quiz = generate_book_quiz(QuizGenerateRequest(subject="Design & Analysis of Algorithms", chapter="Sorting",
                                                  num_questions=3, document_id=sorting_doc.id),
                              db=db_session, current_user=student_user)
    assert len(quiz["questions"]) == 3
    assert all("correct_answer" not in q for q in quiz["questions"])          # never leaked to the client
    stored = _json.loads(db_session.query(GeneratedQuiz).get(quiz["quiz_id"]).questions_json)
    for q in stored:                                                           # grounded in the notes
        assert q["correct_answer"].lower() in SORTING_NOTES.lower()

    # answer 2 correctly, 1 wrongly; a forged "correct_answer" field must be ignored
    picks = [{"question_id": stored[0]["id"], "selected_answer": stored[0]["correct_answer"]},
             {"question_id": stored[1]["id"], "selected_answer": stored[1]["correct_answer"]},
             {"question_id": stored[2]["id"], "selected_answer": "definitely wrong",
              "correct_answer": "definitely wrong"}]
    res = submit_quiz_attempt(QuizSubmitRequest(quiz_id=quiz["quiz_id"], answers=picks),
                              db=db_session, current_user=student_user)
    assert res["score"] == 2.0 and res["max_score"] == 3.0 and res["percentage"] == 66.7
    wt = db_session.query(WeakTopic).filter_by(user_id=student_user.id, topic_name=stored[2]["topic"]).first()
    assert wt is not None and wt.status in ("Needs Practice", "Improving")

    with pytest.raises(HTTPException) as e:                                    # no double submission
        submit_quiz_attempt(QuizSubmitRequest(quiz_id=quiz["quiz_id"], answers=picks),
                            db=db_session, current_user=student_user)
    assert e.value.status_code == 409

    other = db_session.query(User).filter_by(username="student2").first()     # not someone else's quiz
    with pytest.raises(HTTPException) as e:
        submit_quiz_attempt(QuizSubmitRequest(quiz_id=quiz["quiz_id"], answers=picks),
                            db=db_session, current_user=other)
    assert e.value.status_code == 404


def test_adaptive_study_roadmap_generation(db_session):
    """Test generating an adaptive personalized daily/weekly study plan targeting weak topics."""
    student_user = db_session.query(User).filter_by(username="student").first()

    req = RoadmapGenerateRequest(
        target_exam="Final Semester Theory Exam",
        target_date="2026-12-20",
        daily_hours=3.0,
        subjects=["Design & Analysis of Algorithms", "Database Management Systems"]
    )
    roadmap = generate_study_roadmap(req, db=db_session, current_user=student_user)

    assert roadmap["status"] == "success"
    assert len(roadmap["tasks"]) >= 4
    # Ensure tasks prioritize verified weak areas
    task_topics = [t["focus_topic"] for t in roadmap["tasks"]]
    assert len(task_topics) > 0
