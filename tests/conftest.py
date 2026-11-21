"""
pytest configuration and shared fixtures for IntelliGrade-H tests.
"""
import os
import shutil
import sys
import tempfile
import pytest

# Make sure the project root is on sys.path so that `backend.*` imports work
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# ── Isolation: tests must NEVER write into the real database or uploads folder ──
# Runs before any `backend.*` import (database.py reads DATABASE_URL at import time).
# Tests work on a throw-away copy of intelligrade.db inside a temp directory, and the
# working directory is moved there so relative "uploads/…" paths land in the temp dir too.
if not os.getenv("PAPERCHECK_TEST_KEEP_ENV"):
    _TMP = tempfile.mkdtemp(prefix="papercheck_tests_")
    _src_db = os.path.join(_ROOT, "intelligrade.db")
    _db = os.path.join(_TMP, "test.db")
    if os.path.exists(_src_db):
        shutil.copy(_src_db, _db)
    os.environ["DATABASE_URL"] = f"sqlite:///{_db}"
    os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
    os.environ["OMR_DIR"] = os.path.join(_TMP, "uploads", "omr")
    # never call paid cloud AI from unit tests unless explicitly requested
    for _k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
               "GOOGLE_VISION_API_KEY", "MISTRAL_API_KEY", "AZURE_VISION_KEY"):
        if not os.getenv("PAPERCHECK_TEST_LIVE_AI"):
            os.environ[_k] = ""
    # Hermetic AI: tests must not talk to a real local Ollama either (its answers are non-deterministic and the
    # e2e tests assert the offline, flagged-for-review behaviour). Port 9 = nothing listens.
    if not os.getenv("PAPERCHECK_TEST_LIVE_AI"):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:9"
        os.environ["LLM_RETRIES"] = "0"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["HANDWRITING_PROVIDER"] = "local"
    os.chdir(_TMP)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_student_answer():
    return (
        "Gradient descent is an optimization algorithm used to minimize "
        "the loss function by iteratively updating model parameters in the "
        "direction of the negative gradient. The learning rate controls the "
        "step size. When the gradient is zero, the algorithm converges."
    )


@pytest.fixture
def sample_teacher_answer():
    return (
        "Gradient descent is an iterative optimization algorithm that minimizes "
        "a loss function by computing the gradient and moving parameters in the "
        "opposite direction. The learning rate (alpha) determines the step size. "
        "Variants include batch, mini-batch, and stochastic gradient descent."
    )


@pytest.fixture
def sample_rubric():
    return [
        {"criterion": "definition of gradient descent", "marks": 3.0},
        {"criterion": "role of learning rate", "marks": 3.0},
        {"criterion": "convergence explanation", "marks": 2.0},
        {"criterion": "real-world example or application", "marks": 2.0},
    ]


@pytest.fixture
def dummy_image_bytes():
    """Returns minimal 1×1 white PNG bytes for OCR/preprocessing tests."""
    import io
    from PIL import Image
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
