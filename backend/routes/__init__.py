"""
AI Paper Checking API Routes Package
"""
from backend.routes.auth_routes import router as auth_router
from backend.routes.admin_routes import router as admin_router
from backend.routes.examiner_routes import router as examiner_router
from backend.routes.student_routes import router as student_router
from backend.routes.mentor_routes import router as mentor_router

__all__ = [
    "auth_router",
    "admin_router",
    "examiner_router",
    "student_router",
    "mentor_router",
]
