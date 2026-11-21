"""
AI Paper Checking — TestClient API Endpoint Verification
"""
import os
import sys

os.environ["SKIP_OCR_WARMUP"] = "1"

from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.api import app

def test_api_health_and_docs():
    client = TestClient(app)
    
    # Health endpoint
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data.get("status") == "ok"
    
    # OpenAPI schema
    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    schema = r_openapi.json()
    assert "paths" in schema
    assert "/auth/login" in schema["paths"]
    assert "/admin/users" in schema["paths"]
    assert "/examiner/booklets" in schema["paths"]
    assert "/student/results" in schema["paths"]
    assert "/mentor/chat" in schema["paths"]

def test_auth_login_endpoints():
    client = TestClient(app)
    
    # Admin login
    r_admin = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert r_admin.status_code == 200
    admin_data = r_admin.json()
    assert "access_token" in admin_data
    assert admin_data["role"] == "admin"
    admin_token = admin_data["access_token"]
    
    # Admin access
    r_users = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert r_users.status_code == 200
    
    # Student login
    r_student = client.post("/auth/login", json={"username": "student", "password": "student123"})
    assert r_student.status_code == 200
    student_data = r_student.json()
    assert student_data["role"] == "student"
    student_token = student_data["access_token"]
    
    # Student access to student profile
    r_prof = client.get("/student/profile", headers={"Authorization": f"Bearer {student_token}"})
    assert r_prof.status_code == 200
    assert r_prof.json()["registration_number"] == "21BCS101"
    
    # Student CANNOT access admin users endpoint (RBAC check)
    r_forbidden = client.get("/admin/users", headers={"Authorization": f"Bearer {student_token}"})
    assert r_forbidden.status_code == 403
