"""
AI Paper Checking — Authentication Routes
==========================================
Provides /auth/login and /auth/me endpoints supporting JWT tokens.
"""

import json
import os
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db, User, AuditLog
from backend.auth import (
    verify_password, create_access_token, get_current_user, get_password_hash
)
from backend.schemas import LoginRequest, TokenResponse

router = APIRouter()

# ── Login rate limiting (per username + client IP, in-process) ────────────────
_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILURES", "8"))
_WINDOW_S = int(os.getenv("LOGIN_WINDOW_SECONDS", "900"))
_fails = defaultdict(deque)
_lock = threading.Lock()


def _rl_key(request: Request, username: str) -> str:
    ip = request.client.host if request.client else "?"
    return f"{username.lower()}|{ip}"


def _too_many(key: str) -> int:
    """Seconds to wait if locked out, else 0."""
    now = time.time()
    with _lock:
        q = _fails[key]
        while q and now - q[0] > _WINDOW_S:
            q.popleft()
        return int(_WINDOW_S - (now - q[0])) + 1 if len(q) >= _MAX_FAILS else 0


def _record_fail(key: str):
    with _lock:
        _fails[key].append(time.time())


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=200)
    new_password: str = Field(..., min_length=8, max_length=200)
    confirm_password: str = Field(..., min_length=8, max_length=200)


@router.post("/login", response_model=TokenResponse, summary="User authentication")
def login(request: LoginRequest, http: Request, db: Session = Depends(get_db)):
    """
    Authenticate user by username or registration number, return JWT bearer token.
    """
    username = request.username.strip()
    key = _rl_key(http, username)
    wait = _too_many(key)
    if wait:
        raise HTTPException(status_code=429,
                            detail=f"Too many failed sign-in attempts. Try again in {max(1, wait // 60)} minute(s).")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Also check registration_number
        user = db.query(User).filter(User.registration_number == username).first()

    if not user or not verify_password(request.password, user.hashed_password):
        _record_fail(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password. Please verify your credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated. Please contact the administrator."
        )

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role, "user_id": user.id}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "department": user.department,
            "registration_number": user.registration_number
        }
    }


@router.get("/me", summary="Current user profile")
def get_me(current_user: User = Depends(get_current_user)):
    """Returns profile of currently logged-in user."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "department": current_user.department,
        "registration_number": current_user.registration_number,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }


@router.post("/change-password", summary="Change my own password (all roles)")
def change_password(data: ChangePasswordRequest, current_user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    if not verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirmation do not match.")
    if data.new_password == data.current_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one.")
    if not (re.search(r"[A-Za-z]", data.new_password) and re.search(r"\d", data.new_password)):
        raise HTTPException(status_code=400, detail="Use at least 8 characters with both letters and numbers.")
    if data.new_password.lower() in {current_user.username.lower(), (current_user.registration_number or "").lower()}:
        raise HTTPException(status_code=400, detail="Password cannot be your username or registration number.")
    current_user.hashed_password = get_password_hash(data.new_password)
    db.add(AuditLog(user_id=current_user.id, username=current_user.username, role=current_user.role,
                    action="PASSWORD_CHANGE", target_type="user", target_id=str(current_user.id),
                    details=json.dumps({"self_service": True})))
    db.commit()
    return {"status": "success", "message": "Password changed. Use the new password next time you sign in."}
