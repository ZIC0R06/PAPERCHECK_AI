"""
AI Paper Checking — Authentication and RBAC (Role-Based Access Control)
========================================================================
Implements secure password hashing with PBKDF2-HMAC-SHA256 + salt,
JWT access tokens with HS256 signatures, and role-based FastAPI dependencies.
Supported roles: 'admin', 'faculty', 'student'.
"""

import os
import time
import json
import base64
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config import SECRET_KEY
from backend.database import get_db, User, Student

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours


# ─────────────────────────────────────────────────────────────
# Password Hashing (PBKDF2-HMAC-SHA256 with Salt)
# ─────────────────────────────────────────────────────────────

def get_password_hash(password: str) -> str:
    """Hash password with PBKDF2-HMAC-SHA256 and unique 16-byte salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against a salt$hash string (or a legacy sha256 hash)."""
    if not hashed_password:
        return False
    if "$" in hashed_password:
        try:
            salt, key_hex = hashed_password.split("$", 1)
            computed = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000)
            return hmac.compare_digest(computed.hex(), key_hex)
        except Exception:
            return False
    # Legacy salted-sha256 hashes only. Plaintext comparison was removed for security.
    legacy_hash = hashlib.sha256((plain_password + SECRET_KEY).encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, hashed_password)


# ─────────────────────────────────────────────────────────────
# JWT Token Generation & Verification
# ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate signed JWT string using HMAC-SHA256."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": int(expire.timestamp())})

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")

    signature = hmac.new(
        SECRET_KEY.encode(),
        f"{header_b64}.{payload_b64}".encode(),
        hashlib.sha256
    ).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[dict]:
    """Verify signature and expiration of JWT string."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        expected_sig = hmac.new(
            SECRET_KEY.encode(),
            f"{header_b64}.{payload_b64}".encode(),
            hashlib.sha256
        ).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return None

        rem = len(payload_b64) % 4
        if rem > 0:
            payload_b64 += "=" * (4 - rem)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode()).decode())

        if payload.get("exp") and payload["exp"] < time.time():
            return None

        return payload
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# FastAPI Dependencies & RBAC
# ─────────────────────────────────────────────────────────────

def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Resolves authenticated user from Bearer token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload["sub"]
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Fallback search by reg_no
        user = db.query(User).filter(User.registration_number == username).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account is deactivated"
        )
    return user


def get_optional_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Optional[User]:
    """Resolves user if token provided, otherwise None (for flexible endpoints)."""
    if not token:
        return None
    try:
        return get_current_user(token, db)
    except Exception:
        return None


def require_role(allowed_roles: List[str]):
    """Enforces that current user belongs to one of allowed_roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {', '.join(allowed_roles)}"
            )
        return current_user
    return role_checker


def verify_student_access(current_user: User, student: Optional[Student]):
    """
    Ensures student can only access their own records.
    Faculty and Admin can access any record.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if current_user.role in ["admin", "faculty"]:
        return True
    if current_user.role == "student":
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student record not found")
        # Check if student ID matches or registration number matches
        user_reg = (current_user.registration_number or current_user.username or "").strip().lower()
        student_reg = (student.student_code or student.registration_number or "").strip().lower()
        if student.user_id == current_user.id or (user_reg and user_reg == student_reg):
            return True
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authorization violation: You can only view and access your own examination records."
        )
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
