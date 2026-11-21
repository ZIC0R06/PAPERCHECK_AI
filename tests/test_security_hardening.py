"""Regression tests for the security fixes made during the PaperCheckAI upgrade."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.auth import get_password_hash, verify_password


def test_plaintext_stored_password_is_not_accepted():
    # A DB value equal to the typed password must NOT authenticate any more.
    assert verify_password("admin123", "admin123") is False


def test_hashed_password_roundtrip():
    h = get_password_hash("S3cret!pw")
    assert verify_password("S3cret!pw", h) is True
    assert verify_password("wrong", h) is False


def test_each_hash_uses_unique_salt():
    assert get_password_hash("same") != get_password_hash("same")
