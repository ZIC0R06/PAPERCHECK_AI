"""
AI Mentor — Secure chat-attachment handling
===========================================
Validates, stores and extracts text from files a student attaches in the
AI Mentor chat. Files are stored per user under uploads/mentor_files/<user_id>/
with random names; the original name is only kept as display metadata.

Checks performed:
  • extension allow-list (pdf, docx, txt, md, png, jpg, jpeg)
  • size limit (MENTOR_MAX_FILE_MB, default 15 MB) and non-empty
  • content sniffing ("magic bytes") so a renamed .exe cannot pass as .pdf
  • DOCX must be a real OOXML zip containing word/document.xml
  • text files must decode as UTF-8 and contain no NUL bytes
"""

import io
import os
import re
import uuid
import zipfile
from pathlib import Path
from typing import Tuple

MAX_MB = int(os.getenv("MENTOR_MAX_FILE_MB", "15"))
MAX_BYTES = MAX_MB * 1024 * 1024
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
UPLOAD_ROOT = Path(os.getenv("MENTOR_UPLOAD_DIR", "uploads/mentor_files"))

# Marks a StudyDocument as a chat attachment (re-uses the existing table, no new DB)
CHAT_UPLOAD_SUBJECT = "Mentor Chat Upload"


class FileValidationError(ValueError):
    """Raised with a student-friendly message when a file is rejected."""


def safe_display_name(name: str) -> str:
    base = Path(name or "file").name
    base = re.sub(r"[^\w.\- ()\[\]]+", "_", base).strip(" .") or "file"
    return base[:120]


def validate_upload(filename: str, content: bytes) -> str:
    """Return the normalised extension, or raise FileValidationError."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise FileValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: PDF, DOCX, TXT, MD, JPG, PNG."
        )
    if not content:
        raise FileValidationError("The file is empty.")
    if len(content) > MAX_BYTES:
        raise FileValidationError(
            f"File is {len(content) / 1048576:.1f} MB — the limit is {MAX_MB} MB."
        )

    head = content[:8]
    if ext == ".pdf" and not content[:1024].lstrip().startswith(b"%PDF"):
        raise FileValidationError("This file is not a valid PDF.")
    if ext == ".png" and head != b"\x89PNG\r\n\x1a\n":
        raise FileValidationError("This file is not a valid PNG image.")
    if ext in (".jpg", ".jpeg") and not head.startswith(b"\xff\xd8\xff"):
        raise FileValidationError("This file is not a valid JPEG image.")
    if ext == ".docx":
        if not head.startswith(b"PK\x03\x04"):
            raise FileValidationError("This file is not a valid DOCX document.")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                names = z.namelist()
                if "word/document.xml" not in names:
                    raise FileValidationError("This file is not a valid DOCX document.")
                # zip-bomb guard
                total = sum(i.file_size for i in z.infolist())
                if total > 200 * 1024 * 1024:
                    raise FileValidationError("DOCX expands to an unsafe size.")
        except zipfile.BadZipFile:
            raise FileValidationError("This DOCX file is corrupted.")
    if ext in (".txt", ".md"):
        if b"\x00" in content[:65536]:
            raise FileValidationError("This does not look like a plain-text file.")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content.decode("utf-16")
            except UnicodeDecodeError:
                raise FileValidationError("Text file must be UTF-8 encoded.")
    return ext


def store_upload(user_id: int, ext: str, content: bytes) -> Path:
    folder = UPLOAD_ROOT / str(int(user_id))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{ext}"
    with open(path, "wb") as f:
        f.write(content)
    return path


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1048576:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1048576:.1f} MB"
