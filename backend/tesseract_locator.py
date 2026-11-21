"""
Find the Tesseract OCR executable reliably (Windows installs are usually NOT on PATH).

Order: TESSERACT_CMD env var -> PATH -> standard Windows/macOS/Linux install folders.
configure() sets pytesseract's command and returns a status dict, so every OCR module uses the same lookup
and "OCR doesn't work" is reported with a precise reason instead of a silent failure.
"""
from __future__ import annotations

import os
import shutil
from typing import Dict, Optional

_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    "/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract",
]

INSTALL_HINT = ("Tesseract OCR was not found. Install it (Windows: https://github.com/UB-Mannheim/tesseract/wiki), "
                "then either add it to PATH or set TESSERACT_CMD in .env, e.g. "
                r"TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe")


def find_tesseract() -> Optional[str]:
    env = (os.getenv("TESSERACT_CMD") or "").strip().strip('"')
    if env:
        if os.path.isfile(env):
            return env
        found = shutil.which(env)
        if found:
            return found
    found = shutil.which("tesseract")
    if found:
        return found
    for c in _CANDIDATES:
        if c and os.path.isfile(c):
            return c
    return None


def configure() -> Dict:
    """Point pytesseract at the executable. Raises nothing; returns {'available', 'path', 'error'}."""
    path = find_tesseract()
    if not path:
        return {"available": False, "path": None, "error": INSTALL_HINT}
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = path
    except ImportError:
        return {"available": False, "path": path, "error": "Python package 'pytesseract' is not installed "
                                                              "(pip install pytesseract)."}
    return {"available": True, "path": path, "error": None}


class TesseractMissingError(RuntimeError):
    pass


def require() -> str:
    st = configure()
    if not st["available"]:
        raise TesseractMissingError(st["error"])
    return st["path"]
