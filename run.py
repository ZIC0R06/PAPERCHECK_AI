#!/usr/bin/env python3
"""
AI Paper Checking - University AI Examination System
Quick start script. Runs both the API server and Streamlit dashboard in a single command.

Usage:
    python run.py          # start both API + dashboard
    python run.py api      # start API only
    python run.py ui       # start dashboard only
    python run.py init     # initialise database and seed demo data
"""

import sys
import subprocess
import threading
import os

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["PYTHONPATH"] = ROOT_DIR

API_CMD  = [sys.executable, "-m", "uvicorn", "backend.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
UI_CMD   = [sys.executable, "-m", "streamlit", "run", os.path.join("frontend", "dashboard.py")]


def run_api():
    subprocess.run(API_CMD, cwd=ROOT_DIR)


def run_ui():
    subprocess.run(UI_CMD, cwd=ROOT_DIR)


def init_db():
    from backend.database import init_db as _init
    _init()
    print("[OK] Database initialised successfully.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "init":
        init_db()
        return

    if mode == "api":
        run_api()
    elif mode == "ui":
        run_ui()
    else:
        print("=" * 60)
        print(" [AI Paper Checking] University AI Examination System")
        print("=" * 60)
        print("   API  -> http://localhost:8000")
        print("   Docs -> http://localhost:8000/docs")
        print("   UI   -> http://localhost:8501")
        print("   Press Ctrl+C to stop\n")

        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        run_ui()


if __name__ == "__main__":
    main()