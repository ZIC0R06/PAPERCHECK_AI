#!/usr/bin/env python3
"""
Database backup & restore
=========================
  python scripts/backup_db.py backup            # consistent SQLite snapshot → backups/
  python scripts/backup_db.py backup --files    # also zip uploads/ (answer sheets, scans, OMR, study files)
  python scripts/backup_db.py list
  python scripts/backup_db.py restore backups/intelligrade_20261030_190000.db

SQLite snapshots use the online-backup API, so they are consistent even while the app is running.
Restore first saves a safety copy of the current database. Keeps the newest BACKUP_KEEP (default 14).
PostgreSQL: use `pg_dump -Fc "$DATABASE_URL" > backup.dump` and `pg_restore -c -d "$DATABASE_URL" backup.dump`.
"""

import argparse
import os
import shutil
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", ROOT / "backups"))
KEEP = int(os.getenv("BACKUP_KEEP", "14"))


def _db_path() -> Path:
    url = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'intelligrade.db'}")
    if not url.startswith("sqlite:///"):
        sys.exit("DATABASE_URL is not SQLite — use pg_dump / pg_restore (see the docstring).")
    p = Path(url.replace("sqlite:///", "", 1))
    return p if p.is_absolute() else (ROOT / p).resolve()


def backup(with_files: bool):
    src = _db_path()
    if not src.exists():
        sys.exit(f"Database not found: {src}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{src.stem}_{stamp}.db"
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    with sqlite3.connect(dst) as d:
        ok = d.execute("PRAGMA integrity_check").fetchone()[0]
    print(f"[OK] database → {dst} (integrity: {ok})")
    if with_files and (ROOT / "uploads").exists():
        z = BACKUP_DIR / f"uploads_{stamp}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in (ROOT / "uploads").rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(ROOT))
        print(f"[OK] files    → {z}")
    olds = sorted(BACKUP_DIR.glob(f"{src.stem}_*.db"))[:-KEEP]
    for o in olds:
        o.unlink()
        print(f"[..] pruned old backup {o.name}")


def restore(path: str):
    src, dst = Path(path), _db_path()
    if not src.exists():
        sys.exit(f"Backup not found: {src}")
    with sqlite3.connect(src) as c:
        if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            sys.exit("Backup failed its integrity check — not restoring.")
    if dst.exists():
        safety = dst.with_name(f"{dst.stem}_before_restore_{datetime.now():%Y%m%d_%H%M%S}.db")
        shutil.copy2(dst, safety)
        print(f"[OK] current database saved as {safety.name}")
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    print(f"[OK] restored {src.name} → {dst}. Restart the API.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup"); b.add_argument("--files", action="store_true")
    sub.add_parser("list")
    r = sub.add_parser("restore"); r.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "backup":
        backup(a.files)
    elif a.cmd == "list":
        for f in sorted(BACKUP_DIR.glob("*")):
            print(f"{f.name:45s} {f.stat().st_size / 1e6:8.2f} MB")
    else:
        restore(a.path)
