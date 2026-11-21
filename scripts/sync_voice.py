#!/usr/bin/env python3
"""Embed frontend/voice_out.js into frontend/mentor_app.html between the VOICE_OUT markers (single source of truth)."""
import os
import re
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
js = open(os.path.join(root, "frontend", "voice_out.js"), encoding="utf-8").read().strip()
for page in ("mentor_app.html", "exam_assistant_app.html"):
    p = os.path.join(root, "frontend", page)
    s = open(p, encoding="utf-8").read()
    new, n = re.subn(r"/\*VOICE_OUT_BEGIN\*/.*?/\*VOICE_OUT_END\*/",
                     lambda m: "/*VOICE_OUT_BEGIN*/\n" + js + "\n/*VOICE_OUT_END*/", s, flags=re.S)
    if not n:
        sys.exit(f"markers not found in {page}")
    if "--check" in sys.argv:
        sys.exit(0 if new == s else f"{page} is out of sync with voice_out.js — run scripts/sync_voice.py")
    open(p, "w", encoding="utf-8").write(new)
    print("synced", page)
