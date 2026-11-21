"""Exit codes: 0 ready | 2 Ollama unreachable | 3 model not installed.  Used by start.bat."""
import json
import sys
import urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434").rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "llama3.2:3b"
norm = lambda n: n.lower() if ":" in n else n.lower() + ":latest"
try:
    with urllib.request.urlopen(url + "/api/tags", timeout=5) as r:
        names = [m.get("name", "") for m in json.load(r).get("models", [])]
except Exception as e:
    print(f"[X] Ollama not reachable at {url} ({type(e).__name__})")
    sys.exit(2)
if norm(model) not in {norm(n) for n in names}:
    print(f"[X] Ollama is running but model '{model}' is not installed. Installed: {', '.join(names) or 'none'}")
    sys.exit(3)
print(f"[OK] Ollama ready at {url} with model {model}")
