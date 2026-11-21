"""Tiny in-process fake of the Ollama HTTP API (stdlib only) used by the provider tests.
It mimics the real wire format: GET /api/tags, POST /api/chat (stream true/false, format=json)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockOllama:
    def __init__(self, models=("llama3.2:3b",), reply="Hello from mock llama.", fail_first=0, status=200,
                 json_reply=None, delay=0.0):
        self.models, self.reply, self.fail_first, self.status = list(models), reply, fail_first, status
        self.json_reply, self.delay = json_reply, delay
        self.requests = []           # every /api/chat payload received
        self.calls = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m, "model": m} for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path != "/api/chat":
                    return self._send(404, {"error": "not found"})
                outer.calls += 1
                outer.requests.append(body)
                if outer.delay:
                    import time
                    time.sleep(outer.delay)
                if outer.calls <= outer.fail_first:
                    return self._send(503, {"error": "overloaded"})
                if body.get("model") not in outer.models:
                    return self._send(404, {"error": f"model '{body.get('model')}' not found"})
                if outer.status != 200:
                    return self._send(outer.status, {"error": "boom"})
                text = outer.json_reply if (body.get("format") == "json" and outer.json_reply is not None) else outer.reply
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    words = text.split(" ")
                    for i, w in enumerate(words):
                        chunk = {"model": body["model"], "message": {"role": "assistant",
                                 "content": w + (" " if i < len(words) - 1 else "")}, "done": False}
                        self.wfile.write((json.dumps(chunk) + "\n").encode())
                        self.wfile.flush()
                    self.wfile.write((json.dumps({"model": body["model"], "message": {"role": "assistant",
                                      "content": ""}, "done": True}) + "\n").encode())
                    return
                self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": text},
                                 "done": True, "prompt_eval_count": 11, "eval_count": 7})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()
