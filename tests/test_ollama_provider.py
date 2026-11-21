"""Ollama provider tests — run against a faithful in-process mock of the Ollama HTTP API.
(The real-model smoke test is scripts/ollama_smoke_test.py and needs your local Ollama.)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.mock_ollama import MockOllama
from backend.llm_provider import (LLMClient, LLMConfig, LLMError, LLMModelMissingError, LLMParseError,
                                  LLMUnavailableError, parse_json_text)


def client(url, **kw):
    cfg = LLMConfig(provider="ollama", base_url=url, model=kw.pop("model", "llama3.2:3b"),
                    timeout=kw.pop("timeout", 5), retries=kw.pop("retries", 2), **kw)
    return LLMClient(cfg)


class TestDefaults(unittest.TestCase):
    def test_env_defaults_are_ollama(self):
        for k in ("LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LLM_TEMPERATURE", "LLM_TIMEOUT", "LLM_MAX_TOKENS"):
            os.environ.pop(k, None)
        c = LLMConfig.from_env()
        self.assertEqual((c.provider, c.base_url, c.model), ("ollama", "http://localhost:11434", "llama3.2:3b"))
        self.assertEqual((c.temperature, c.timeout, c.max_tokens), (0.2, 120, 4096))

    def test_no_cloud_key_needed(self):
        for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
            os.environ[k] = ""
        self.assertEqual(LLMClient(LLMConfig())._order(), ["ollama"])


class TestChat(unittest.TestCase):
    def test_complete_sends_system_history_and_options(self):
        with MockOllama(reply="Binary tree is a hierarchical structure.") as m:
            c = client(m.url)
            out = c.complete("SYS PROMPT", "What is a tree?", max_tokens=300,
                             history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
            self.assertIn("Binary tree", out)
            req = m.requests[-1]
            self.assertEqual([x["role"] for x in req["messages"]], ["system", "user", "assistant", "user"])
            self.assertEqual(req["messages"][0]["content"], "SYS PROMPT")
            self.assertEqual(req["model"], "llama3.2:3b")
            self.assertFalse(req["stream"])
            self.assertEqual(req["options"]["num_predict"], 300)
            self.assertEqual(req["options"]["temperature"], 0.2)
            self.assertGreaterEqual(req["options"]["num_ctx"], 4096)   # default 2048 would truncate documents

    def test_max_tokens_is_capped_by_config(self):
        with MockOllama() as m:
            client(m.url, max_tokens=1000).complete("s", "u", max_tokens=6000)
            self.assertEqual(m.requests[-1]["options"]["num_predict"], 1000)

    def test_generate_returns_metadata(self):
        with MockOllama() as m:
            r = client(m.url).generate("hi")
            self.assertEqual((r.provider, r.model, r.prompt_tokens, r.completion_tokens), ("ollama", "llama3.2:3b", 11, 7))


class TestStreaming(unittest.TestCase):
    def test_stream_collects_tokens_in_order(self):
        with MockOllama(reply="one two three four") as m:
            got = []
            full = client(m.url).stream("sys", "q", on_token=got.append)
            self.assertEqual("".join(got).strip(), "one two three four")
            self.assertEqual(full, "one two three four")
            self.assertTrue(len(got) >= 4)
            self.assertTrue(m.requests[-1]["stream"])


class TestJson(unittest.TestCase):
    def test_json_mode_flag_and_parse(self):
        with MockOllama(json_reply='```json\n{"score": 7.5, "confidence": 0.8}\n```') as m:
            d = client(m.url).generate_json("grade this")
            self.assertEqual(d["score"], 7.5)
            self.assertEqual(m.requests[-1].get("format"), "json")

    def test_bad_json_raises_instead_of_inventing_a_score(self):
        with MockOllama(json_reply="I think the answer deserves 5 marks.") as m:
            with self.assertRaises(LLMParseError):
                client(m.url).generate_json("grade this")
            self.assertEqual(m.calls, 2)          # one retry, then an honest error

    def test_parse_helper(self):
        self.assertEqual(parse_json_text('noise {"a": 1} tail'), {"a": 1})
        with self.assertRaises(LLMParseError):
            parse_json_text("[1,2]")


class TestHealthAndErrors(unittest.TestCase):
    def test_health_ok(self):
        with MockOllama(models=["llama3.2:3b", "other:7b"]) as m:
            h = client(m.url).health(force=True)
            self.assertTrue(h["reachable"] and h["model_available"])
            self.assertIn("other:7b", h["installed_models"])
            self.assertEqual(client(m.url).active_provider, "ollama(llama3.2:3b)")

    def test_model_not_installed(self):
        with MockOllama(models=["mistral:7b"]) as m:
            c = client(m.url)
            h = c.health(force=True)
            self.assertTrue(h["reachable"])
            self.assertFalse(h["model_available"])
            self.assertIn("ollama pull llama3.2:3b", h["error"])
            self.assertFalse(c.has_real_provider)
            self.assertEqual(c.active_provider, "offline")
            with self.assertRaises(LLMModelMissingError):
                c.complete("s", "u")

    def test_latest_tag_alias(self):
        with MockOllama(models=["llama3.2:latest"]) as m:
            self.assertTrue(client(m.url, model="llama3.2").health(force=True)["model_available"])

    def test_unreachable_is_reported_honestly(self):
        c = client("http://127.0.0.1:9", retries=1)          # nothing listens on port 9
        h = c.health(force=True)
        self.assertFalse(h["reachable"])
        self.assertIn("Cannot reach Ollama", h["error"])
        self.assertFalse(c.has_real_provider)
        with self.assertRaises(LLMUnavailableError):         # never a fabricated answer
            c.complete("s", "u")
        with self.assertRaises(LLMUnavailableError):
            c.generate_json("x")
        with self.assertRaises(LLMUnavailableError):
            list(c.chat_stream([{"role": "user", "content": "x"}]))

    def test_retry_on_503_then_success(self):
        with MockOllama(fail_first=1, reply="recovered") as m:
            import backend.llm_provider as lp
            real_sleep, lp.time.sleep = lp.time.sleep, lambda s: None
            try:
                self.assertEqual(client(m.url, retries=2).complete("s", "u"), "recovered")
            finally:
                lp.time.sleep = real_sleep
            self.assertEqual(m.calls, 2)

    def test_gives_up_after_retries(self):
        with MockOllama(fail_first=99) as m:
            import backend.llm_provider as lp
            real_sleep, lp.time.sleep = lp.time.sleep, lambda s: None
            try:
                with self.assertRaises(LLMUnavailableError):
                    client(m.url, retries=2).complete("s", "u")
            finally:
                lp.time.sleep = real_sleep
            self.assertEqual(m.calls, 3)

    def test_timeout(self):
        with MockOllama(delay=1.5) as m:
            with self.assertRaises(LLMUnavailableError):
                client(m.url, timeout=0.3, retries=0).complete("s", "u")


if __name__ == "__main__":
    unittest.main()
