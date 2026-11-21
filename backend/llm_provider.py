"""
PaperCheckAI — Central LLM provider (Ollama first)
===================================================
One shared interface for every AI module in the project.

DEFAULT: a local Ollama server — no API key, no cloud, no cost.
    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3.2:3b

Cloud providers (groq / claude / gemini / openai) are OPTIONAL. They are only used when
LLM_PROVIDER names them, or when listed in LLM_FALLBACKS *and* their API key is set.
No cloud SDK packages are needed: everything is plain HTTPS through `requests`.

Public interface (unchanged names, so existing modules keep working)
    get_llm_client() / LLMClient.from_env()
    complete(system, user, max_tokens)          -> str
    generate(prompt, max_tokens)                -> LLMResponse
    generate_json(prompt, max_tokens)           -> dict
    stream(system, user, max_tokens, on_token)  -> str
    has_real_provider / active_provider
Added
    chat(messages, ...)  -> LLMResponse         (system prompt + conversation history)
    chat_stream(messages, ...)  -> iterator of text deltas
    health()             -> dict                (reachable? model installed? latency, error)

FAILURE POLICY: if no provider can answer, an LLMError is RAISED. This module never
invents a result (the old code returned a made-up `score: 5.0`). Callers decide what to
do — typically flag the item for teacher review.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Base class: the AI provider could not produce a usable answer."""


class LLMUnavailableError(LLMError):
    """Provider unreachable / timed out / all retries failed."""


class LLMModelMissingError(LLMUnavailableError):
    """Ollama is running but the configured model is not installed."""


class LLMParseError(LLMError):
    """The model answered, but not with valid JSON when JSON was required."""


# ── Configuration (read lazily so .env / test overrides always apply) ─────────

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except ValueError:
        return default


def load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:                                   # python-dotenv is optional here
        pass


load_dotenv_once()

CLOUD_PROVIDERS = ("groq", "claude", "gemini", "openai")
_CLOUD_KEYS = {"groq": "GROQ_API_KEY", "claude": "ANTHROPIC_API_KEY",
               "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
_CLOUD_MODEL_DEFAULTS = {"groq": ("GROQ_MODEL", "llama-3.3-70b-versatile"),
                         "claude": ("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
                         "gemini": ("GEMINI_MODEL", "gemini-2.5-flash"),
                         "openai": ("OPENAI_MODEL", "gpt-4.1-mini")}


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.2
    timeout: float = 120.0
    max_tokens: int = 4096
    num_ctx: int = 8192
    retries: int = 2
    keep_alive: str = "30m"
    fallbacks: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = _env("LLM_PROVIDER", "ollama").lower() or "ollama"
        fb = [p.strip().lower() for p in _env("LLM_FALLBACKS", "").split(",") if p.strip()]
        return cls(
            provider=provider,
            base_url=(_env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/"),
            model=_env("OLLAMA_MODEL", "llama3.2:3b") or "llama3.2:3b",
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            timeout=_env_float("LLM_TIMEOUT", 120),
            max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
            num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
            retries=max(0, _env_int("LLM_RETRIES", 2)),
            keep_alive=_env("OLLAMA_KEEP_ALIVE", "30m") or "30m",
            fallbacks=fb,
        )


@dataclass
class LLMResponse:
    """Standardised response returned by generate()/chat()."""
    text: str
    provider: str = "ollama"
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


Message = Dict[str, str]      # {"role": "system"|"user"|"assistant", "content": "..."}


def _norm_model(name: str) -> str:
    """'llama3.2:3b' stays, 'llama3.2' == 'llama3.2:latest'."""
    name = (name or "").strip().lower()
    return name if ":" in name else name + ":latest"


# ── Ollama backend ───────────────────────────────────────────────────────────

class OllamaBackend:
    name = "ollama"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @property
    def model(self) -> str:
        return self.cfg.model

    def _http(self):
        import requests
        return requests

    def _payload(self, messages: List[Message], max_tokens: int, json_mode: bool,
                 stream: bool, temperature: Optional[float]) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": int(min(max_tokens or self.cfg.max_tokens, self.cfg.max_tokens)),   # LLM_MAX_TOKENS is the ceiling
                # Ollama's default context is only 2048 tokens and silently TRUNCATES longer
                # prompts (documents, rubrics). Raise it explicitly.
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if json_mode:
            p["format"] = "json"
        return p

    # -- health ---------------------------------------------------------------
    def health(self, timeout: float = 5.0) -> Dict[str, Any]:
        t0 = time.time()
        out: Dict[str, Any] = {"provider": "ollama", "base_url": self.cfg.base_url, "model": self.cfg.model,
                               "reachable": False, "model_available": False, "installed_models": [],
                               "latency_ms": None, "error": None}
        try:
            r = self._http().get(f"{self.cfg.base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
            names = [m.get("name") or m.get("model") or "" for m in (r.json().get("models") or [])]
            out.update(reachable=True, installed_models=names,
                       model_available=_norm_model(self.cfg.model) in {_norm_model(n) for n in names},
                       latency_ms=round((time.time() - t0) * 1000, 1))
            if not out["model_available"]:
                out["error"] = (f"Model '{self.cfg.model}' is not installed in Ollama. "
                                f"Run: ollama pull {self.cfg.model}")
        except Exception as e:
            out["error"] = (f"Cannot reach Ollama at {self.cfg.base_url} ({type(e).__name__}). "
                            "Start it with the Ollama app or `ollama serve`.")
            logger.debug("Ollama health check failed: %s", e)
        return out

    # -- one request with retry -----------------------------------------------
    def _post_with_retry(self, payload: Dict[str, Any], stream: bool):
        requests = self._http()
        last: Optional[Exception] = None
        attempts = self.cfg.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(f"{self.cfg.base_url}/api/chat", json=payload, stream=stream,
                                  timeout=(10, self.cfg.timeout))
                if r.status_code == 404:                       # model not pulled
                    detail = ""
                    try:
                        detail = r.json().get("error", "")
                    except Exception:
                        pass
                    raise LLMModelMissingError(
                        f"Ollama model '{self.cfg.model}' not found ({detail or '404'}). "
                        f"Run: ollama pull {self.cfg.model}")
                if r.status_code >= 500 or r.status_code == 429:
                    raise LLMUnavailableError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise LLMError(f"Ollama rejected the request (HTTP {r.status_code}): {r.text[:300]}")
                return r
            except LLMModelMissingError:
                raise
            except LLMError as e:
                if not isinstance(e, LLMUnavailableError):
                    raise
                last = e
            except (requests.ConnectionError, requests.Timeout) as e:
                last = LLMUnavailableError(
                    f"Ollama is not reachable at {self.cfg.base_url} ({type(e).__name__}). "
                    "Start Ollama and check OLLAMA_BASE_URL.")
            if attempt < attempts:
                wait = min(2 ** (attempt - 1), 8)
                logger.warning("Ollama attempt %d/%d failed: %s — retrying in %ss", attempt, attempts, last, wait)
                time.sleep(wait)
        raise last or LLMUnavailableError("Ollama request failed")

    # -- public ---------------------------------------------------------------
    def chat(self, messages: List[Message], max_tokens: int, json_mode: bool = False,
             temperature: Optional[float] = None) -> LLMResponse:
        t0 = time.time()
        r = self._post_with_retry(self._payload(messages, max_tokens, json_mode, False, temperature), False)
        try:
            data = r.json()
        except ValueError as e:
            raise LLMError(f"Ollama sent a non-JSON response: {r.text[:200]}") from e
        if data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty answer.")
        resp = LLMResponse(text=text, provider="ollama", model=data.get("model") or self.cfg.model,
                           latency_ms=round((time.time() - t0) * 1000, 1),
                           prompt_tokens=int(data.get("prompt_eval_count") or 0),
                           completion_tokens=int(data.get("eval_count") or 0))
        logger.info("ollama/%s answered in %.0f ms (%d prompt + %d completion tokens)",
                    resp.model, resp.latency_ms, resp.prompt_tokens, resp.completion_tokens)
        return resp

    def chat_stream(self, messages: List[Message], max_tokens: int,
                    temperature: Optional[float] = None) -> Iterator[str]:
        r = self._post_with_retry(self._payload(messages, max_tokens, False, True, temperature), True)
        got_any = False
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("error"):
                    raise LLMError(f"Ollama error: {obj['error']}")
                delta = (obj.get("message") or {}).get("content") or ""
                if delta:
                    got_any = True
                    yield delta
                if obj.get("done"):
                    break
        finally:
            r.close()
        if not got_any:
            raise LLMError("Ollama returned an empty streamed answer.")


# ── Optional cloud backends (plain HTTPS, no SDKs) ───────────────────────────

class CloudBackend:
    def __init__(self, name: str, cfg: LLMConfig):
        self.name = name
        self.cfg = cfg
        self.key = _env(_CLOUD_KEYS[name])
        env_name, default = _CLOUD_MODEL_DEFAULTS[name]
        self.model = _env(env_name, default)

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def chat(self, messages: List[Message], max_tokens: int, json_mode: bool = False,
             temperature: Optional[float] = None) -> LLMResponse:
        import requests
        t = self.cfg.temperature if temperature is None else temperature
        t0 = time.time()
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        try:
            if self.name in ("groq", "openai"):
                url = ("https://api.groq.com/openai/v1/chat/completions" if self.name == "groq"
                       else "https://api.openai.com/v1/chat/completions")
                body = {"model": self.model, "messages": messages, "temperature": t, "max_tokens": max_tokens}
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                r = requests.post(url, headers={"Authorization": f"Bearer {self.key}"}, json=body,
                                  timeout=self.cfg.timeout)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
            elif self.name == "claude":
                r = requests.post("https://api.anthropic.com/v1/messages",
                                  headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                                  json={"model": self.model, "max_tokens": min(max_tokens, 4096), "system": system,
                                        "temperature": t, "messages": convo}, timeout=self.cfg.timeout)
                r.raise_for_status()
                text = "".join(b.get("text", "") for b in r.json().get("content", []))
            else:  # gemini
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": self.key},
                    json={"systemInstruction": {"parts": [{"text": system}]},
                          "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                                        "parts": [{"text": m["content"]}]} for m in convo],
                          "generationConfig": {"temperature": t, "maxOutputTokens": max_tokens}},
                    timeout=self.cfg.timeout)
                r.raise_for_status()
                parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
        except Exception as e:
            raise LLMUnavailableError(f"{self.name} request failed: {str(e)[:200]}") from e
        text = (text or "").strip()
        if not text:
            raise LLMError(f"{self.name} returned an empty answer.")
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_ms=round((time.time() - t0) * 1000, 1))


# ── JSON helper ──────────────────────────────────────────────────────────────

def parse_json_text(raw: str) -> Dict[str, Any]:
    """Parse a model's JSON answer (tolerates ``` fences and surrounding prose)."""
    cleaned = re.sub(r"```(?:json)?", "", raw or "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
    except ValueError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            raise LLMParseError(f"No JSON object in model output: {cleaned[:200]!r}")
        try:
            obj = json.loads(m.group(0))
        except ValueError as e:
            raise LLMParseError(f"Invalid JSON from model: {e}") from e
    if not isinstance(obj, dict):
        raise LLMParseError("Model returned JSON that is not an object.")
    return obj


# ── The shared client ────────────────────────────────────────────────────────

DEFAULT_SYSTEM = "You are a careful, accurate assistant for a university examination platform."


class LLMClient:
    """Provider-agnostic client. Order: LLM_PROVIDER first, then LLM_FALLBACKS (cloud, only if keyed)."""

    HEALTH_TTL = 10.0

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig.from_env()
        self._ollama = OllamaBackend(self.cfg)
        self._cloud = {n: CloudBackend(n, self.cfg) for n in CLOUD_PROVIDERS}
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_at = 0.0
        self._lock = threading.Lock()
        logger.info("LLMClient ready: provider=%s model=%s url=%s", self.cfg.provider, self.cfg.model,
                    self.cfg.base_url)

    # -- provider order ---------------------------------------------------------
    def _order(self) -> List[str]:
        order = [self.cfg.provider] if self.cfg.provider in ("ollama",) + CLOUD_PROVIDERS else ["ollama"]
        for p in self.cfg.fallbacks:
            if p in ("ollama",) + CLOUD_PROVIDERS and p not in order:
                order.append(p)
        return [p for p in order if p == "ollama" or self._cloud[p].configured]

    def _backend(self, name: str):
        return self._ollama if name == "ollama" else self._cloud[name]

    # -- status ---------------------------------------------------------------
    def health(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if force or not self._health_cache or time.time() - self._health_at > self.HEALTH_TTL:
                if "ollama" in self._order():
                    h = self._ollama.health()
                else:       # cloud-only configuration
                    name = self._order()[0] if self._order() else None
                    h = {"provider": name or "none", "model": self._cloud[name].model if name else "",
                         "reachable": bool(name), "model_available": bool(name), "installed_models": [],
                         "base_url": "", "latency_ms": None,
                         "error": None if name else "No AI provider configured."}
                h["configured_provider"] = self.cfg.provider
                h["fallbacks"] = [p for p in self._order() if p != self.cfg.provider]
                self._health_cache, self._health_at = h, time.time()
            return dict(self._health_cache)

    def is_available(self) -> bool:
        h = self.health()
        return bool(h.get("reachable") and h.get("model_available"))

    @property
    def has_real_provider(self) -> bool:
        """True only when a provider can answer RIGHT NOW (Ollama up AND model installed)."""
        return self.is_available()

    @property
    def active_provider(self) -> str:
        """'ollama(llama3.2:3b)' while usable, else 'offline'."""
        if not self.is_available():
            return "offline"
        h = self.health()
        return f"{h['provider']}({h['model']})"

    # -- core call with fallback chain ----------------------------------------------
    def chat(self, messages: List[Message], max_tokens: Optional[int] = None, json_mode: bool = False,
             temperature: Optional[float] = None) -> LLMResponse:
        mt = int(max_tokens or self.cfg.max_tokens)
        errors: List[str] = []
        last: Optional[LLMError] = None
        for name in self._order():
            try:
                resp = self._backend(name).chat(messages, mt, json_mode, temperature)
                return resp
            except LLMError as e:
                logger.warning("Provider %s failed: %s", name, e)
                errors.append(f"{name}: {e}")
                last = e
                if name == "ollama":
                    self._health_cache = None          # re-check next time
        if isinstance(last, LLMModelMissingError) and len(errors) == 1:
            raise last                                  # keep the precise "ollama pull ..." error
        raise LLMUnavailableError("No AI provider could answer. " + " | ".join(errors or ["none configured"]))

    def chat_stream(self, messages: List[Message], max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None) -> Iterator[str]:
        mt = int(max_tokens or self.cfg.max_tokens)
        errors: List[str] = []
        for name in self._order():
            started = False
            try:
                if name == "ollama":
                    for d in self._ollama.chat_stream(messages, mt, temperature):
                        started = True
                        yield d
                else:        # cloud: single-shot, delivered as one chunk
                    yield self._cloud[name].chat(messages, mt, False, temperature).text
                return
            except LLMError as e:
                if started:            # cannot restart mid-answer without duplicating text
                    raise
                errors.append(f"{name}: {e}")
                logger.warning("Streaming via %s failed: %s", name, e)
        raise LLMUnavailableError("No AI provider could answer. " + " | ".join(errors or ["none configured"]))

    # -- convenience API used across the project ----------------------------------------
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None,
                 history: Optional[List[Message]] = None, temperature: Optional[float] = None) -> str:
        return self.chat(self._messages(system_prompt, user_prompt, history), max_tokens,
                         temperature=temperature).text

    def stream(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None,
               on_token: Optional[Callable[[str], None]] = None,
               history: Optional[List[Message]] = None) -> str:
        parts: List[str] = []
        for delta in self.chat_stream(self._messages(system_prompt, user_prompt, history), max_tokens):
            parts.append(delta)
            if on_token:
                on_token(delta)
        return "".join(parts).strip()

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None,
                 json_mode: bool = False) -> LLMResponse:
        return self.chat(self._messages(system or DEFAULT_SYSTEM, prompt, None), max_tokens, json_mode=json_mode)

    def generate_json(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None,
                      temperature: Optional[float] = None) -> Dict[str, Any]:
        """JSON-mode call. Retries once on malformed JSON, then raises LLMParseError (never fakes a result)."""
        msgs = self._messages(system or (DEFAULT_SYSTEM + " Reply with a single valid JSON object only."),
                              prompt, None)
        last: Optional[LLMParseError] = None
        for _ in range(2):
            resp = self.chat(msgs, max_tokens, json_mode=True, temperature=temperature)
            try:
                return parse_json_text(resp.text)
            except LLMParseError as e:
                last = e
                logger.warning("generate_json: bad JSON (%s) — retrying once", e)
        raise last or LLMParseError("unparseable JSON")

    @staticmethod
    def _messages(system: str, user: str, history: Optional[List[Message]]) -> List[Message]:
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in history or []:
            if m.get("content"):
                msgs.append({"role": "assistant" if m.get("role") == "assistant" else "user",
                             "content": m["content"]})
        msgs.append({"role": "user", "content": user})
        return msgs

    @classmethod
    def from_env(cls) -> "LLMClient":
        return get_llm_client()


# ── Singleton ────────────────────────────────────────────────────────────────────

_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = LLMClient()
        return _client


def reset_llm_client() -> None:
    """Drop the singleton (used by tests and after .env changes)."""
    global _client
    with _client_lock:
        _client = None
