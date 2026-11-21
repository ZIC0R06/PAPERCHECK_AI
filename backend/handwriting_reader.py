"""
Handwritten answer-sheet reader
===============================
Turns a scanned / photographed answer booklet (PDF or image) into question-wise answer text,
with the page and vertical position of every answer (used to annotate the evaluated PDF) and an
honest per-answer reading confidence.

Engines (first one that is configured wins; a page that fails falls back to the next engine):

1. **Vision AI** — reads handwriting like a human examiner and splits the page by question number.
   Configure ONE key in .env:
       ANTHROPIC_API_KEY   → Claude        (HANDWRITING_CLAUDE_MODEL,  default claude-sonnet-5-5)
       GEMINI_API_KEY      → Google Gemini (HANDWRITING_GEMINI_MODEL,  default gemini-2.5-flash)
       OPENAI_API_KEY      → OpenAI        (HANDWRITING_OPENAI_MODEL,  default gpt-4.1-mini)
       GROQ_API_KEY        → Groq (Llama 4 vision) (HANDWRITING_GROQ_MODEL)
          HANDWRITING_PROVIDER=ollama + HANDWRITING_OLLAMA_MODEL=<vision model>  → LOCAL vision (no key)
           e.g.  ollama pull llama3.2-vision   (llama3.2:3b itself is text-only and cannot read images)
   HANDWRITING_PROVIDER=auto|ollama|anthropic|gemini|openai|groq|local   (default auto)

2. **Local OCR** (Tesseract, no internet, no key) — line-aware reading on the ink channel with ruled
   lines removed, question markers ("Q1", "Ans 2", "3)") detected in the margin and reconciled with
   the question paper's numbering. Handwriting accuracy is limited: every answer read this way is
   flagged so the teacher compares the transcription with the original page before approving.

The AI is told to transcribe exactly what the student wrote (no corrections), because the
transcription is what gets marked.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 200
VISION_MAX_SIDE = int(os.getenv("HANDWRITING_MAX_SIDE", "1800"))
HTTP_TIMEOUT = float(os.getenv("HANDWRITING_TIMEOUT", "120"))

PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}
DEFAULT_MODELS = {
    "anthropic": ("HANDWRITING_CLAUDE_MODEL", "claude-sonnet-5-5"),
    "gemini": ("HANDWRITING_GEMINI_MODEL", "gemini-2.5-flash"),
    "openai": ("HANDWRITING_OPENAI_MODEL", "gpt-4.1-mini"),
    "groq": ("HANDWRITING_GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "ollama": ("HANDWRITING_OLLAMA_MODEL", ""),          # no default: needs an installed vision model
}
LOCAL_ENGINE = "local-tesseract"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageSegment:
    page: int                         # 1-based
    question_number: Optional[int]    # None = continues the previous answer
    label: str
    text: str
    y0: float                         # fraction of page height
    y1: float
    confidence: float                 # 0..1 reading confidence
    engine: str


@dataclass
class ReadAnswer:
    question_number: int
    text: str
    page_refs: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = ""
    illegible_words: int = 0
    labels: List[str] = field(default_factory=list)


@dataclass
class SheetReading:
    answers: List[ReadAnswer]
    pages: int
    engine: str                       # e.g. "anthropic:claude-sonnet-5-5" or "local-tesseract"
    page_engines: List[str]
    full_text: str
    unassigned_text: str = ""         # text that could not be attributed to a question
    warnings: List[str] = field(default_factory=list)

    @property
    def used_local(self) -> bool:
        return any(e.startswith("local") for e in self.page_engines)

    def report(self) -> Dict:
        return {"pages": self.pages, "engine": self.engine, "page_engines": self.page_engines,
                "answers": len(self.answers), "warnings": self.warnings,
                "unassigned_text": self.unassigned_text[:4000]}


# ─────────────────────────────────────────────────────────────────────────────
# Page images
# ─────────────────────────────────────────────────────────────────────────────

def load_pages(path: str, dpi: int = RENDER_DPI) -> List[Image.Image]:
    """Render every page of a PDF (or load an image) as RGB PIL images."""
    if path.lower().endswith(".pdf"):
        try:
            try:
                import pymupdf as fitz
            except ImportError:  # pragma: no cover
                import fitz
        except ImportError:      # PyMuPDF missing → pypdfium2 fallback (pure wheel, works on Windows)
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(path)
            try:
                return [doc[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(doc))]
            finally:
                doc.close()
        out = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB" if pix.n < 4 else "RGBA", (pix.width, pix.height), pix.samples)
                out.append(img.convert("RGB"))
        return out
    from PIL import ImageOps
    return [ImageOps.exif_transpose(Image.open(path)).convert("RGB")]


def _jpeg_b64(img: Image.Image) -> str:
    im = img.copy()
    im.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Vision AI providers
# ─────────────────────────────────────────────────────────────────────────────

def configured_providers() -> List[str]:
    want = os.getenv("HANDWRITING_PROVIDER", "auto").strip().lower()
    if want == "local":
        return []
    avail = [p for p, k in PROVIDER_KEYS.items() if os.getenv(k, "").strip()]
    # Local Ollama vision is opt-in (explicit model required) and only used when the model is really installed.
    if os.getenv("HANDWRITING_OLLAMA_MODEL", "").strip() and want in ("ollama", "auto"):
        if _ollama_vision_ready():
            avail = ["ollama"] + avail if want == "ollama" else avail + ["ollama"]
    if want == "ollama":
        return avail
    if want in PROVIDER_KEYS:
        return [want] + [p for p in avail if p != want] if want in avail else avail
    return avail


def _ollama_vision_ready() -> bool:
    try:
        from backend.llm_provider import get_llm_client, _norm_model
        h = get_llm_client().health()
        m = _norm_model(os.getenv("HANDWRITING_OLLAMA_MODEL", ""))
        return bool(h.get("reachable") and m in {_norm_model(n) for n in h.get("installed_models") or []})
    except Exception:
        return False


def _call_ollama(img_b64: str, prompt: str) -> str:
    import requests
    base = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
    r = requests.post(f"{base}/api/chat", timeout=HTTP_TIMEOUT, json={
        "model": _model("ollama"), "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192") or 8192)},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt, "images": [img_b64]}]})
    if r.status_code >= 400:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    return (r.json().get("message") or {}).get("content") or ""


def _model(provider: str) -> str:
    env, default = DEFAULT_MODELS[provider]
    return os.getenv(env, default)


def _questions_brief(questions: List[Dict]) -> str:
    if not questions:
        return "(question paper not available — use the numbers the student wrote)"
    rows = []
    for q in questions:
        t = re.sub(r"\s+", " ", q.get("question_text") or "")[:140]
        rows.append(f"Q{q['question_number']} ({q.get('marks') or '?'} marks): {t}")
    return "\n".join(rows)


SYSTEM_PROMPT = (
    "You are an exam-script transcriber for a university. You read handwritten student answer sheets "
    "and transcribe them EXACTLY as written, so that an examiner can mark the transcription. Never correct "
    "spelling, grammar or facts, never complete unfinished sentences, never add content. Output JSON only."
)


def build_page_prompt(page_no: int, total: int, questions: List[Dict], prev_question: Optional[int]) -> str:
    prev = f"The previous page ended inside the answer to Q{prev_question}." if prev_question else \
        "No answer has started before this page."
    return f"""This is page {page_no} of {total} of a student's handwritten answer sheet.

Questions on the official question paper:
{_questions_brief(questions)}

{prev}

Instructions:
- Transcribe every handwritten answer on this page exactly as written (keep the student's mistakes).
- Write a word you cannot read as [illegible]. Leave out text the student struck through.
- Describe a drawn diagram briefly as [diagram: …]; write formulas/equations in plain text.
- Split the page into segments, one per question the student is answering. A segment that continues
  the previous page's answer (no new question number written) has "question_number": null.
- Use the question number the student wrote (e.g. "Q3", "Ans 3", "3)", "3(b)" → 3).
- Ignore printed headers, page numbers, cover-page boxes and the examiner's margin.

Return ONLY this JSON:
{{"is_blank": false,
  "segments": [{{"question_number": 3, "label": "Q3(b)", "text": "…exact transcription…",
                 "y_start": 0.10, "y_end": 0.42, "legibility": 0.9}}]}}
y_start / y_end = top / bottom of the segment as a fraction of the page height (0 = top, 1 = bottom).
legibility = how sure you are of the transcription (1 = completely clear, 0.5 = many guesses)."""


def _http_post(url: str, headers: Dict, payload: Dict) -> Dict:
    import httpx
    r = httpx.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def _call_anthropic(img_b64: str, prompt: str) -> str:
    data = _http_post(
        os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages"),
        {"x-api-key": os.getenv("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": _model("anthropic"), "max_tokens": 4096, "temperature": 0, "system": SYSTEM_PROMPT,
         "messages": [{"role": "user", "content": [
             {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
             {"type": "text", "text": prompt}]}]})
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _call_gemini(img_b64: str, prompt: str) -> str:
    model = _model("gemini")
    data = _http_post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": os.getenv("GEMINI_API_KEY", ""), "content-type": "application/json"},
        {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
         "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                                                 {"text": prompt}]}],
         "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _call_openai_compatible(base_url: str, key_env: str, provider: str, img_b64: str, prompt: str) -> str:
    payload = {"model": _model(provider), "temperature": 0, "max_tokens": 4096,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]}
    if provider == "openai":
        payload["response_format"] = {"type": "json_object"}
    data = _http_post(base_url, {"Authorization": f"Bearer {os.getenv(key_env, '')}",
                                 "content-type": "application/json"}, payload)
    return data["choices"][0]["message"]["content"] or ""


CALLERS: Dict[str, Callable[[str, str], str]] = {
    "ollama": _call_ollama,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "openai": lambda b, p: _call_openai_compatible("https://api.openai.com/v1/chat/completions",
                                                   "OPENAI_API_KEY", "openai", b, p),
    "groq": lambda b, p: _call_openai_compatible("https://api.groq.com/openai/v1/chat/completions",
                                                 "GROQ_API_KEY", "groq", b, p),
}


def _json_from_text(raw: str) -> Dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("model did not return JSON")
        return json.loads(m.group(0))


def _qnum(v) -> Optional[int]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d{1,3}", str(v))
    return int(m.group(0)) if m else None


def parse_vision_page(raw: str, page_no: int, engine: str) -> List[PageSegment]:
    data = _json_from_text(raw)
    segs = []
    for s in data.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        y0 = float(np.clip(float(s.get("y_start", 0) or 0), 0, 1))
        y1 = float(np.clip(float(s.get("y_end", 1) or 1), 0, 1))
        if y1 <= y0:
            y1 = min(1.0, y0 + 0.1)
        leg = float(np.clip(float(s.get("legibility", 0.8) or 0.8), 0, 1))
        segs.append(PageSegment(page_no, _qnum(s.get("question_number")), str(s.get("label") or ""),
                                text, y0, y1, leg, engine))
    return segs


def read_page_vision(provider: str, img: Image.Image, page_no: int, total: int, questions: List[Dict],
                     prev_q: Optional[int]) -> List[PageSegment]:
    raw = CALLERS[provider](_jpeg_b64(img), build_page_prompt(page_no, total, questions, prev_q))
    return parse_vision_page(raw, page_no, f"{provider}:{_model(provider)}")


# ─────────────────────────────────────────────────────────────────────────────
# Local OCR (Tesseract, line-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _ink_image(img: Image.Image) -> Image.Image:
    """Keep pen strokes, drop ruled lines / margins / shadows (red channel + background division)."""
    import cv2
    arr = np.asarray(img.convert("RGB"))
    # blue/black ink is dark in the red channel; light-blue rules and red margins are light there
    ch = arr[:, :, 0].copy()
    k = max(15, (min(ch.shape) // 60) | 1)
    bg = cv2.medianBlur(cv2.dilate(ch, np.ones((k, k), np.uint8)), 31)
    flat = cv2.divide(ch, bg, scale=255)
    return Image.fromarray(flat)


def _tesseract_lines(img: Image.Image) -> List[Dict]:
    import pytesseract
    from backend.tesseract_locator import require
    require()                          # finds Tesseract even when it is not on PATH (typical on Windows)
    d = pytesseract.image_to_data(_ink_image(img), config="--oem 1 --psm 4", output_type=pytesseract.Output.DICT)
    lines: Dict[tuple, List] = {}
    for i, w in enumerate(d["text"]):
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not w.strip() or conf < 0:
            continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(k, []).append((d["left"][i], d["top"][i], d["left"][i] + d["width"][i],
                                        d["top"][i] + d["height"][i], w, conf))
    out = []
    for ws in lines.values():
        ws.sort(key=lambda t: t[0])
        out.append({"x0": min(t[0] for t in ws), "y0": min(t[1] for t in ws), "x1": max(t[2] for t in ws),
                    "y1": max(t[3] for t in ws), "words": [t[4] for t in ws], "conf": float(np.mean([t[5] for t in ws]))})
    out.sort(key=lambda L: (L["y0"], L["x0"]))
    return out


_DIGITLIKE = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "i": "1", "!": "1",
                            "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9", "T": "7"})
_MARKER = re.compile(r"^[\(\[]?\s*(?:q(?:ue(?:s(?:tion)?)?)?|a[nvuwmh][s5$](?:wer)?|a)\s*[.:\-–#]?\s*(?:no\.?\s*)?"
                     r"([0-9OoIl|iZzSsBgqT!]{1,2})\s*(?:\(?[a-hivx]{1,3}\)|[a-hivx]{1,3}(?=[.):\]]))?\s*[.):\]\-–]?\s*(.*)$", re.I)
# sub-part labels ("(a)", "b)", "ii.") need a bracket / full stop, so "Q3 Deadlock…" keeps its first word
_NUM_MARKER = re.compile(r"^[\(\[]?\s*(\d{1,2})\s*(?:\(?[a-h]\)?)?\s*[.):\]]\s*(.*)$")


def _marker(line: Dict, margin_x: float, page_w: int, body_x: Optional[float] = None) -> Optional[tuple]:
    """(raw_number_or_None, rest_of_line) if the line starts with a question marker."""
    text = " ".join(line["words"]).strip()
    m = _MARKER.match(text)
    if m:
        # a letter that only LOOKS like a digit and runs into a word is not a number: "Answer Sheet", "Ans Below"
        if not any(ch.isdigit() for ch in m.group(1)) and text[m.end(1):m.end(1) + 1].isalpha():
            return None
        raw = m.group(1).translate(_DIGITLIKE)
        rest = m.group(2).strip()
        # "a" alone is too ambiguous unless the line starts at the margin
        if text[:1].lower() == "a" and not re.match(r"a[nvuwmh][s5$]", text.lower()) and line["x0"] > margin_x + 0.02 * page_w:
            return None
        return (int(raw) if raw.isdigit() else None, rest)
    at_margin = line["x0"] <= margin_x + 0.03 * page_w
    m = _NUM_MARKER.match(text)
    if m and at_margin:
        return int(m.group(1)), m.group(2).strip()
    # A tiny token written alone, left of where the answer text starts ("1)", "Q2", "3." — often misread as
    # "))", "G2", "iS;") is a question marker; its number is reconciled with the question paper later.
    compact = text.replace(" ", "")
    outdented = body_x is not None and line["x0"] <= body_x - 0.012 * page_w
    if len(compact) <= 4 and (outdented or (at_margin and body_x is None)):
        d = re.search(r"\d{1,2}", compact.lstrip("|!([{-_~'\".,").translate(_DIGITLIKE))
        return (int(d.group(0)) if d else None), ""
    return None


def read_page_local(img: Image.Image, page_no: int) -> List[PageSegment]:
    lines = _tesseract_lines(img)
    W, H = img.size
    if not lines:
        return []
    margin_x = float(np.percentile([L["x0"] for L in lines], 10))
    body = [L["x0"] for L in lines if len(" ".join(L["words"])) > 8]
    body_x = float(np.median(body)) if body else None
    segs: List[PageSegment] = []
    cur: Optional[PageSegment] = None
    cur_conf: List[float] = []

    def close():
        if cur and cur.text.strip():
            cur.confidence = round(float(np.mean(cur_conf)) / 100.0 * 0.85, 3) if cur_conf else 0.3
            segs.append(cur)

    for L in lines:
        text = " ".join(L["words"]).strip()
        mk = _marker(L, margin_x, W, body_x) if text else None
        if mk is None and (not text or (len(text) <= 2 and not re.search(r"\w", text))):
            continue
        if mk is not None:
            close()
            raw_n, rest = mk
            rest = rest.lstrip(" ,.;:-–")
            # number resolved later against the paper; keep raw guess (-1 = unreadable marker)
            cur = PageSegment(page_no, raw_n if raw_n is not None else -1, text.split()[0], rest,
                              L["y0"] / H, L["y1"] / H, 0.0, LOCAL_ENGINE)
            cur_conf = [L["conf"]] if rest else []
            continue
        if cur is None:
            cur = PageSegment(page_no, None, "", "", L["y0"] / H, L["y1"] / H, 0.0, LOCAL_ENGINE)
            cur_conf = []
        cur.text = (cur.text + "\n" + text).strip()
        cur.y1 = max(cur.y1, L["y1"] / H)
        cur_conf.append(L["conf"])
    close()
    return segs


def resolve_question_numbers(segs: List[PageSegment], expected: List[int]) -> None:
    """
    Reconcile OCR-read markers with the question paper. Handwritten "1" is often read as "7", "2" as "z"…
    A marker whose number is a valid, not-yet-seen paper question (and moves forward) is kept; otherwise the
    next unseen question of the paper is assigned. Without a paper, raw numbers are kept.
    """
    if not expected:
        for s in segs:
            if s.question_number == -1:
                s.question_number = None
        return
    order = sorted(set(expected))
    seen: List[int] = []
    for s in segs:
        if s.question_number is None:
            continue
        n = s.question_number
        last = seen[-1] if seen else 0
        if n in order and n not in seen and n > last:
            pass
        elif n in order and n in seen and not [q for q in order if q > last and q not in seen]:
            pass                                       # student returned to an earlier question at the end
        else:
            nxt = [q for q in order if q > last and q not in seen]
            if not nxt:
                nxt = [q for q in order if q not in seen]
            if not nxt:
                s.question_number = None               # extra marker → treat as continuation
                continue
            s.question_number = nxt[0]
            s.label = (s.label + f" (read as Q{n}, matched to Q{nxt[0]})") if n not in (-1, None) else s.label
        seen.append(s.question_number)


# ─────────────────────────────────────────────────────────────────────────────
# Whole sheet
# ─────────────────────────────────────────────────────────────────────────────

def _merge(segs: List[PageSegment]) -> tuple:
    answers: Dict[int, ReadAnswer] = {}
    weights: Dict[int, List[tuple]] = {}
    unassigned: List[str] = []
    current: Optional[int] = None
    for s in segs:
        q = s.question_number if s.question_number is not None else current
        if q is None:
            unassigned.append(s.text)
            continue
        current = q
        a = answers.setdefault(q, ReadAnswer(question_number=q, text=""))
        a.text = (a.text + "\n" + s.text).strip()
        if a.page_refs and a.page_refs[-1]["page"] == s.page and s.question_number is None:
            a.page_refs[-1]["y1"] = round(max(a.page_refs[-1]["y1"], s.y1), 4)
        else:
            a.page_refs.append({"page": s.page, "y0": round(s.y0, 4), "y1": round(s.y1, 4)})
        if s.label:
            a.labels.append(s.label)
        weights.setdefault(q, []).append((s.confidence, max(1, len(s.text))))
        a.engine = s.engine if not a.engine or a.engine == s.engine else "mixed"
    for q, a in answers.items():
        w = weights[q]
        a.confidence = round(sum(c * n for c, n in w) / sum(n for _, n in w), 3)
        a.illegible_words = len(re.findall(r"\[illegible\]", a.text, re.I))
        if a.illegible_words:
            a.confidence = round(a.confidence * max(0.4, 1 - 0.05 * a.illegible_words), 3)
    return [answers[k] for k in sorted(answers)], "\n".join(unassigned).strip()


def read_answer_sheet(path: str, questions: Optional[List[Dict]] = None,
                      pages: Optional[List[Image.Image]] = None) -> SheetReading:
    """
    questions: [{"question_number": 1, "question_text": "...", "marks": 10}, ...] from the official paper.
    """
    questions = questions or []
    imgs = pages if pages is not None else load_pages(path)
    providers = configured_providers()
    expected = [int(q["question_number"]) for q in questions if q.get("question_number") is not None]
    all_segs: List[PageSegment] = []
    page_engines: List[str] = []
    warnings: List[str] = []
    prev_q: Optional[int] = None
    disabled: set = set()
    local_failures = 0
    for i, img in enumerate(imgs, start=1):
        segs: Optional[List[PageSegment]] = None
        for p in providers:
            if p in disabled:
                continue
            try:
                segs = read_page_vision(p, img, i, len(imgs), questions, prev_q)
                page_engines.append(f"{p}:{_model(p)}")
                break
            except Exception as e:  # network / quota / bad key → try the next engine
                msg = f"page {i}: {p} vision reading failed ({str(e)[:160]})"
                logger.warning(msg)
                warnings.append(msg)
                if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                    disabled.add(p)
        if segs is None:
            try:
                segs = read_page_local(img, i)
            except Exception as e:
                from backend.tesseract_locator import TesseractMissingError
                if isinstance(e, TesseractMissingError):
                    raise                  # precise install instructions; never grade a sheet we could not read
                logger.exception("local OCR failed")
                warnings.append(f"page {i}: local OCR failed ({type(e).__name__}: {e})")
                local_failures += 1
                segs = []
            page_engines.append(LOCAL_ENGINE)
        all_segs.extend(segs)
        for s in segs:
            if s.question_number not in (None, -1):
                prev_q = s.question_number
    if imgs and local_failures == len(imgs) and not all_segs:
        # Every page failed to read: stop here. Silently continuing would grade blank answers as "not attempted".
        raise RuntimeError("The answer sheet could not be read (OCR failed on every page): " + "; ".join(warnings[-3:]))
    # numbers read by local OCR are reconciled with the paper; AI-read numbers are trusted
    local = [s for s in all_segs if s.engine == LOCAL_ENGINE]
    if local:
        resolve_question_numbers(all_segs if all(s.engine == LOCAL_ENGINE for s in all_segs) else local, expected)
        warnings.append("Some pages were read with local OCR (no vision-AI key configured or the AI call failed). "
                        "Handwriting transcription must be checked against the original page.")
    answers, unassigned = _merge(all_segs)
    if expected:
        extra = [a.question_number for a in answers if a.question_number not in expected]
        if extra:
            warnings.append(f"Answers labelled {extra} do not exist on the question paper.")
    engines = sorted(set(page_engines))
    full = "\n\n".join(f"[Page {s.page}] " + (f"Q{s.question_number}: " if s.question_number else "") + s.text
                       for s in all_segs)
    return SheetReading(answers=answers, pages=len(imgs), engine="+".join(engines) or "none",
                        page_engines=page_engines, full_text=full, unassigned_text=unassigned, warnings=warnings)
