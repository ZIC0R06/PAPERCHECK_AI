"""
Question-wise answer grading against the official paper, model answer and marking scheme
========================================================================================
grade_answer() returns justified marks for ONE answer:

  * rubric_breakdown — every marking point / criterion with marks available, marks awarded,
                       met | partial | missing | incorrect, the evidence quoted from the answer and why
  * deductions       — marks lost and the reason (derived from the breakdown, so they always add up)
  * corrections      — statements in the answer that are wrong, with the correct version (AI only)
  * strengths / missing_concepts / feedback for the student
  * confidence + flags that tell the teacher what to verify

Engines
  1. AI language model (local Ollama by default; cloud providers optional) for semantic marking.
  2. Without any key: deterministic marking-point matching (keywords with OCR-tolerant fuzzy matching
     + character n-gram similarity). It cannot judge meaning, so every answer it marks is flagged
     "verify" for the teacher. Objective questions (MCQ, true/false) are always marked deterministically.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MARKS_STEP = float(os.getenv("MARKS_STEP", "0.5"))        # awarded marks are rounded to this step
OCR_VERIFY_BELOW = float(os.getenv("OCR_VERIFY_BELOW", "0.7"))
AI_VERIFY_BELOW = float(os.getenv("AI_VERIFY_BELOW", "0.6"))

STOP = set("""a an the is are was were be been being of to in on at for from by with as and or but not no nor so if
then than that this these those it its into onto over under about above below between each every any all some such
which who whom whose what when where why how can could should would may might must will shall do does did done has
have had having also very more most less least only just other others same own both few many much one two their there
they them we our you your he she his her i me my us e.g eg ie etc used use using called known""".split())


@dataclass
class GradeResult:
    score: float
    max_marks: float
    confidence: float
    feedback: str
    strengths: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    rubric_breakdown: List[Dict] = field(default_factory=list)
    deductions: List[Dict] = field(default_factory=list)
    corrections: List[Dict] = field(default_factory=list)
    method: str = ""            # "ai", "objective", "keyword-matching"
    provider: str = ""          # e.g. "claude(claude-haiku-4-5)" or "offline"
    flags: List[str] = field(default_factory=list)

    def evaluation_json(self) -> str:
        d = asdict(self)
        for k in ("score", "max_marks", "confidence", "feedback", "strengths", "missing_concepts"):
            d.pop(k, None)
        return json.dumps(d)


def round_marks(x: float, max_marks: float) -> float:
    step = MARKS_STEP if MARKS_STEP > 0 else 0.01
    v = round(round(max(0.0, min(max_marks, x)) / step) * step, 2)
    return min(v, max_marks)


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ization", "isation", "ations", "ation", "ments", "ment", "ness", "ities", "ity", "ings", "ing",
                "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\d*[A-Za-z][A-Za-z0-9+#\-]*|\d+(?:\.\d+)?", text or "") if w.lower() not in STOP
            and (len(w) >= 3 or w.isdigit() or re.match(r"\d", w))]


# Common equivalent terms in answers (offline marking only — the AI path understands paraphrase itself)
SYNONYM_GROUPS = [
    {"row", "record", "tuple"}, {"column", "attribute", "field"}, {"table", "relation"},
    {"function", "method", "procedure", "routine"}, {"array", "list"}, {"error", "exception"},
    {"fast", "quick", "efficient"}, {"big", "large"}, {"small", "little"}, {"begin", "start"},
    {"end", "finish", "terminate"}, {"forever", "indefinitely", "infinitely"}, {"prevent", "avoid"},
    {"increase", "raise", "grow"}, {"decrease", "reduce", "lower"}, {"output", "result"},
    {"cpu", "processor"}, {"memory", "ram"}, {"node", "vertex"}, {"edge", "link"},
]
_SYN = {}
for _g in SYNONYM_GROUPS:
    for _w in _g:
        _SYN.setdefault(_w, set()).update(_g)


def _word_present(word: str, stems: set, raw_words: List[str]) -> float:
    s = _stem(word)
    if s in stems:
        return 1.0
    for alt in _SYN.get(word.lower(), ()):
        if _stem(alt) in stems:
            return 0.9
    if len(word) >= 5:   # tolerate OCR / spelling slips: "normalisation", "redundency"
        best = max((difflib.SequenceMatcher(None, s, t).ratio() for t in stems), default=0.0)
        if best >= 0.84:
            return 0.9
    if re.fullmatch(r"\d+[a-z]+", word.lower()):           # "3NF" read as "3N/F", "1NF" as "INF", "2NF" as "2NE"
        compact = word.lower()
        for w in raw_words:
            c2 = re.sub(r"[^a-z0-9]", "", w.lower().replace("i", "1").replace("l", "1").replace("|", "1"))
            if c2 == compact:
                return 0.9
            if len(c2) == len(compact) and c2[0] == compact[0] and sum(a != b for a, b in zip(c2, compact)) <= 1:
                return 0.7
            letters = re.sub(r"^\d+", "", compact)            # "1NF" read as "NF": the digit was lost
            if len(letters) >= 2 and c2 == letters:
                return 0.7
    return 0.0


def _char_ngrams(text: str, n: int = 4) -> Dict[str, int]:
    t = re.sub(r"\s+", " ", (text or "").lower())
    grams: Dict[str, int] = {}
    for i in range(max(0, len(t) - n + 1)):
        g = t[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.;!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 2]


def derive_marking_points(model_answer: str, max_marks: float) -> List[Dict]:
    """Without a marking scheme, each statement of the model answer becomes a marking point (equal weight)."""
    pts = []
    for s in _sentences(model_answer):
        # split "Prevention: lock ordering, timeouts, wait-die" style lists into one point (keeps weight sane)
        if len(_content_words(s)) >= 2:
            pts.append(s)
    if not pts and model_answer.strip():
        pts = [model_answer.strip()]
    pts = pts[:12]
    if not pts:
        return []
    each = max_marks / len(pts)
    return [{"criterion": p, "marks": round(each, 2)} for p in pts]


# ─────────────────────────────────────────────────────────────────────────────
# Offline (no AI key) marking
# ─────────────────────────────────────────────────────────────────────────────

def _point_coverage(point: str, student: str) -> tuple:
    words = list(dict.fromkeys(_content_words(point)))
    raw_student = re.findall(r"\S+", student or "")
    stems = {_stem(w) for w in _content_words(student)}
    if not words:
        return 0.0, [], []
    hits = [(w, _word_present(w, stems, raw_student)) for w in words]
    found = [w for w, h in hits if h > 0]
    missing = [w for w, h in hits if h == 0]
    kw = sum(h for _, h in hits) / len(words)
    pg = _char_ngrams(point)
    sent_sim = max((_cosine(pg, _char_ngrams(s)) for s in _sentences(student) or [student]), default=0.0)
    cov = max(kw, 0.6 * kw + 0.4 * sent_sim)
    # "Prevention: lock ordering, timeouts, wait-die" — naming some of the listed items earns part of the point
    m = re.search(r"(?::|such as|include[s]?|e\.g\.?)\s*(.+)$", point, re.I)
    if m:
        items = [i.strip() for i in re.split(r",|/|\bor\b|\band\b", m.group(1)) if len(i.strip()) > 2]
        if len(items) >= 2:
            hit = sum(1 for i in items if _point_coverage(i, student)[0] >= 0.7)
            if hit:
                cov = max(cov, 0.4 + 0.6 * hit / len(items))
    return cov, found, missing


def _grade_offline(question: Dict, student: str, max_marks: float, criteria: List[Dict]) -> GradeResult:
    breakdown, strengths, missing_c = [], [], []
    total = 0.0
    scale = max_marks / max(1e-9, sum(float(c["marks"]) for c in criteria))
    for c in criteria:
        avail = float(c["marks"]) * scale
        cov, found, miss = _point_coverage(c["criterion"], student)
        if cov >= 0.7:
            frac, status = 1.0, "met"
        elif cov >= 0.4:
            frac, status = 0.5, "partial"
        else:
            frac, status = 0.0, "missing"
        got = avail * frac
        total += got
        reason = {"met": "Key terms of this point are present in the answer.",
                  "partial": f"Only partly covered — not found: {', '.join(miss[:5])}.",
                  "missing": "This point was not found in the answer."}[status]
        breakdown.append({"criterion": c["criterion"], "max_marks": round(avail, 2), "awarded": round(got, 2),
                          "status": status, "evidence": ", ".join(found[:6]), "reason": reason})
        (strengths if status == "met" else missing_c).append(c["criterion"][:160])
    score = round_marks(total, max_marks)
    pct = int(round(100 * score / max_marks)) if max_marks else 0
    feedback = (f"Marked against {len(criteria)} marking point(s): {sum(b['status'] == 'met' for b in breakdown)} "
                f"covered, {sum(b['status'] == 'partial' for b in breakdown)} partly covered, "
                f"{sum(b['status'] == 'missing' for b in breakdown)} missing ({pct}%).")
    return GradeResult(score=score, max_marks=max_marks, confidence=0.45, feedback=feedback,
                       strengths=strengths, missing_concepts=missing_c, rubric_breakdown=breakdown,
                       method="keyword-matching", provider="offline",
                       flags=["The AI model (Ollama) was not available: marks were estimated by matching the answer "
                              "against the marking points (wording, not meaning). Teacher verification REQUIRED "
                              "before approving."])


# ─────────────────────────────────────────────────────────────────────────────
# Objective questions
# ─────────────────────────────────────────────────────────────────────────────

def _grade_objective(qtype: str, question: Dict, student: str, max_marks: float) -> Optional[GradeResult]:
    from backend.evaluator import _extract_mcq_answer, _extract_true_false
    if qtype == "mcq":
        key = (question.get("correct_option") or "").strip().upper()[:1]
        if not key:
            m = re.match(r"^\s*\(?([A-Ea-e])\)?(?:[.)\s]|$)", question.get("model_answer") or "")
            key = m.group(1).upper() if m else ""
        if not key:
            return None
        got = _extract_mcq_answer(student or "")
        ok = got == key
        return GradeResult(score=max_marks if ok else 0.0, max_marks=max_marks, confidence=0.95 if got else 0.5,
                           feedback=f"Selected option {got or '—'}; correct option {key}.",
                           strengths=["Correct option chosen."] if ok else [],
                           missing_concepts=[] if ok else [f"Correct option is {key}."],
                           rubric_breakdown=[{"criterion": f"Correct option ({key})", "max_marks": max_marks,
                                              "awarded": max_marks if ok else 0.0,
                                              "status": "met" if ok else ("missing" if not got else "incorrect"),
                                              "evidence": got or "", "reason": "Option compared with the key."}],
                           corrections=[] if ok or not got else [{"student_wrote": got, "correct_version": key,
                                                                  "explanation": "Wrong option selected."}],
                           method="objective", provider="answer-key",
                           flags=[] if got else ["No option letter could be read from the answer."])
    if qtype == "true_false":
        key = (question.get("correct_option") or question.get("model_answer") or "").strip().upper()
        key = "TRUE" if key.startswith("T") else ("FALSE" if key.startswith("F") else "")
        if not key:
            return None
        got = _extract_true_false(student or "")
        ok = got == key
        return GradeResult(score=max_marks if ok else 0.0, max_marks=max_marks, confidence=0.95 if got else 0.5,
                           feedback=f"Answered {got or '—'}; correct answer {key}.",
                           rubric_breakdown=[{"criterion": f"Correct answer ({key})", "max_marks": max_marks,
                                              "awarded": max_marks if ok else 0.0,
                                              "status": "met" if ok else "incorrect", "evidence": got or "",
                                              "reason": "Compared with the key."}],
                           method="objective", provider="answer-key",
                           flags=[] if got else ["No True/False answer could be read."])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AI marking
# ─────────────────────────────────────────────────────────────────────────────

GRADER_SYSTEM = ("You are a strict but fair university examiner. You mark one answer against the official model "
                 "answer and marking scheme, awarding marks only for what the student actually wrote, and you "
                 "justify every mark. Output JSON only.")


def build_grading_prompt(question: Dict, student: str, max_marks: float, criteria: List[Dict],
                         scheme_given: bool) -> str:
    crit = "\n".join(f"  {i + 1}. {c['criterion']}  [{float(c['marks']):g} marks]" for i, c in enumerate(criteria))
    scheme = ("Official marking scheme:\n" + crit) if scheme_given else (
        "No separate marking scheme was supplied. Use these key points taken from the model answer "
        "(you may merge or re-word them, but keep the total):\n" + crit)
    return f"""QUESTION ({max_marks:g} marks, type: {question.get('question_type') or 'open_ended'}):
{question.get('question_text') or '(question text not available)'}

MODEL ANSWER:
{question.get('model_answer') or '(none — rely on the marking scheme)'}

{scheme}

STUDENT ANSWER (transcribed from handwriting; may contain transcription/spelling slips — do not penalise
obvious spelling or transcription errors; [illegible] marks words that could not be read — do not guess them):
\"\"\"{student}\"\"\"

Mark it criterion by criterion. Award partial marks where a point is partly correct. Give 0 for a point that
is missing. If the student states something factually wrong, mark that point "incorrect" and add a correction.
Do not reward length or padding; do reward correct points expressed in different words.

"evidence" must be words COPIED EXACTLY from the STUDENT ANSWER above (not from the model answer, not a description).
Use "" when the student did not write that point.

Return ONLY JSON:
{{"rubric_breakdown": [{{"criterion": "...", "max_marks": 2, "awarded": 1.5, "status": "met|partial|missing|incorrect",
   "evidence": "<exact words from the student answer>", "reason": "why these marks"}}],
 "corrections": [{{"student_wrote": "...", "correct_version": "...", "explanation": "..."}}],
 "strengths": ["..."], "missing_concepts": ["..."],
 "feedback": "2-4 sentences addressed to the student: what earned marks, what lost marks, how to improve",
 "confidence": 0.0-1.0}}
The sum of "awarded" must be the total marks (0 to {max_marks:g})."""


def _parse_json(raw: str) -> Dict:
    raw = re.sub(r"```(?:json)?|```", "", raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


def _num(v, default: float = 0.0) -> float:
    """Tolerant number parsing — small local models often answer '7', '7.5', '7/10' or '7 marks'."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v if v is not None else ""))
    return float(m.group(0)) if m else default


def _evidence_in_answer(evidence: str, student: str) -> bool:
    """True when the quoted evidence really occurs (loosely) in the student's text. Guards against a model
    'quoting' words the student never wrote — marks without evidence are not awarded."""
    ev = re.findall(r"[a-z0-9]{3,}", (evidence or "").lower())
    if not ev:
        return False
    st = set(re.findall(r"[a-z0-9]{3,}", (student or "").lower()))
    return sum(1 for w in ev if w in st) / len(ev) >= 0.6


_PLACEHOLDER = re.compile(r"\bquote\b|student'?s? answer|exact words|copied|from the (student|answer)|^\W*$|^\s*<.*>\s*$|"
                          r"^\s*(n/?a|none|null|-+|\.{2,}|…)\s*$", re.I)


def _is_placeholder(evidence: str) -> bool:
    """The model pasted the template / a description instead of quoting the student."""
    return bool(_PLACEHOLDER.search((evidence or "").strip()))


def _locate_evidence(point: str, student: str):
    """(coverage, best matching sentence) when the student's own words cover this marking point, else None."""
    if not point.strip() or not student.strip():
        return None
    cov, _found, _miss = _point_coverage(point, student)
    if cov < 0.55:                        # sharing a couple of words is not enough
        return None
    pg = _char_ngrams(point)
    sents = _sentences(student) or [student]
    best = max(sents, key=lambda x: _cosine(pg, _char_ngrams(x)))
    return cov, best[:300]


def _words(t: str) -> set:
    return {_stem(w) for w in _content_words(t)}


def _consistent(r: Dict, avail: float) -> Dict:
    """The verdict decides the marks: met = full, missing/incorrect = 0, partial = strictly between."""
    r = dict(r)
    st = (r.get("status") or "").lower()
    frac = (r["awarded"] / r["max_marks"]) if r.get("max_marks") else (1.0 if r.get("awarded") else 0.0)
    if st == "met":
        frac = 1.0
    elif st in ("missing", "incorrect"):
        frac = 0.0
    elif st == "partial":
        frac = frac if 0.0 < frac < 1.0 else 0.5
    r.update({"max_marks": round(avail, 2), "awarded": round(avail * frac, 2)})
    return r


_NEG = re.compile(r"\b(not|no|never|cannot|can't|isn't|doesn't|don't|without)\b", re.I)


def _student_sentences(student: str) -> List[str]:
    flat = " ".join((student or "").split())             # handwriting lines break mid-sentence
    return [x.strip() for x in re.split(r"(?<=[.;!?])\s+", flat) if len(x.strip()) > 2] or [flat]


def _near_verbatim(point: str, student: str):
    """The best student sentence when it states this marking point almost word for word (≥85% of the point's key
    words, very similar wording, same negations) — else None. Used to correct a small model that marks such a
    point 'missing'."""
    words = list(dict.fromkeys(_content_words(re.sub(r"^\s*[^:]{1,25}:\s*", "", point) or point)))
    if len(words) < 3:
        return None
    raw = re.findall(r"\S+", student or "")
    stems = {_stem(w) for w in _content_words(student)}
    kw = sum(_word_present(w, stems, raw) for w in words) / len(words)
    if kw < 0.85:
        return None
    pg = _char_ngrams(point)
    best = max(_student_sentences(student), key=lambda x: _cosine(pg, _char_ngrams(x)))
    if _cosine(pg, _char_ngrams(best)) < 0.75:
        return None
    if len(_NEG.findall(best)) != len(_NEG.findall(point)):
        return None
    return best[:300]


def _dedupe_points(rb: List[Dict]) -> List[Dict]:
    """A small model sometimes lists the same point twice (once 'partial', once 'missing'): keep the better one."""
    out: List[Dict] = []
    for r in rb:
        rw = _words(r["criterion"])
        dup = None
        for i, o in enumerate(out):
            ow = _words(o["criterion"])
            if rw and ow and len(rw & ow) / len(rw | ow) >= 0.8:
                dup = i
                break
        if dup is None:
            out.append(r)
        else:
            o = out[dup]
            fr = (r["awarded"] / r["max_marks"]) if r.get("max_marks") else 0
            fo = (o["awarded"] / o["max_marks"]) if o.get("max_marks") else 0
            if fr > fo:
                out[dup] = r
    return out


def _anchor_to_scheme(rb: List[Dict], criteria: List[Dict], max_marks: float, scheme_given: bool) -> List[Dict]:
    """
    Small local models are inconsistent: "met" with half marks, invented 0-mark points, their own point values.
      * always: the verdict decides the marks of each point, and points worth 0 marks are dropped;
      * with an official marking scheme: every scheme point keeps the scheme's marks, points the model invented
        are dropped, and a scheme point the model skipped counts as missing.
    Without a scheme the model may merge / re-word the key points of the model answer (the prompt allows it).
    """
    if not rb:
        return rb
    if not (scheme_given and criteria):
        # Without a scheme the model may merge / re-word points. Points it gave no marks value get the marks that
        # are left (small models often leave "max_marks" out); a point worth nothing that earned nothing is dropped.
        given = sum(r.get("max_marks", 0) for r in rb)
        blank = [r for r in rb if not r.get("max_marks")]
        left = max_marks - given
        if blank and left > 0.01:            # when the given values already cover the question, blanks are extras
            share = left / len(blank)
            for r in blank:
                r["max_marks"] = round(share, 2)
                if r.get("awarded", 0) > r["max_marks"]:
                    r["awarded"] = r["max_marks"]
        return [_consistent(r, r["max_marks"]) for r in rb if r.get("max_marks", 0) > 0]
    scale = max_marks / max(1e-9, sum(float(c["marks"]) for c in criteria))
    used: set = set()
    out: List[Dict] = []
    for i, c in enumerate(criteria):
        cw = _words(c["criterion"])
        best, best_sc = None, 0.0
        for j, r in enumerate(rb):
            if j in used:
                continue
            rw = _words(r["criterion"])
            sc = len(cw & rw) / max(1, len(cw | rw))
            if sc > best_sc:
                best, best_sc = j, sc
        if (best is None or best_sc < 0.3) and len(rb) == len(criteria) and i not in used:
            best = i                                  # same number of points: fall back to the order
        elif best_sc < 0.3:
            best = None
        avail = float(c["marks"]) * scale
        if best is None:
            out.append({"criterion": c["criterion"][:300], "max_marks": round(avail, 2), "awarded": 0.0,
                        "status": "missing", "evidence": "",
                        "reason": "The AI did not assess this point; treated as missing — check it."})
            continue
        used.add(best)
        r = _consistent(rb[best], avail)
        r["criterion"] = c["criterion"][:300]
        out.append(r)
    return out


def _grade_ai(client, question: Dict, student: str, max_marks: float, criteria: List[Dict],
              scheme_given: bool) -> Optional[GradeResult]:
    # JSON mode + low temperature: marking must be reproducible. Raises LLMError when Ollama is down/unparseable
    # (the caller then flags the answer instead of inventing marks).
    data = client.generate_json(build_grading_prompt(question, student, max_marks, criteria, scheme_given),
                                max_tokens=2500, system=GRADER_SYSTEM, temperature=0.0)
    unsupported: List[str] = []
    weak_evidence: List[str] = []
    rb = []
    for item in data.get("rubric_breakdown") or []:
        if not isinstance(item, dict):
            continue
        mx = _num(item.get("max_marks"))
        aw = _num(item.get("awarded"))
        mx = max(mx, 0.0)
        aw = max(0.0, min(aw, mx)) if mx else max(0.0, aw)
        if aw > 0 and not _evidence_in_answer(str(item.get("evidence") or ""), student):
            # Small models often paste the template ("short quote from the student answer") instead of a quote.
            # Check the point against the student's own words before taking the marks away.
            # A specific quote the student never wrote is a hallucination → no marks. Only a template / empty
            # "evidence" (the model did not quote at all) is checked against the student's own words.
            found = (_locate_evidence(str(item.get("criterion") or ""), student)
                     if _is_placeholder(str(item.get("evidence") or "")) else None)
            if found:
                cov, sentence = found
                item["evidence"] = sentence
                item["reason"] = (str(item.get("reason") or "").strip() + " [Evidence located in the answer by the "
                                  "system — the AI did not quote it.]").strip()
                if cov < 0.75:            # only partly supported by the student's words → at most half
                    aw = min(aw, mx * 0.5) if mx else aw * 0.5
                    item["status"] = "partial"
                    weak_evidence.append(str(item.get("criterion") or "")[:80])
            else:
                unsupported.append(str(item.get("criterion") or "")[:80])
                aw = 0.0                  # no verifiable evidence in the student's answer → no marks
                item["reason"] = "No supporting evidence found in the student's answer for this point."
                item["status"] = "missing"
        st = str(item.get("status") or ("met" if mx and aw >= mx else "partial" if aw > 0 else "missing")).lower()
        rb.append({"criterion": str(item.get("criterion") or "")[:300], "max_marks": round(mx, 2),
                   "awarded": round(aw, 2), "status": st, "evidence": str(item.get("evidence") or "")[:300],
                   "reason": str(item.get("reason") or "")[:400]})
    rb = _dedupe_points(rb)
    upgraded: List[str] = []
    def _single_point(r):                 # the model's point is ONE key point of the key, not several merged
        rw = _words(r["criterion"])
        for c in criteria:
            cw = _words(c["criterion"])
            if rw and cw and len(rw & cw) / len(rw | cw) >= 0.6:
                return not r.get("max_marks") or r["max_marks"] <= float(c["marks"]) * 1.25 + 0.01
        return False

    for r in rb:                          # the student wrote this point almost word for word → it is met
        if r["status"] in ("missing", "partial") and _single_point(r):
            sent = _near_verbatim(r["criterion"], student)
            if sent:
                r.update({"status": "met", "awarded": r["max_marks"], "evidence": sent,
                          "reason": "The answer states this point almost word for word (checked by the system; "
                                    "the AI had marked it " + r["status"] + ")."})
                upgraded.append(r["criterion"][:80])
    rb = _anchor_to_scheme(rb, criteria, max_marks, scheme_given)
    total_avail = sum(r["max_marks"] for r in rb)
    if rb and total_avail > 0 and abs(total_avail - max_marks) > 0.01:     # rescale to the paper's marks
        f = max_marks / total_avail
        for r in rb:
            r["max_marks"] = round(r["max_marks"] * f, 2)
            r["awarded"] = round(r["awarded"] * f, 2)
    score = sum(r["awarded"] for r in rb) if rb else _num(data.get("score"))
    if not rb and score > 0:
        unsupported.append("overall score given without a per-point breakdown")
    score = round_marks(score, max_marks)
    corrections = [{"student_wrote": str(c.get("student_wrote") or "")[:300],
                    "correct_version": str(c.get("correct_version") or "")[:300],
                    "explanation": str(c.get("explanation") or "")[:300]}
                   for c in (data.get("corrections") or []) if isinstance(c, dict)]
    conf = _num(data.get("confidence"), 0.7)
    if conf > 1.0:                        # some models answer 85 instead of 0.85
        conf = conf / 100.0
    extra_flags = []
    if upgraded:
        extra_flags.append("The AI marked these points missing/partial although the answer states them almost word "
                           "for word; the system gave the marks — confirm: " + "; ".join(upgraded[:4]))
    if weak_evidence:
        conf = min(conf, 0.6)
        extra_flags.append("The AI gave no usable quote and the answer only partly matches: "
                           + "; ".join(weak_evidence[:4]) + " — half marks at most, check these points.")
    if unsupported:
        conf = min(conf, 0.5)
        extra_flags.append("AI awarded marks without verifiable evidence in the answer for: "
                           + "; ".join(unsupported[:4]) + " — those marks were withheld / need teacher check.")
    return GradeResult(score=score, max_marks=max_marks, confidence=max(0.0, min(1.0, conf)),
                       feedback=str(data.get("feedback") or "")[:1500],
                       strengths=[str(x)[:200] for x in (data.get("strengths") or [])][:8],
                       missing_concepts=[str(x)[:200] for x in (data.get("missing_concepts") or [])][:8],
                       rubric_breakdown=rb, corrections=corrections, method="ai", flags=extra_flags,
                       provider=getattr(client, "active_provider", "ai"))


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def deductions_from_breakdown(rb: List[Dict]) -> List[Dict]:
    out = []
    for r in rb:
        lost = round(float(r.get("max_marks", 0)) - float(r.get("awarded", 0)), 2)
        if lost > 0.001:
            out.append({"marks": lost, "criterion": r.get("criterion", ""),
                        "reason": r.get("reason") or ("Point missing" if r.get("status") == "missing" else
                                                      "Point only partly correct")})
    return out


def grade_answer(question: Dict, student_text: str, *, ocr_confidence: Optional[float] = None,
                 ocr_engine: str = "", llm_client=None) -> GradeResult:
    """
    question: {"question_text", "question_type", "marks", "model_answer", "correct_option",
               "rubric": [{"criterion", "marks"}], "reference_note"}
    """
    max_marks = float(question.get("marks") or 0) or 10.0
    student = (student_text or "").strip()
    qtype = (question.get("question_type") or "open_ended").lower()
    flags: List[str] = []
    if question.get("reference_note"):
        flags.append(question["reference_note"])

    if not student or re.fullmatch(r"(\[illegible\]\s*)+", student, re.I):
        r = GradeResult(score=0.0, max_marks=max_marks, confidence=0.9 if not student else 0.3,
                        feedback="No answer was detected for this question." if not student else
                        "The answer could not be read (illegible).",
                        rubric_breakdown=[], method="objective", provider="—",
                        flags=flags + ([] if not student else ["Answer illegible — check the original page."]))
        r.deductions = [{"marks": max_marks, "criterion": "Whole question",
                         "reason": "Not attempted / no answer found on the sheet" if not student else "Illegible"}]
        return r

    result = _grade_objective(qtype, question, student, max_marks)
    if result is None:
        rubric = [c for c in (question.get("rubric") or []) if str(c.get("criterion", "")).strip()]
        scheme_given = bool(rubric)
        criteria = rubric or derive_marking_points(question.get("model_answer") or "", max_marks)
        if not criteria:
            raise ValueError("No model answer or marking scheme for this question.")
        if llm_client is None:
            from backend.llm_provider import get_llm_client
            llm_client = get_llm_client()
        if getattr(llm_client, "has_real_provider", False):
            try:
                result = _grade_ai(llm_client, question, student, max_marks, criteria, scheme_given)
            except Exception as e:
                logger.warning("AI grading failed (%s) — falling back to marking-point matching", e)
                result = None
            if result is None:
                flags.append("The AI model could not produce a valid marking; marks were estimated offline.")
        if result is None:
            result = _grade_offline(question, student, max_marks, criteria)

    result.deductions = deductions_from_breakdown(result.rubric_breakdown) if result.rubric_breakdown else result.deductions
    result.flags = flags + result.flags
    if ocr_engine.startswith("local"):
        result.flags.append("Handwriting was read by local OCR — compare the transcription with the original page.")
    if ocr_confidence is not None and ocr_confidence < OCR_VERIFY_BELOW:
        result.flags.append(f"Low handwriting-reading confidence ({ocr_confidence:.0%}).")
    if "[illegible]" in student.lower():
        result.flags.append("Some words were illegible.")
    if result.method == "ai" and result.confidence < AI_VERIFY_BELOW:
        result.flags.append(f"Low AI marking confidence ({result.confidence:.0%}).")
    # overall confidence = marking confidence × reading confidence
    if ocr_confidence is not None:
        result.confidence = round(result.confidence * max(0.3, min(1.0, ocr_confidence / 0.9)), 3)
    return result