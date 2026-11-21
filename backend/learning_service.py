"""
Mentor AI — Grounded learning service
=====================================
Quiz generation, chapter explanations and study roadmaps built ONLY from:
  • the student's own uploaded study documents (StudyDocument / DocumentChunk)
  • shared department material (StudyDocument.user_id IS NULL) and CourseMaterial
  • the student's real published results and quiz history (for roadmaps)

When an LLM key is configured the LLM writes the content from retrieved passages;
otherwise deterministic, sentence-grounded builders from mentor_grounding are used.
Nothing is invented: with no material the caller gets an explicit error.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend import mentor_grounding as G
from backend.database import CourseMaterial, DocumentChunk, StudyDocument, User, WeakTopic
from backend.llm_provider import get_llm_client

MAX_CONTEXT = 12000


class NoMaterialError(ValueError):
    """Raised when there is no study material to ground the request in."""


def llm_available() -> bool:
    try:
        return get_llm_client().is_available()
    except Exception:
        return False


def _llm(system: str, user: str, max_tokens: int = 1800) -> Optional[str]:
    if not llm_available():
        return None
    try:
        out = get_llm_client().complete(system, user, max_tokens=max_tokens)
    except Exception:
        return None
    return out.strip() if out and out.strip() else None


# ── material ──────────────────────────────────────────────────────────────────

def list_subjects(db: Session, user: User) -> List[Dict]:
    docs = db.query(StudyDocument).filter(
        (StudyDocument.user_id == user.id) | (StudyDocument.user_id == None)  # noqa: E711
    ).all()
    out: Dict[str, Dict] = {}
    for d in docs:
        s = out.setdefault(d.subject or "General", {"subject": d.subject or "General", "chapters": set(), "documents": []})
        if d.chapter:
            s["chapters"].add(d.chapter)
        s["documents"].append({"id": d.id, "title": d.title, "chapter": d.chapter, "mine": d.user_id == user.id})
    return [{"subject": v["subject"], "chapters": sorted(v["chapters"]), "documents": v["documents"]}
            for v in sorted(out.values(), key=lambda x: x["subject"].lower())]


def gather_passages(db: Session, user: User, subject: Optional[str] = None, chapter: Optional[str] = None,
                    document_id: Optional[int] = None, top_k: int = 30) -> List[Dict]:
    """Passages the user may see, narrowed by document / subject and ranked by chapter relevance."""
    q = db.query(DocumentChunk).join(StudyDocument).filter(
        (StudyDocument.user_id == user.id) | (StudyDocument.user_id == None)  # noqa: E711
    )
    if document_id:
        q = q.filter(StudyDocument.id == document_id)
    elif subject and subject not in ("All Subjects", "All"):
        q = q.filter(StudyDocument.subject == subject)
    passages = [{"title": c.document.title if c.document else "Document", "page": c.page_number or 1,
                 "chapter": c.chapter or "", "content": c.content, "doc_id": c.document_id}
                for c in q.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index).limit(4000).all()]
    if not passages and not document_id:
        for m in db.query(CourseMaterial).all():
            if subject and subject not in ("All Subjects", "All") and subject.lower() not in (m.topic or "").lower() \
                    and subject.lower() not in (m.course_code or "").lower():
                continue
            passages.append({"title": f"Course notes: {m.topic}", "page": 1, "chapter": m.topic,
                             "content": m.content_text, "doc_id": None})
    if not passages:
        raise NoMaterialError(
            "No study material found for this selection. Upload your notes or book in "
            "“My Books & Study Documents” (or attach it in the AI Mentor) and try again.")
    if chapter and chapter.strip():
        same = [p for p in passages if chapter.strip().lower() in (p["chapter"] or "").lower()]
        if same:
            return same[:top_k * 2]
        ranked = [p for _, p in G.retrieve(passages, chapter, top_k=top_k)]
        if ranked:
            return ranked
    return passages[:top_k * 2]


def _context(passages: List[Dict]) -> str:
    out, size = [], 0
    for p in passages:
        if size + len(p["content"]) > MAX_CONTEXT:
            break
        out.append(f"[{p['title']} — page {p['page']}]\n{p['content']}")
        size += len(p["content"])
    return "\n\n".join(out)


def _sources(passages: List[Dict]) -> List[str]:
    seen, res = set(), []
    for p in passages:
        k = f"{p['title']} (p.{p['page']})"
        if k not in seen:
            seen.add(k)
            res.append(k)
    return res[:8]


def _parse_json_array(text: str) -> Optional[list]:
    t = re.sub(r"```(?:json)?|```", "", text or "").strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
        return data if isinstance(data, list) else None
    except Exception:
        return None


# ── quiz ──────────────────────────────────────────────────────────────────────

def generate_mcqs(passages: List[Dict], n: int, difficulty: str, chapter: str) -> Tuple[List[Dict], str]:
    """Returns (questions with answers, mode). Each question: question, options, correct_answer, explanation, topic, source."""
    ctx = _context(passages)
    system = ("You write university multiple-choice questions STRICTLY from the material given. "
              "Every question, option and explanation must be supported by the material; never use outside facts. "
              "Return ONLY a JSON array.")
    user = (f"Create {n} {difficulty.lower()} MCQs about '{chapter or 'this material'}'. JSON array of objects with keys: "
            '"question", "options" (exactly 4 strings), "correct_answer" (one of the options, verbatim), '
            '"explanation" (1-2 sentences citing the page like (p.3)), "topic" (2-5 words).\n\nMATERIAL:\n' + ctx)
    raw = _llm(system, user, 2200)
    qs: List[Dict] = []
    if raw:
        for item in _parse_json_array(raw) or []:
            try:
                opts = [str(o).strip() for o in item["options"]][:4]
                ans = str(item["correct_answer"]).strip()
                if len(opts) == 4 and ans in opts and str(item["question"]).strip():
                    qs.append({"question": str(item["question"]).strip(), "options": opts, "correct_answer": ans,
                               "explanation": str(item.get("explanation", "")).strip(),
                               "topic": str(item.get("topic") or chapter or "General").strip()[:150],
                               "source": ""})
            except Exception:
                continue
        if len(qs) >= max(1, n // 2):
            return qs[:n], "llm"
    # offline / LLM failure → cloze MCQs from real sentences
    built = G.make_mcqs(passages, n=n, seed=len(passages) + n)
    qs = [{"question": q["question"], "options": q["options"], "correct_answer": q["answer"],
           "explanation": f"\"{q['explanation']}\" ({q['source']})", "topic": q["answer"][:150],
           "source": q["source"]} for q in built]
    return qs, "offline"


# ── chapter explanation ──────────────────────────────────────────────────────

def explain(passages: List[Dict], subject: str, chapter: str, style: str) -> Tuple[str, List[str], str]:
    ctx = _context(passages)
    style_l = (style or "").lower()
    lang = ("Write in simple Hinglish (Hindi in Roman script mixed with English terms)." if "hinglish" in style_l
            else "Use very simple English." if "simple" in style_l
            else "Focus on formulas, definitions and likely exam points." if "formula" in style_l or "exam" in style_l
            else "Give a detailed, structured explanation.")
    system = ("You are Mentor AI. Explain ONLY what is in the material; if something is missing say so. "
              "Use Markdown with these sections: Overview, Key Concepts, Important Definitions, Exam Points. "
              "Cite pages like (p.2). " + lang)
    out = _llm(system, f"Subject: {subject}\nChapter/topic: {chapter}\n\nMATERIAL:\n{ctx}", 1800)
    if out:
        return out, _sources(passages), "llm"
    points, terms = G.extractive_summary(passages, max_points=8)
    defs = G.make_short_questions(passages, n=6)
    imp = G.make_important_questions(passages, n=5)
    hing = "hinglish" in style_l
    parts = [f"### 🎓 {chapter or subject}", "",
             "#### 📌 Overview" if not hing else "#### 📌 Overview (Mukhya points)",
             *[f"- {s} *(p.{p['page']})*" for s, p in points]]
    if terms:
        parts += ["", "#### 🔑 Key Concepts", ", ".join(terms)]
    if defs:
        parts += ["", "#### 📖 Important Definitions", *[f"- **{d['question'].replace('What is ', '').rstrip('?')}** — {d['answer']} *({d['source']})*" for d in defs]]
    if imp:
        parts += ["", "#### 📝 Exam Points", *[f"- {q['question']}" for q in imp]]
    parts += ["", "_Offline mode: no LLM API key configured — this explanation is assembled from sentences in your material"
              + (" (Hinglish rewriting needs the LLM)." if hing else ".") + "_"]
    return "\n".join(parts), _sources(passages), "offline"


# ── roadmap ───────────────────────────────────────────────────────────────────

def weak_and_strong(db: Session, user: User, published_weak: List[str]) -> Tuple[List[Dict], List[str]]:
    weak, seen = [], set()
    for t in published_weak:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            weak.append({"topic": t.strip(), "evidence": "missed in a published exam", "accuracy": None})
    strong = []
    for w in db.query(WeakTopic).filter(WeakTopic.user_id == user.id).order_by(WeakTopic.accuracy.asc()).all():
        k = w.topic_name.strip().lower()
        if w.status == "Mastered":
            strong.append(w.topic_name)
        elif k not in seen:
            seen.add(k)
            weak.append({"topic": w.topic_name, "evidence": f"quiz accuracy {w.accuracy:.0f}% over {w.attempts_count} attempt(s)",
                         "accuracy": w.accuracy})
    return weak, strong


def build_roadmap(weak: List[Dict], daily_hours: float, days: int = 7) -> List[Dict]:
    """Deterministic plan: weakest first, learn → practise → test cycles, final mixed review."""
    if not weak:
        return []
    steps = [("Concept study", "Re-read notes; write definitions and one example in your own words"),
             ("Practice", "Solve 8–10 practice MCQs from your material in the Quiz Center"),
             ("Exam writing", "Write one full exam-style answer and compare with your notes")]
    tasks, i = [], 0
    for d in range(days - 1):
        w = weak[(d // len(steps)) % len(weak)] if len(weak) * len(steps) > d else weak[d % len(weak)]
        kind, what = steps[d % len(steps)]
        tasks.append({"day": f"Day {d + 1}", "focus_topic": w["topic"], "allocated_time": f"{daily_hours:g} hrs",
                      "task_type": kind, "details": what,
                      "reason": f"Weak topic — {w['evidence']}.", "completed": False})
        i += 1
    tasks.append({"day": f"Day {days}", "focus_topic": ", ".join(x["topic"] for x in weak[:3]),
                  "allocated_time": f"{daily_hours:g} hrs", "task_type": "Self-test & revision",
                  "details": "Timed quiz covering all focus topics, then revise every mistake",
                  "reason": "Checks improvement before the roadmap updates.", "completed": False})
    return tasks
