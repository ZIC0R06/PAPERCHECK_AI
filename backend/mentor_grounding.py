"""
AI Mentor — Grounded document tools
===================================
Deterministic, file-grounded helpers used by the AI Mentor:

  • lexical retrieval of the most relevant passages inside a student's own files
  • extractive summary, key terms
  • MCQs / short-answer / important questions built *only* from sentences in the file

They power the offline path (no LLM key configured) and supply the evidence
passages that are sent to the LLM in the online path. Nothing here invents
content: every question and answer is traceable to a sentence + page of the file.
"""

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just let me more most my myself
no nor not now of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours yourself
yourselves within without upon using used use may might must shall via etc e.g i.e eg ie one two three
first second third however therefore thus hence also another many much several various example examples
called known following given make makes made like way ways page chapter figure table section unit week
means mean runs run provides provide include includes including defines define offers offer creates create
serves serve allows allow enables enable based according lists list refers different important main
""".split())

# Hinglish / chat filler words that should never become a "topic"
QUERY_FILLER = STOPWORDS | set("""
mujhe mera meri mere hai hain ka ki ke ko se me mein kya kyu kyun kaise batao samjha samjhao samjhado
karo kar do de dena please plz explain tell give show simple language bhai yaar important topics topic
file document notes pdf summary summarize question questions about
""".split())

Passage = Dict  # {"doc_id", "title", "page", "content"}


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower()) if w not in STOPWORDS]


def unwrap(text: str) -> str:
    """Re-join lines that a PDF wrapped mid-sentence."""
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"-\n(?=[a-z])", "", text)                      # hyphenated wrap
    text = re.sub(r"(?<=[^\n.!?:;])\n(?=[a-z0-9(])", " ", text)    # next line continues lowercase
    text = re.sub(r"(?<=[,;])\n", " ", text)                       # line ends with a comma
    return text


def split_sentences(text: str) -> List[str]:
    text = unwrap(text)
    raw = re.split(r"(?<=[.!?])\s+|\n{1,}|•|•", text)
    out = []
    for s in raw:
        s = s.strip(" -*#\t\r\n")
        if 25 <= len(s) <= 400 and len(tokenize(s)) >= 3:
            out.append(s)
    return out


def key_terms(text: str, n: int = 12) -> List[Tuple[str, int]]:
    """Frequent content words + frequent capitalised/bigram phrases."""
    words = tokenize(text)
    uni = Counter(w for w in words if len(w) >= 4)
    bi = Counter()
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
    for a, b in zip(toks, toks[1:]):
        la, lb = a.lower(), b.lower()
        if la in STOPWORDS or lb in STOPWORDS or len(la) < 3 or len(lb) < 3:
            continue
        bi[f"{la} {lb}"] += 1
    scored: Dict[str, float] = {}
    for w, c in uni.items():
        if c >= 2:
            scored[w] = c
    for p, c in bi.items():
        if c >= 2:
            scored[p] = c * 1.6
            for part in p.split():  # don't double count the parts
                if part in scored:
                    scored[part] = max(0, scored[part] - c)
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    phrases = {t for t, sc in ranked if " " in t and sc > 0}
    covered = {w for ph in phrases for w in ph.split()}
    out, seen = [], set()
    for term, sc in ranked:
        if sc <= 0 or term in seen or (" " not in term and term in covered):
            continue
        seen.add(term)
        out.append((term, int(round(sc))))
        if len(out) >= n:
            break
    return out


def retrieve(passages: List[Passage], query: str, top_k: int = 5) -> List[Tuple[float, Passage]]:
    """Simple BM25-flavoured lexical ranking restricted to the given passages."""
    q_terms = [t for t in tokenize(query) if t not in QUERY_FILLER]
    if not q_terms or not passages:
        return []
    n = len(passages)
    df = Counter()
    toks = []
    for p in passages:
        t = tokenize(p["content"])
        toks.append(t)
        df.update(set(t))
    import math
    avg = sum(len(t) for t in toks) / max(1, n)
    scored = []
    for p, t in zip(passages, toks):
        tf = Counter(t)
        s = 0.0
        for q in set(q_terms):
            if tf[q] == 0:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * (tf[q] * 2.2) / (tf[q] + 1.2 * (0.25 + 0.75 * len(t) / max(1, avg)))
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def coverage(query: str, passages: List[Passage]) -> float:
    """Fraction of meaningful query terms that appear anywhere in the passages."""
    q_terms = {t for t in tokenize(query) if t not in QUERY_FILLER}
    if not q_terms:
        return 1.0
    blob = set(tokenize(" ".join(p["content"] for p in passages)))
    return len(q_terms & blob) / len(q_terms)


def _sentences_with_pages(passages: List[Passage]) -> List[Tuple[str, Passage]]:
    out, seen = [], set()
    for p in passages:
        for s in split_sentences(p["content"]):
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append((s, p))
    return out


def _cite(p: Passage) -> str:
    return f"{p['title']}, p.{p['page']}"


def extractive_summary(passages: List[Passage], max_points: int = 7) -> Tuple[List[Tuple[str, Passage]], List[str]]:
    text = "\n".join(p["content"] for p in passages)
    freq = Counter(tokenize(text))
    sents = _sentences_with_pages(passages)
    if not sents:
        return [], []
    scored = []
    for idx, (s, p) in enumerate(sents):
        t = tokenize(s)
        if not t:
            continue
        sc = sum(freq[w] for w in t) / (len(t) ** 0.7)
        sc *= 1.15 if idx < 3 else 1.0  # opening lines often define the topic
        scored.append((sc, idx, s, p))
    top = sorted(scored, key=lambda x: -x[0])[:max_points]
    top.sort(key=lambda x: x[1])
    return [(s, p) for _, _, s, p in top], [t for t, _ in key_terms(text, 8)]


def make_mcqs(passages: List[Passage], n: int = 5, seed: int = 7) -> List[Dict]:
    """Cloze-style MCQs: a key term is blanked out of a real sentence from the file."""
    text = "\n".join(p["content"] for p in passages)
    terms = [t for t, _ in key_terms(text, 30)]
    if len(terms) < 4:
        return []
    rng = random.Random(seed)
    sents = _sentences_with_pages(passages)
    mcqs, used_terms, used_sents = [], set(), set()
    for term in terms:
        if len(mcqs) >= n:
            break
        if term in used_terms:
            continue
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for s, p in sents:
            if s in used_sents or not pat.search(s):
                continue
            stem = pat.sub("_____", s, count=1)
            if len(tokenize(stem)) < 4:
                continue
            distractors = [t for t in terms if t != term and not re.search(r"\b" + re.escape(t) + r"\b", s, re.I)
                           and t not in term and term not in t]
            if len(distractors) < 3:
                continue
            opts = [term] + rng.sample(distractors[:10], 3)
            rng.shuffle(opts)
            mcqs.append({
                "question": f"Fill in the blank: {stem}",
                "options": opts,
                "answer": term,
                "answer_letter": "ABCD"[opts.index(term)],
                "explanation": s,
                "source": _cite(p),
            })
            used_terms.add(term)
            used_sents.add(s)
            break
    return mcqs


def _defining_sentence(term: str, sents: List[Tuple[str, Passage]]) -> Optional[Tuple[str, Passage]]:
    pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    best = None
    for s, p in sents:
        if not pat.search(s):
            continue
        score = 0
        if re.search(r"\b(is|are|refers to|means|defined as|is called|consists of)\b", s, re.I):
            score += 2
        if s.lower().startswith(term.lower()):
            score += 1
        if best is None or score > best[0]:
            best = (score, s, p)
    return (best[1], best[2]) if best else None


def make_short_questions(passages: List[Passage], n: int = 6) -> List[Dict]:
    text = "\n".join(p["content"] for p in passages)
    sents = _sentences_with_pages(passages)
    out = []
    for term, _ in key_terms(text, 20):
        d = _defining_sentence(term, sents)
        if not d:
            continue
        s, p = d
        verb = "What is" if re.search(r"\b(is|refers to|means|defined as)\b", s, re.I) else "Explain"
        out.append({"question": f"{verb} {term}?" if verb == "What is" else f"Explain {term} briefly.",
                    "answer": s, "source": _cite(p)})
        if len(out) >= n:
            break
    return out


def make_important_questions(passages: List[Passage], n: int = 6) -> List[Dict]:
    text = "\n".join(p["content"] for p in passages)
    sents = _sentences_with_pages(passages)
    out = []
    for term, freq in key_terms(text, 20):
        d = _defining_sentence(term, sents)
        if not d:
            continue
        s, p = d
        out.append({"question": f"Explain {term} in detail with a suitable example.",
                    "weight": freq, "hint": s, "source": _cite(p)})
        if len(out) >= n:
            break
    return out
