"""
AI Mentor — reply-language policy (pure Python, no dependencies)
=================================================================
Single place that decides which language the Mentor answers in.

Priority (highest first)
  1. An explicit selection (English / Hindi / Hinglish / any other named language) ALWAYS wins.
  2. "Auto" → detected from the student's LATEST message only.
     Old chat history, uploaded documents and retrieved notes never influence the choice.

Languages
  en        English
  hi        Hindi in Devanagari script
  hinglish  Hindi written in Roman (Latin) script, mixed with English terms
  other     any other explicitly selected language (Tamil, Spanish, …) — passed through by name
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z']+")

# Common Roman-script Hindi words. Deliberately excludes words that are also frequent English
# words ("me", "to", "the", "main" is kept out; "hai", "kya" etc. are unambiguous).
HINGLISH_MARKERS = frozenset("""
kya kyu kyun kyon kaise kaisa kaisi kab kahan kaun kitna kitne kitni kidhar
hai hain ho hoon hun tha thi the_ raha rahi rahe hota hoti hote gaya gayi gaye
mujhe mujhko mera meri mere hamara hamari tumhara tumhare aapka aapki aapke apna apni
nahi nahin nhi mat sirf bhi hi toh lekin par magar aur ya
karo karna karein kariye kijiye kar karke kiya kiye karta karti
batao batana bataiye samjhao samjhana samjhaiye dikhao likho likhna padho padhna
chahiye chahta chahti sakta sakti sakte milega milenge mila mili
isme isse isko iska iski iske uska uski uske unka unki
yeh ye woh wo yahan wahan abhi pehle baad phir tak se ko ka ki ke mein
accha achha theek thik bahut bohot zyada kam sab sabhi kuch koi
number marks_ kate kata kaat kaate sawal jawab uttar prashn
""".split())
_STRONG_HINGLISH = frozenset("""kya kyu kyun kaise kaisa hai hain mujhe mera meri mere nahi nahin nhi batao samjhao
chahiye karo kijiye bataiye samjhaiye kate kata sawal jawab isme isse abhi bahut bohot toh""".split())

_ENGLISH_NAMES = ("english",)
_HINDI_NAMES = ("hindi", "हिन्दी", "हिंदी")


@dataclass(frozen=True)
class Lang:
    code: str                 # en | hi | hinglish | other | unknown
    name: str                 # human name used in prompts
    explicit: bool            # True when the user picked it in the selector


def detect_language(text: str) -> str:
    """Classify one message as 'en', 'hi', 'hinglish' or 'unknown' (other scripts / empty)."""
    t = (text or "").strip()
    if not t:
        return "unknown"
    dev = len(_DEVANAGARI.findall(t))
    lat = len(_LATIN_LETTER.findall(t))
    if dev and dev >= lat * 0.5:
        return "hi"
    if lat == 0:
        return "unknown"
    words = [w.lower() for w in _WORD.findall(t)]
    if not words:
        return "unknown"
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    strong = sum(1 for w in words if w in _STRONG_HINGLISH)
    # Hindi-in-Roman: at least 2 markers (or 1 strong marker in a very short message) and a fair ratio.
    if (hits >= 2 and hits / len(words) >= 0.25) or (strong >= 1 and len(words) <= 4) or strong >= 2:
        return "hinglish"
    if dev:                       # mixed Devanagari + Latin, Latin dominates
        return "hinglish" if hits else "en"
    return "en"


def resolve_language(selected: Optional[str], latest_message: str) -> Lang:
    """Apply the priority rules. `selected` is the raw value from the UI selector."""
    sel = (selected or "").strip()
    low = sel.lower()
    if low in ("", "auto", "same as question", "auto-detect", "auto detect"):
        code = detect_language(latest_message)
        if code == "unknown":
            return Lang("unknown", "the same language and script as the student's latest message", False)
        return Lang(code, _NAME[code], False)
    if low.startswith("hinglish") or low in ("hi-latn", "hindi (latin)", "roman hindi"):
        return Lang("hinglish", _NAME["hinglish"], True)
    if any(low.startswith(n) for n in _ENGLISH_NAMES):
        return Lang("en", _NAME["en"], True)
    if any(low.startswith(n) for n in _HINDI_NAMES) or low in ("hi", "hi-in"):
        return Lang("hi", _NAME["hi"], True)
    return Lang("other", sel, True)


_NAME = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish (Hindi in Roman script)"}


def language_instruction(lang: Lang) -> str:
    """The firm instruction appended to every Mentor prompt. Names the priority explicitly so a
    small model does not drift toward the language of notes, documents or earlier turns."""
    ignore = ("The language of the question source, uploaded documents, retrieved notes and earlier chat "
              "messages must NOT change the reply language.")
    if lang.code == "en":
        return ("LANGUAGE RULE (highest priority): Write the ENTIRE reply in English only. Do not use Hindi or "
                "Devanagari script, and do not mix in Hindi words. " + ignore +
                " If a source is in another language, explain it in English.")
    if lang.code == "hi":
        return ("LANGUAGE RULE (highest priority): Write the ENTIRE reply in Hindi using Devanagari script (हिन्दी). "
                "Technical terms, formulas, code and file names may stay in English. " + ignore)
    if lang.code == "hinglish":
        return ("LANGUAGE RULE (highest priority): Write the ENTIRE reply in Hinglish — Hindi written in ROMAN/Latin "
                "letters, mixed naturally with English technical terms (example: 'Is answer mein definition missing "
                "thi, isliye marks kate.'). Never use Devanagari script. " + ignore)
    if lang.code == "other":
        return (f"LANGUAGE RULE (highest priority): Write the ENTIRE reply in {lang.name}, in its native script. "
                "Standard technical terms, formulas, code and file names may stay as they are. " + ignore)
    return ("LANGUAGE RULE: Reply in the same language and script the student used in their latest message "
            "(ignore the language of documents and earlier turns).")


def reply_matches(lang: Lang, reply: str) -> bool:
    """Cheap post-check that the model obeyed the language rule (used to trigger one stricter retry)."""
    r = (reply or "").strip()
    if not r or lang.code in ("other", "unknown"):
        return True
    # ignore code blocks and quoted source text when judging script
    body = re.sub(r"```.*?```", " ", r, flags=re.S)
    dev = len(_DEVANAGARI.findall(body))
    lat = len(_LATIN_LETTER.findall(body))
    if lang.code in ("en", "hinglish"):
        return dev <= max(3, 0.02 * (dev + lat))
    if lang.code == "hi":
        return dev >= max(5, 0.2 * (dev + lat)) if (dev + lat) > 20 else True
    return True


# ── Static (non-LLM) messages, so offline paths also respect the language ──────────

OFFLINE_NOTICE = {
    "en": "The AI model (Ollama) is not reachable right now, so this answer is built only from your own material, "
          "without AI wording. Start Ollama and try again for a full AI answer.",
    "hinglish": "AI model (Ollama) abhi connect nahi hai, isliye yeh jawab sirf aapke apne material se bana hai, "
                "AI ke bina. Poora AI jawab chahiye to Ollama start karke dobara try karein.",
    "hi": "AI मॉडल (Ollama) अभी कनेक्ट नहीं है, इसलिए यह उत्तर केवल आपकी अपनी सामग्री से बना है, AI के बिना। "
          "पूरा AI उत्तर पाने के लिए Ollama चालू करके दोबारा प्रयास करें।",
}


def offline_notice(lang: Lang) -> str:
    if lang.code in OFFLINE_NOTICE:
        return OFFLINE_NOTICE[lang.code]
    if lang.code == "other":
        return OFFLINE_NOTICE["en"] + f" (A reply in {lang.name} needs the AI model.)"
    return OFFLINE_NOTICE["en"]
