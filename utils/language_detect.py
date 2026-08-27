"""
utils/language_detect.py
------------------------
Lightweight, practical language detection for the sentiment router.

Strategy (cheapest signal first, no model download):

  1. Devanagari script present            -> Hindi (or Hinglish if mixed
                                             with a lot of Latin script)
  2. Romanised Hindi marker words present -> Hinglish
  3. langdetect (pure-Python, ~1MB)       -> ISO code + confidence
  4. Too short / low confidence           -> Unknown

Buckets returned: "English", "Hindi", "Hinglish", "Other", "Unknown".
Only "English" is routed to VADER; everything else goes to the
multilingual model.

Public API:
    detect_language(text) -> dict
"""

import re
from functools import lru_cache
from typing import Dict

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# langdetect is probabilistic; seeding makes results reproducible.
DetectorFactory.seed = 0

ENGLISH = "English"
HINDI = "Hindi"
HINGLISH = "Hinglish"
OTHER = "Other"
UNKNOWN = "Unknown"

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0000FE00-\U0000FE0F\U00002190-\U000021FF\U00002B00-\U00002BFF]"
)

# Romanised Hindi/Urdu markers that are *not* also common English words.
# Deliberately excludes ambiguous tokens ("the", "to", "are", "is", "main"
# is kept because "mai/main" as a pronoun is overwhelmingly Hinglish in
# context with other markers — the ratio threshold protects against
# false positives on single hits.
_HINGLISH_MARKERS = {
    "abe", "accha", "acha", "achha", "agar", "andar", "apna", "apne", "arre",
    "aur", "bacha", "badhiya", "bahut", "bakwas", "banao", "bas", "bekar",
    "bhai", "bhaiya", "bhut", "bilkul", "bohot", "chahiye", "chal", "chalo",
    "dekh", "dekha", "dekhna", "dekho", "didi", "dost", "faltu", "fir",
    "galat", "ghatiya", "haan", "hai", "hain", "hamara", "hamare", "han",
    "hoga", "hona", "hone", "hua", "hui", "isliye", "jaisa", "jhakaas",
    "jyada", "kaam", "kaise", "kaisa", "kamaal", "kar", "karke", "karna",
    "karo", "kitna", "koi", "kuch", "kyu", "kyun", "kyunki", "lekin",
    "log", "magar", "mast", "matlab", "mera", "mere", "mujhe", "nahi",
    "nahin", "pata", "phir", "raha", "rahe", "rahi", "sahi", "samajh",
    "shandar", "tera", "tere", "tha", "thi", "thoda", "toh", "tujhe",
    "tumhara", "tumhe", "wah", "yaar", "yeh", "zyada",
}

_MIN_CHARS_FOR_LANGDETECT = 12
_MIN_CONFIDENCE = 0.75
_HINGLISH_RATIO = 0.18


def _bucket_from_code(code: str) -> str:
    if code == "en":
        return ENGLISH
    if code in {"hi", "mr", "ne", "bn"}:
        # Devanagari-family / Indic — the multilingual model handles these.
        return HINDI if code == "hi" else OTHER
    return OTHER


def _detect_uncached(text: str) -> Dict:
    """Classify the language of *text*.

    Returns:
        {
          "language": "English" | "Hindi" | "Hinglish" | "Other" | "Unknown",
          "code": ISO-639-1 code or "und",
          "confidence": float 0-1,
          "method": which signal decided it,
        }
    """
    if not text or not str(text).strip():
        return {
            "language": UNKNOWN,
            "code": "und",
            "confidence": 0.0,
            "method": "empty",
        }

    text = str(text)

    # 0. No letters at all, but emoji present ("😂😂🔥"). VADER carries an
    #    emoji/emoticon lexicon and is the best available handler for this,
    #    so it is routed there rather than to the transformer.
    if not _LETTER_RE.search(text) and _EMOJI_RE.search(text):
        return {
            "language": UNKNOWN,
            "code": "zxx",
            "confidence": 0.5,
            "method": "emoji-only",
        }

    devanagari = len(_DEVANAGARI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))

    # 1. Native Devanagari script.
    if devanagari:
        # A lot of Latin alongside Devanagari => code-switched.
        if latin > devanagari:
            return {
                "language": HINGLISH,
                "code": "hi",
                "confidence": 0.9,
                "method": "script-mixed",
            }
        return {
            "language": HINDI,
            "code": "hi",
            "confidence": 0.99,
            "method": "script-devanagari",
        }

    # 2. Romanised Hindi markers.
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if words:
        hits = sum(1 for w in words if w in _HINGLISH_MARKERS)
        ratio = hits / len(words)
        if hits >= 2 or (hits >= 1 and ratio >= _HINGLISH_RATIO):
            return {
                "language": HINGLISH,
                "code": "hi-Latn",
                "confidence": round(min(0.55 + ratio, 0.95), 3),
                "method": "romanised-markers",
            }

    # 3. Statistical detection — only worth it on enough characters.
    letters = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    if len(letters) < _MIN_CHARS_FOR_LANGDETECT:
        # Short strings defeat statistical detection. A short, plain-ASCII
        # string with no Hinglish markers is overwhelmingly likely to be
        # English ("ok", "This is okay.", "great!"), so send it to VADER
        # instead of spinning up the transformer for every short comment.
        if text.isascii():
            return {
                "language": ENGLISH,
                "code": "en",
                "confidence": 0.5,
                "method": "short-ascii",
            }
        return {
            "language": UNKNOWN,
            "code": "und",
            "confidence": 0.0,
            "method": "too-short",
        }

    try:
        ranked = detect_langs(text)
    except LangDetectException:
        return {
            "language": UNKNOWN,
            "code": "und",
            "confidence": 0.0,
            "method": "detector-failed",
        }

    if not ranked:
        return {
            "language": UNKNOWN,
            "code": "und",
            "confidence": 0.0,
            "method": "no-candidates",
        }

    best = ranked[0]
    confidence = round(float(best.prob), 3)

    # English is the top candidate: accept it even at moderate confidence.
    # Low confidence on Latin-script text usually means English competing
    # with other Latin languages, not genuine uncertainty about the script.
    if best.lang == "en":
        return {
            "language": ENGLISH,
            "code": "en",
            "confidence": confidence,
            "method": "langdetect",
        }

    # ASCII-English bias. langdetect is trained on long documents and is
    # unreliable on short ones -- "Terrible experience" is confidently
    # labelled French. When the text is plain ASCII (so definitely not
    # Devanagari) and carries no Hinglish markers, prefer English if English
    # is a plausible candidate or the text is short. This keeps short English
    # comments on VADER instead of the transformer. Trade-off: short,
    # accent-free French/Spanish/etc. is treated as English -- see the
    # Limitations section of the README.
    if best.lang != "en" and text.isascii():
        english_prob = next(
            (float(c.prob) for c in ranked if c.lang == "en"), 0.0
        )
        if english_prob >= 0.25 or len(letters) < 30:
            return {
                "language": ENGLISH,
                "code": "en",
                "confidence": round(max(english_prob, 0.5), 3),
                "method": "ascii-english-bias",
            }

    if confidence < _MIN_CONFIDENCE:
        # 4. Low confidence -> treat as uncertain, use multilingual model.
        return {
            "language": UNKNOWN,
            "code": best.lang,
            "confidence": confidence,
            "method": "low-confidence",
        }

    return {
        "language": _bucket_from_code(best.lang),
        "code": best.lang,
        "confidence": confidence,
        "method": "langdetect",
    }


# Real-world comment datasets repeat themselves heavily ("Nice video", "First",
# emoji-only replies), and detection is the most expensive per-row step in a
# large batch. A bounded cache over short strings removes that duplicate work
# without holding whole documents in memory.
_CACHE_MAX_TEXT = 400


@lru_cache(maxsize=8192)
def _detect_cached(text: str) -> Dict:
    return _detect_uncached(text)


def detect_language(text: str) -> Dict:
    """Detect the language of *text* and return a routing decision.

    Returns {"language", "code", "confidence", "method"}. `language` is one of
    English / Hindi / Hinglish / Other / Unknown and is what `utils.engine`
    routes on. Short strings are cached; long ones are not, to bound memory.

    The returned dict is a fresh copy, so callers may mutate it safely even
    when the result came from the cache.
    """
    if not isinstance(text, str) or len(text) > _CACHE_MAX_TEXT:
        return _detect_uncached(text)
    return dict(_detect_cached(text))
