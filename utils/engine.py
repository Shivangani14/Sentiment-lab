"""
utils/engine.py
---------------
The single sentiment pipeline shared by all four content sources.

    Text | Dataset | YouTube | Blog
                 |
        clean / normalise text
                 |
          language detection
                 |
        +--------+---------+
        |                  |
     English        Hindi / Hinglish / Other / Unknown
        |                  |
      VADER        multilingual transformer
        |                  |
        +--------+---------+
                 |
      normalised result records
                 |
          calculate_statistics()

Nothing else in the project scores text directly — routes call into here.

Public API:
    analyze_one(text) -> dict
    analyze_many(texts, progress=None) -> list[dict]
    calculate_statistics(results) -> dict
    engine_status() -> dict
"""

import logging
from typing import Callable, Dict, List, Optional

from config import SENTIMENT_MODEL_NAME
from utils import multilingual
from utils.language_detect import ENGLISH, detect_language
from utils.preprocessing import clean_text, is_meaningful
from utils.sentiment_analyzer import vader_normalized

logger = logging.getLogger(__name__)

RESULT_FIELDS = [
    "language",
    "sentiment",
    "compound",
    "polarity",
    "confidence",
    "positive",
    "neutral",
    "negative",
    "model",
]

_SKIPPED = {
    "sentiment": "UNSCORED",
    "compound": None,
    "polarity": None,
    "confidence": None,
    "positive": None,
    "neutral": None,
    "negative": None,
    "model": "none",
    "score_type": "none",
    "note": "No analysable text in this row.",
}


def engine_status() -> Dict:
    """Report which engines are live — surfaced in the UI status chips.

    Deliberately uses `multilingual.status()` rather than `available()`:
    this runs on every page load and must NOT be what triggers the lazy
    model download.
    """
    ml = multilingual.status()
    return {
        "vader": True,
        "multilingual_ready": ml["expected"],
        "multilingual_loaded": ml["loaded"],
        "multilingual_model": SENTIMENT_MODEL_NAME,
        "multilingual_error": ml["error"],
    }


def _route(language: str, method: str = "") -> str:
    """Decide which engine scores a piece of text.

    English -> VADER. Everything else (Hindi, Hinglish, other languages,
    low-confidence/unknown) -> the multilingual model.

    One exception: emoji-only text is routed to VADER, which ships an
    emoji/emoticon lexicon and handles it better than a text transformer.
    """
    if language == ENGLISH or method == "emoji-only":
        return "vader"
    return "multilingual"


def analyze_one(text: str) -> Dict:
    """Detect the language of *text*, route it, and return a unified result."""
    cleaned = clean_text(text)
    target = cleaned if cleaned else (text or "")

    if not is_meaningful(target):
        return {"text": text, "language": "Unknown", **_SKIPPED}

    lang = detect_language(target)
    record = {"text": text, "language": lang["language"]}

    if _route(lang["language"], lang["method"]) == "vader":
        record.update(vader_normalized(target))
        record["text"] = text
        record["language"] = lang["language"]
        record["language_code"] = lang["code"]
        record["language_confidence"] = lang["confidence"]
        return record

    # Non-English -> multilingual model, with an honest VADER fallback.
    if multilingual.available():
        try:
            scored = multilingual.analyze_batch([target])[0]
        except Exception:
            logger.exception("Multilingual scoring failed")
            scored = None
        if scored:
            record.update(scored)
            record["text"] = text
            record["language"] = lang["language"]
            record["language_code"] = lang["code"]
            record["language_confidence"] = lang["confidence"]
            return record

    record.update(vader_normalized(target))
    record["text"] = text
    record["language"] = lang["language"]
    record["language_code"] = lang["code"]
    record["language_confidence"] = lang["confidence"]
    record["model"] = "VADER (fallback)"
    record["fallback"] = True
    record["note"] = (
        "The multilingual model was unavailable, so this non-English text "
        "was scored with VADER. Treat the result as unreliable."
    )
    return record


def analyze_many(
    texts: List[str],
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    """Score a list of texts efficiently.

    English items are scored inline with VADER (microseconds each) while
    non-English items are collected and sent to the transformer in batches,
    so the model runs once over many texts instead of once per text.

    Args:
        texts: raw strings (may contain None / NaN / blanks).
        progress: optional callback(done, total) for progress reporting.

    Returns:
        A list of unified result dicts, in the same order as *texts*.
    """
    total = len(texts)
    results: List[Optional[Dict]] = [None] * total
    pending_idx: List[int] = []
    pending_txt: List[str] = []
    langs: Dict[int, Dict] = {}

    for i, raw in enumerate(texts):
        cleaned = clean_text(raw)
        target = cleaned if cleaned else (raw if isinstance(raw, str) else "")

        if not is_meaningful(target):
            results[i] = {"text": raw, "language": "Unknown", **_SKIPPED}
            continue

        lang = detect_language(target)
        langs[i] = lang

        if _route(lang["language"], lang["method"]) == "vader":
            record = vader_normalized(target)
            record["text"] = raw
            results[i] = record
        else:
            pending_idx.append(i)
            pending_txt.append(target)

        if progress and (i + 1) % 100 == 0:
            progress(i + 1, total)

    if pending_txt:
        scored_batch: List[Dict] = []
        if multilingual.available():
            try:
                scored_batch = multilingual.analyze_batch(pending_txt)
            except Exception:
                logger.exception("Batch multilingual scoring failed")
                scored_batch = []

        if len(scored_batch) == len(pending_txt):
            for idx, scored in zip(pending_idx, scored_batch):
                scored["text"] = texts[idx]
                results[idx] = scored
        else:
            note = (
                "The multilingual model was unavailable, so non-English text "
                "was scored with VADER. Treat these rows as unreliable."
            )
            for idx, target in zip(pending_idx, pending_txt):
                record = vader_normalized(target)
                record["text"] = texts[idx]
                record["model"] = "VADER (fallback)"
                record["fallback"] = True
                record["note"] = note
                results[idx] = record

    # Attach language metadata uniformly.
    for i, record in enumerate(results):
        if record is None:
            results[i] = {"text": texts[i], "language": "Unknown", **_SKIPPED}
            continue
        lang = langs.get(i)
        if lang:
            record["language"] = lang["language"]
            record["language_code"] = lang["code"]
            record["language_confidence"] = lang["confidence"]

    if progress:
        progress(total, total)

    return results  # type: ignore[return-value]


def _pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 1) if whole else 0.0


def calculate_statistics(results: List[Dict]) -> Dict:
    """Aggregate normalised results into dashboard statistics.

    VADER compound scores and transformer confidences are averaged
    *separately* and reported under distinct keys — a model confidence is
    never presented as a compound score.
    """
    scored = [r for r in results if r.get("sentiment") in
              {"POSITIVE", "NEUTRAL", "NEGATIVE"}]
    total = len(results)
    analysed = len(scored)

    counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    for r in scored:
        counts[r["sentiment"]] += 1

    compounds = [r["compound"] for r in scored
                 if isinstance(r.get("compound"), (int, float))]
    confidences = [r["confidence"] for r in scored
                   if isinstance(r.get("confidence"), (int, float))]
    polarities = [r["polarity"] for r in scored
                  if isinstance(r.get("polarity"), (int, float))]

    languages: Dict[str, int] = {}
    models: Dict[str, int] = {}
    for r in results:
        languages[r.get("language") or "Unknown"] = (
            languages.get(r.get("language") or "Unknown", 0) + 1
        )
        models[r.get("model") or "none"] = models.get(r.get("model") or "none", 0) + 1

    def avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "total": total,
        "analyzed": analysed,
        "skipped": total - analysed,
        "positive": counts["POSITIVE"],
        "neutral": counts["NEUTRAL"],
        "negative": counts["NEGATIVE"],
        "positive_pct": _pct(counts["POSITIVE"], analysed),
        "neutral_pct": _pct(counts["NEUTRAL"], analysed),
        "negative_pct": _pct(counts["NEGATIVE"], analysed),
        # Averaged over VADER-scored English rows only.
        "avg_compound": avg(compounds),
        "vader_rows": len(compounds),
        # Averaged over transformer-scored rows only.
        "avg_confidence": avg(confidences),
        "model_rows": len(confidences),
        # Cross-engine comparable polarity (compound OR pos-neg probability).
        "avg_polarity": avg(polarities),
        "languages": dict(sorted(languages.items(), key=lambda kv: -kv[1])),
        "models": dict(sorted(models.items(), key=lambda kv: -kv[1])),
        "dominant": max(counts, key=counts.get) if analysed else "NONE",
    }
