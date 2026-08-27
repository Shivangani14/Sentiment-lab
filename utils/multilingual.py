"""
utils/multilingual.py
---------------------
Lazily-loaded multilingual transformer sentiment model.

Design notes
============
* The model is **never** downloaded or loaded at application start-up. It is
  loaded on the first non-English request and then cached for the lifetime
  of the process (`_STATE`), so it is not reloaded per comment or per batch.
* If transformers/torch are not installed, or the download fails (offline,
  no disk space, gated repo), the module degrades gracefully: callers get
  `available() -> False` plus a human-readable reason, and the engine falls
  back to VADER with an explicit `fallback` flag rather than crashing.
* The default model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) is an
  XLM-RoBERTa base fine-tuned on ~198M multilingual tweets across 8
  languages including Hindi, so it handles Devanagari Hindi and multilingual
  social text far better than VADER. It is **not** a dedicated Hinglish
  model — romanised Hinglish is out of its training distribution and results
  there should be treated as indicative, not authoritative.
* Override with the SENTIMENT_MODEL_NAME environment variable. Any
  Hugging Face sequence-classification model with positive/neutral/negative
  (or positive/negative) labels will work.

Public API:
    available() -> bool
    load_error() -> str | None
    model_name() -> str
    analyze_batch(texts) -> list[dict]   # unified v2 schema
"""

import logging
import threading
from typing import Dict, List, Optional

from config import (
    ENABLE_MULTILINGUAL,
    MULTILINGUAL_BATCH_SIZE,
    SENTIMENT_MODEL_NAME,
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE = {
    "attempted": False,
    "pipeline": None,
    "error": None,
    "label_map": {},
}

# Normalises the many label conventions used by HF sentiment checkpoints.
_LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "label_0": "negative",
    "1 star": "negative",
    "2 stars": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "label_1": "neutral",
    "3 stars": "neutral",
    "positive": "positive",
    "pos": "positive",
    "label_2": "positive",
    "4 stars": "positive",
    "5 stars": "positive",
}


def model_name() -> str:
    return SENTIMENT_MODEL_NAME


def load_error() -> Optional[str]:
    return _STATE["error"]


def _canonical(label: str) -> str:
    key = str(label).strip().lower()
    return _LABEL_ALIASES.get(key, key)


def _load() -> None:
    """Load the pipeline exactly once. Safe to call concurrently."""
    if _STATE["attempted"]:
        return
    with _LOCK:
        if _STATE["attempted"]:
            return
        _STATE["attempted"] = True

        if not ENABLE_MULTILINGUAL:
            _STATE["error"] = (
                "Multilingual analysis is disabled (ENABLE_MULTILINGUAL=0). "
                "Non-English text is being scored with VADER, which is "
                "unreliable for Hindi and Hinglish."
            )
            return

        try:
            from transformers import pipeline  # heavy import, done lazily
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("transformers unavailable: %s", exc)
            _STATE["error"] = (
                "The multilingual model requires the optional 'transformers' "
                "and 'torch' packages. Install them with: "
                "pip install -r requirements-multilingual.txt"
            )
            return

        try:
            logger.info("Loading multilingual model %s", SENTIMENT_MODEL_NAME)
            _STATE["pipeline"] = pipeline(
                task="sentiment-analysis",
                model=SENTIMENT_MODEL_NAME,
                tokenizer=SENTIMENT_MODEL_NAME,
                top_k=None,
                truncation=True,
                max_length=512,
                use_fast=False,

            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Could not load %s: %s", SENTIMENT_MODEL_NAME, exc)
            _STATE["error"] = (
                f"Could not load the multilingual model "
                f"'{SENTIMENT_MODEL_NAME}'. Check your internet connection "
                f"on first run, or set SENTIMENT_MODEL_NAME to a model you "
                f"have cached locally."
            )


def available() -> bool:
    """True when the multilingual model is loaded and usable.

    Calling this TRIGGERS the lazy load. Only call it when you are about to
    score non-English text. For UI/status reporting use `status()`, which
    never loads anything.
    """
    _load()
    return _STATE["pipeline"] is not None


def status() -> Dict:
    """Report readiness WITHOUT loading the model.

    /api/status is hit on every page load, so it must never be the thing
    that pulls a 500 MB model off the network. This inspects installed
    packages with importlib (which does not import torch) and only reports
    real load state if a load has already been attempted.

    Returns: {"expected": bool, "loaded": bool, "error": str | None}
      expected -> the model should work when first needed
      loaded   -> it is already in memory right now
    """
    if not ENABLE_MULTILINGUAL:
        return {
            "expected": False,
            "loaded": False,
            "error": (
                "Multilingual analysis is disabled (ENABLE_MULTILINGUAL=0). "
                "Non-English text will be scored with VADER, which is "
                "unreliable for Hindi and Hinglish."
            ),
        }

    if _STATE["attempted"]:
        loaded = _STATE["pipeline"] is not None
        return {
            "expected": loaded,
            "loaded": loaded,
            "error": None if loaded else _STATE["error"],
        }

    import importlib.util

    missing = [
        name
        for name in ("transformers", "torch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return {
            "expected": False,
            "loaded": False,
            "error": (
                "The multilingual model requires the optional "
                + " and ".join(f"'{m}'" for m in missing)
                + " package(s). Install them with: "
                "pip install -r requirements-multilingual.txt"
            ),
        }

    return {"expected": True, "loaded": False, "error": None}


def _to_result(text: str, scored: List[Dict]) -> Dict:
    """Map raw pipeline output onto the unified v2 schema."""
    probs = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for item in scored:
        canon = _canonical(item.get("label", ""))
        if canon in probs:
            probs[canon] += float(item.get("score", 0.0))

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    top = max(probs, key=probs.get)
    # Derived polarity, comparable to VADER's compound in *direction and
    # range* but explicitly NOT a compound score. Never labelled as one.
    polarity = probs["positive"] - probs["negative"]

    return {
        "text": text,
        "sentiment": top.upper(),
        "compound": None,  # the model does not produce a VADER compound
        "polarity": round(polarity, 4),
        "confidence": round(probs[top], 4),
        "positive": round(probs["positive"], 4),
        "neutral": round(probs["neutral"], 4),
        "negative": round(probs["negative"], 4),
        "model": SENTIMENT_MODEL_NAME,
        "score_type": "model_probability",
    }


def analyze_batch(texts: List[str]) -> List[Dict]:
    """Score a list of texts in batches.

    Raises RuntimeError if the model is unavailable — callers should check
    `available()` first and fall back to VADER.
    """
    if not available():
        raise RuntimeError(_STATE["error"] or "Multilingual model unavailable.")

    pipe = _STATE["pipeline"]
    results: List[Dict] = []
    step = max(1, MULTILINGUAL_BATCH_SIZE)

    for start in range(0, len(texts), step):
        chunk = [t if t else " " for t in texts[start : start + step]]
        raw = pipe(chunk)
        # top_k=None yields a list-of-lists; guard against single-dict output.
        for text, scored in zip(chunk, raw):
            if isinstance(scored, dict):
                scored = [scored]
            results.append(_to_result(text, scored))

    return results
