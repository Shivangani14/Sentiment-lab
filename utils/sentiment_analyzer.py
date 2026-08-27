"""
utils/sentiment_analyzer.py
---------------------------
VADER wrapper and classification logic.

This module is unchanged in behaviour from Sentiment Lab v1 — `analyze_text`
still returns the same VADER payload — but the analyzer is now created once
at import time and reused, and a small normalisation helper is exposed so
the unified engine can emit one schema across engines.

Public API:
    analyze_text(text) -> dict            # v1 contract, preserved
    classify(compound) -> str
    vader_normalized(text) -> dict        # v2 unified schema
"""

from functools import lru_cache
from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

MODEL_NAME = "VADER"


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return the process-wide VADER analyzer.

    VADER loads a ~7k-term lexicon from disk on construction, so building a
    new instance per request (as a naive implementation would) is wasteful.
    lru_cache makes this a singleton without a module-level global.
    """
    return SentimentIntensityAnalyzer()


def classify(compound: float) -> str:
    """Map a VADER compound score to a human label."""
    if compound >= POSITIVE_THRESHOLD:
        return "POSITIVE"
    if compound <= NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def analyze_text(text: str) -> Dict:
    """Score *text* with VADER.

    Preserved v1 contract — returns:
        {
          "text": str,
          "compound": float,
          "positive": float,
          "neutral": float,
          "negative": float,
          "sentiment": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
        }
    """
    scores = get_analyzer().polarity_scores(text or "")
    compound = round(float(scores["compound"]), 4)
    return {
        "text": text,
        "compound": compound,
        "positive": round(float(scores["pos"]), 4),
        "neutral": round(float(scores["neu"]), 4),
        "negative": round(float(scores["neg"]), 4),
        "sentiment": classify(compound),
    }


def vader_normalized(text: str) -> Dict:
    """Score *text* with VADER and return the unified v2 schema.

    `compound` is a genuine VADER compound score. `confidence` is None
    because VADER is rule-based and does not produce a probability — the
    UI relies on this distinction so a model confidence is never displayed
    as if it were a compound score.
    """
    base = analyze_text(text)
    return {
        "text": text,
        "sentiment": base["sentiment"],
        "compound": base["compound"],
        "polarity": base["compound"],
        "confidence": None,
        "positive": base["positive"],
        "neutral": base["neutral"],
        "negative": base["negative"],
        "model": MODEL_NAME,
        "score_type": "vader_compound",
    }


def warm_up() -> None:
    """Pre-build the VADER lexicon at application startup (cheap, ~10ms)."""
    get_analyzer().polarity_scores("warm up")
