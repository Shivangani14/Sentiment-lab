"""
utils/multilingual.py
---------------------
Remote multilingual sentiment analysis using Hugging Face Inference Providers.

The model is NOT downloaded to the Render server. Instead, requests are sent
to Hugging Face using the HF_TOKEN environment variable.
"""

import logging
import os
import threading
from typing import Dict, List, Optional

from config import (
    ENABLE_MULTILINGUAL,
    SENTIMENT_MODEL_NAME,
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

_STATE = {
    "attempted": False,
    "client": None,
    "error": None,
}

# Normalise the many label conventions used by Hugging Face
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
    """
    Create the Hugging Face InferenceClient exactly once.

    The actual sentiment model stays on Hugging Face. Render only creates
    a lightweight API client.
    """
    if _STATE["attempted"]:
        return

    with _LOCK:
        if _STATE["attempted"]:
            return

        _STATE["attempted"] = True

        if not ENABLE_MULTILINGUAL:
            _STATE["error"] = (
                "Multilingual analysis is disabled "
                "(ENABLE_MULTILINGUAL=0)."
            )
            return

        hf_token = os.getenv("HF_TOKEN", "").strip()

        if not hf_token:
            _STATE["error"] = (
                "HF_TOKEN is not configured. Add the Hugging Face "
                "access token to the Render environment variables."
            )
            return

        try:
            from huggingface_hub import InferenceClient

            logger.info(
                "Initialising Hugging Face InferenceClient for model %s",
                SENTIMENT_MODEL_NAME,
            )

            _STATE["client"] = InferenceClient(
                provider="auto",
                api_key=hf_token,
            )

        except Exception as exc:
            logger.exception(
                "Could not initialise Hugging Face InferenceClient"
            )

            _STATE["error"] = (
                "Could not initialise Hugging Face InferenceClient: "
                f"{exc}"
            )


def available() -> bool:
    """
    Return True when the Hugging Face client is ready.

    This does not download or load the model locally.
    """
    _load()
    return _STATE["client"] is not None


def status() -> Dict:
    """
    Report multilingual service readiness without making an inference call.

    Returns:
        {
            "expected": bool,
            "loaded": bool,
            "error": str | None
        }
    """

    if not ENABLE_MULTILINGUAL:
        return {
            "expected": False,
            "loaded": False,
            "error": (
                "Multilingual analysis is disabled "
                "(ENABLE_MULTILINGUAL=0)."
            ),
        }

    if _STATE["attempted"]:
        loaded = _STATE["client"] is not None

        return {
            "expected": loaded,
            "loaded": loaded,
            "error": None if loaded else _STATE["error"],
        }

    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not hf_token:
        return {
            "expected": False,
            "loaded": False,
            "error": (
                "HF_TOKEN is not configured. Add the Hugging Face "
                "access token to Render environment variables."
            ),
        }

    return {
        "expected": True,
        "loaded": False,
        "error": None,
    }


def _item_value(item, name: str, default=None):
    """
    Read a value from either a Hugging Face output object or a dictionary.
    """

    if isinstance(item, dict):
        return item.get(name, default)

    return getattr(item, name, default)


def _to_result(text: str, scored) -> Dict:
    """
    Convert Hugging Face classification output into the unified
    sentiment schema used by the rest of the application.
    """

    if isinstance(scored, dict):
        scored = [scored]

    probs = {
        "positive": 0.0,
        "neutral": 0.0,
        "negative": 0.0,
    }

    for item in scored:
        label = _item_value(item, "label", "")
        score = _item_value(item, "score", 0.0)

        canon = _canonical(label)

        if canon in probs:
            try:
                probs[canon] += float(score)
            except (TypeError, ValueError):
                pass

    total = sum(probs.values())

    if total > 0:
        probs = {
            key: value / total
            for key, value in probs.items()
        }

    top = max(probs, key=probs.get)

    polarity = probs["positive"] - probs["negative"]

    return {
        "text": text,
        "sentiment": top.upper(),
        "compound": None,
        "polarity": round(polarity, 4),
        "confidence": round(probs[top], 4),
        "positive": round(probs["positive"], 4),
        "neutral": round(probs["neutral"], 4),
        "negative": round(probs["negative"], 4),
        "model": SENTIMENT_MODEL_NAME,
        "score_type": "model_probability",
    }


def analyze_batch(texts: List[str]) -> List[Dict]:
    """
    Send multilingual texts to Hugging Face for sentiment analysis.

    Each text is sent as a remote inference request. The model itself is
    not downloaded or stored on the Render server.

    Raises RuntimeError if the Hugging Face client is unavailable.
    """

    if not available():
        raise RuntimeError(
            _STATE["error"]
            or "Hugging Face multilingual service unavailable."
        )

    client = _STATE["client"]

    results: List[Dict] = []

    for text in texts:
        clean_text = text if text else " "

        try:
            scored = client.text_classification(
                clean_text,
                model=SENTIMENT_MODEL_NAME,
                top_k=3,
            )

            results.append(
                _to_result(clean_text, scored)
            )

        except Exception as exc:
            logger.exception(
                "Hugging Face inference failed for text: %s",
                exc,
            )

            raise RuntimeError(
                f"Hugging Face inference failed: {exc}"
            ) from exc

    return results
