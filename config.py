"""
config.py
---------
Central configuration for Sentiment Lab.

All tunables live here so nothing has to be hard-coded in routes or
utility modules. Secrets are read from the environment (via a .env file
in local development) and are never written to templates or JSON
responses.
"""

import os

from dotenv import load_dotenv

# Load .env from the project root if present. Never fails if absent.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

# --------------------------------------------------------------------------
# Uploads / limits
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = _int_env("MAX_UPLOAD_MB", 10) * 1024 * 1024
PREVIEW_ROW_LIMIT = _int_env("PREVIEW_ROW_LIMIT", 1000)
MAX_TEXT_LENGTH = _int_env("MAX_TEXT_LENGTH", 20000)
MAX_ROWS = _int_env("MAX_ROWS", 50000)

# --------------------------------------------------------------------------
# Secrets / integrations
# --------------------------------------------------------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Allowed comment limits offered in the UI. "all" is capped by
# YOUTUBE_HARD_LIMIT so a single request can never drain the daily quota.
YOUTUBE_COMMENT_CHOICES = [100, 500, 1000, 5000]
YOUTUBE_HARD_LIMIT = _int_env("YOUTUBE_HARD_LIMIT", 10000)
YOUTUBE_PAGE_SIZE = 100  # maximum the API allows per page
REQUEST_TIMEOUT = _int_env("REQUEST_TIMEOUT", 15)

# --------------------------------------------------------------------------
# NLP
# --------------------------------------------------------------------------
# Multilingual sentiment model used for Hindi / Hinglish / uncertain text.
# Swap via the SENTIMENT_MODEL_NAME environment variable.
SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-xlm-roberta-base-sentiment"
)
# Set ENABLE_MULTILINGUAL=0 to keep the app VADER-only (no torch download).
ENABLE_MULTILINGUAL = os.getenv("ENABLE_MULTILINGUAL", "1").lower() not in {
    "0",
    "false",
    "no",
}
MULTILINGUAL_BATCH_SIZE = _int_env("MULTILINGUAL_BATCH_SIZE", 32)

# VADER classification thresholds (unchanged from v1).
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

USER_AGENT = os.getenv(
    "USER_AGENT",
    "SentimentLab/2.0 (+https://github.com/; research/portfolio project)",
)

# --- Dev server ---------------------------------------------------------
# Only used by `python app.py`. In production run gunicorn, which manages
# its own bind address and never reads these.
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int_env("PORT", 5000)
DEBUG = os.getenv("FLASK_DEBUG", "1").lower() not in {"0", "false", "no"}
# Flask sessions are not used today, but a real key should be set in
# production so that any future use of flash()/session is safe.
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me-in-production")
