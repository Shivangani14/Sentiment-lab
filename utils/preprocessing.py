"""
utils/preprocessing.py
----------------------
Conservative text cleaning and input validation.

Cleaning is deliberately gentle: VADER treats capitalisation, punctuation
emphasis ("good!!!") and emoticons as intensity signals, so we only remove
genuine noise (URLs, HTML entities/tags, redundant whitespace) and leave
the sentiment-bearing surface features intact. The same conservative
treatment also benefits the transformer model, which was trained on raw
social text.

Public API (unchanged from v1):
    clean_text(text) -> str
    validate_text_input(text) -> (bool, str | None)
"""

import html
import re
from typing import Optional, Tuple

from config import MAX_TEXT_LENGTH

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_MENTION_HANDLE_RE = re.compile(r"@\w{1,30}")
_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")


def clean_text(text: str, strip_handles: bool = False) -> str:
    """Remove noise from *text* while preserving sentiment signals.

    Args:
        text: raw input of any type; non-strings are coerced safely.
        strip_handles: also remove @mentions (useful for YouTube replies
            where the first token is often the parent author's handle).

    Returns:
        Cleaned string. May be empty if the input was pure noise.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    if strip_handles:
        text = _MENTION_HANDLE_RE.sub(" ", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def validate_text_input(text: str) -> Tuple[bool, Optional[str]]:
    """Validate user-submitted text before analysis.

    Returns:
        (is_valid, error_message). error_message is None when valid.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return False, "Please enter some text to analyze."

    stripped = text.strip()
    if len(stripped) > MAX_TEXT_LENGTH:
        return (
            False,
            f"That text is too long. Please keep it under "
            f"{MAX_TEXT_LENGTH:,} characters.",
        )
    return True, None


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF]"
)


def is_meaningful(text: str) -> bool:
    """True when a string carries any sentiment-bearing content.

    Letters, digits and emoji all count — an emoji-only comment ("\U0001F602\U0001F602\U0001F525")
    is genuinely analysable because VADER ships an emoji/emoticon lexicon.
    Blank, NaN and punctuation-only rows return False and are marked
    UNSCORED rather than crashing the pipeline.
    """
    if not text:
        return False
    value = str(text)
    return bool(re.search(r"[^\W_]", value, re.UNICODE) or _EMOJI_RE.search(value))
