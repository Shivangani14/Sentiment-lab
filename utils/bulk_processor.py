"""
utils/bulk_processor.py
-----------------------
Dataset (CSV) handling: text-column discovery, batch scoring, and summaries.

Two generations of API live here on purpose:

* `find_text_column`, `process_dataframe`, `summarize` are the **v1**
  functions. `/bulk-upload` still calls them, so existing integrations and
  the original five appended columns (`sentiment_compound`, `sentiment_pos`,
  `sentiment_neu`, `sentiment_neg`, `sentiment_label`) keep working exactly
  as before.
* `analyze_dataframe` is the **v2** function used by the unified /analyze
  page. It routes through `utils.engine`, so a dataset containing Hindi or
  Hinglish gets the multilingual model, and it appends the normalised
  columns (`language`, `sentiment`, `compound`, ...). Original columns are
  never modified or dropped.

Public API:
    find_text_column(df) -> str | None
    candidate_text_columns(df) -> list[dict]
    read_csv_safely(path) -> pandas.DataFrame
    process_dataframe(df, text_column) -> DataFrame        # v1
    summarize(df) -> dict                                  # v1
    analyze_dataframe(df, text_column, limit=None, progress=None)
        -> (DataFrame, list[dict])                         # v2
"""

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from utils.engine import analyze_many
from utils.preprocessing import is_meaningful
from utils.sentiment_analyzer import analyze_text

# Ordered by preference — the first match wins and is preselected in the UI.
TEXT_COLUMN_CANDIDATES = [
    "comment",
    "comments",
    "text",
    "review",
    "review_text",
    "reviews",
    "feedback",
    "message",
    "content",
    "description",
    "body",
    "tweet",
    "post",
    "title",
]

V1_COLUMNS = [
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neu",
    "sentiment_neg",
    "sentiment_label",
]

V2_COLUMNS = [
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


class DatasetError(Exception):
    """Raised with a user-facing message when a CSV cannot be used."""


# --------------------------------------------------------------------------
# Reading / column discovery
# --------------------------------------------------------------------------
def read_csv_safely(path: str) -> pd.DataFrame:
    """Read a CSV, tolerating encoding quirks and malformed rows.

    Raises DatasetError with a friendly message rather than leaking a
    pandas traceback.
    """
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]
    last_error: Optional[Exception] = None

    for opts in attempts:
        try:
            return pd.read_csv(path, on_bad_lines="skip", **opts)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError:
            raise DatasetError("That CSV file is empty.")
        except pd.errors.ParserError:
            raise DatasetError(
                "That CSV could not be parsed. Check the delimiter and "
                "that every row has the same number of columns."
            )
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

    raise DatasetError(
        "That file could not be read as CSV. Try re-saving it as UTF-8 CSV."
    )


def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely text column, or None if nothing looks right.

    v1 contract preserved: exact/lowercase name matching against the
    candidate list, in preference order.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    # Fall back to a substring match ("customer_review_text").
    for candidate in TEXT_COLUMN_CANDIDATES:
        for low, original in lowered.items():
            if candidate in low:
                return original
    return None


def candidate_text_columns(df: pd.DataFrame) -> List[Dict]:
    """Describe every column so the user can pick the text column.

    Each entry carries a sample value, the count of non-empty values and an
    average length, which makes the right column obvious in the dropdown.
    """
    out: List[Dict] = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        strings = non_null.astype(str)
        filled = int(strings.map(is_meaningful).sum())
        avg_len = round(float(strings.str.len().mean()), 1) if len(strings) else 0.0
        sample = ""
        for value in strings:
            if is_meaningful(value):
                sample = value[:120]
                break
        out.append(
            {
                "name": str(col),
                "filled": filled,
                "avg_length": avg_len,
                "sample": sample,
                "is_text_like": avg_len >= 15,
            }
        )
    return out


# --------------------------------------------------------------------------
# v1 pipeline (preserved for /bulk-upload)
# --------------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Append the five original VADER columns to *df*. Unchanged from v1."""
    enriched = df.copy()
    records = [analyze_text(str(v)) for v in enriched[text_column].fillna("")]
    enriched["sentiment_compound"] = [r["compound"] for r in records]
    enriched["sentiment_pos"] = [r["positive"] for r in records]
    enriched["sentiment_neu"] = [r["neutral"] for r in records]
    enriched["sentiment_neg"] = [r["negative"] for r in records]
    enriched["sentiment_label"] = [r["sentiment"] for r in records]
    return enriched


def summarize(df: pd.DataFrame) -> Dict:
    """v1 summary payload built from the `sentiment_label` column."""
    total = int(len(df))
    labels = df["sentiment_label"] if "sentiment_label" in df else pd.Series(dtype=str)
    counts = labels.value_counts().to_dict()
    positive = int(counts.get("POSITIVE", 0))
    neutral = int(counts.get("NEUTRAL", 0))
    negative = int(counts.get("NEGATIVE", 0))

    def pct(value: int) -> float:
        return round((value / total) * 100, 1) if total else 0.0

    avg = (
        round(float(df["sentiment_compound"].mean()), 4)
        if total and "sentiment_compound" in df
        else 0.0
    )
    return {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_pct": pct(positive),
        "neutral_pct": pct(neutral),
        "negative_pct": pct(negative),
        "average_compound": avg,
    }


# --------------------------------------------------------------------------
# v2 pipeline (unified /analyze page)
# --------------------------------------------------------------------------
def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Score *text_column* and append the normalised analysis columns.

    Rows whose text is empty/NaN/unusable are kept but marked `UNSCORED`
    instead of being dropped, so the user never silently loses data.

    Returns:
        (enriched DataFrame, list of normalised result dicts)
    """
    if text_column not in df.columns:
        raise DatasetError(
            f"Column '{text_column}' is not in this file. "
            f"Available columns: {', '.join(str(c) for c in df.columns)}."
        )

    working = df.copy()
    if limit is not None and limit > 0:
        working = working.head(limit)

    if working.empty:
        raise DatasetError("There are no rows to analyze in that file.")

    texts = ["" if pd.isna(v) else str(v) for v in working[text_column]]
    results = analyze_many(texts, progress=progress)

    for column in V2_COLUMNS:
        working[column] = [r.get(column) for r in results]

    return working, results
