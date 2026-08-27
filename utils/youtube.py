"""
utils/youtube.py
----------------
YouTube comment retrieval via the official **YouTube Data API v3**.

No HTML is fetched or parsed — everything goes through the documented REST
endpoints (`videos`, `commentThreads`). The API key is read from the
YOUTUBE_API_KEY environment variable and is never returned to the client,
rendered into a template, or written to a log line.

Quota discipline
================
`commentThreads.list` costs 1 unit per call and returns at most 100 items,
so a 5,000-comment request costs ~50 units of the default 10,000/day quota.
Requests are hard-capped by `YOUTUBE_HARD_LIMIT` so "All available" can
never issue an unbounded number of page requests.

Public API:
    extract_video_id(url) -> str
    get_video_details(video_id) -> dict
    fetch_comments(video_id, limit, include_replies=True, progress=None)
        -> list[dict]
    YouTubeError  (carries a user-facing message)
"""

import logging
import re
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests

from config import (
    REQUEST_TIMEOUT,
    USER_AGENT,
    YOUTUBE_API_BASE,
    YOUTUBE_API_KEY,
    YOUTUBE_HARD_LIMIT,
    YOUTUBE_PAGE_SIZE,
)

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# API reason code -> message shown to the user.
_REASON_MESSAGES = {
    "commentsDisabled": "Comments are turned off for this video.",
    "quotaExceeded": (
        "The YouTube API daily quota has been used up. Try again after the "
        "quota resets (midnight Pacific Time) or use a different API key."
    ),
    "rateLimitExceeded": "YouTube is rate-limiting requests. Try again shortly.",
    "videoNotFound": "That video does not exist, or it has been removed.",
    "forbidden": (
        "YouTube refused the request. The video may be private, "
        "age-restricted, or region-blocked."
    ),
    "keyInvalid": "The configured YOUTUBE_API_KEY is not valid.",
    "badRequest": "The configured YOUTUBE_API_KEY is not valid.",
    "ipRefererBlocked": (
        "This API key has HTTP referrer or IP restrictions that block this "
        "server. Loosen the key restrictions in Google Cloud Console."
    ),
    "accessNotConfigured": (
        "The YouTube Data API v3 is not enabled for this Google Cloud "
        "project. Enable it and try again."
    ),
    "processingFailure": "YouTube could not process that request. Try again.",
}


class YouTubeError(Exception):
    """A YouTube failure with a message that is safe to show a user."""


def api_key_configured() -> bool:
    return bool(YOUTUBE_API_KEY)


def _require_key() -> str:
    if not YOUTUBE_API_KEY:
        raise YouTubeError(
            "No YouTube API key is configured. Add YOUTUBE_API_KEY to your "
            ".env file (see .env.example) and restart the app."
        )
    return YOUTUBE_API_KEY


def extract_video_id(url: str) -> str:
    """Pull the 11-character video ID out of any common YouTube URL form.

    Supports watch?v=, youtu.be/, /shorts/, /embed/, /live/, and a bare ID.
    """
    if not url or not str(url).strip():
        raise YouTubeError("Please paste a YouTube video URL.")

    raw = str(url).strip()

    # A bare video ID pasted on its own.
    if _ID_RE.match(raw):
        return raw

    if "://" not in raw:
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
    except ValueError:
        raise YouTubeError("That does not look like a valid URL.")

    host = (parsed.hostname or "").lower().replace("www.", "")
    path = parsed.path or ""

    candidate = None
    if host in {"youtu.be"}:
        candidate = path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            parts = [p for p in path.split("/") if p]
            if parts and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1] if len(parts) > 1 else None
    else:
        raise YouTubeError(
            "That URL is not a YouTube link. Paste a youtube.com or "
            "youtu.be video URL."
        )

    if not candidate or not _ID_RE.match(candidate):
        raise YouTubeError(
            "Could not find a video ID in that URL. Supported formats: "
            "youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID."
        )
    return candidate


def _get(endpoint: str, params: Dict) -> Dict:
    """Call the API and translate every failure into a YouTubeError."""
    params = {**params, "key": _require_key()}
    url = f"{YOUTUBE_API_BASE}/{endpoint}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.Timeout:
        raise YouTubeError("YouTube did not respond in time. Please try again.")
    except requests.RequestException:
        raise YouTubeError(
            "Could not reach the YouTube API. Check your internet connection."
        )

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise YouTubeError("YouTube returned an unreadable response.")

    reason = ""
    detail = ""
    try:
        error = response.json().get("error", {})
        detail = error.get("message", "")
        errors = error.get("errors") or []
        if errors:
            reason = errors[0].get("reason", "")
    except ValueError:
        pass

    # Log the technical detail server-side only; never echo the key.
    logger.warning(
        "YouTube API %s -> %s (reason=%s)", endpoint, response.status_code, reason
    )

    if reason in _REASON_MESSAGES:
        raise YouTubeError(_REASON_MESSAGES[reason])
    if response.status_code == 403:
        raise YouTubeError(
            "YouTube refused the request (403). This usually means the API "
            "key is restricted, the quota is exhausted, or comments are "
            "disabled."
        )
    if response.status_code == 404:
        raise YouTubeError("That video could not be found.")
    raise YouTubeError(
        f"YouTube returned an error ({response.status_code})."
        + (f" {detail}" if detail else "")
    )


def get_video_details(video_id: str) -> Dict:
    """Fetch title/channel/stats so the UI can confirm the right video."""
    data = _get(
        "videos",
        {"part": "snippet,statistics", "id": video_id},
    )
    items = data.get("items") or []
    if not items:
        raise YouTubeError(
            "That video is unavailable — it may be private, deleted, or "
            "region-blocked."
        )

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    thumbs = snippet.get("thumbnails", {})
    thumb = (
        thumbs.get("medium", {}).get("url")
        or thumbs.get("default", {}).get("url")
        or ""
    )

    def as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "video_id": video_id,
        "title": snippet.get("title", "Untitled video"),
        "channel": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "thumbnail": thumb,
        "view_count": as_int(stats.get("viewCount")),
        "like_count": as_int(stats.get("likeCount")),
        "comment_count": as_int(stats.get("commentCount")),
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def _flatten_comment(comment_id: str, snippet: Dict, parent_id: str = "") -> Dict:
    return {
        "comment_id": comment_id,
        "author": snippet.get("authorDisplayName", ""),
        "comment": snippet.get("textOriginal") or snippet.get("textDisplay") or "",
        "published_at": snippet.get("publishedAt", ""),
        "like_count": int(snippet.get("likeCount") or 0),
        "is_reply": bool(parent_id),
        "parent_id": parent_id,
    }


def fetch_comments(
    video_id: str,
    limit: Optional[int] = 100,
    include_replies: bool = True,
    order: str = "relevance",
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> List[Dict]:
    """Retrieve comments for *video_id* with pagination.

    Args:
        video_id: 11-character YouTube video ID.
        limit: maximum comments to return. None means "all available",
            still bounded by YOUTUBE_HARD_LIMIT.
        include_replies: include the replies returned alongside each thread
            (free — they arrive in the same response).
        order: "relevance" (default) or "time".
        progress: callback(fetched, limit).

    Raises:
        YouTubeError with a user-facing message.
    """
    hard_cap = YOUTUBE_HARD_LIMIT
    target = hard_cap if not limit or limit <= 0 else min(int(limit), hard_cap)

    collected: List[Dict] = []
    page_token = None
    seen_ids = set()

    while len(collected) < target:
        params = {
            "part": "snippet,replies" if include_replies else "snippet",
            "videoId": video_id,
            "maxResults": min(YOUTUBE_PAGE_SIZE, target - len(collected)) or 1,
            "textFormat": "plainText",
            "order": order if order in {"relevance", "time"} else "relevance",
        }
        if page_token:
            params["pageToken"] = page_token

        data = _get("commentThreads", params)
        items = data.get("items") or []
        if not items and not collected:
            raise YouTubeError("This video has no comments to analyze.")

        for item in items:
            top = item.get("snippet", {}).get("topLevelComment", {}) or {}
            cid = top.get("id") or item.get("id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                collected.append(_flatten_comment(cid, top.get("snippet", {})))

            if include_replies:
                for reply in (item.get("replies", {}) or {}).get("comments", []):
                    rid = reply.get("id", "")
                    if rid and rid not in seen_ids and len(collected) < target:
                        seen_ids.add(rid)
                        collected.append(
                            _flatten_comment(
                                rid, reply.get("snippet", {}), parent_id=cid
                            )
                        )

            if len(collected) >= target:
                break

        if progress:
            progress(len(collected), None if not limit else target)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    if not collected:
        raise YouTubeError("This video has no comments to analyze.")

    return collected[:target]
