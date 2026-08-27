"""
utils/blog.py
-------------
Best-effort comment retrieval from blogs and articles.

Detection order (most reliable first)
=====================================
1. **WordPress REST API** — `/wp-json/wp/v2/comments?post=<id>`. Fully
   supported, paginated, structured author/date/content.
2. **Structured data** — schema.org `comment` / `Comment` entries embedded
   in JSON-LD.
3. **Accessible HTML comment sections** — a conservative, generic
   extraction layer that looks for standard comment markup (`<ol
   class="comment-list">`, `[itemprop=comment]`, `#comments`, elements whose
   class/id matches a comment pattern). Nothing is hard-coded to one site.
4. **Known third-party embeds** — Disqus, Facebook Comments, Hyvor, Commento
   are *detected and reported*, not bypassed: their comments load from a
   separate authenticated service and are not in the page HTML.

Ethics and safety
=================
* `robots.txt` is fetched and honoured for our user agent; a disallowed path
  is refused with an explanation.
* Crawl-delay is respected between requests, requests time out, and total
  pages are capped.
* No attempt is made to defeat CAPTCHAs, logins, paywalls, or anti-bot
  protection. If content is gated, the user is told why.
* Only comment text is extracted — never the whole page as fake "comments".

Public API:
    fetch_blog_comments(url, limit=500) -> dict
    BlogError  (carries a user-facing message)
"""

import ipaddress
import json
import logging
import re
import socket
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from config import REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)

MAX_PAGES = 10
WP_PER_PAGE = 100
MIN_COMMENT_CHARS = 3
MAX_COMMENT_CHARS = 5000

_COMMENT_CLASS_RE = re.compile(
    r"(^|[-_ ])(comment|comments|commentlist|comment-list|comment-body|"
    r"comment-content|comment-text|respond|discussion|replies)([-_ ]|$)",
    re.IGNORECASE,
)
_SKIP_RE = re.compile(
    r"(comment-form|comment-respond|reply-title|comments-title|"
    r"comment-count|comment-nav|leave-a-comment|add-comment)",
    re.IGNORECASE,
)

_EMBED_SIGNATURES = {
    "Disqus": ("disqus.com/embed", "disqus_thread", "disquscdn"),
    "Facebook Comments": ("fb-comments", "connect.facebook.net"),
    "Hyvor Talk": ("hyvor",),
    "Commento": ("commento",),
    "Utterances / Giscus": ("utteranc.es", "giscus"),
    "Spot.IM / OpenWeb": ("spot.im", "openweb"),
}


class BlogError(Exception):
    """A blog-fetch failure with a message that is safe to show a user."""


# --------------------------------------------------------------------------
# URL validation (also protects against SSRF against internal hosts)
# --------------------------------------------------------------------------
def validate_url(url: str) -> str:
    if not url or not str(url).strip():
        raise BlogError("Please paste a blog or article URL.")

    raw = str(url).strip()
    # Bare hostnames are a convenience ('example.com/post'), but only add a
    # scheme when there genuinely isn't one. Without this guard,
    # 'javascript:alert(1)' would be rewritten into an https URL with the
    # hostname 'javascript' and produce a confusing DNS error instead of a
    # clear "unsupported scheme" message.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", raw):
        raw = "https://" + raw

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise BlogError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname:
        raise BlogError("That URL is missing a hostname.")

    host = parsed.hostname
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise BlogError("Local addresses cannot be analyzed.")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise BlogError(f"Could not resolve '{host}'. Check the URL.")

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise BlogError("Private and internal addresses cannot be analyzed.")

    return raw


# --------------------------------------------------------------------------
# robots.txt
# --------------------------------------------------------------------------
def _robots(url: str):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    try:
        response = requests.get(
            robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        if response.status_code >= 400:
            return None, 0.0  # no robots.txt => allowed
        parser.parse(response.text.splitlines())
    except requests.RequestException:
        return None, 0.0

    delay = 0.0
    try:
        raw_delay = parser.crawl_delay(USER_AGENT)
        if raw_delay:
            delay = min(float(raw_delay), 5.0)
    except Exception:
        delay = 0.0
    return parser, delay


def _assert_allowed(parser, url: str) -> None:
    if parser is None:
        return
    if not parser.can_fetch(USER_AGENT, url):
        raise BlogError(
            "This site's robots.txt asks automated clients not to fetch that "
            "page, so it was not requested."
        )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _fetch(session: requests.Session, url: str, as_json: bool = False):
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.Timeout:
        raise BlogError("That site took too long to respond.")
    except requests.TooManyRedirects:
        raise BlogError("That URL redirected too many times.")
    except requests.RequestException:
        raise BlogError("Could not connect to that site.")

    if response.status_code in {401, 403}:
        raise BlogError(
            "That site blocked the request (it requires a login or has "
            "anti-bot protection). Access controls are not bypassed."
        )
    if response.status_code == 404:
        raise BlogError("That page could not be found (404).")
    if response.status_code == 429:
        raise BlogError("That site is rate-limiting requests. Try again later.")
    if response.status_code >= 500:
        raise BlogError(f"That site returned a server error ({response.status_code}).")
    if response.status_code >= 400:
        raise BlogError(f"That site returned an error ({response.status_code}).")

    if as_json:
        try:
            return response, response.json()
        except ValueError:
            return response, None
    return response, None


def _clean_html_to_text(fragment: str) -> str:
    soup = BeautifulSoup(fragment or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _valid_comment(text: str) -> bool:
    return bool(text) and MIN_COMMENT_CHARS <= len(text) <= MAX_COMMENT_CHARS


# --------------------------------------------------------------------------
# Strategy 1 - WordPress REST API
# --------------------------------------------------------------------------
def _wp_post_id(html: str) -> Optional[str]:
    match = re.search(r'rel=["\']shortlink["\'][^>]*href=["\'][^"\']*[?&]p=(\d+)', html)
    if match:
        return match.group(1)
    match = re.search(r"postid-(\d+)", html)
    if match:
        return match.group(1)
    match = re.search(r'name=["\']comment_post_ID["\'][^>]*value=["\'](\d+)', html)
    if match:
        return match.group(1)
    return None


def _try_wordpress(session, parser, delay, url, html, limit) -> Optional[List[Dict]]:
    if "wp-json" not in html and "wp-content" not in html and "wp-includes" not in html:
        return None

    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    post_id = _wp_post_id(html)

    if not post_id:
        slug = [p for p in urlparse(url).path.split("/") if p]
        if slug:
            probe = urljoin(base, f"/wp-json/wp/v2/posts?slug={slug[-1]}&_fields=id")
            try:
                _assert_allowed(parser, probe)
                _, data = _fetch(session, probe, as_json=True)
                if isinstance(data, list) and data:
                    post_id = str(data[0].get("id"))
            except BlogError:
                post_id = None

    if not post_id:
        return None

    comments: List[Dict] = []
    for page in range(1, MAX_PAGES + 1):
        endpoint = urljoin(
            base,
            f"/wp-json/wp/v2/comments?post={post_id}"
            f"&per_page={WP_PER_PAGE}&page={page}&orderby=date&order=desc",
        )
        try:
            _assert_allowed(parser, endpoint)
            response, data = _fetch(session, endpoint, as_json=True)
        except BlogError:
            break

        if not isinstance(data, list) or not data:
            break

        for item in data:
            text = _clean_html_to_text(
                (item.get("content") or {}).get("rendered", "")
            )
            if not _valid_comment(text):
                continue
            comments.append(
                {
                    "comment_id": str(item.get("id", "")),
                    "author": item.get("author_name") or "Anonymous",
                    "comment": text,
                    "published_at": item.get("date_gmt") or item.get("date") or "",
                    "parent_id": str(item.get("parent") or ""),
                }
            )
            if len(comments) >= limit:
                return comments

        total_pages = int(response.headers.get("X-WP-TotalPages") or 0)
        if total_pages and page >= total_pages:
            break
        if len(data) < WP_PER_PAGE:
            break
        if delay:
            time.sleep(delay)

    return comments or None


# --------------------------------------------------------------------------
# Strategy 2 - schema.org JSON-LD
# --------------------------------------------------------------------------
def _walk_jsonld(node, out: List[Dict]) -> None:
    if isinstance(node, list):
        for child in node:
            _walk_jsonld(child, out)
        return
    if not isinstance(node, dict):
        return

    node_type = node.get("@type") or ""
    types = node_type if isinstance(node_type, list) else [node_type]
    if any(str(t).lower() == "comment" for t in types):
        author = node.get("author")
        if isinstance(author, dict):
            author = author.get("name", "")
        text = _clean_html_to_text(node.get("text") or node.get("description") or "")
        if _valid_comment(text):
            out.append(
                {
                    "comment_id": str(node.get("@id") or len(out) + 1),
                    "author": str(author or "Anonymous"),
                    "comment": text,
                    "published_at": node.get("dateCreated")
                    or node.get("datePublished")
                    or "",
                    "parent_id": "",
                }
            )

    for value in node.values():
        if isinstance(value, (dict, list)):
            _walk_jsonld(value, out)


def _try_structured_data(soup: BeautifulSoup, limit: int) -> Optional[List[Dict]]:
    found: List[Dict] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        _walk_jsonld(payload, found)
        if len(found) >= limit:
            break
    return found[:limit] or None


# --------------------------------------------------------------------------
# Strategy 3 - generic accessible HTML
# --------------------------------------------------------------------------
def _try_html(soup: BeautifulSoup, limit: int) -> Optional[List[Dict]]:
    candidates = []

    # Microdata / ARIA first — the most explicit signals.
    candidates += soup.select("[itemtype*='Comment'], [itemprop='comment']")
    candidates += soup.select("article.comment, li.comment, div.comment")
    candidates += soup.select(
        "ol.comment-list > li, ul.comment-list > li, "
        ".comments-list > li, .commentlist > li"
    )

    if not candidates:
        for element in soup.find_all(["article", "li", "div", "section"]):
            identifier = " ".join(
                filter(
                    None,
                    [
                        " ".join(element.get("class") or []),
                        element.get("id") or "",
                    ],
                )
            )
            if not identifier or _SKIP_RE.search(identifier):
                continue
            if _COMMENT_CLASS_RE.search(identifier):
                candidates.append(element)

    comments: List[Dict] = []
    seen = set()

    for element in candidates:
        body = element.select_one(
            ".comment-content, .comment-body, .comment-text, "
            "[itemprop='text'], .comment__body, p"
        )
        text = _clean_html_to_text(str(body if body else element))
        if not _valid_comment(text):
            continue

        fingerprint = text[:160].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        author_el = element.select_one(
            ".comment-author, .fn, [itemprop='author'], .author, cite"
        )
        time_el = element.find("time")

        comments.append(
            {
                "comment_id": element.get("id") or str(len(comments) + 1),
                "author": (
                    _clean_html_to_text(str(author_el))[:120]
                    if author_el
                    else "Anonymous"
                ),
                "comment": text,
                "published_at": (
                    time_el.get("datetime") or _clean_html_to_text(str(time_el))
                    if time_el
                    else ""
                ),
                "parent_id": "",
            }
        )
        if len(comments) >= limit:
            break

    # Two or fewer hits on a generic page is almost certainly a false
    # positive (nav item, "Leave a comment" box), so don't claim success.
    return comments if len(comments) >= 2 else None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _detect_embed(html: str) -> Optional[str]:
    lowered = html.lower()
    for name, signatures in _EMBED_SIGNATURES.items():
        if any(sig in lowered for sig in signatures):
            return name
    return None


def fetch_blog_comments(url: str, limit: int = 500) -> Dict:
    """Retrieve comments from a blog/article URL.

    Returns:
        {
          "url": final URL,
          "title": page title,
          "method": "WordPress REST API" | "Structured data (schema.org)"
                    | "HTML comment section",
          "comments": [ {comment_id, author, comment, published_at, parent_id} ]
        }

    Raises:
        BlogError with a clear, user-facing explanation on failure.
    """
    safe_url = validate_url(url)
    limit = max(1, min(int(limit or 500), 2000))

    parser, delay = _robots(safe_url)
    _assert_allowed(parser, safe_url)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9",
            "Accept-Language": "en,hi;q=0.8",
        }
    )

    response, _ = _fetch(session, safe_url)
    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "xml" not in content_type:
        raise BlogError(
            f"That URL returned '{content_type or 'unknown content'}', not a "
            f"web page."
        )

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    title = _clean_html_to_text(str(soup.title)) if soup.title else safe_url

    if delay:
        time.sleep(delay)

    strategies = [
        ("WordPress REST API", lambda: _try_wordpress(
            session, parser, delay, response.url, html, limit)),
        ("Structured data (schema.org)", lambda: _try_structured_data(soup, limit)),
        ("HTML comment section", lambda: _try_html(soup, limit)),
    ]

    for method, run in strategies:
        try:
            found = run()
        except BlogError:
            raise
        except Exception:
            logger.exception("Blog strategy '%s' failed", method)
            found = None

        if found:
            return {
                "url": response.url,
                "title": title,
                "method": method,
                "comments": found[:limit],
            }

    embed = _detect_embed(html)
    if embed:
        raise BlogError(
            f"This page uses {embed} for comments. Those comments are loaded "
            f"from {embed}'s own service and are not present in the page "
            f"HTML, so they cannot be read without that platform's API "
            f"credentials. Export them from {embed} to CSV and use the "
            f"Dataset source instead."
        )

    raise BlogError(
        "No comments were found on that page. It may have no comments, they "
        "may be loaded by JavaScript after page load, or they may live behind "
        "a login. Nothing was guessed or fabricated."
    )
