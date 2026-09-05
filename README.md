# Sentiment Lab

A Flask web app that runs **one sentiment pipeline over four content
sources** — pasted text, uploaded CSV datasets, YouTube video comments, and
blog/article comments — with language-aware model routing.

# Sentiment Lab

## 🚀 Live Demo

[Open Sentiment Lab](https://sentiment-lab.onrender.com)

English text is scored with
[VADER](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, ICWSM
2014). Hindi, Hinglish and other non-English text is routed to a
transformer-based multilingual model instead. Every result — regardless of
source or engine — is normalised into the same shape and rendered by the
same dashboard.

---

## Contents

- [What's new in v2](#whats-new-in-v2)
- [Architecture](#architecture)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Using each source](#using-each-source)
- [How scores are reported (read this)](#how-scores-are-reported-read-this)
- [Language detection](#language-detection)
- [Routes and API](#routes-and-api)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Performance](#performance)
- [Limitations](#limitations)
- [Production notes](#production-notes)

---

## What's new in v2

v1 had two separate pages (`/` for single text, `/bulk` for CSVs) and a
VADER-only, English-only engine. v2 replaces that with:

- **One unified page** at `/analyze` with a source selector. The input area
  changes with the selected source; the results dashboard does not.
- **Two new sources**: YouTube comments (official Data API v3) and blog
  comments (WordPress REST API → schema.org structured data → generic HTML).
- **Language-aware routing**: English → VADER, Hindi/Hinglish/uncertain →
  multilingual transformer, loaded lazily and cached.
- **Honest score reporting**: VADER compound scores and model confidence
  scores are never conflated (see below).
- **Zero-config datasets**: the text column is auto-detected, blank and
  invalid rows are marked `UNSCORED` rather than crashing or being dropped.
- **Async jobs with progress** for datasets, YouTube and blogs.

The old routes still work: `/` redirects to `/analyze?source=text`, `/bulk`
redirects to `/analyze?source=dataset`, and the v1 JSON endpoints
`POST /predict`, `POST /bulk-upload` and `GET /download/<id>` are unchanged.

---

## Architecture

Every source funnels into the same five stages:

```
  Text  ─┐
Dataset ─┤
YouTube ─┼──▶ Text extraction ──▶ Language detection ──▶ Sentiment engine ──▶ Normalised results ──▶ Dashboard / charts / CSV export
   Blog ─┘    (per-source        (utils/language_      (utils/engine.py)      (RESULT_FIELDS)         (static/js/analyze.js)
               adapters)          detect.py)            ├─ VADER
                                                        └─ multilingual model
```

The sentiment logic lives in exactly one place — `utils/engine.py`. Routes
are thin adapters that turn a source into a list of strings and hand it to
`analyze_many()`. No route scores text itself.

---

## Setup

**1. Create and activate a virtual environment** (Python 3.12)

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Optional — install multilingual support**

Needed only for Hindi / Hinglish / other non-English text. This pulls in
PyTorch (~2 GB), so it is deliberately kept out of the base install:

```bash
pip install -r requirements-multilingual.txt
```

Without it the app runs fine; non-English text is flagged with a visible
warning instead of being silently mis-scored.

**4. Configure the environment**

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then open `.env` and fill in `YOUTUBE_API_KEY` if you want the YouTube
source. Everything else has a working default.

**5. Run**

```bash
python app.py
```

Open **http://127.0.0.1:5000** — it redirects to `/analyze`.

For production: `gunicorn -w 4 -b 0.0.0.0:8000 app:app`

---

## Environment variables

All are optional except `YOUTUBE_API_KEY` (required only for the YouTube
source). Set them in `.env`; see `.env.example` for the annotated template.

| Variable | Default | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | *(none)* | YouTube Data API v3 key. Without it the YouTube tab shows a setup notice and is disabled. **Never committed — `.env` is gitignored.** |
| `SENTIMENT_MODEL_NAME` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Hugging Face model id used for non-English text. Swap it without touching code. |
| `ENABLE_MULTILINGUAL` | `true` | Set to `false` to force VADER-only mode (non-English rows are then clearly flagged as fallbacks). |
| `SECRET_KEY` | dev value | Flask session key. Set a real one in production. |
| `FLASK_DEBUG` | `true` | Set to `false` in production. |
| `MAX_UPLOAD_MB` | `10` | CSV upload cap. |
| `PREVIEW_ROW_LIMIT` | `1000` | Rows sent to the browser. Downloaded CSVs always contain every row. |
| `YOUTUBE_MAX_COMMENTS` | `500` | Default comment budget in the UI. |
| `YOUTUBE_HARD_LIMIT` | `10000` | Absolute server-side ceiling, so no request can quietly drain your quota. |
| `BLOG_MAX_COMMENTS` | `500` | Comment ceiling for blog extraction. |
| `REQUEST_TIMEOUT` | `15` | Per-request HTTP timeout, in seconds. |

### Getting a YouTube API key

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and
   create (or pick) a project.
2. **APIs & Services → Library →** search *YouTube Data API v3* → **Enable**.
3. **APIs & Services → Credentials → Create credentials → API key**.
4. Copy the key into `.env` as `YOUTUBE_API_KEY=...` and restart the app.

The free tier gives 10,000 quota units/day. A `commentThreads.list` call
costs 1 unit and returns up to 100 comments, so ~1,000,000 comments/day.
The UI tells you the cost before you run anything.

---

## Using each source

### Text
Paste anything and hit **Analyze Sentiment**. Works with English, Devanagari
Hindi and romanised Hinglish. Returns a single verdict card with the score
breakdown, the detected language, and which engine scored it.

### Dataset
Drop in a CSV. The app inspects the header, **auto-selects** the most
likely text column (`comment`, `text`, `review`, `review_text`, `feedback`,
`message`, `content`, `description`, then falls back to the longest-average
string column), and shows you a dropdown of every column with fill counts
and average lengths so you can override it. You never have to construct
sentences by hand.

Blank cells, `NaN`, and non-text values are kept as rows and labelled
`UNSCORED` — your row count never silently changes. All of your original
columns are preserved in both the preview and the download; sentiment
columns are appended, never substituted.

### YouTube
Paste a video URL — `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`
or a bare 11-character id all work. Choose a comment budget (100 / 500 /
1000 / 5000) and whether to include reply threads. Comments are read through
the **official YouTube Data API v3**; no HTML is read and no scraping
happens. Pagination is handled for you up to your chosen budget.

Handled cleanly with plain-language messages: invalid URLs, private or
deleted videos, videos with comments disabled, quota exhaustion, invalid
keys, and videos with zero comments.

### Blog
Paste an article URL. Three strategies are tried in order:

1. **WordPress REST API** (`/wp-json/wp/v2/comments?post=<id>`) — the clean
   path, used whenever the site exposes it.
2. **schema.org structured data** — JSON-LD `comment` / `Comment` nodes.
3. **Generic HTML comment markup** — standard patterns such as
   `.comment`, `[itemprop=comment]`, `#comments li`, `.commentlist`.

No single site's markup is hard-coded. `robots.txt` is fetched and obeyed,
including `Crawl-delay`. Requests are timed out, rate-limited, and sent with
an identifying user agent. Loopback, private and link-local addresses are
refused.

If comments are behind Disqus, Facebook Comments, Hyvor, Commento or giscus,
the app **tells you which system it detected and stops** rather than
returning misleading results. CAPTCHAs, logins, paywalls and anti-bot
protections are never bypassed.

---

## How scores are reported (read this)

This is the part most sentiment dashboards get wrong, so it is worth being
explicit.

**VADER produces a `compound` score in `[-1, +1]`.** It is a real, signed
polarity value derived from a lexicon.

**A transformer classifier produces class probabilities.** Its "confidence"
is the probability of the winning class — a number in `[0, 1]` that says
nothing about direction. A 0.95 confidence on a *negative* prediction is not
a +0.95 compound score.

So the app reports them in separate fields and **never fills one in from the
other**:

| Field | VADER rows | Model rows |
|---|---|---|
| `sentiment` | POSITIVE / NEUTRAL / NEGATIVE | POSITIVE / NEUTRAL / NEGATIVE |
| `compound` | real VADER compound | `null` — shown as `—` in the UI |
| `confidence` | `null` — shown as `—` | winning-class probability |
| `positive` / `neutral` / `negative` | VADER proportions | class probabilities |
| `polarity` | same as `compound` | derived as `p(pos) − p(neg)` |
| `score_type` | `vader_compound` | `model_probability` |
| `model` | `VADER` | the model id |

The table header says **COMPOUND (VADER)** and **CONFIDENCE (Model)** so the
distinction is visible, not buried. `polarity` exists purely so mixed-engine
datasets can be averaged and charted on one axis; it is labelled "average
sentiment (cross-engine polarity)", not "average compound". The summary
reports `avg_compound` over VADER rows only and `avg_confidence` over model
rows only.

**If the multilingual model is unavailable** (packages not installed,
download failed, or `ENABLE_MULTILINGUAL=false`), non-English text falls
back to VADER — but the row is marked `model: "VADER (fallback)"`,
`fallback: true`, carries an explanatory note, and the dashboard shows a
banner reading *"Some non-English rows fell back to VADER because the
multilingual model was unavailable — treat those as unreliable."* Nothing is
quietly mis-scored.

### Model loading

The multilingual model is **not** loaded at startup. It loads on the first
non-English input, behind a lock, and is cached in a module-level singleton
for the process lifetime. VADER is a single `lru_cache`d instance. Neither
is re-created per comment. Non-English text is scored in batches; English
text is scored inline (VADER is cheap).

---

## Language detection

`utils/language_detect.py` runs a cascade, cheapest signal first:

| Order | Signal | Result |
|---|---|---|
| 1 | No letters but emoji present | `Unknown` → **VADER** (it ships an emoji lexicon) |
| 2 | Devanagari-dominant script | `Hindi` |
| 3 | Devanagari **and** Latin mixed | `Hinglish` |
| 4 | Romanised Hinglish marker lexicon (~100 unambiguous tokens: `bhai`, `yaar`, `kya`, `nahi`, `mast`, …) | `Hinglish` |
| 5 | `langdetect` says English | `English` |
| 6 | Plain-ASCII text, short or with English as a plausible candidate | `English` |
| 7 | `langdetect` confident (≥ 0.75) on another language | `Other` |
| 8 | Anything else | `Unknown` → multilingual model |

`langdetect`'s `DetectorFactory.seed` is pinned to `0` so results are
deterministic across runs.

Routing: `English` → VADER. Everything else → multilingual model. The single
exception is emoji-only text, which goes to VADER by design.

---

## Routes and API

### Pages
| Route | Purpose |
|---|---|
| `GET /analyze` | The unified analysis page. `?source=text\|dataset\|youtube\|blog` preselects a tab. |
| `GET /` | **302 →** `/analyze?source=text` (v1 compatibility) |
| `GET /bulk` | **302 →** `/analyze?source=dataset` (v1 compatibility) |

### JSON API
| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Engine availability: VADER, multilingual readiness, whether a YouTube key is configured. Never returns the key itself. |
| `POST /api/analyze/text` | Score one string. Synchronous. |
| `POST /api/dataset/inspect` | Upload a CSV, get column stats plus the suggested text column. |
| `POST /api/dataset/analyze` | Start a dataset job. Returns `job_id`. |
| `POST /api/youtube/analyze` | Start a YouTube job. Returns `job_id`. |
| `POST /api/blog/analyze` | Start a blog job. Returns `job_id`. |
| `GET /api/job/<job_id>` | Poll a job: `status`, `stage`, `progress`, and `result` when done. |

### v1 endpoints (preserved)
| Endpoint | Notes |
|---|---|
| `POST /predict` | Same request/response shape as v1, now language-aware. |
| `POST /bulk-upload` | Same shape as v1, still appends the original five `sentiment_*` columns. |
| `GET /download/<file_id>` | Unchanged. |

Errors return `{"success": false, "error": "<plain sentence>"}` with a real
HTTP status. Handlers exist for 400, 404, 413 and 500. **Tracebacks are
logged server-side and never sent to the browser** — including in async
jobs, where the exception is caught, logged and replaced with a
user-readable message.

---

## Project structure

```
sentiment-lab/
├── app.py                       # routes only — thin adapters over the engine
├── config.py                    # env-driven configuration, one source of truth
├── requirements.txt             # base install
├── requirements-multilingual.txt# optional transformers + torch
├── .env.example                 # annotated template, no real secrets
├── .gitignore
├── utils/
│   ├── preprocessing.py         # clean_text, validate_text_input, is_meaningful
│   ├── sentiment_analyzer.py    # cached VADER instance + classification
│   ├── language_detect.py       # script + lexicon + langdetect cascade
│   ├── multilingual.py          # lazy, cached, thread-safe transformer wrapper
│   ├── engine.py                # THE pipeline: routing, batching, statistics
│   ├── bulk_processor.py        # CSV reading, column detection, dataframe scoring
│   ├── youtube.py               # YouTube Data API v3 client
│   ├── blog.py                  # robots-aware, multi-strategy comment extraction
│   └── jobs.py                  # in-process async job registry with progress
├── templates/
│   ├── base.html
│   ├── analyze.html             # the one unified page
│   └── 404.html
├── static/
│   ├── css/style.css            # paper/ink design system, light + dark
│   └── js/analyze.js            # tabs, polling, charts, filters, sort, paging
├── tests/
│   ├── test_sentiments.csv
│   └── test_mixed.csv           # English + Hindi + Hinglish + a blank row
├── uploads/                     # incoming CSVs (gitignored)
└── results/                     # downloadable enriched CSVs (gitignored)
```

Charts are hand-built SVG and CSS in `analyze.js`. **No charting library was
added** — there are no frontend dependencies at all.

### Classification thresholds (unchanged from v1)

| Compound score | Label |
|---|---|
| `>= 0.05` | Positive |
| `-0.05` to `0.05` | Neutral |
| `<= -0.05` | Negative |

---

## Testing

### 1. Single text
Go to **Text**, enter `I absolutely love this product!` → **POSITIVE**,
compound ≈ `+0.70`, language `English`, engine `VADER`.

### 2. Dataset
Upload `tests/test_mixed.csv`. Expect: `review_text` auto-selected; 7 rows
in, 7 rows out; row 4 (blank) labelled `UNSCORED`; `I love this.` positive,
`This is terrible.` negative; every original column (`id`, `product`,
`rating`, `date`) still present; **Download Results CSV** returns all rows.

> Note on `"This is okay."` — VADER scores this `+0.23` (POSITIVE), because
> its lexicon rates *okay* as mildly positive. That is genuine, unmodified
> VADER behaviour carried over from v1, not a routing bug. For a neutral
> reading, `"It was okay, nothing special."` scores `0.00`.

### 3. YouTube
Requires `YOUTUBE_API_KEY`. Paste any public video URL with comments
enabled, set the budget to 100, and run. Expect real comments, correct
author/likes/published columns, sentiment per comment, the dashboard, and a
CSV download. Then test the failure paths: a malformed URL, a video with
comments disabled, and a deleted video — each should give a specific,
readable message.

### 4. Hindi / Hinglish routing
Enter `बहुत अच्छा वीडियो है।` and `Bhai kya hi video hai 😂`.

- With `requirements-multilingual.txt` installed: the language chip reads
  **Hindi** / **Hinglish**, `model` is the transformer id, `score_type` is
  `model_probability`, and the compound column shows `—`. This confirms the
  text was **not** forced through VADER.
- Without it: the language is still detected correctly, but the row is
  marked `VADER (fallback)` and the warning banner appears — the honest
  degradation path.

Either way you can verify the routing decision directly:

```bash
python -c "from utils.language_detect import detect_language; \
print(detect_language('Bhai kya hi video hai'))"
# {'language': 'Hinglish', 'code': 'hi-Latn', 'confidence': 0.95, 'method': 'romanised-markers'}
```

### 5. Backward compatibility
```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"text":"I love this"}'
curl -i localhost:5000/          # 302 -> /analyze?source=text
curl -i localhost:5000/bulk      # 302 -> /analyze?source=dataset
```

---

## Performance

Measured in this project on 5,000 rows, English text, VADER path:

| Workload | Time | Throughput |
|---|---|---|
| 5,000 rows, all unique | ~17.7 s | ~283 rows/s |
| 5,000 rows, duplicate-heavy | ~0.5 s | cache hits |

Language detection is the most expensive per-row step, so `detect_language`
is wrapped in a bounded `lru_cache` (8,192 entries, strings under 400 chars).
Real comment datasets repeat themselves heavily — "Nice video", "First",
emoji-only replies — so this is a large practical win with no effect on
results. Longer texts bypass the cache to bound memory.

Other resource decisions:

- VADER is one `lru_cache`d `SentimentIntensityAnalyzer`, created at startup.
- The multilingual model is one process-wide cached pipeline, loaded on first
  need behind a lock.
- Non-English text is batched (`MULTILINGUAL_BATCH_SIZE`, default 32);
  English is scored inline.
- One `requests.Session` per outbound source, so connections are reused.
- Dataset, YouTube and blog runs report real incremental progress
  (verified stepping 2% → 100%) rather than a fake spinner.
- Browser previews are capped at `PREVIEW_ROW_LIMIT` (1,000) and flagged
  `truncated`; the downloaded CSV always contains every row.

---

## Limitations

Being straight about what this does and does not do:

- **The multilingual model is not purpose-built for Hinglish.**
  `cardiffnlp/twitter-xlm-roberta-base-sentiment` is trained on multilingual
  tweets, which makes it far better than VADER on Devanagari Hindi and
  code-mixed text — but romanised Hinglish is under-represented in its
  training data. Treat Hinglish results as indicative, not authoritative.
  Swap in a Hinglish-specific checkpoint via `SENTIMENT_MODEL_NAME` if you
  find a better one.
- **VADER is English-only and lexicon-based.** It has no model of sarcasm,
  domain jargon, or negation spanning clauses.
- **Language detection on very short text is a heuristic.** Plain-ASCII
  strings under ~30 characters are assumed English, which means short,
  accent-free French or Spanish ("Muy bueno") is treated as English. This is
  a deliberate trade-off: it keeps thousands of short English comments off
  the transformer. Longer non-English text is detected correctly.
- **The Hinglish marker lexicon is finite** (~100 tokens). Hinglish written
  entirely in English-looking words will be classified English.
- **Blog comment extraction cannot reach JavaScript-rendered comments.**
  Disqus, Facebook Comments, Hyvor, Commento and giscus are detected and
  reported, not scraped. Sites with fully custom, non-semantic markup may
  yield nothing — the app says so rather than inventing rows.
- **YouTube quota is finite and shared** across your Google Cloud project.
  Server-side ceilings help, but a busy day can still exhaust 10,000 units.
- **Jobs are in-process and in-memory.** Restarting the server loses
  in-flight jobs. That is fine for single-instance use; multi-worker
  deployments need Celery/Redis or equivalent (see below).
- **`UNSCORED` is a real category.** Empty, NaN and punctuation-only rows
  are counted separately and excluded from percentage distributions, so
  percentages describe scored rows, not total rows.

---

## Production notes

Carried forward from v1 and still accurate:

- Run behind **gunicorn** rather than the dev server:
  `gunicorn -w 4 -b 0.0.0.0:8000 app:app`. Note that the in-memory job
  registry is per-worker; with multiple workers, move jobs to Celery + Redis
  or pin sessions.
- Move large CSV processing to a real job queue so uploads never block a
  request.
- Swap `uploads/` and `results/` for object storage with a TTL policy.
- Add rate limiting (e.g. Flask-Limiter) on the analysis endpoints if the app
  is public — the YouTube and blog routes make outbound requests on a user's
  behalf.
- Set `SECRET_KEY` and `FLASK_DEBUG=false`.

## Why text cleaning stays conservative

VADER (Hutto & Gilbert, *"VADER: A Parsimonious Rule-Based Model for
Sentiment Analysis of Social Media Text"*, ICWSM 2014) treats
capitalisation, punctuation emphasis (`"good!!!"`) and emoticons as
sentiment **intensifiers**. Lowercasing or stripping punctuation before
scoring throws away exactly the signal the model is built to use, so
`utils/preprocessing.py` removes only genuine noise: URLs, HTML entities and
redundant whitespace. The multilingual model is fed the same conservatively
cleaned text for consistency.

---

## Credits

- **VADER** — [cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, ICWSM 2014)
- **Multilingual model** — [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- **Language detection** — [langdetect](https://pypi.org/project/langdetect/)
- **YouTube comments** — [YouTube Data API v3](https://developers.google.com/youtube/v3/docs/commentThreads/list)
- **Blog comments** — [WordPress REST API](https://developer.wordpress.org/rest-api/reference/comments/) and [schema.org/Comment](https://schema.org/Comment)
