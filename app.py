"""
app.py
------
Flask entrypoint for Sentiment Lab v2 — Unified Content Intelligence.

Routes
======
Pages
  GET  /                        -> redirect to /analyze  (v1 route preserved)
  GET  /bulk                    -> redirect to /analyze?source=dataset  (v1)
  GET  /analyze                 -> the unified analysis UI

Unified API
  GET  /api/status              -> which engines are live
  POST /api/analyze/text        -> analyze a single piece of text
  POST /api/dataset/inspect     -> upload a CSV, get its columns back
  POST /api/dataset/analyze     -> analyze a chosen column (async job)
  POST /api/youtube/analyze     -> analyze YouTube comments (async job)
  POST /api/blog/analyze        -> analyze blog comments (async job)
  GET  /api/job/<job_id>        -> progress + result for an async job

Legacy API (v1 contracts, unchanged)
  POST /predict                 -> single-text JSON API
  POST /bulk-upload             -> one-shot CSV analysis
  GET  /download/<file_id>      -> download an enriched results CSV

Run locally with:
  python app.py
"""

import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from utils import jobs
from utils.blog import BlogError, fetch_blog_comments
from utils.blog import validate_url as validate_blog_url
from utils.bulk_processor import (
    DatasetError,
    V2_COLUMNS,
    analyze_dataframe,
    candidate_text_columns,
    find_text_column,
    process_dataframe,
    read_csv_safely,
    summarize,
)
from utils.engine import analyze_many, analyze_one, calculate_statistics, engine_status
from utils.preprocessing import validate_text_input
from utils.sentiment_analyzer import warm_up
from utils.youtube import (
    YouTubeError,
    api_key_configured,
    extract_video_id,
    fetch_comments,
    get_video_details,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sentimentlab")

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = config.RESULTS_FOLDER
app.config["JSON_SORT_KEYS"] = False

# VADER's lexicon is loaded once here; the multilingual model is NOT loaded
# at start-up — it is pulled in lazily on the first non-English text.
warm_up()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _fail(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _int_arg(payload: Dict, key: str, default: int, maximum: int) -> int:
    raw = payload.get(key, default)
    if raw in (None, "", "all", "All"):
        return maximum
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return maximum
    return min(value, maximum)


def _save_results_csv(df: pd.DataFrame) -> str:
    """Persist an enriched DataFrame and return its download id."""
    file_id = uuid.uuid4().hex
    path = os.path.join(config.RESULTS_FOLDER, f"sentiment_results_{file_id}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return file_id


def _prune_folder(folder: str, max_age_seconds: int = 6 * 3600) -> None:
    """Best-effort cleanup of old scratch/result files."""
    now = time.time()
    try:
        for name in os.listdir(folder):
            if name == ".gitkeep":
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
                os.remove(path)
    except OSError:
        pass


def _preview(records: List[Dict]) -> Dict:
    """Trim rows sent to the browser; the CSV download always has them all."""
    limit = config.PREVIEW_ROW_LIMIT
    return {
        "rows": records[:limit],
        "truncated": len(records) > limit,
        "preview_row_limit": limit,
    }


def _sanitize(records: List[Dict]) -> List[Dict]:
    """Replace NaN/NaT with empty strings so the JSON is valid."""
    clean = []
    for record in records:
        row = {}
        for key, value in record.items():
            if value is None:
                row[key] = None
            elif isinstance(value, float) and pd.isna(value):
                row[key] = None
            elif isinstance(value, (pd.Timestamp,)):
                row[key] = str(value)
            else:
                row[key] = value
        clean.append(row)
    return clean


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/analyze")
def analyze_page():
    """The single unified analysis interface."""
    status = engine_status()
    source = request.args.get("source", "text")
    if source not in {"text", "dataset", "youtube", "blog"}:
        source = "text"

    return render_template(
        "analyze.html",
        active_source=source,
        youtube_ready=api_key_configured(),
        youtube_choices=config.YOUTUBE_COMMENT_CHOICES,
        youtube_hard_limit=config.YOUTUBE_HARD_LIMIT,
        multilingual_model=status["multilingual_model"],
        multilingual_enabled=config.ENABLE_MULTILINGUAL,
        max_upload_mb=config.MAX_CONTENT_LENGTH // (1024 * 1024),
    )


@app.route("/")
def index():
    """v1 single-text page — now folded into the unified interface."""
    return redirect(url_for("analyze_page", source="text"))


@app.route("/bulk")
def bulk():
    """v1 bulk page — now the Dataset source on the unified interface."""
    return redirect(url_for("analyze_page", source="dataset"))


# --------------------------------------------------------------------------
# Unified API
# --------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    """Engine availability for the UI status chips.

    Reports only booleans and the (public) model name. The YouTube API key
    is never returned, echoed or logged -- only whether one is configured.
    This endpoint does NOT trigger the lazy multilingual load.
    """
    status = engine_status()
    return jsonify(
        {
            "success": True,
            "vader": True,
            "multilingual_enabled": config.ENABLE_MULTILINGUAL,
            "multilingual_ready": status["multilingual_ready"],
            "multilingual_loaded": status["multilingual_loaded"],
            "multilingual_model": status["multilingual_model"],
            "multilingual_note": status["multilingual_error"],
            "youtube_ready": api_key_configured(),
        }
    )


@app.route("/api/analyze/text", methods=["POST"])
def api_analyze_text():
    """Analyze one piece of text through the unified engine."""
    try:
        payload = request.get_json(silent=True) or {}
        raw_text = payload.get("text", "")

        is_valid, error = validate_text_input(raw_text)
        if not is_valid:
            return _fail(error)

        result = analyze_one(raw_text)
        if result.get("sentiment") == "UNSCORED":
            return _fail(
                "There is no analysable text in that input — it looks like "
                "only punctuation, symbols or a bare URL."
            )

        return jsonify(
            {
                "success": True,
                "source": "text",
                "result": result,
                "summary": calculate_statistics([result]),
            }
        )
    except Exception:
        logger.exception("Unhandled error in /api/analyze/text")
        return _fail(
            "Something went wrong while analyzing that text. Please try again.",
            500,
        )


@app.route("/api/dataset/inspect", methods=["POST"])
def api_dataset_inspect():
    """Accept a CSV upload and report its columns for selection."""
    try:
        _prune_folder(config.UPLOAD_FOLDER)

        if "file" not in request.files:
            return _fail("No file was uploaded.")

        uploaded = request.files["file"]
        if not uploaded.filename:
            return _fail("No file was selected.")
        if not _allowed_file(uploaded.filename):
            return _fail("Only .csv files are supported.")

        upload_id = uuid.uuid4().hex
        filename = secure_filename(uploaded.filename) or "upload.csv"
        path = os.path.join(config.UPLOAD_FOLDER, f"{upload_id}_{filename}")
        uploaded.save(path)

        try:
            df = read_csv_safely(path)
        except DatasetError as exc:
            os.remove(path)
            return _fail(str(exc))

        if df.empty or not len(df.columns):
            os.remove(path)
            return _fail("That CSV has no rows or no header row.")

        if len(df) > config.MAX_ROWS:
            df = df.head(config.MAX_ROWS)

        suggested = find_text_column(df)
        columns = candidate_text_columns(df)

        return jsonify(
            {
                "success": True,
                "upload_id": upload_id,
                "filename": filename,
                "rows": int(len(df)),
                "columns": columns,
                "suggested_column": suggested,
                "row_choices": [
                    n for n in (100, 500, 1000, 5000) if n < len(df)
                ],
                "preview": _sanitize(df.head(5).fillna("").to_dict(orient="records")),
            }
        )
    except Exception:
        logger.exception("Unhandled error in /api/dataset/inspect")
        return _fail("Could not read that file. Please try another CSV.", 500)


def _find_upload(upload_id: str) -> str:
    safe_id = secure_filename(upload_id or "")
    if not safe_id:
        raise DatasetError("That upload is no longer available. Please re-upload.")
    for name in os.listdir(config.UPLOAD_FOLDER):
        if name.startswith(safe_id + "_"):
            return os.path.join(config.UPLOAD_FOLDER, name)
    raise DatasetError("That upload has expired. Please upload the file again.")


@app.route("/api/dataset/analyze", methods=["POST"])
def api_dataset_analyze():
    """Start an async job that scores the chosen column of an upload."""
    payload = request.get_json(silent=True) or {}
    text_column = (payload.get("text_column") or "").strip()
    if not text_column:
        return _fail("Please choose which column contains the text.")

    try:
        path = _find_upload(payload.get("upload_id", ""))
    except DatasetError as exc:
        return _fail(str(exc))

    limit = _int_arg(payload, "limit", config.MAX_ROWS, config.MAX_ROWS)

    def worker(handle: jobs.JobHandle) -> Dict:
        handle.stage("Reading CSV…")
        df = read_csv_safely(path)

        handle.stage("Analyzing rows…")
        enriched, results = analyze_dataframe(
            df, text_column, limit=limit, progress=handle.progress
        )

        handle.stage("Building results…")
        summary = calculate_statistics(results)
        file_id = _save_results_csv(enriched)

        records = _sanitize(enriched.fillna("").to_dict(orient="records"))
        return {
            "source": "dataset",
            "summary": summary,
            "columns": [str(c) for c in enriched.columns],
            "analysis_columns": V2_COLUMNS,
            "text_column": text_column,
            "file_id": file_id,
            "meta": {"filename": os.path.basename(path).split("_", 1)[-1]},
            **_preview(records),
        }

    job_id = jobs.start(worker, label="dataset")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/youtube/analyze", methods=["POST"])
def api_youtube_analyze():
    """Start an async job that fetches and scores YouTube comments."""
    payload = request.get_json(silent=True) or {}

    if not api_key_configured():
        return _fail(
            "No YouTube API key is configured. Add YOUTUBE_API_KEY to your "
            ".env file (see .env.example) and restart the app."
        )

    try:
        video_id = extract_video_id(payload.get("url", ""))
    except YouTubeError as exc:
        return _fail(str(exc))

    limit = _int_arg(payload, "limit", 100, config.YOUTUBE_HARD_LIMIT)
    include_replies = bool(payload.get("include_replies", True))
    order = payload.get("order", "relevance")

    def worker(handle: jobs.JobHandle) -> Dict:
        handle.stage("Looking up the video…")
        video = get_video_details(video_id)

        handle.stage("Fetching comments…")
        comments = fetch_comments(
            video_id,
            limit=limit,
            include_replies=include_replies,
            order=order,
            progress=handle.progress,
        )

        handle.stage(f"Analyzing {len(comments):,} comments…")
        results = analyze_many(
            [c["comment"] for c in comments], progress=handle.progress
        )

        handle.stage("Building results…")
        rows = []
        for comment, result in zip(comments, results):
            row = {
                "comment_id": comment["comment_id"],
                "author": comment["author"],
                "comment": comment["comment"],
                "published_at": comment["published_at"],
                "like_count": comment["like_count"],
                "is_reply": comment["is_reply"],
            }
            for field in V2_COLUMNS:
                row[field] = result.get(field)
            rows.append(row)

        summary = calculate_statistics(results)
        file_id = _save_results_csv(pd.DataFrame(rows))

        return {
            "source": "youtube",
            "summary": summary,
            "columns": list(rows[0].keys()) if rows else [],
            "analysis_columns": V2_COLUMNS,
            "text_column": "comment",
            "file_id": file_id,
            "meta": video,
            **_preview(rows),
        }

    job_id = jobs.start(worker, label="youtube")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/blog/analyze", methods=["POST"])
def api_blog_analyze():
    """Start an async job that reads and scores blog/article comments."""
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    if not url:
        return _fail("Please paste a blog or article URL.")

    # Validate synchronously so obviously-bad or unsafe URLs fail instantly
    # with a clear message, instead of the user watching a progress bar for a
    # job that was never going to run. The same check runs again inside the
    # worker, so this is a UX shortcut, not the security boundary.
    try:
        url = validate_blog_url(url)
    except BlogError as exc:
        return _fail(str(exc))

    limit = _int_arg(payload, "limit", 500, 2000)

    def worker(handle: jobs.JobHandle) -> Dict:
        handle.stage("Checking robots.txt and fetching the page…")
        found = fetch_blog_comments(url, limit=limit)
        comments = found["comments"]

        handle.stage(f"Analyzing {len(comments):,} comments…")
        results = analyze_many(
            [c["comment"] for c in comments], progress=handle.progress
        )

        handle.stage("Building results…")
        rows = []
        for comment, result in zip(comments, results):
            row = {
                "comment_id": comment["comment_id"],
                "author": comment["author"],
                "comment": comment["comment"],
                "published_at": comment["published_at"],
            }
            for field in V2_COLUMNS:
                row[field] = result.get(field)
            rows.append(row)

        summary = calculate_statistics(results)
        file_id = _save_results_csv(pd.DataFrame(rows))

        return {
            "source": "blog",
            "summary": summary,
            "columns": list(rows[0].keys()) if rows else [],
            "analysis_columns": V2_COLUMNS,
            "text_column": "comment",
            "file_id": file_id,
            "meta": {
                "title": found["title"],
                "url": found["url"],
                "method": found["method"],
            },
            **_preview(rows),
        }

    job_id = jobs.start(worker, label="blog")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    """Poll an async job for progress and, once finished, its result."""
    job = jobs.get(secure_filename(job_id or ""))
    if not job:
        return _fail("That analysis job is no longer available.", 404)

    response = make_response(jsonify({"success": job["status"] != "error", **job}))
    response.headers["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------------------------------
# Legacy API (v1 contracts preserved)
# --------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """v1 single-text API.

    Still returns `{"success": true, "result": {...}}` with all the original
    keys (text, sentiment, compound, positive, neutral, negative). The result
    is now a superset — it also carries `language`, `model` and `polarity` —
    and non-English text is routed to the multilingual model, where
    `compound` is null because that model does not produce one.
    """
    try:
        payload = request.get_json(silent=True) or {}
        raw_text = payload.get("text", "")

        is_valid, error = validate_text_input(raw_text)
        if not is_valid:
            return _fail(error)

        result = analyze_one(raw_text)
        return jsonify({"success": True, "result": result})
    except Exception:
        logger.exception("Unhandled error in /predict")
        return _fail(
            "Something went wrong while analyzing that text. Please try again.",
            500,
        )


@app.route("/bulk-upload", methods=["POST"])
def bulk_upload():
    """v1 one-shot CSV API — unchanged output shape and VADER columns."""
    temp_path: Optional[str] = None
    try:
        if "file" not in request.files:
            return _fail("No file was uploaded.")

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return _fail("No file was selected.")
        if not _allowed_file(uploaded_file.filename):
            return _fail("Only .csv files are supported.")

        filename = secure_filename(uploaded_file.filename)
        temp_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}"
        )
        uploaded_file.save(temp_path)

        try:
            df = read_csv_safely(temp_path)
        except DatasetError as exc:
            return _fail(str(exc))

        if df.empty:
            return _fail("The uploaded CSV has no rows.")

        text_column = find_text_column(df)
        if text_column is None:
            return _fail(
                "No 'comment' or 'text' column was found. Please include a "
                "column named 'comment', 'text', or 'review'."
            )

        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        if df.empty:
            return _fail(
                f"The '{text_column}' column doesn't contain any usable text."
            )

        enriched_df = process_dataframe(df, text_column)
        summary = summarize(enriched_df)
        file_id = _save_results_csv(enriched_df)

        preview_records = _sanitize(
            enriched_df.head(config.PREVIEW_ROW_LIMIT)
            .fillna("")
            .to_dict(orient="records")
        )

        return jsonify(
            {
                "success": True,
                "summary": summary,
                "preview": preview_records,
                "columns": [str(c) for c in enriched_df.columns],
                "text_column": text_column,
                "file_id": file_id,
                "truncated": len(enriched_df) > config.PREVIEW_ROW_LIMIT,
                "preview_row_limit": config.PREVIEW_ROW_LIMIT,
            }
        )
    except Exception:
        logger.exception("Unhandled error in /bulk-upload")
        return _fail(
            "Something went wrong while processing the file. Please try again.",
            500,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/download/<file_id>")
def download(file_id):
    """Download an enriched results CSV produced by any source."""
    _prune_folder(config.RESULTS_FOLDER, max_age_seconds=24 * 3600)

    safe_id = secure_filename(file_id or "")
    filename = f"sentiment_results_{safe_id}.csv"
    file_path = os.path.join(config.RESULTS_FOLDER, filename)

    if not safe_id or not os.path.exists(file_path):
        abort(404)

    return send_from_directory(
        config.RESULTS_FOLDER,
        filename,
        as_attachment=True,
        download_name="sentiment_analysis_results.csv",
    )


# --------------------------------------------------------------------------
# Error handlers — users never see a traceback
# --------------------------------------------------------------------------
def _wants_json() -> bool:
    return request.path.startswith(
        ("/api/", "/predict", "/bulk-upload", "/download")
    ) or request.accept_mimetypes.best == "application/json"


@app.errorhandler(400)
def bad_request(_error):
    if _wants_json():
        return _fail("That request could not be understood.", 400)
    return render_template("404.html", code=400,
                           message="That request could not be understood."), 400


@app.errorhandler(404)
def not_found(_error):
    if _wants_json():
        return _fail("Resource not found.", 404)
    return render_template("404.html", code=404,
                           message="That page doesn't exist."), 404


@app.errorhandler(413)
def file_too_large(_error):
    return _fail(
        f"File is too large. Maximum upload size is "
        f"{config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB.",
        413,
    )


@app.errorhandler(500)
def server_error(_error):
    if _wants_json():
        return _fail("Internal server error.", 500)
    return render_template("404.html", code=500,
                           message="Something went wrong on our side."), 500


if __name__ == "__main__":
    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
        use_reloader=False
    )
