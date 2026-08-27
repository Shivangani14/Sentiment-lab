"""
utils/jobs.py
-------------
A tiny in-process job registry so long analyses report real progress.

Dataset, YouTube and Blog analyses can take seconds to minutes (fetching
pages, running a transformer over thousands of comments). Rather than
blocking a request until it finishes, routes start a job here and the
browser polls `/api/job/<id>` for stage + percentage updates.

This is deliberately in-memory and single-process — the right size for a
portfolio/local app. For multi-worker production, swap this module for
Celery/RQ with Redis; the route contract would not change.
"""

import threading
import time
import traceback
import uuid
from typing import Callable, Dict, Optional

_LOCK = threading.RLock()
_JOBS: Dict[str, Dict] = {}
_MAX_JOBS = 50
_TTL_SECONDS = 3600


def _prune() -> None:
    now = time.time()
    stale = [
        jid
        for jid, job in _JOBS.items()
        if now - job["updated_at"] > _TTL_SECONDS
    ]
    for jid in stale:
        _JOBS.pop(jid, None)
    if len(_JOBS) > _MAX_JOBS:
        for jid in sorted(_JOBS, key=lambda j: _JOBS[j]["updated_at"])[
            : len(_JOBS) - _MAX_JOBS
        ]:
            _JOBS.pop(jid, None)


class JobHandle:
    """Passed to the worker so it can publish progress."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def stage(self, message: str) -> None:
        with _LOCK:
            job = _JOBS.get(self.job_id)
            if job:
                job["stage"] = message
                job["updated_at"] = time.time()

    def progress(self, done: int, total: Optional[int]) -> None:
        with _LOCK:
            job = _JOBS.get(self.job_id)
            if not job:
                return
            job["done"] = int(done)
            job["total"] = int(total) if total else None
            job["percent"] = (
                round(min(100.0, (done / total) * 100), 1) if total else None
            )
            job["updated_at"] = time.time()


def start(worker: Callable[[JobHandle], Dict], label: str = "") -> str:
    """Run *worker* on a daemon thread and return its job id."""
    job_id = uuid.uuid4().hex
    with _LOCK:
        _prune()
        _JOBS[job_id] = {
            "id": job_id,
            "label": label,
            "status": "running",
            "stage": "Starting…",
            "done": 0,
            "total": None,
            "percent": None,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    handle = JobHandle(job_id)

    def run() -> None:
        try:
            result = worker(handle)
            with _LOCK:
                job = _JOBS.get(job_id)
                if job:
                    job["status"] = "done"
                    job["stage"] = "Complete"
                    job["percent"] = 100.0
                    job["result"] = result
                    job["updated_at"] = time.time()
        except Exception as exc:  # noqa: BLE001 - message is curated below
            message = str(exc) or "Something went wrong during analysis."
            with _LOCK:
                job = _JOBS.get(job_id)
                if job:
                    job["status"] = "error"
                    job["stage"] = "Failed"
                    job["error"] = message
                    job["trace"] = traceback.format_exc()
                    job["updated_at"] = time.time()

    threading.Thread(target=run, daemon=True, name=f"job-{job_id[:8]}").start()
    return job_id


def get(job_id: str) -> Optional[Dict]:
    """Public view of a job — never exposes the stack trace."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return {
            "id": job["id"],
            "status": job["status"],
            "stage": job["stage"],
            "done": job["done"],
            "total": job["total"],
            "percent": job["percent"],
            "result": job["result"],
            "error": job["error"],
        }
