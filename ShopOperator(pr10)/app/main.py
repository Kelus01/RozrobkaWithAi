"""FastAPI routes and the public error contract."""

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import gateway, jobs, limits, samples

logger = logging.getLogger(__name__)
app = FastAPI(title="Помічник оператора — ПР10")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    """A sample request and an in-page demo client identifier."""

    request_id: str = Field(min_length=1, max_length=64)
    client: str = Field(default="anonymous", min_length=1, max_length=128)


class RequestNotFound(Exception):
    """A sample request identifier does not exist."""


def _error_body(kind: str, message: str, retryable: bool, request_id: str) -> dict:
    return {
        "error": {
            "kind": kind,
            "message": message,
            "retryable": retryable,
            "request_id": request_id,
        }
    }


def _error_response(kind: str, message: str, status: int, request_id: str,
                    *, retryable: bool = False, retry_after: float | None = None) -> JSONResponse:
    logger.info("API error request_id=%s kind=%s status=%s", request_id, kind, status)
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    return JSONResponse(
        status_code=status,
        content=_error_body(kind, message, retryable, request_id),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    return _error_response(
        "bad_request", "Перевірте формат запиту.", 400, uuid.uuid4().hex[:8]
    )


@app.exception_handler(limits.RateLimited)
async def rate_limit_handler(_: Request, exc: limits.RateLimited) -> JSONResponse:
    return _error_response(
        "rate_limit",
        str(exc),
        429,
        uuid.uuid4().hex[:8],
        retryable=True,
        retry_after=exc.retry_after,
    )


@app.exception_handler(RequestNotFound)
async def request_not_found_handler(_: Request, __: RequestNotFound) -> JSONResponse:
    return _error_response(
        "not_found", "Звернення не знайдено.", 404, uuid.uuid4().hex[:8]
    )


@app.exception_handler(jobs.DraftNotFound)
async def draft_not_found_handler(_: Request, __: jobs.DraftNotFound) -> JSONResponse:
    return _error_response(
        "not_found", "Чернетку не знайдено.", 404, uuid.uuid4().hex[:8]
    )


@app.exception_handler(jobs.DraftConflict)
async def draft_conflict_handler(_: Request, __: jobs.DraftConflict) -> JSONResponse:
    return _error_response(
        "conflict",
        "Неможливо скасувати чернетку: вона вже завершена або скасована.",
        409,
        uuid.uuid4().hex[:8],
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = uuid.uuid4().hex[:8]
    logger.exception("Unhandled API error request_id=%s path=%s", request_id, request.url.path)
    return _error_response(
        "unknown",
        "Сталася внутрішня помилка. Спробуйте пізніше.",
        500,
        request_id,
    )


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(sample) for sample in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    sample = samples.get(payload.request_id)
    if sample is None:
        raise RequestNotFound(payload.request_id)
    limits.acquire(payload.client)
    draft = jobs.start(sample, payload.client)
    snapshot = jobs.snapshot(draft.draft_id)
    if snapshot is None:
        raise RuntimeError("Draft disappeared immediately after being registered")
    return snapshot


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return jobs.snapshots()


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    snapshot = jobs.snapshot(draft_id)
    if snapshot is None:
        raise jobs.DraftNotFound(draft_id)
    return snapshot


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    if jobs.get(draft_id) is None:
        raise jobs.DraftNotFound(draft_id)
    try:
        since = max(0, int(request.headers.get("last-event-id") or 0))
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    draft = jobs.cancel(draft_id)
    snapshot = jobs.snapshot(draft.draft_id)
    if snapshot is None:
        raise jobs.DraftNotFound(draft_id)
    return snapshot
