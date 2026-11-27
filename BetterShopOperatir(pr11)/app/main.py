"""FastAPI routes and the public error contract."""

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import budget, cache, costs, gateway, jobs, limits, samples, telemetry

logger = logging.getLogger(__name__)
app = FastAPI(title="Помічник оператора — ПР11")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
telemetry.configure_logging()


class DraftIn(BaseModel):
    """A sample request and an in-page demo client identifier."""

    request_id: str = Field(min_length=1, max_length=64)
    client: str = Field(default="anonymous", min_length=1, max_length=128)
    fresh: bool = False


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
                    *, retryable: bool = False, retry_after: float | None = None,
                    reset_at: str | None = None) -> JSONResponse:
    telemetry.log_event(
        "http.error", level=logging.WARNING if status < 500 else logging.ERROR,
        request_id=request_id, kind=kind, status=status,
    )
    headers = {}
    headers["X-Request-ID"] = request_id
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    body = _error_body(kind, message, retryable, request_id)
    if reset_at:
        body["error"]["reset_at"] = reset_at
    return JSONResponse(
        status_code=status,
        content=body,
        headers=headers,
    )


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    token = telemetry.set_request_id(request_id)
    started = time.monotonic()
    telemetry.log_event(
        "http.request", request_id=request_id, method=request.method,
        path=request.url.path,
    )
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        telemetry.log_event(
            "http.response", request_id=request_id, method=request.method,
            path=request.url.path, status=response.status_code,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
        )
        return response
    finally:
        telemetry.reset_request_id(token)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    return _error_response(
        "bad_request", "Перевірте формат запиту.", 400,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        kind, message = "not_found", "Ресурс не знайдено."
    elif exc.status_code == 405:
        kind, message = "bad_request", "Метод запиту не підтримується."
    else:
        kind, message = "unknown", "Запит не вдалося виконати."
    response = _error_response(
        kind, message, exc.status_code,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
    )
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(limits.RateLimited)
async def rate_limit_handler(_: Request, exc: limits.RateLimited) -> JSONResponse:
    return _error_response(
        "rate_limit",
        str(exc),
        429,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
        retryable=True,
        retry_after=exc.retry_after,
    )


@app.exception_handler(RequestNotFound)
async def request_not_found_handler(_: Request, __: RequestNotFound) -> JSONResponse:
    return _error_response(
        "not_found", "Звернення не знайдено.", 404,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
    )


@app.exception_handler(jobs.DraftNotFound)
async def draft_not_found_handler(_: Request, __: jobs.DraftNotFound) -> JSONResponse:
    return _error_response(
        "not_found", "Чернетку не знайдено.", 404,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
    )


@app.exception_handler(jobs.DraftConflict)
async def draft_conflict_handler(_: Request, __: jobs.DraftConflict) -> JSONResponse:
    return _error_response(
        "conflict",
        "Неможливо скасувати чернетку: вона вже завершена або скасована.",
        409,
        telemetry.current_request_id() or uuid.uuid4().hex[:12],
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = telemetry.current_request_id() or uuid.uuid4().hex[:12]
    logger.exception("Unhandled API error request_id=%s path=%s", request_id, request.url.path)
    telemetry.log_event(
        "http.exception", level=logging.ERROR, request_id=request_id,
        path=request.url.path, exception_type=type(exc).__name__,
    )
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
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "budget": budget.snapshot(),
        "costs": {
            "input_usd_per_million": costs.INPUT_USD_PER_MILLION,
            "output_usd_per_million": costs.OUTPUT_USD_PER_MILLION,
            "cached_input_usd_per_million": costs.CACHED_INPUT_USD_PER_MILLION,
        },
        "cache": {
            "enabled": cache.ENABLED,
            "ttl_seconds": cache.TTL_SECONDS,
            "max_entries": cache.MAX_ENTRIES,
        },
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(sample) for sample in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    sample = samples.get(payload.request_id)
    if sample is None:
        raise RequestNotFound(payload.request_id)
    limits.acquire(payload.client)
    draft = jobs.start(
        sample, payload.client,
        request_id=telemetry.current_request_id(),
        fresh=payload.fresh,
    )
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
    telemetry.log_event(
        "draft.viewed", request_id=telemetry.current_request_id(),
        draft_id=draft_id,
    )
    return snapshot


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    if jobs.get(draft_id) is None:
        raise jobs.DraftNotFound(draft_id)
    telemetry.log_event(
        "draft.events_subscribed", request_id=telemetry.current_request_id(),
        draft_id=draft_id,
    )
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
    draft = jobs.cancel(draft_id, request_id=telemetry.current_request_id())
    snapshot = jobs.snapshot(draft.draft_id)
    if snapshot is None:
        raise jobs.DraftNotFound(draft_id)
    return snapshot
