"""In-memory draft registry, background execution, cancellation and SSE events."""

import datetime
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterator

from .providers.base import ProviderError
from . import telemetry

logger = logging.getLogger(__name__)

TERMINAL = {"done", "failed", "cancelled"}
MAX_RETAINED_DRAFTS = 100


class DraftNotFound(Exception):
    """A requested draft does not exist in this process."""


class DraftConflict(Exception):
    """The requested transition is not valid for the draft's current state."""


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str
    trace_request_id: str = ""
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    cost_estimated: bool = False
    cached: bool = False
    cache_key: str | None = None
    queue_ms: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancellations: dict[str, threading.Event] = {}
_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _emit_locked(draft: Draft, event: str, data: dict | None = None) -> None:
    draft.events.append({
        "id": len(draft.events) + 1,
        "event": event,
        "data": data or {},
    })
    draft.updated_at = _now()
    _changed.notify_all()


def _prune_locked() -> None:
    if len(_order) <= MAX_RETAINED_DRAFTS:
        return
    for draft_id in reversed(_order):
        if len(_order) <= MAX_RETAINED_DRAFTS:
            break
        draft = _drafts.get(draft_id)
        if draft is not None and draft.status in TERMINAL:
            _order.remove(draft_id)
            _drafts.pop(draft_id, None)


def new_draft(request_id: str, client: str, trace_request_id: str | None = None) -> Draft:
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        trace_request_id=trace_request_id or uuid.uuid4().hex[:12],
        created_at=_now(),
        updated_at=_now(),
    )
    with _changed:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        _emit_locked(draft, "status", {"status": "queued"})
        _prune_locked()
    telemetry.log_event(
        "draft.created", request_id=draft.trace_request_id,
        draft_id=draft.draft_id, request_sample_id=draft.request_id,
        client_id_hash=telemetry.hash_identifier(client),
    )
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def snapshot(draft_id: str) -> dict | None:
    with _lock:
        draft = _drafts.get(draft_id)
        if draft is None:
            return None
        data = asdict(draft)
        data["events"] = len(draft.events)
        return data


def snapshots(limit: int = 50) -> list[dict]:
    with _lock:
        result = []
        for draft_id in _order[:limit]:
            draft = _drafts.get(draft_id)
            if draft is not None:
                data = asdict(draft)
                data["events"] = len(draft.events)
                result.append(data)
        return result


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [_drafts[draft_id] for draft_id in _order[:limit] if draft_id in _drafts]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    with _changed:
        _emit_locked(draft, event, data)


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Replay events after ``since`` and then yield new events as they arrive."""
    cursor = max(0, since)
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[cursor:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[cursor:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        cursor += len(pending)


def _finish_error(draft: Draft, error: dict) -> None:
    with _changed:
        if draft.status in TERMINAL:
            return
        draft.status = "failed"
        draft.error = error
        draft.cost_usd = float(error.get("cost_usd") or 0.0)
        draft.cost_estimated = bool(error.get("cost_estimated", False))
        _emit_locked(draft, "error", error)


def _run(draft: Draft, sample: dict, cancellation: threading.Event,
         fresh: bool, queued_monotonic: float) -> None:
    from .drafting import build_request, check
    from .gateway import generate

    context_token = telemetry.set_request_id(draft.trace_request_id)
    started = time.monotonic()
    draft.queue_ms = round(max(0.0, started - queued_monotonic) * 1000, 3)
    first_token: float | None = None

    with _changed:
        if draft.status == "cancelled":
            _cancellations.pop(draft.draft_id, None)
            telemetry.reset_request_id(context_token)
            return
        draft.status = "running"
        _emit_locked(draft, "status", {"status": "running"})
    telemetry.log_event(
        "draft.started", request_id=draft.trace_request_id,
        draft_id=draft.draft_id, client_id_hash=telemetry.hash_identifier(draft.client),
        queue_ms=draft.queue_ms,
    )

    def on_delta(text: str) -> None:
        nonlocal first_token
        with _changed:
            if draft.status in TERMINAL or cancellation.is_set():
                return
            if first_token is None:
                first_token = time.monotonic()
            draft.text += text
            _emit_locked(draft, "delta", {"text": text})

    def on_status(data: dict) -> None:
        with _changed:
            if draft.status in TERMINAL or cancellation.is_set():
                return
            if data.get("reset"):
                draft.text = ""
            if isinstance(data.get("attempt"), int):
                draft.attempts = data["attempt"]
            if data.get("provider"):
                draft.provider = data["provider"]
            if data.get("model"):
                draft.model = data["model"]
            if "fallback" in data:
                draft.fallback = bool(data["fallback"])
            if data.get("cache_hit"):
                draft.cached = True
            if data.get("cache_key"):
                draft.cache_key = data["cache_key"]
            event_data = {"status": draft.status, **data}
            _emit_locked(draft, "status", event_data)

    try:
        draft_request = build_request(sample)
        result = generate(
            draft_request,
            on_delta=on_delta,
            on_status=on_status,
            is_cancelled=cancellation.is_set,
            request_id=draft.trace_request_id,
            draft_id=draft.draft_id,
            client=draft.client,
            fresh=fresh,
            queued_ms=draft.queue_ms,
        )
        if cancellation.is_set():
            return

        finished = time.monotonic()
        elapsed = {
            "first_token": round(first_token - started, 3) if first_token is not None else None,
            "total": round(finished - started, 3),
        }
        warnings = check(result.text, sample)
        with _changed:
            if cancellation.is_set() or draft.status in TERMINAL:
                return
            draft.status = "done"
            draft.text = result.text
            draft.provider = result.provider
            draft.model = result.model
            draft.usage = result.usage
            draft.elapsed = elapsed
            draft.warnings = warnings
            draft.cost_usd = result.cost_usd
            draft.cost_estimated = result.cost_estimated
            draft.cached = result.cached
            draft.cache_key = result.cache_key
            _emit_locked(draft, "done", {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "usage": result.usage,
                "elapsed": elapsed,
                "attempts": draft.attempts,
                "fallback": draft.fallback,
                "warnings": warnings,
                "cost_usd": draft.cost_usd,
                "cost_estimated": draft.cost_estimated,
                "cached": draft.cached,
                "cache_key": draft.cache_key,
                "queue_ms": draft.queue_ms,
            })
        telemetry.log_event(
            "draft.done", request_id=draft.trace_request_id,
            draft_id=draft.draft_id, client_id_hash=telemetry.hash_identifier(draft.client),
            provider=result.provider, model=result.model, attempts=draft.attempts,
            status="done", fallback=draft.fallback, cached=result.cached,
            prompt_version=draft_request.metadata["prompt_version"],
            duration_ms=round(elapsed["total"] * 1000, 3),
            first_token_ms=(
                round(elapsed["first_token"] * 1000, 3)
                if elapsed["first_token"] is not None else None
            ),
            input_tokens=(result.usage or {}).get("prompt_tokens"),
            output_tokens=(result.usage or {}).get("completion_tokens"),
            cost_usd=result.cost_usd, cost_estimated=result.cost_estimated,
        )
    except ProviderError as exc:
        if exc.kind == "cancelled" or cancellation.is_set():
            return
        logger.warning(
            "Draft %s failed: kind=%s retryable=%s detail=%s",
            draft.draft_id, exc.kind, exc.retryable, exc.detail,
        )
        _finish_error(draft, {
            "kind": exc.kind,
            "message": exc.message,
            "retryable": exc.retryable,
            "request_id": draft.trace_request_id,
            "draft_id": draft.draft_id,
            "cost_usd": exc.cost_usd,
            "cost_estimated": exc.cost_estimated,
            "reset_at": exc.reset_at,
        })
        telemetry.log_event(
            "draft.failed", level=40, request_id=draft.trace_request_id,
            draft_id=draft.draft_id, client_id_hash=telemetry.hash_identifier(draft.client),
            provider=draft.provider, model=draft.model, attempts=draft.attempts,
            status="failed", kind=exc.kind, fallback=draft.fallback,
            prompt_version="drafts-v1",
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            cost_usd=exc.cost_usd, cost_estimated=exc.cost_estimated,
        )
    except Exception:
        logger.exception("Unexpected background failure for draft %s", draft.draft_id)
        _finish_error(draft, {
            "kind": "unknown",
            "message": "Не вдалося створити чернетку через внутрішню помилку.",
            "retryable": False,
            "request_id": draft.trace_request_id,
            "draft_id": draft.draft_id,
        })
        telemetry.log_event(
            "draft.failed", level=40, request_id=draft.trace_request_id,
            draft_id=draft.draft_id, client_id_hash=telemetry.hash_identifier(draft.client),
            status="failed", kind="unknown", attempts=draft.attempts,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            exception_type="unexpected",
        )
    finally:
        with _changed:
            if draft.status in {"queued", "running"}:
                draft.status = "failed"
                draft.error = {
                    "kind": "unknown",
                    "message": "Генерація завершилася без результату.",
                    "retryable": False,
                    "request_id": draft.trace_request_id,
                    "draft_id": draft.draft_id,
                }
                _emit_locked(draft, "error", draft.error)
            _cancellations.pop(draft.draft_id, None)
        telemetry.reset_request_id(context_token)


def start(sample: dict, client: str, request_id: str | None = None,
          fresh: bool = False) -> Draft:
    """Register a draft and return before its background worker completes."""
    if not sample or not sample.get("id"):
        raise DraftNotFound("Unknown support request")

    draft = new_draft(sample["id"], client, request_id)
    queued_monotonic = time.monotonic()
    cancellation = threading.Event()
    with _lock:
        _cancellations[draft.draft_id] = cancellation

    worker = threading.Thread(
        target=_run,
        args=(draft, sample, cancellation, fresh, queued_monotonic),
        name=f"draft-{draft.draft_id}",
        daemon=True,
    )
    try:
        worker.start()
    except RuntimeError:
        logger.exception("Failed to start worker for draft %s", draft.draft_id)
        _finish_error(draft, {
            "kind": "unknown",
            "message": "Не вдалося запустити генерацію.",
            "retryable": True,
            "request_id": draft.trace_request_id,
            "draft_id": draft.draft_id,
        })
        with _lock:
            _cancellations.pop(draft.draft_id, None)
    return draft


def cancel(draft_id: str, request_id: str | None = None) -> Draft:
    with _changed:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise DraftNotFound(draft_id)
        if draft.status in TERMINAL:
            raise DraftConflict(draft_id)
        cancellation = _cancellations.get(draft_id)
        if cancellation is None:
            raise DraftConflict(draft_id)
        cancellation.set()
        draft.status = "cancelled"
        _emit_locked(draft, "cancelled", {})
        telemetry.log_event(
            "draft.cancelled", request_id=request_id or draft.trace_request_id,
            draft_id=draft.draft_id,
            origin_request_id=draft.trace_request_id,
            client_id_hash=telemetry.hash_identifier(draft.client),
        )
        return draft
