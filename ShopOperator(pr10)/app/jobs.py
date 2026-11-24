"""In-memory draft registry, background execution, cancellation and SSE events."""

import datetime
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterator

from .providers.base import ProviderError

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


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        created_at=_now(),
        updated_at=_now(),
    )
    with _changed:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        _emit_locked(draft, "status", {"status": "queued"})
        _prune_locked()
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
        _emit_locked(draft, "error", error)


def _run(draft: Draft, sample: dict, cancellation: threading.Event) -> None:
    import time

    from .drafting import build_request, check
    from .gateway import generate

    started = time.monotonic()
    first_token: float | None = None

    with _changed:
        if draft.status == "cancelled":
            _cancellations.pop(draft.draft_id, None)
            return
        draft.status = "running"
        _emit_locked(draft, "status", {"status": "running"})

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
            event_data = {"status": draft.status, **data}
            _emit_locked(draft, "status", event_data)

    try:
        result = generate(
            build_request(sample),
            on_delta=on_delta,
            on_status=on_status,
            is_cancelled=cancellation.is_set,
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
            _emit_locked(draft, "done", {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "usage": result.usage,
                "elapsed": elapsed,
                "attempts": draft.attempts,
                "fallback": draft.fallback,
                "warnings": warnings,
            })
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
            "request_id": draft.draft_id,
        })
    except Exception:
        logger.exception("Unexpected background failure for draft %s", draft.draft_id)
        _finish_error(draft, {
            "kind": "unknown",
            "message": "Не вдалося створити чернетку через внутрішню помилку.",
            "retryable": False,
            "request_id": draft.draft_id,
        })
    finally:
        with _changed:
            if draft.status in {"queued", "running"}:
                draft.status = "failed"
                draft.error = {
                    "kind": "unknown",
                    "message": "Генерація завершилася без результату.",
                    "retryable": False,
                    "request_id": draft.draft_id,
                }
                _emit_locked(draft, "error", draft.error)
            _cancellations.pop(draft.draft_id, None)


def start(sample: dict, client: str) -> Draft:
    """Register a draft and return before its background worker completes."""
    if not sample or not sample.get("id"):
        raise DraftNotFound("Unknown support request")

    draft = new_draft(sample["id"], client)
    cancellation = threading.Event()
    with _lock:
        _cancellations[draft.draft_id] = cancellation

    worker = threading.Thread(
        target=_run,
        args=(draft, sample, cancellation),
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
            "request_id": draft.draft_id,
        })
        with _lock:
            _cancellations.pop(draft.draft_id, None)
    return draft


def cancel(draft_id: str) -> Draft:
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
        return draft
