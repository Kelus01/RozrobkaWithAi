"""Structured JSON-lines telemetry with request correlation."""

import contextvars
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_lock = threading.Lock()
_configured_path: Path | None = None


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "event": getattr(record, "event_name", record.getMessage()),
        }
        request_id = getattr(record, "request_id", None) or request_id_var.get()
        if request_id:
            event["request_id"] = request_id
        fields = getattr(record, "event_fields", {})
        event.update(fields)
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_logging() -> Path:
    global _configured_path
    configured = Path(os.getenv("LOG_FILE", "logs/app.jsonl"))
    if not configured.is_absolute():
        configured = Path.cwd() / configured
    with _lock:
        if _configured_path == configured:
            return configured
        configured.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("app.telemetry")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(configured, encoding="utf-8")
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _configured_path = configured
    return configured


def log_event(event: str, *, level: int = logging.INFO,
              request_id: str | None = None, **fields: Any) -> None:
    logger = logging.getLogger("app.telemetry")
    if not logger.handlers:
        configure_logging()
    logger.log(
        level,
        event,
        extra={
            "event_name": event,
            "request_id": request_id or request_id_var.get(),
            "event_fields": fields,
        },
    )


def set_request_id(request_id: str) -> contextvars.Token:
    return request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    request_id_var.reset(token)


def current_request_id() -> str | None:
    return request_id_var.get()


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
