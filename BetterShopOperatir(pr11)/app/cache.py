"""Exact response cache with a bounded lifetime and atomic persistence."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .providers.base import Request

ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() not in {
    "0", "false", "no", "off",
}
TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("LLM_CACHE_MAX_ENTRIES", "1000"))
CACHE_FILE = Path(os.getenv("CACHE_FILE", "state/response_cache.json"))
if TTL_SECONDS < 0 or MAX_ENTRIES < 1:
    raise ValueError("LLM_CACHE_TTL_SECONDS must be non-negative and MAX_ENTRIES positive")
_lock = threading.RLock()
_entries: dict[str, dict[str, Any]] | None = None


def key_for(provider: str, model: str, prompt_version: str,
            request: Request) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_locked() -> dict[str, dict[str, Any]]:
    global _entries
    if _entries is not None:
        return _entries
    path = CACHE_FILE
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains invalid cache JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _entries = raw
    return _entries


def _write_locked(entries: dict[str, dict[str, Any]]) -> None:
    path = CACHE_FILE
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def get(key: str) -> dict[str, Any] | None:
    if not ENABLED:
        return None
    now = time.time()
    with _lock:
        entries = _load_locked()
        entry = entries.get(key)
        expired = [
            cache_key for cache_key, cached_entry in entries.items()
            if now - cached_entry.get("created_at", 0) > TTL_SECONDS
        ]
        if expired:
            for cache_key in expired:
                entries.pop(cache_key, None)
            _write_locked(entries)
        if entry is None or key not in entries:
            return None
        return dict(entry["result"])


def put(key: str, result: dict[str, Any]) -> None:
    if not ENABLED:
        return
    with _lock:
        entries = _load_locked()
        now = time.time()
        expired = [
            cache_key for cache_key, entry in entries.items()
            if now - entry.get("created_at", 0) > TTL_SECONDS
        ]
        for cache_key in expired:
            entries.pop(cache_key, None)
        entries[key] = {"created_at": now, "result": result}
        while len(entries) > MAX_ENTRIES:
            oldest = min(entries, key=lambda cache_key: entries[cache_key].get("created_at", 0))
            entries.pop(oldest)
        _write_locked(entries)


def clear() -> None:
    global _entries
    with _lock:
        _entries = {}
        _write_locked(_entries)
