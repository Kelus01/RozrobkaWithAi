"""Persistent, process-wide daily spend limit with atomic reservations."""

import json
import os
import threading
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

DAILY_LIMIT_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))
WARNING_FRACTION = float(os.getenv("LLM_BUDGET_WARNING_FRACTION", "0.8"))
STATE_FILE = Path(os.getenv("BUDGET_STATE_FILE", "state/daily_budget.json"))
if DAILY_LIMIT_USD <= 0 or not 0 < WARNING_FRACTION < 1:
    raise ValueError("Daily budget must be positive and warning fraction between 0 and 1")
_lock = threading.RLock()
_state: dict[str, Any] | None = None


class BudgetExceeded(Exception):
    def __init__(self, reset_at: str, remaining_usd: float) -> None:
        super().__init__("Daily model budget exhausted")
        self.reset_at = reset_at
        self.remaining_usd = remaining_usd


def _today() -> str:
    return date.today().isoformat()


def _next_reset() -> str:
    tomorrow = date.today() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min).astimezone().isoformat(timespec="seconds")


def _load_locked() -> dict[str, Any]:
    global _state
    if _state is not None:
        if _state["date"] != _today():
            _state = {"date": _today(), "spent_usd": 0.0, "reserved_usd": 0.0,
                      "warning_emitted": False}
            _write_locked(_state)
        return _state
    path = STATE_FILE if STATE_FILE.is_absolute() else Path.cwd() / STATE_FILE
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loaded = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains invalid budget JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if loaded.get("date") != _today():
        loaded = {"date": _today(), "spent_usd": 0.0, "reserved_usd": 0.0,
                  "warning_emitted": False}
    loaded.setdefault("spent_usd", 0.0)
    loaded.setdefault("reserved_usd", 0.0)
    loaded.setdefault("warning_emitted", False)
    _state = loaded
    _write_locked(_state)
    return _state


def _write_locked(state: dict[str, Any]) -> None:
    path = STATE_FILE if STATE_FILE.is_absolute() else Path.cwd() / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(state, separators=(",", ":")) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reserve(upper_bound_usd: float) -> tuple[str, bool]:
    if upper_bound_usd < 0:
        raise ValueError("budget reservation cannot be negative")
    with _lock:
        state = _load_locked()
        current = state["spent_usd"] + state["reserved_usd"]
        remaining = max(0.0, DAILY_LIMIT_USD - current)
        if current + upper_bound_usd > DAILY_LIMIT_USD:
            raise BudgetExceeded(_next_reset(), round(remaining, 8))
        state["reserved_usd"] += upper_bound_usd
        warning = (
            not state["warning_emitted"]
            and state["spent_usd"] + state["reserved_usd"]
            >= DAILY_LIMIT_USD * WARNING_FRACTION
        )
        if warning:
            state["warning_emitted"] = True
        _write_locked(state)
        return uuid.uuid4().hex, warning


def settle(reservation_id: str, reserved_usd: float, actual_usd: float) -> None:
    del reservation_id
    if actual_usd < 0:
        raise ValueError("actual cost cannot be negative")
    with _lock:
        state = _load_locked()
        state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved_usd)
        state["spent_usd"] += actual_usd
        _write_locked(state)


def snapshot() -> dict[str, Any]:
    with _lock:
        state = _load_locked()
        spent = float(state["spent_usd"])
        reserved = float(state["reserved_usd"])
        return {
            "date": state["date"],
            "limit_usd": DAILY_LIMIT_USD,
            "spent_usd": round(spent, 8),
            "reserved_usd": round(reserved, 8),
            "remaining_usd": round(max(0.0, DAILY_LIMIT_USD - spent - reserved), 8),
            "warning": state["warning_emitted"],
            "warning_fraction": WARNING_FRACTION,
            "reset_at": _next_reset(),
        }
