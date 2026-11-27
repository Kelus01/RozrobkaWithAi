"""Per-client sliding-window rate limiting for the draft endpoint."""

import collections
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
WINDOW_SECONDS = 60.0
_requests: dict[str, collections.deque[float]] = {}
_lock = threading.Lock()
_last_cleanup = 0.0

if PER_MINUTE < 1:
    raise ValueError("RATE_LIMIT_PER_MINUTE must be positive")


class RateLimited(Exception):
    """Ліміт вичерпано. `retry_after` — через скільки секунд пробувати."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    """Чинні обмеження — для сторінки."""
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    """Count accepted requests in a rolling 60-second window."""
    global _last_cleanup
    now = time.monotonic()
    with _lock:
        if now - _last_cleanup >= WINDOW_SECONDS:
            cutoff = now - WINDOW_SECONDS
            for key, history in list(_requests.items()):
                while history and history[0] <= cutoff:
                    history.popleft()
                if not history:
                    _requests.pop(key, None)
            _last_cleanup = now

        history = _requests.setdefault(client, collections.deque())
        cutoff = now - WINDOW_SECONDS
        while history and history[0] <= cutoff:
            history.popleft()
        if not history:
            _requests.pop(client, None)
            history = _requests.setdefault(client, collections.deque())

        if len(history) >= PER_MINUTE:
            raise RateLimited(max(0.0, history[0] + WINDOW_SECONDS - now))

        history.append(now)
