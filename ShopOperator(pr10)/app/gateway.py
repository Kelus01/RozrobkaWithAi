"""Provider selection, deadlines, retries, concurrency and fallback."""

import logging
import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, ProviderError, Request, Result

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

if min(TIMEOUT, IDLE_TIMEOUT, TOTAL_TIMEOUT) <= 0:
    raise ValueError("LLM timeouts must be positive")
if MAX_ATTEMPTS < 1 or MAX_CONCURRENT < 1:
    raise ValueError("LLM_MAX_ATTEMPTS and LLM_MAX_CONCURRENT must be positive")

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def settings() -> dict:
    """Public gateway settings; never expose credentials."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str, *, timeout: float | None = None,
                  idle_timeout: float | None = None) -> Provider:
    """Build a provider; a non-special name selects an alternate model."""
    if name == "mock":
        from .providers.mock import MockProvider

        return MockProvider(timeout=timeout or TIMEOUT)

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = MODEL if name == "openai" else name
    if not base_url or not api_key or not model:
        raise ProviderError(
            "auth",
            "Налаштування основного провайдера неповні.",
            retryable=False,
            detail="LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required",
        )

    from .providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout or TIMEOUT,
        idle_timeout=idle_timeout or IDLE_TIMEOUT,
    )


def _put_result(channel: queue.Queue, stopped: threading.Event,
                item: tuple[str, object]) -> bool:
    while not stopped.is_set():
        try:
            channel.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


def _stream_attempt(provider: Provider, request: Request, *,
                    attempt_timeout: float, idle_timeout: float,
                    total_deadline: float, on_delta: Callable[[str], None],
                    is_cancelled: Callable[[], bool]) -> Result:
    slots = _slots
    if not slots.acquire(blocking=False):
        raise ProviderError(
            "unavailable",
            "Усі канали генерації зайняті. Спробуйте ще раз за мить.",
            retryable=True,
        )

    channel: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    stopped = threading.Event()
    stream_started = time.monotonic()

    def produce() -> None:
        stream = None
        try:
            stream = iter(provider.stream(request))
            for chunk in stream:
                if stopped.is_set():
                    break
                if not _put_result(channel, stopped, ("chunk", chunk)):
                    break
            _put_result(channel, stopped, ("end", None))
        except ProviderError as exc:
            _put_result(channel, stopped, ("error", exc))
        except Exception as exc:
            logger.exception("Unexpected provider failure (%s)", provider.name)
            error = ProviderError(
                "unknown",
                "Сталася неочікувана помилка під час генерації.",
                retryable=False,
                detail=type(exc).__name__,
            )
            _put_result(channel, stopped, ("error", error))
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close provider stream (%s)", provider.name)
            slots.release()

    worker = threading.Thread(target=produce, name="provider-stream", daemon=True)
    worker.start()

    chunks: list[str] = []
    usage: dict | None = None
    finish_reason: str | None = None
    last_activity = stream_started
    attempt_deadline = min(stream_started + attempt_timeout, total_deadline)
    completed = False

    try:
        while True:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)

            now = time.monotonic()
            remaining = min(attempt_deadline - now, last_activity + idle_timeout - now)
            if remaining <= 0:
                raise ProviderError(
                    "timeout",
                    "Провайдер не надіслав відповідь вчасно.",
                    retryable=True,
                )

            try:
                kind, value = channel.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue

            if kind == "error":
                raise value
            if kind == "end":
                completed = True
                text = "".join(chunks).strip()
                if not text:
                    raise ProviderError(
                        "unknown", "Провайдер повернув порожню відповідь.", retryable=False
                    )
                return Result(
                    text=text,
                    provider=provider.name,
                    model=getattr(provider, "model", request.model or "unknown"),
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=time.monotonic() - stream_started,
                )

            chunk = value
            last_activity = time.monotonic()
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if chunk.text:
                chunks.append(chunk.text)
                on_delta(chunk.text)
    finally:
        if not completed:
            stopped.set()


def _interruptible_wait(seconds: float, is_cancelled: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if is_cancelled():
            return False
        time.sleep(min(0.05, deadline - time.monotonic()))
    return not is_cancelled()


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    """Generate with bounded retries, a total deadline, and an optional fallback."""
    started = time.monotonic()
    total_deadline = started + TOTAL_TIMEOUT
    attempts = 0
    last_error = ProviderError("unknown", "Генерацію не вдалося виконати.", retryable=False)
    has_visible_text = False

    def run_provider(name: str, limit: int, fallback: bool) -> Result:
        nonlocal attempts, last_error, has_visible_text
        provider_attempt = 0
        while provider_attempt < limit:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
            remaining = total_deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(
                    "timeout", "Вичерпано загальний час очікування.", retryable=True
                )

            provider_attempt += 1
            attempts += 1
            on_status({
                "attempt": attempts,
                "provider_attempt": provider_attempt,
                "provider": "openai" if name != "mock" else "mock",
                "model": MODEL if name == "openai" else (None if name == "mock" else name),
                "fallback": fallback,
                "detail": f"спроба {provider_attempt} з {limit}",
            })

            try:
                provider = make_provider(
                    name,
                    timeout=min(TIMEOUT, remaining),
                    idle_timeout=min(IDLE_TIMEOUT, remaining),
                )

                def deliver(text: str) -> None:
                    nonlocal has_visible_text
                    has_visible_text = True
                    on_delta(text)

                return _stream_attempt(
                    provider,
                    request,
                    attempt_timeout=min(TIMEOUT, remaining),
                    idle_timeout=min(IDLE_TIMEOUT, remaining),
                    total_deadline=total_deadline,
                    on_delta=deliver,
                    is_cancelled=is_cancelled,
                )
            except ProviderError as exc:
                last_error = exc
                if exc.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)

                can_retry = exc.retryable and provider_attempt < limit
                if not can_retry:
                    break

                remaining = total_deadline - time.monotonic()
                if exc.retry_after is not None:
                    delay = max(0.0, exc.retry_after)
                else:
                    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (provider_attempt - 1)))
                    delay *= random.uniform(0.5, 1.5)
                if delay >= remaining:
                    break

                if has_visible_text:
                    on_status({
                        "reset": True,
                        "fallback": fallback,
                        "detail": "спроба починається заново; попередній текст скинуто",
                    })
                    has_visible_text = False
                if not _interruptible_wait(delay, is_cancelled):
                    raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)

        raise last_error

    try:
        return run_provider(PROVIDER, MAX_ATTEMPTS, False)
    except ProviderError as primary_error:
        if primary_error.kind == "cancelled" or is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
        if FALLBACK == "none" or FALLBACK == PROVIDER:
            raise primary_error
        if (
            primary_error.kind == "rate_limit"
            and primary_error.retry_after is not None
            and primary_error.retry_after >= total_deadline - time.monotonic()
        ):
            raise primary_error
        if total_deadline <= time.monotonic():
            raise ProviderError(
                "timeout", "Вичерпано загальний час очікування.", retryable=True
            ) from None

        if has_visible_text:
            on_status({
                "reset": True,
                "fallback": True,
                "detail": "основний провайдер не відповів; текст буде згенеровано заново",
            })
            has_visible_text = False
        on_status({"fallback": True, "detail": "перехід до запасного варіанта"})
        try:
            return run_provider(FALLBACK, 1, True)
        except ProviderError as fallback_error:
            if fallback_error.kind == "cancelled" or is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
            raise fallback_error
