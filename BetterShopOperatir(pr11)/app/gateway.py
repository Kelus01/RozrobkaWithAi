"""Provider selection, deadlines, retries, concurrency and fallback."""

import logging
import hashlib
import os
import queue
import random
import threading
import time
from typing import Callable

from . import budget, cache, costs, telemetry
from .providers.base import Provider, ProviderError, Request, Result

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
                    on_usage: Callable[[dict], None],
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
                on_usage(chunk.usage)
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
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool], *,
             request_id: str | None = None, draft_id: str | None = None,
             client: str | None = None, fresh: bool = False,
             queued_ms: float = 0.0) -> Result:
    """Generate with bounded retries, a total deadline, and an optional fallback."""
    started = time.monotonic()
    total_deadline = started + TOTAL_TIMEOUT
    attempts = 0
    last_error = ProviderError("unknown", "Генерацію не вдалося виконати.", retryable=False)
    has_visible_text = False
    total_cost = 0.0
    estimated_cost = False
    prompt_version = str(request.metadata.get("prompt_version", "unknown"))
    requested_model = (MODEL or "") if PROVIDER == "openai" else PROVIDER
    cache_key = cache.key_for(PROVIDER, requested_model, prompt_version, request)
    safe_client_id = (
        hashlib.sha256(client.encode("utf-8")).hexdigest()[:16] if client else None
    )

    if not fresh:
        cached = cache.get(cache_key)
        if cached is not None:
            on_status({
                "cache_hit": True,
                "provider": cached["provider"],
                "model": cached["model"],
                "detail": "відповідь отримано з кешу",
            })
            on_delta(cached["text"])
            telemetry.log_event(
                "cache.hit", request_id=request_id, draft_id=draft_id,
                client_id_hash=safe_client_id, provider=cached["provider"],
                model=cached["model"], prompt_version=prompt_version,
                cache_key=cache_key, cached=True, cost_usd=0.0,
            )
            return Result(
                text=cached["text"],
                provider=cached["provider"],
                model=cached["model"],
                usage=cached.get("usage"),
                elapsed=0.0,
                cached=True,
                cache_key=cache_key,
            )
    else:
        telemetry.log_event(
            "cache.bypass", request_id=request_id, draft_id=draft_id,
            client_id_hash=safe_client_id, prompt_version=prompt_version,
            cache_key=cache_key, reason="fresh",
        )

    def record_attempt(*, provider_name: str, model_name: str, attempt: int,
                       fallback: bool, attempt_started: float,
                       first_token_at: float | None, text_parts: list[str],
                       usage: dict | None, status: str, kind: str | None,
                       reservation_id: str, reservation_usd: float) -> costs.UsageCost:
        nonlocal total_cost, estimated_cost
        measured = costs.calculate(usage, request.messages, "".join(text_parts))
        charge = measured.cost_usd if not measured.estimated else reservation_usd
        budget.settle(reservation_id, reservation_usd, charge)
        total_cost += charge
        estimated_cost = estimated_cost or measured.estimated
        telemetry.log_event(
            "llm.attempt", level=(
                logging.WARNING if attempt > 1 or fallback else (
                    logging.ERROR if status == "failed" else logging.INFO
                )
            ),
            request_id=request_id, draft_id=draft_id,
            client_id_hash=safe_client_id, provider=provider_name,
            model=model_name, prompt_version=prompt_version, attempt=attempt,
            queue_ms=round(max(0.0, queued_ms), 3),
            first_token_ms=(
                round((first_token_at - attempt_started) * 1000, 3)
                if first_token_at is not None else None
            ),
            duration_ms=round((time.monotonic() - attempt_started) * 1000, 3),
            status=status, kind=kind, fallback=fallback, cached=False,
            input_tokens=measured.input_tokens,
            output_tokens=measured.output_tokens,
            cached_input_tokens=measured.cached_input_tokens,
            cost_usd=round(charge, 10),
            cost_estimated=measured.estimated or usage is None,
            cache_key=cache_key,
        )
        return measured

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

            model_name = MODEL if name == "openai" else name
            try:
                provider = make_provider(
                    name,
                    timeout=min(TIMEOUT, remaining),
                    idle_timeout=min(IDLE_TIMEOUT, remaining),
                )
            except ProviderError as exc:
                last_error = exc
                telemetry.log_event(
                    "llm.preflight_failed", level=logging.ERROR,
                    request_id=request_id, draft_id=draft_id,
                    client_id_hash=safe_client_id,
                    provider="openai" if name != "mock" else "mock",
                    model=model_name or "unknown", prompt_version=prompt_version,
                    status="failed", kind=exc.kind, fallback=fallback,
                    cost_usd=0.0, cost_estimated=False, cache_key=cache_key,
                )
                break

            try:
                reservation_usd = costs.reservation_upper_bound(
                    request.messages, request.max_tokens
                )
                reservation_id, warn_budget = budget.reserve(reservation_usd)
            except budget.BudgetExceeded as exc:
                telemetry.log_event(
                    "budget.denied", level=logging.WARNING,
                    request_id=request_id, draft_id=draft_id,
                    client_id_hash=safe_client_id, provider=name,
                    model=MODEL if name == "openai" else name,
                    prompt_version=prompt_version, kind="budget",
                    remaining_usd=exc.remaining_usd, reset_at=exc.reset_at,
                )
                raise ProviderError(
                    "budget",
                    "Денний бюджет генерації вичерпано. Спробуйте після його оновлення.",
                    retryable=False,
                    detail=f"Бюджет оновиться {exc.reset_at}.",
                    reset_at=exc.reset_at,
                    cost_usd=total_cost,
                    cost_estimated=estimated_cost,
                ) from None
            if warn_budget:
                state = budget.snapshot()
                telemetry.log_event(
                    "budget.warning", level=logging.WARNING,
                    request_id=request_id, draft_id=draft_id,
                    client_id_hash=safe_client_id, spent_usd=state["spent_usd"],
                    reserved_usd=state["reserved_usd"],
                    limit_usd=state["limit_usd"],
                )
                on_status({
                    "budget_warning": True,
                    "budget": state,
                    "detail": "наближається денний ліміт витрат",
                })

            provider_attempt += 1
            attempts += 1
            attempt_started = time.monotonic()
            attempt_text: list[str] = []
            attempt_usage: dict | None = None
            first_token_at: float | None = None
            on_status({
                "attempt": attempts,
                "provider_attempt": provider_attempt,
                "provider": "openai" if name != "mock" else "mock",
                "model": model_name,
                "fallback": fallback,
                "detail": f"спроба {provider_attempt} з {limit}",
            })

            try:
                def deliver(text: str) -> None:
                    nonlocal has_visible_text, first_token_at
                    if first_token_at is None and text:
                        first_token_at = time.monotonic()
                    attempt_text.append(text)
                    has_visible_text = True
                    on_delta(text)

                def capture_usage(value: dict) -> None:
                    nonlocal attempt_usage
                    attempt_usage = value

                result = _stream_attempt(
                    provider,
                    request,
                    attempt_timeout=min(TIMEOUT, remaining),
                    idle_timeout=min(IDLE_TIMEOUT, remaining),
                    total_deadline=total_deadline,
                    on_delta=deliver,
                    on_usage=capture_usage,
                    is_cancelled=is_cancelled,
                )
                used = result.usage if result.usage is not None else attempt_usage
                measured = record_attempt(
                    provider_name=result.provider, model_name=result.model,
                    attempt=attempts, fallback=fallback,
                    attempt_started=attempt_started, first_token_at=first_token_at,
                    text_parts=attempt_text, usage=used, status="done", kind=None,
                    reservation_id=reservation_id, reservation_usd=reservation_usd,
                )
                result.cost_usd = total_cost
                result.cost_estimated = estimated_cost or measured.estimated
                result.cache_key = cache_key
                result.attempts = attempts
                if not fallback and result.provider == PROVIDER:
                    cache.put(cache_key, {
                        "text": result.text,
                        "provider": result.provider,
                        "model": result.model,
                        "usage": result.usage,
                    })
                return result
            except ProviderError as exc:
                measured = record_attempt(
                    provider_name="openai" if name != "mock" else "mock",
                    model_name=model_name or "unknown", attempt=attempts,
                    fallback=fallback, attempt_started=attempt_started,
                    first_token_at=first_token_at, text_parts=attempt_text,
                    usage=attempt_usage, status="failed", kind=exc.kind,
                    reservation_id=reservation_id, reservation_usd=reservation_usd,
                )
                last_error = exc
                last_error.cost_usd = total_cost
                last_error.cost_estimated = estimated_cost or measured.estimated
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
        primary_error.cost_usd = total_cost
        primary_error.cost_estimated = estimated_cost or primary_error.cost_estimated
        if primary_error.kind == "cancelled" or is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
        if primary_error.kind == "budget" or FALLBACK == "none" or FALLBACK == PROVIDER:
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
