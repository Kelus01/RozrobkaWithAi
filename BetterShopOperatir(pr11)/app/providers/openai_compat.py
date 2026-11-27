"""OpenAI-compatible API implementation; SDK-specific errors stay here."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterator

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from .base import Chunk, ProviderError, Request, Result


def _usage_dict(usage: object | None) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return None


def _retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _provider_error(exc: Exception) -> ProviderError:
    if isinstance(exc, APITimeoutError):
        return ProviderError("timeout", "Провайдер не відповів вчасно.", retryable=True)
    if isinstance(exc, RateLimitError):
        response = exc.response
        return ProviderError(
            "rate_limit",
            "Провайдер тимчасово обмежив кількість запитів.",
            retryable=True,
            retry_after=_retry_after(response),
            detail=f"HTTP {response.status_code}",
        )
    if isinstance(exc, AuthenticationError):
        return ProviderError("auth", "Не вдалося пройти автентифікацію у провайдера.")
    if isinstance(exc, BadRequestError):
        return ProviderError("bad_request", "Провайдер відхилив запит.")
    if isinstance(exc, APIConnectionError):
        return ProviderError("unavailable", "Не вдалося зʼєднатися з провайдером.", retryable=True)
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status >= 500:
            return ProviderError(
                "unavailable", "Провайдер тимчасово недоступний.", retryable=True,
                detail=f"HTTP {status}",
            )
        return ProviderError(
            "unknown", "Провайдер не зміг обробити запит.", retryable=False,
            detail=f"HTTP {status}",
        )
    return ProviderError(
        "unknown",
        "Сталася неочікувана помилка під час звернення до провайдера.",
        retryable=False,
        detail=type(exc).__name__,
    )


class OpenAICompatProvider:
    """Translate the local provider contract to the OpenAI-compatible SDK."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float,
                 idle_timeout: float | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.idle_timeout = idle_timeout or timeout

    def _client(self) -> OpenAI:
        timeout = httpx.Timeout(
            self.timeout,
            connect=self.timeout,
            read=min(self.timeout, self.idle_timeout),
        )
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
        )

    def _parameters(self, request: Request) -> dict:
        return {
            "model": request.model or self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

    def complete(self, request: Request) -> Result:
        try:
            with self._client() as client:
                response = client.chat.completions.create(**self._parameters(request))
            if not response.choices or not response.choices[0].message.content:
                raise ProviderError("unknown", "Провайдер повернув порожню відповідь.")
            return Result(
                text=response.choices[0].message.content,
                provider=self.name,
                model=response.model or request.model or self.model,
                usage=_usage_dict(response.usage),
                finish_reason=response.choices[0].finish_reason,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _provider_error(exc) from None

    def stream(self, request: Request) -> Iterator[Chunk]:
        client = self._client()
        stream = None
        try:
            stream = client.chat.completions.create(
                **self._parameters(request),
                stream=True,
                stream_options={"include_usage": True},
            )
            for event in stream:
                text = ""
                finish_reason = None
                if event.choices:
                    choice = event.choices[0]
                    text = choice.delta.content or ""
                    finish_reason = choice.finish_reason
                yield Chunk(
                    text=text,
                    usage=_usage_dict(event.usage),
                    finish_reason=finish_reason,
                )
        except ProviderError:
            raise
        except Exception as exc:
            raise _provider_error(exc) from None
        finally:
            if stream is not None:
                stream.close()
            client.close()
