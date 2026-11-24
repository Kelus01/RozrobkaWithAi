"""Network-free provider used for demos and deterministic failure tests."""

import os
import random
import re
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from .. import samples

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")
RETRY_AFTER = float(os.getenv("MOCK_RETRY_AFTER", "1"))
FAILURE_KINDS = {"timeout", "rate_limit", "unavailable", "auth", "bad_request", "mid_stream"}

if not 0 <= FAILURE_RATE <= 1:
    raise ValueError("MOCK_FAILURE_RATE must be between 0 and 1")
if FAILURE_KIND not in FAILURE_KINDS:
    raise ValueError("MOCK_FAILURE_KIND is not supported")
if RETRY_AFTER < 0:
    raise ValueError("MOCK_RETRY_AFTER must not be negative")


class MockProvider:
    """Return canonical sample replies, with configurable delays and faults."""

    name = "mock"
    model = "mock-1"

    def __init__(self, timeout: float = 20) -> None:
        self.timeout = timeout

    @staticmethod
    def _reply(request: Request) -> tuple[dict, str]:
        request_id = request.metadata.get("request_id")
        sample = samples.get(request_id) if isinstance(request_id, str) else None
        if sample is None:
            raise ProviderError(
                "bad_request", "Не вдалося знайти звернення для чернетки.", retryable=False
            )
        return sample, sample["reply_example"]

    @staticmethod
    def _usage(request: Request, text: str) -> dict:
        prompt = " ".join(str(message.get("content", "")) for message in request.messages)
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _delay(milliseconds: int) -> None:
        if milliseconds > 0:
            time.sleep(milliseconds / 1000)

    def complete(self, request: Request) -> Result:
        started = time.monotonic()
        text = "".join(chunk.text for chunk in self.stream(request))
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=self._usage(request, text),
            finish_reason="stop",
            elapsed=time.monotonic() - started,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        _, text = self._reply(request)
        failed = random.random() < FAILURE_RATE
        failure_kind = FAILURE_KIND if failed else None

        if failure_kind == "timeout":
            self._delay(max(FIRST_DELAY_MS, int(self.timeout * 1000) + 100))
            raise ProviderError("timeout", "Підставний провайдер перевищив таймаут.", retryable=True)
        if failure_kind == "rate_limit":
            raise ProviderError(
                "rate_limit",
                "Підставний провайдер повідомив про обмеження частоти.",
                retryable=True,
                retry_after=RETRY_AFTER,
            )
        if failure_kind == "auth":
            raise ProviderError("auth", "Підставний провайдер відхилив доступ.", retryable=False)
        if failure_kind == "bad_request":
            raise ProviderError(
                "bad_request", "Підставний провайдер відхилив запит.", retryable=False
            )
        if failure_kind == "unavailable":
            raise ProviderError(
                "unavailable", "Підставний провайдер тимчасово недоступний.", retryable=True
            )

        words = re.findall(r"\S+\s*", text)
        pieces = ["".join(words[index:index + 3]) for index in range(0, len(words), 3)]
        if not pieces:
            raise ProviderError("unknown", "Порожній текст зразка.", retryable=False)

        self._delay(FIRST_DELAY_MS)
        for index, piece in enumerate(pieces):
            if failure_kind == "mid_stream" and index == min(2, len(pieces) - 1):
                raise ProviderError(
                    "unavailable",
                    "Підставний потік обірвався під час відповіді.",
                    retryable=True,
                )
            yield Chunk(text=piece)
            if index < len(pieces) - 1:
                self._delay(DELAY_MS)

        yield Chunk(
            text="",
            usage=self._usage(request, text),
            finish_reason="stop",
        )
