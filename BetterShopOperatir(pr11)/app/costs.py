"""Token accounting and configurable per-model call cost estimates."""

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

INPUT_USD_PER_MILLION = float(os.getenv("LLM_INPUT_USD_PER_MILLION", "0.5"))
OUTPUT_USD_PER_MILLION = float(os.getenv("LLM_OUTPUT_USD_PER_MILLION", "3.0"))
CACHED_INPUT_USD_PER_MILLION = float(
    os.getenv("LLM_CACHED_INPUT_USD_PER_MILLION", str(INPUT_USD_PER_MILLION))
)
if min(INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION,
       CACHED_INPUT_USD_PER_MILLION) < 0:
    raise ValueError("LLM token prices cannot be negative")


@dataclass(frozen=True)
class UsageCost:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    estimated: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_estimate(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def prompt_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{message.get('role', '')}:{message.get('content', '')}"
        for message in messages
    )


def calculate(usage: dict[str, Any] | None, messages: list[dict[str, Any]],
              completion: str) -> UsageCost:
    prompt = prompt_text(messages)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    output_tokens = usage.get("completion_tokens") if usage else None
    estimated = not (
        isinstance(prompt_tokens, int)
        and isinstance(output_tokens, int)
        and prompt_tokens >= 0
        and output_tokens >= 0
    )
    if estimated:
        prompt_tokens = _token_estimate(prompt)
        output_tokens = _token_estimate(completion)

    details = usage.get("prompt_tokens_details", {}) if usage else {}
    cached_tokens = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        cached_tokens = 0
    cached_tokens = min(cached_tokens, prompt_tokens)
    chargeable_input = prompt_tokens - cached_tokens
    cost = (
        chargeable_input * INPUT_USD_PER_MILLION
        + cached_tokens * CACHED_INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return UsageCost(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        cost_usd=round(cost, 10),
        estimated=estimated,
    )


def reservation_upper_bound(messages: list[dict[str, Any]], max_output_tokens: int) -> float:
    input_bytes = len(prompt_text(messages).encode("utf-8")) + len(messages) * 32
    output_tokens = max(0, max_output_tokens)
    return round(
        (input_bytes * INPUT_USD_PER_MILLION
         + output_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000,
        10,
    )
