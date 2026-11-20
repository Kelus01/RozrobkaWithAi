"""Цикл помічника: від питання клієнта до відповіді, спертої на дані магазину."""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента — для сторінки й для перевірки."""

    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Результат циклу — те, з чим працює веб-рівень."""

    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _make_error_answer(message: str, model: str | None = None) -> Answer:
    return Answer(text=message, calls=[], rounds=0, stopped="error", model=model, elapsed={"model": 0.0, "tools": 0.0}, usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові, за потреби викликаючи інструменти."""
    if not question or not str(question).strip():
        return Answer(text="Питання порожнє. Напишіть, що саме хочете перевірити.", calls=[], rounds=0, stopped="empty", model=None, elapsed={"model": 0.0, "tools": 0.0}, usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    try:
        from shop import service

        customers = service.list_customers()
        if not any(c["customer_id"] == customer_id for c in customers):
            return Answer(text=f"Клієнта {customer_id} не знайдено в системі.", calls=[], rounds=0, stopped="invalid_customer", model=None, elapsed={"model": 0.0, "tools": 0.0}, usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    except Exception:
        return _make_error_answer("Сервіс магазину недоступний. Спробуйте пізніше.")

    messages = llm.build_messages(question)
    traces: list[ToolTrace] = []
    total_model_time = 0.0
    total_tool_time = 0.0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    ctx = tools.Context(customer_id=customer_id)

    for round_index in range(1, MAX_ROUNDS + 1):
        try:
            started = time.perf_counter()
            response = llm.chat(messages, tools.specs(), tool_choice="auto")
            model_elapsed = time.perf_counter() - started
            total_model_time += model_elapsed
            total_prompt += response["usage"].get("prompt_tokens", 0)
            total_completion += response["usage"].get("completion_tokens", 0)
            total_tokens += response["usage"].get("total_tokens", 0)

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                text = response.get("content") or "Я не можу дати точну відповідь на це питання без потрібних даних."
                return Answer(
                    text=text.strip(),
                    calls=traces,
                    rounds=round_index,
                    stopped="answer",
                    model=response.get("model"),
                    elapsed={"model": round(total_model_time, 3), "tools": round(total_tool_time, 3)},
                    usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion, "total_tokens": total_tokens},
                )

            messages.append(response["assistant_message"])

            for tool_call in tool_calls:
                call_name = tool_call["function"]["name"]
                raw_arguments = tool_call["function"].get("arguments") or "{}"
                call_started = time.perf_counter()
                result = tools.call(call_name, raw_arguments, ctx)
                elapsed = time.perf_counter() - call_started
                total_tool_time += elapsed

                traces.append(
                    ToolTrace(
                        round=round_index,
                        name=call_name,
                        arguments=raw_arguments,
                        status=result.status,
                        reason=result.reason,
                        result=result.content,
                        elapsed=elapsed,
                    )
                )

                content_value = result.content
                if content_value is None:
                    content_value = {"status": result.status, "reason": result.reason}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": call_name,
                    "content": json.dumps(content_value, ensure_ascii=False),
                })

            continue
        except llm.LLMError as exc:
            return Answer(text=f"Модель зараз недоступна: {exc}.", calls=traces, rounds=round_index - 1 if round_index > 0 else 0, stopped="llm_error", model=None, elapsed={"model": round(total_model_time, 3), "tools": round(total_tool_time, 3)}, usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion, "total_tokens": total_tokens})

    fallback = "Я не можу дати вірну відповідь без потрібних даних або ліміт звернень до моделі вичерпано."
    return Answer(
        text=fallback,
        calls=traces,
        rounds=MAX_ROUNDS,
        stopped="limit",
        model=None,
        elapsed={"model": round(total_model_time, 3), "tools": round(total_tool_time, 3)},
        usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion, "total_tokens": total_tokens},
    )
