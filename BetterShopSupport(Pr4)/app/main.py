"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит до неї, — усе це лишається в `app/llm.py`. Тут
вирішується інше: що застосунок віддає клієнтові та з яким HTTP-статусом.
"""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

from . import llm

app = FastAPI(title="Помічник служби підтримки")
logger = logging.getLogger(__name__)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"
SESSION_STORE: dict[str, dict[str, Any]] = {}


class Question(BaseModel):
    """Звернення користувача."""

    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None

    @validator("question")
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Звернення не може бути порожнім.")
        return value


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель."""
    return CONTEXT_FILE.read_text(encoding="utf-8")


def _ensure_session(session_id: str | None) -> str:
    session_key = session_id or uuid4().hex
    session = SESSION_STORE.setdefault(session_key, {"history": [], "order_number": None})
    if not isinstance(session.get("history"), list):
        session["history"] = []
    if not isinstance(session.get("order_number"), (str, type(None))):
        session["order_number"] = None
    return session_key


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути структуровану відповідь і картку звернення для клієнта."""
    try:
        context = load_context()
    except OSError as exc:
        logger.error("Could not load store rules (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Не вдалося завантажити правила магазину.",
        ) from exc

    session_id = _ensure_session(payload.session_id)
    session = SESSION_STORE[session_id]

    try:
        result = llm.ask_structured(
            payload.question,
            context,
            history=session["history"],
            order_number=session.get("order_number"),
        )
    except llm.LLMError as exc:
        logger.warning("LLM request failed (%s)", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    session["history"].append({"role": "user", "content": payload.question})
    session["history"].append({"role": "assistant", "content": result["answer"]})
    if result.get("order_number"):
        session["order_number"] = result["order_number"]

    while len(session["history"]) > 20:
        session["history"] = session["history"][2:]

    card = {
        "session_id": session_id,
        "topic": result["topic"],
        "based_on_rules": result["based_on_rules"],
        "needs_clarification": result["needs_clarification"],
        "transfer_to_operator": result["transfer_to_operator"],
        "order_number": result["order_number"] or session.get("order_number"),
    }

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "topic": result["topic"],
        "based_on_rules": result["based_on_rules"],
        "needs_clarification": result["needs_clarification"],
        "transfer_to_operator": result["transfer_to_operator"],
        "order_number": result["order_number"] or session.get("order_number"),
        "model": result["model"],
        "elapsed": result["elapsed"],
        "usage": result.get("usage"),
        "card": card,
    }
