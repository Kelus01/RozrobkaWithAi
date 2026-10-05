"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит до неї, — усе це лишається в `app/llm.py`. Тут
вирішується інше: що застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr3:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР3")
logger = logging.getLogger(__name__)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    """Звернення користувача."""

    question: str = Field(min_length=1, max_length=4000)

    @validator("question")
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Звернення не може бути порожнім.")
        return value


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом зі зверненням.
    """
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь або безпечну для клієнта помилку."""
    try:
        context = load_context()
    except OSError as exc:
        logger.error("Could not load store rules (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Не вдалося завантажити правила магазину.",
        ) from exc

    try:
        return llm.ask(payload.question, context)
    except llm.LLMError as exc:
        logger.warning("LLM request failed (%s)", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
