"""FastAPI routes for the PR6 knowledge-base assistant."""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from . import embeddings, index, keyword, llm, rag

app = FastAPI(title="Помічник за базою знань — ПР6")
logger = logging.getLogger(__name__)
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Питання не може бути порожнім.")
        return value

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - {"product"}
        if unknown:
            raise ValueError(
                f"Недозволені поля фільтра: {', '.join(sorted(unknown))}."
            )
        if any(not item.strip() for item in value.values()):
            raise ValueError("Значення фільтрів не можуть бути порожніми.")
        return {key: item.strip() for key, item in value.items()}


class SearchRequest(BaseModel):
    """Compatibility endpoint for the original PR5 search interface."""

    query: str = Field(min_length=1, max_length=2000)
    mode: str = "both"
    top_k: int = Field(default=index.DEFAULT_TOP_K, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)
    threshold: float | None = Field(default=None, ge=-1, le=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Пошуковий запит не може бути порожнім.")
        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in {"semantic", "keyword", "both"}:
            raise ValueError("mode має бути semantic, keyword або both.")
        return value

    @field_validator("filters")
    @classmethod
    def validate_search_filters(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - {"category", "product"}
        if unknown:
            raise ValueError(f"Невідомі фільтри: {', '.join(sorted(unknown))}.")
        if any(not item.strip() for item in value.values()):
            raise ValueError("Значення фільтрів не можуть бути порожніми.")
        return {key: item.strip() for key, item in value.items()}


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {
                "ref": source.ref,
                "score": source.score,
                "source": source.chunk.source,
                "metadata": source.chunk.metadata,
                "text": source.chunk.text,
            }
            for source in result.sources
        ],
        "retrieved": [hit_to_dict(hit) for hit in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    app.state.index = None
    app.state.search_index = None
    app.state.keyword_index = None
    try:
        loaded_index = index.load()
    except index.IndexNotFoundError as exc:
        app.state.index_error = str(exc)
        logger.info("Search index is not built: %s", exc)
        return
    except (index.IndexFormatError, OSError):
        app.state.index_error = (
            "Індекс пошкоджений або не читається. "
            "Перебудуйте його командою python ingest.py."
        )
        logger.exception("Could not load search index")
        return

    app.state.index = loaded_index
    app.state.search_index = loaded_index
    app.state.keyword_index = keyword.build(loaded_index.chunks)
    app.state.index_error = None


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    search_index = getattr(app.state, "index", None)
    if search_index is None:
        return {
            "ready": False,
            "hint": getattr(
                app.state,
                "index_error",
                "Індекс не збудовано — виконайте python ingest.py.",
            ),
        }
    return {
        "ready": True,
        "chunks": len(search_index),
        "documents": len({chunk.source for chunk in search_index.chunks}),
        "model": search_index.model_name,
        "embedding_model_matches": search_index.model_name == embeddings.MODEL_NAME,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest, request: Request) -> dict:
    search_index = getattr(request.app.state, "index", None)
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if search_index is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(
                request.app.state,
                "index_error",
                "Індекс не збудовано — виконайте python ingest.py.",
            ),
        )
    if search_index.model_name != embeddings.MODEL_NAME:
        raise HTTPException(
            status_code=409,
            detail="Модель ембедінгів не збігається з індексом. Виконайте python ingest.py.",
        )
    try:
        result = rag.answer(
            payload.question,
            search_index,
            keyword_index,
            filters=payload.filters or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except embeddings.EmbeddingError as exc:
        logger.exception("Could not encode user question")
        raise HTTPException(
            status_code=503,
            detail="Модель пошуку недоступна. Перевірте налаштування ембедінгів.",
        ) from exc
    except llm.LLMError as exc:
        logger.warning("RAG generation failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return answer_to_dict(result)


@app.post("/api/search")
def api_search(payload: SearchRequest, request: Request) -> dict:
    """Keep the PR5 search API available for comparisons and regression tests."""
    search_index = getattr(request.app.state, "search_index", None)
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if search_index is None or keyword_index is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(
                request.app.state,
                "index_error",
                "Індекс не збудовано — виконайте python ingest.py.",
            ),
        )
    if (
        payload.mode in {"semantic", "both"}
        and search_index.model_name != embeddings.MODEL_NAME
    ):
        raise HTTPException(
            status_code=409,
            detail="Модель ембедінгів не збігається з індексом. Виконайте python ingest.py.",
        )

    result: dict = {
        "query": payload.query,
        "semantic": None,
        "keyword": None,
        "elapsed": {},
    }
    filters = {
        **payload.filters,
        "audience": "клієнти",
        "status": "чинний",
    }
    if payload.mode in {"semantic", "both"}:
        started = time.perf_counter()
        try:
            vector = embeddings.embed_query(payload.query)
        except embeddings.EmbeddingError as exc:
            logger.exception("Could not encode PR5 search query")
            raise HTTPException(
                status_code=503,
                detail="Модель пошуку недоступна. Перевірте налаштування ембедінгів.",
            ) from exc
        try:
            hits = index.search(
                search_index,
                vector,
                top_k=payload.top_k,
                filters=filters,
                threshold=(
                    payload.threshold
                    if payload.threshold is not None
                    else index.SIMILARITY_THRESHOLD
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["elapsed"]["semantic"] = time.perf_counter() - started
        result["semantic"] = [hit_to_dict(hit) for hit in hits]

    if payload.mode in {"keyword", "both"}:
        started = time.perf_counter()
        try:
            hits = keyword.search(
                keyword_index,
                payload.query,
                top_k=payload.top_k,
                filters=filters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["elapsed"]["keyword"] = time.perf_counter() - started
        result["keyword"] = [hit_to_dict(hit) for hit in hits]
    return result
