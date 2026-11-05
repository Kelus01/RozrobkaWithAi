"""FastAPI web layer for the store knowledge-base search."""

import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from . import embeddings, index, keyword

app = FastAPI(title="База знань магазину")
logger = logging.getLogger(__name__)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
ALLOWED_FILTERS = {"category", "product", "audience", "status"}


class SearchRequest(BaseModel):
    """Validated request sent by the search page."""

    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["semantic", "keyword", "both"] = "both"
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

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - ALLOWED_FILTERS
        if unknown:
            raise ValueError(f"Невідомі фільтри: {', '.join(sorted(unknown))}.")
        if any(not item.strip() for item in value.values()):
            raise ValueError("Значення фільтрів не можуть бути порожніми.")
        return {key: item.strip() for key, item in value.items()}


def load_indexes() -> None:
    """Load the saved index and build its keyword counterpart at startup."""
    app.state.search_index = None
    app.state.keyword_index = None
    try:
        search_index = index.load()
    except index.IndexNotFoundError as exc:
        app.state.search_index_error = str(exc)
        logger.info("Search index is not built: %s", exc)
        return
    except (index.IndexFormatError, OSError):
        app.state.search_index_error = (
            "Індекс пошкоджений або не читається. "
            "Перебудуйте його командою python ingest.py."
        )
        logger.exception("Could not load search index")
        return

    app.state.search_index = search_index
    app.state.keyword_index = keyword.build(search_index.chunks)
    app.state.search_index_error = None


@app.on_event("startup")
def startup() -> None:
    load_indexes()


def _hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Show the PR5 search interface."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    search_index = getattr(app.state, "search_index", None)
    if search_index is None:
        return {
            "ready": False,
            "hint": getattr(
                app.state,
                "search_index_error",
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


@app.post("/api/search")
def api_search(payload: SearchRequest, request: Request) -> dict:
    """Search the vector and keyword indexes and return their timings."""
    search_index = getattr(request.app.state, "search_index", None)
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if search_index is None or keyword_index is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(
                request.app.state,
                "search_index_error",
                "Індекс не збудовано — виконайте python ingest.py.",
            ),
        )
    if (
        payload.mode in ("semantic", "both")
        and search_index.model_name != embeddings.MODEL_NAME
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Індекс побудовано моделлю {search_index.model_name}, а в налаштуваннях "
                f"вказано {embeddings.MODEL_NAME}. Виконайте python ingest.py."
            ),
        )

    result: dict = {
        "query": payload.query,
        "semantic": None,
        "keyword": None,
        "elapsed": {},
    }
    filters = payload.filters or None

    if payload.mode in ("semantic", "both"):
        started = time.perf_counter()
        try:
            query_vector = embeddings.embed_query(payload.query)
        except embeddings.EmbeddingError as exc:
            logger.exception("Could not encode search query")
            raise HTTPException(
                status_code=503,
                detail="Модель ембедінгів недоступна. Перевірте налаштування й модель.",
            ) from exc
        try:
            semantic_hits = index.search(
                search_index,
                query_vector,
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
        result["semantic"] = [_hit_to_dict(hit) for hit in semantic_hits]

    if payload.mode in ("keyword", "both"):
        started = time.perf_counter()
        try:
            keyword_hits = keyword.search(
                keyword_index,
                payload.query,
                top_k=payload.top_k,
                filters=filters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["elapsed"]["keyword"] = time.perf_counter() - started
        result["keyword"] = [_hit_to_dict(hit) for hit in keyword_hits]
    return result
