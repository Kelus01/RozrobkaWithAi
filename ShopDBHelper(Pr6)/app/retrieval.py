"""Select safe search results and build a bounded context for the LLM."""

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from . import embeddings, index, keyword
from .documents import Chunk
from .index import Hit, SearchIndex
from .keyword import KeywordIndex

load_dotenv()


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} має бути цілим числом.") from exc
    if value < 1:
        raise ValueError(f"{name} має бути більшим за нуль.")
    return value


CONTEXT_CHUNKS = _positive_int("RAG_CONTEXT_CHUNKS", 4)
CONTEXT_BUDGET = _positive_int("RAG_CONTEXT_BUDGET", 1500)
RRF_K = 60
NO_ANSWER_TEXT = "У базі знань немає достатньо інформації, щоб відповісти на це питання."


@dataclass
class Source:
    """A numbered chunk shown to the model and cited in its answer."""

    ref: int
    chunk: Chunk
    score: float


def retrieve(
    query: str,
    search_index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Run hybrid retrieval with server-enforced customer and current-only filters."""
    requested_filters = dict(filters or {})
    unknown = set(requested_filters) - {"product"}
    if unknown:
        raise ValueError(f"Недозволені поля фільтра: {', '.join(sorted(unknown))}.")
    if search_index.model_name != embeddings.MODEL_NAME:
        raise ValueError(
            "Модель ембедінгів не збігається з індексом. Виконайте python ingest.py."
        )

    safe_filters = {
        **requested_filters,
        "audience": "клієнти",
        "status": "чинний",
    }
    vector = embeddings.embed_query(query)
    semantic_hits = index.search(
        search_index,
        vector,
        top_k=index.DEFAULT_TOP_K,
        filters=safe_filters,
        threshold=index.SIMILARITY_THRESHOLD,
    )
    keyword_hits = (
        keyword.search(
            keyword_index,
            query,
            top_k=index.DEFAULT_TOP_K,
            filters=safe_filters,
        )
        if keyword_index is not None
        else []
    )

    fused_scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], Chunk] = {}
    for ranked_hits in (semantic_hits, keyword_hits):
        seen_sources: dict[str, int] = {}
        for rank, hit in enumerate(ranked_hits, start=1):
            source = hit.chunk.source
            seen_sources[source] = seen_sources.get(source, 0) + 1
            if seen_sources[source] > 2:
                continue
            key = (source, int(hit.chunk.metadata.get("chunk_index", rank)))
            fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (RRF_K + rank)
            chunks[key] = hit.chunk

    ranked = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        Hit(chunk=chunks[key], score=score)
        for key, score in ranked[: max(index.DEFAULT_TOP_K, CONTEXT_CHUNKS)]
    ]


def _estimate_tokens(text: str) -> int:
    """Conservative estimate for mixed Ukrainian text without a model tokenizer."""
    return max(1, math.ceil(len(text) / 4))


def build_context(
    hits: list[Hit],
    budget: int = CONTEXT_BUDGET,
) -> tuple[str, list[Source]]:
    """Build verbatim, numbered source blocks that fit the token budget."""
    if budget < 1:
        raise ValueError("Бюджет контексту має бути більшим за нуль.")

    blocks: list[str] = []
    sources: list[Source] = []
    for hit in hits[:CONTEXT_CHUNKS]:
        metadata = hit.chunk.metadata
        title = str(metadata.get("title") or hit.chunk.source)
        heading = str(metadata.get("heading") or "Без розділу")
        updated = str(metadata.get("updated") or "дата невідома")
        ref = len(sources) + 1
        block = (
            f"[{ref}] Документ: {title} | Розділ: {heading} | "
            f"Редакція: {updated}\n{hit.chunk.text}"
        )
        candidate = "\n\n".join([*blocks, block])
        if _estimate_tokens(candidate) > budget:
            continue
        blocks.append(block)
        sources.append(Source(ref=ref, chunk=hit.chunk, score=hit.score))

    return "\n\n".join(blocks), sources
