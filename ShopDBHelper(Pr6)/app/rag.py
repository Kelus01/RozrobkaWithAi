"""RAG orchestration: retrieve, build context, generate, and verify citations."""

from dataclasses import dataclass, field
import logging
import time

from . import llm, retrieval
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Response exposed to the web layer, including retrieval diagnostics."""

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    search_index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Answer only from safe, retrieved context; reject unverified citations."""
    started = time.perf_counter()
    hits = retrieval.retrieve(question, search_index, keyword_index, filters)
    retrieval_elapsed = time.perf_counter() - started
    context, context_sources = retrieval.build_context(hits)
    if not context_sources:
        return Answer(
            text=retrieval.NO_ANSWER_TEXT,
            found=False,
            elapsed={"retrieval": retrieval_elapsed, "generation": 0.0},
        )

    generated = llm.ask(question, context)
    refs = generated["source_refs"]
    source_by_ref = {source.ref: source for source in context_sources}
    unknown_refs = set(refs) - source_by_ref.keys()
    if unknown_refs:
        logger.warning("Model returned citations not present in context: %s", sorted(unknown_refs))
        return Answer(
            text=retrieval.NO_ANSWER_TEXT,
            found=False,
            retrieved=[Hit(chunk=source.chunk, score=source.score) for source in context_sources],
            model=generated["model"],
            elapsed={
                "retrieval": retrieval_elapsed,
                "generation": generated["elapsed"],
            },
            usage=generated["usage"],
        )

    if generated["found"] and not refs:
        logger.warning("Model claimed a grounded answer without citing a source")
        return Answer(
            text=retrieval.NO_ANSWER_TEXT,
            found=False,
            retrieved=[Hit(chunk=source.chunk, score=source.score) for source in context_sources],
            model=generated["model"],
            elapsed={
                "retrieval": retrieval_elapsed,
                "generation": generated["elapsed"],
            },
            usage=generated["usage"],
        )

    if not generated["found"]:
        return Answer(
            text=retrieval.NO_ANSWER_TEXT,
            found=False,
            retrieved=[Hit(chunk=source.chunk, score=source.score) for source in context_sources],
            model=generated["model"],
            elapsed={
                "retrieval": retrieval_elapsed,
                "generation": generated["elapsed"],
            },
            usage=generated["usage"],
        )

    return Answer(
        text=generated["answer"],
        found=True,
        sources=[source_by_ref[ref] for ref in refs],
        retrieved=[Hit(chunk=source.chunk, score=source.score) for source in context_sources],
        model=generated["model"],
        elapsed={
            "retrieval": retrieval_elapsed,
            "generation": generated["elapsed"],
        },
        usage=generated["usage"],
    )
