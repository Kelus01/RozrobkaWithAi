"""Пошук за ключовими словами BM25 — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Токенізатор зберігає артикули з дефісами як одне слово й однаково
застосовується до документів та запитів. BM25 оцінки не можна напряму
порівнювати з косинусною схожістю.
"""

from dataclasses import dataclass, field
import re

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit
from .index import filtered_positions

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-][^\W_]+)*", re.UNICODE)


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами. Що саме тут зберігати — вирішуєте ви."""

    chunks: list[Chunk]
    bm25: BM25Okapi
    tokens: list[list[str]]
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """
    normalized = text.casefold().replace("’", "'").replace("ʼ", "'")
    return TOKEN_PATTERN.findall(normalized)


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    return KeywordIndex(
        chunks=chunks,
        bm25=BM25Okapi(tokenized_chunks),
        tokens=tokenized_chunks,
    )


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими, щоб сторінка показувала обидва
    способи поруч.
    """
    if top_k < 1:
        raise ValueError("top_k має бути більшим за нуль.")
    query_tokens = tokenize(query)
    if not query_tokens or not index.chunks:
        return []

    positions = filtered_positions(index.chunks, filters)
    if not positions:
        return []
    scores = index.bm25.get_scores(query_tokens)
    query_set = set(query_tokens)
    ranked = sorted(
        (
            (position, float(scores[position]))
            for position in positions
            if query_set.intersection(index.tokens[position])
        ),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    return [Hit(chunk=index.chunks[position], score=score) for position, score in ranked]
