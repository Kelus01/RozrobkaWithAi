"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Веб-рівень (`app/main.py`) звертається сюди з вектором запиту й отримує
список влучень із оцінкою схожості. Звідки береться вектор — не справа
індексу; що показувати клієнтові — не його справа теж.

Для цієї невеликої колекції використовується повний перебір нормалізованих
векторів і cosine similarity. Фільтри застосовуються до ранжування, а
метадані й назва моделі зберігаються поруч із матрицею векторів.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Кількість результатів і поріг можна перевизначити через `.env`.
def _search_settings() -> tuple[int, float | None]:
    try:
        top_k = int(os.getenv("SEARCH_TOP_K", "5"))
        raw_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
        threshold = float(raw_threshold) if raw_threshold else None
    except ValueError as exc:
        raise ValueError(
            "SEARCH_TOP_K має бути цілим числом, а SIMILARITY_THRESHOLD — числом."
        ) from exc
    if not 1 <= top_k <= 50:
        raise ValueError("SEARCH_TOP_K має бути в межах від 1 до 50.")
    if threshold is not None and (
        not np.isfinite(threshold) or threshold < -1 or threshold > 1
    ):
        raise ValueError("SIMILARITY_THRESHOLD має бути в межах від -1 до 1.")
    return top_k, threshold


DEFAULT_TOP_K, SIMILARITY_THRESHOLD = _search_settings()
FILTER_FIELDS = {"category", "product", "audience", "status"}
INDEX_VERSION = 1


class IndexNotFoundError(FileNotFoundError):
    """Індекс ще не побудовано."""


class IndexFormatError(ValueError):
    """Файли індексу пошкоджені або мають несумісний формат."""


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Тут перевіряється, що векторів стільки ж, скільки фрагментів, і
    вирішується, що зробити з векторами перед пошуком (наприклад,
    нормалізувати, якщо схожість — косинусна).
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Очікувалася матриця векторів, отримано форму {matrix.shape}.")
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Кількість векторів ({matrix.shape[0]}) не відповідає "
            f"кількості фрагментів ({len(chunks)})."
        )
    if not model_name.strip():
        raise ValueError("Назва моделі ембедінгів не може бути порожньою.")
    if not np.isfinite(matrix).all():
        raise ValueError("Матриця векторів містить нескінченні значення.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return SearchIndex(chunks=chunks, vectors=normalized, model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск.

    Формат — ваше рішення: `numpy` для векторів і JSON для фрагментів з
    метаданими — цілком достатньо. Назву моделі зберігайте обовʼязково.
    """
    path.mkdir(parents=True, exist_ok=True)
    vectors_tmp = path / "vectors.npy.tmp"
    chunks_tmp = path / "chunks.json.tmp"
    metadata = {
        "version": INDEX_VERSION,
        "model_name": index.model_name,
        "dimension": int(index.vectors.shape[1]) if index.vectors.ndim == 2 else 0,
        "chunks": [
            {
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in index.chunks
        ],
        "extra": index.extra,
    }
    with vectors_tmp.open("wb") as file:
        np.save(file, index.vectors, allow_pickle=False)
    chunks_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    vectors_tmp.replace(path / "vectors.npy")
    chunks_tmp.replace(path / "chunks.json")


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підняти зрозумілу помилку: веб-рівень має
    сказати користувачеві «індекс не збудовано», а не впасти з
    `FileNotFoundError` десь усередині.
    """
    vectors_path = path / "vectors.npy"
    chunks_path = path / "chunks.json"
    if not vectors_path.is_file() or not chunks_path.is_file():
        raise IndexNotFoundError(
            f"Індекс не знайдено в {path}; виконайте python ingest.py."
        )

    try:
        metadata = json.loads(chunks_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IndexFormatError(f"Не вдалося прочитати файли індексу: {exc}") from exc

    if not isinstance(metadata, dict) or metadata.get("version") != INDEX_VERSION:
        raise IndexFormatError("Версія формату індексу невідома або несумісна.")
    raw_chunks = metadata.get("chunks")
    model_name = metadata.get("model_name")
    if not isinstance(raw_chunks, list) or not isinstance(model_name, str) or not model_name:
        raise IndexFormatError("У файлі індексу немає фрагментів або назви моделі.")
    try:
        chunks = [
            Chunk(
                text=str(item["text"]),
                source=str(item["source"]),
                metadata=dict(item["metadata"]),
            )
            for item in raw_chunks
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexFormatError("Некоректні метадані фрагментів в індексі.") from exc

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise IndexFormatError(
            "Розмірність матриці векторів не відповідає кількості фрагментів."
        )
    expected_dimension = metadata.get("dimension")
    if expected_dimension != vectors.shape[1]:
        raise IndexFormatError("Розмірність векторів не відповідає метаданим індексу.")
    if not np.isfinite(vectors).all():
        raise IndexFormatError("Матриця векторів містить нескінченні значення.")
    return SearchIndex(
        chunks=chunks,
        vectors=np.asarray(vectors, dtype=np.float32),
        model_name=model_name,
        extra=dict(metadata.get("extra") or {}),
    )


def filtered_positions(chunks: list[Chunk], filters: dict | None) -> list[int]:
    if not filters:
        return list(range(len(chunks)))
    unknown = set(filters) - FILTER_FIELDS
    if unknown:
        raise ValueError(f"Невідомі поля фільтра: {', '.join(sorted(unknown))}.")
    normalized = {
        key: str(value).strip().casefold()
        for key, value in filters.items()
        if value is not None and str(value).strip()
    }
    return [
        position
        for position, chunk in enumerate(chunks)
        if all(
            str(chunk.metadata.get(key, "")).strip().casefold() == value
            for key, value in normalized.items()
        )
    ]


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    `filters` — умови на метадані фрагмента, наприклад
    `{"category": "інструкція", "audience": "клієнти"}`. Як трактувати
    поле з датою і що робити з невідомим полем — ваше рішення.

    Повертає щонайбільше `top_k` влучень, упорядкованих за спаданням
    оцінки. Оцінка — та сама міра схожості, що й для ранжування, і саме
    вона показується користувачеві.
    """
    if top_k < 1:
        raise ValueError("top_k має бути більшим за нуль.")
    if threshold is not None and (
        not np.isfinite(threshold) or threshold < -1 or threshold > 1
    ):
        raise ValueError("Поріг схожості має бути в межах від -1 до 1.")
    if index.vectors.ndim != 2:
        raise ValueError("Матриця векторів індексу має бути двовимірною.")

    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != index.vectors.shape[1]:
        raise ValueError(
            f"Розмірність запиту ({vector.shape[0]}) не відповідає "
            f"індексу ({index.vectors.shape[1]})."
        )
    if not np.isfinite(vector).all():
        raise ValueError("Вектор запиту містить нескінченні значення.")
    norm = np.linalg.norm(vector)
    normalized_query = vector / norm if norm > 0 else np.zeros_like(vector)

    positions = filtered_positions(index.chunks, filters)
    if not positions:
        return []
    scores = index.vectors[positions] @ normalized_query
    ranked = sorted(
        (
            (position, float(score))
            for position, score in zip(positions, scores)
            if threshold is None or float(score) >= threshold
        ),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    return [Hit(chunk=index.chunks[position], score=score) for position, score in ranked]
