"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.

Локальна модель завантажується один раз. Для моделей сімейства E5
застосовуються префікси `passage:` і `query:`, а повернуті вектори
нормалізуються, щоб їхній скалярний добуток дорівнював косинусній
схожості. Назва моделі читається з `.env`; після її зміни індекс треба
перебудувати.
"""

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Назва моделі ембедінгів на Hugging Face Hub.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


class EmbeddingError(RuntimeError):
    """Модель ембедінгів не завантажилася або повернула некоректний результат."""


def _prefixes() -> tuple[str, str]:
    if "e5" in MODEL_NAME.lower():
        return "passage: ", "query: "
    return "", ""


def _encode(texts: list[str], *, is_query: bool) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_model()
    passage_prefix, query_prefix = _prefixes()
    prefix = query_prefix if is_query else passage_prefix
    try:
        vectors = model.encode(
            [prefix + text for text in texts],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=not is_query and len(texts) > 16,
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Не вдалося закодувати текст моделлю {MODEL_NAME}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise EmbeddingError(
            f"Модель {MODEL_NAME} повернула вектори неочікуваної форми {array.shape}."
        )
    if not np.isfinite(array).all():
        raise EmbeddingError(f"Модель {MODEL_NAME} повернула нескінченні значення.")
    return array


@lru_cache(maxsize=1)
def get_model():
    """Повернути кешовану локальну модель ембедінгів."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, device=os.getenv("EMBEDDING_DEVICE") or None)
    except Exception as exc:
        raise EmbeddingError(
            f"Не вдалося завантажити модель {MODEL_NAME}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
    return _encode(texts, is_query=False)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    if not text.strip():
        raise ValueError("Пошуковий запит не може бути порожнім.")
    return _encode([text.strip()], is_query=True)[0]
