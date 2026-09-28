"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.
"""

from __future__ import annotations

import io
import time
from typing import Any

from PIL import Image, UnidentifiedImageError

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

_MODEL = None


class DetectionError(Exception):
    """Базовий тип помилки детекції."""


class InvalidImageError(DetectionError):
    """Невірний або порожній файл із зображенням."""


class ModelLoadError(DetectionError):
    """Помилка завантаження ваг моделі."""


class ModelInferenceError(DetectionError):
    """Помилка виконання inference моделі."""


def load_model():
    """Повернути готову до роботи модель, завантажену лише один раз."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ModelLoadError("Не вдалося імпортувати ultralytics.") from exc

    try:
        _MODEL = YOLO(WEIGHTS)
    except Exception as exc:
        raise ModelLoadError(f"Не вдалося завантажити ваги {WEIGHTS}.") from exc

    return _MODEL


def _open_valid_image(image_bytes: bytes) -> Image.Image:
    """Перевірити, що вхідні дані дійсно є зображенням."""
    if not image_bytes:
        raise InvalidImageError("Порожній файл. Завантажте зображення.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Невідомий або пошкоджений формат зображення.") from exc

    return rgb_image


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE):
    """Знайти обʼєкти на зображенні і повернути структурований результат."""
    if confidence < 0 or confidence > 1:
        raise InvalidImageError("Поріг упевненості має бути в межах від 0 до 1.")

    try:
        image = _open_valid_image(image_bytes)
    except InvalidImageError:
        raise

    model = load_model()
    started = time.perf_counter()

    try:
        results = model(image, conf=confidence, verbose=False)
    except Exception as exc:
        raise ModelInferenceError("Модель не змогла обробити зображення.") from exc

    elapsed = time.perf_counter() - started
    detections: list[dict[str, Any]] = []

    try:
        names = results[0].names if hasattr(results[0], "names") else {}
    except (IndexError, TypeError):
        names = {}

    for result in results:
        for box in getattr(result, "boxes", []):
            if box is None or len(box.conf) == 0:
                continue

            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id))
            conf = float(box.conf[0].item())

            if conf < confidence:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    return {
        "count": len(detections),
        "processing_time_sec": round(elapsed, 4),
        "objects": detections,
    }
