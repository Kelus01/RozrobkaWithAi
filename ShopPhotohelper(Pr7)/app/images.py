"""Підготовка зображення до надсилання моделі."""

import io
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання."""

    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі.

    Непридатний файл — `ImageError` з поясненням, що не так.
    """
    if content is None or len(content) == 0:
        raise ImageError("Файл порожній. Завантажте зображення рахунку.")
    if len(content) > UPLOAD_MAX_BYTES:
        raise ImageError(
            f"Файл занадто великий: {len(content)} байт, а максимум {UPLOAD_MAX_BYTES} байт."
        )

    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as exc:  # pragma: no cover - перехоплюється вище у веб-API
        raise ImageError("Файл не є допустимим зображенням.") from exc

    try:
        with Image.open(io.BytesIO(content)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "L", "RGBA", "P"}:
                img = img.convert("RGB")
            width, height = img.size
            original = {"width": width, "height": height, "bytes": len(content)}

            max_side = max(width, height)
            if max_side > IMAGE_MAX_SIDE:
                scale = IMAGE_MAX_SIDE / max_side
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if img.mode == "RGBA":
                img = img.convert("RGB")
            if img.mode == "P":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            payload = buffer.getvalue()

            sent = {"width": img.size[0], "height": img.size[1], "bytes": len(payload)}
            return PreparedImage(data=payload, mime="image/png", original=original, sent=sent)
    except Exception as exc:  # pragma: no cover - резерв для нестандартних зображень
        raise ImageError("Не вдалося підготувати зображення для аналізу.") from exc
