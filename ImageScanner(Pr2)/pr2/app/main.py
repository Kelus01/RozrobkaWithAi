"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from . import detector

app = FastAPI(title="Детекція обʼєктів — ПР2")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Query(detector.DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
):
    """Повернути знайдені обʼєкти або обробити помилку вводу."""
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="Файл не надіслано.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Порожній файл. Завантажте зображення.")

    try:
        return detector.detect(content, confidence=confidence)
    except detector.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except detector.ModelInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except detector.DetectionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
