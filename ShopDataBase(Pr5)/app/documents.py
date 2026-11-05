"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.

Розбір метаданих і поділ на фрагменти реалізовані тут. Для фрагментів
використовуються заголовки розділів і абзаци, а довгі абзаци діляться
на вікна з перекриттям.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу вимірюються символами; заголовки розділів є межами.


def _chunk_settings() -> tuple[int, int]:
    try:
        size = int(os.getenv("CHUNK_SIZE", "600"))
        overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    except ValueError as exc:
        raise ValueError("CHUNK_SIZE і CHUNK_OVERLAP мають бути цілими числами.") from exc
    if size < 1:
        raise ValueError("CHUNK_SIZE має бути більшим за нуль.")
    if overlap < 0 or overlap >= size:
        raise ValueError("CHUNK_OVERLAP має бути невід'ємним і меншим за CHUNK_SIZE.")
    return size, overlap


CHUNK_SIZE, CHUNK_OVERLAP = _chunk_settings()


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        required_fields = {"title", "category", "product", "audience", "updated", "status"}
        missing_fields = required_fields - metadata.keys()
        if missing_fields:
            raise ValueError(
                f"{path.name}: у блоці метаданих бракує полів "
                f"{', '.join(sorted(missing_fields))}."
            )
        if not body.strip():
            raise ValueError(f"{path.name}: документ не містить основного тексту.")
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Ділити за розділами й абзацами, з вікном для довгих абзаців.

    Назва документа й заголовок розділу додаються до кожного фрагмента,
    щоб його зміст залишався зрозумілим поза оригінальним файлом.
    """
    size, overlap = _chunk_settings()
    document_title = str(metadata.get("title") or Path(source).stem)
    heading = document_title
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []

    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((heading, section_text))
            heading = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append((heading, section_text))

    chunks: list[Chunk] = []
    for section_heading, section_text in sections:
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", section_text)
            if paragraph.strip()
        ]
        section_chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > size:
                if current:
                    section_chunks.append(current)
                    current = ""
                start = 0
                while start < len(paragraph):
                    section_chunks.append(paragraph[start : start + size].strip())
                    if start + size >= len(paragraph):
                        break
                    start += size - overlap
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= size:
                current = candidate
                continue

            if current:
                section_chunks.append(current)
            carry = ""
            if overlap and current:
                carry = current[-overlap:].lstrip()
            current = f"{carry}\n\n{paragraph}".strip() if carry else paragraph
            if len(current) > size:
                section_chunks.append(current[:size].strip())
                current = current[size - overlap :].strip() if overlap else ""

        if current:
            section_chunks.append(current)

        for section_index, body in enumerate(section_chunks):
            chunk_metadata = {
                **metadata,
                "heading": section_heading,
                "chunk_index": len(chunks),
                "section_chunk": section_index,
            }
            chunks.append(
                Chunk(
                    text=f"{document_title}\n{section_heading}\n{body}",
                    source=source,
                    metadata=chunk_metadata,
                )
            )
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
