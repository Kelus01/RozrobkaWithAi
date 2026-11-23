"""Сховище стану звернень: зберегти, прочитати, перелічити.

Стан живе поза памʼяттю процесу, бо звернення чекає на оператора
довше, ніж живе один процес застосунку. Модуль не знає ані про модель,
ані про магазин, ані про кроки workflow: він зберігає `Run` і
повертає його таким, яким зберіг.

Каталог — `RUNS_DIR` з `.env` (за замовчуванням `runs/` у папці
роботи). До репозиторію він не потрапляє: там тексти звернень клієнтів.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* у чому зберігати: JSON-файл на звернення чи SQLite (`sqlite3` зі
  стандартної бібліотеки) — і що це змінює для переліку за статусом;
* як записувати, щоб збій посередині запису не лишив зіпсованого файлу;
* як перетворити `Run` на JSON і назад, разом зі списком `Step`;
* що робити, якщо два запити одночасно змінюють те саме звернення —
  наприклад, оператор двічі натиснув «підтвердити». Чи це справа
  сховища, чи workflow?
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))
_LOCK = threading.Lock()


def _run_path(run_id: str) -> Path:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_.")
    if not safe or safe != run_id:
        raise ValueError("Некоректний run_id")
    return RUNS_DIR / f"{safe}.json"


def _to_dict(run: Run) -> dict:
    data = {
        "run_id": run.run_id,
        "customer_id": run.customer_id,
        "text": run.text,
        "status": run.status,
        "route": run.route,
        "steps": [vars(step).copy() for step in run.steps],
        "data": run.data,
        "pending": run.pending,
        "decision": run.decision,
        "reply": run.reply,
        "stopped": run.stopped,
        "usage": run.usage,
        "elapsed": run.elapsed,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }
    return data


def _from_dict(data: dict) -> Run:
    return Run(
        run_id=data["run_id"],
        customer_id=data["customer_id"],
        text=data["text"],
        status=data.get("status", "running"),
        route=data.get("route"),
        steps=[Step(**step) for step in data.get("steps", [])],
        data=data.get("data", {}),
        pending=data.get("pending"),
        decision=data.get("decision"),
        reply=data.get("reply"),
        stopped=data.get("stopped"),
        usage=data.get("usage", {}),
        elapsed=data.get("elapsed", {}),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def save(run: Run) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = _run_path(run.run_id)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{run.run_id}-", suffix=".tmp",
                                    dir=str(RUNS_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_to_dict(run), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        with _LOCK:
            os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load(run_id: str) -> Run | None:
    try:
        path = _run_path(run_id)
        with path.open("r", encoding="utf-8") as handle:
            return _from_dict(json.load(handle))
    except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def list_runs(status: str | None = None) -> list[Run]:
    if not RUNS_DIR.exists():
        return []
    runs: list[Run] = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            run = _from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            continue
        if status is None or run.status == status:
            runs.append(run)
    runs.sort(key=lambda item: item.created_at or "", reverse=True)
    return runs
