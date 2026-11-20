"""Run eval/questions.json and save the actual assistant responses and tool calls."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import assistant, llm
from shop import service

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
DEFAULT_OUTPUT = ROOT / "eval" / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогнати питання з eval/questions.json і зберегти фактичні результати."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=13.0,
        help="мінімальна кількість секунд між зверненнями до моделі (за замовчуванням: 13)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="файл для результатів (за замовчуванням: eval/results.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="перезаписати файл результатів, якщо він уже існує",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="виконати лише перші N питань (зручно для короткої перевірки)",
    )
    parser.add_argument(
        "--start-at",
        help="почати з питання з указаним id, наприклад q02",
    )
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval не може бути від’ємним")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit має бути додатним числом")
    return args


def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        print(
            f"Файл {output_path} уже існує. Вкажи інший --output або додай --overwrite.",
            file=sys.stderr,
        )
        return 2
    if not QUESTIONS_PATH.is_file():
        print(f"Не знайдено файл із питаннями: {QUESTIONS_PATH}", file=sys.stderr)
        return 2

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        print("questions.json має містити JSON-масив питань.", file=sys.stderr)
        return 2
    start_index = 0
    if args.start_at:
        start_index = next(
            (i for i, question in enumerate(questions) if question.get("id") == args.start_at),
            -1,
        )
        if start_index < 0:
            print(f"Не знайдено питання з id {args.start_at!r}.", file=sys.stderr)
            return 2
    questions = questions[start_index:]
    if args.limit is not None:
        questions = questions[:args.limit]

    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(QUESTIONS_PATH.relative_to(ROOT)),
        "start_at": args.start_at or (questions[0].get("id") if questions else None),
        "question_limit": args.limit,
        "request_interval_seconds": args.interval,
        "results": [],
        "completed": False,
    }

    original_chat = llm.chat
    last_request_started = 0.0

    def paced_chat(*chat_args, **chat_kwargs):
        nonlocal last_request_started
        wait = args.interval - (time.monotonic() - last_request_started)
        if wait > 0:
            time.sleep(wait)
        last_request_started = time.monotonic()
        return original_chat(*chat_args, **chat_kwargs)

    llm.chat = paced_chat
    try:
        for index, question in enumerate(questions, start=1):
            question_id = question.get("id", f"q{index:02d}")
            print(f"[{index}/{len(questions)}] {question_id}: {question['question']}")
            service.reset()
            started = time.perf_counter()
            try:
                answer = assistant.answer(
                    question=question["question"],
                    customer_id=question["customer"],
                )
                entry = {
                    "id": question_id,
                    "kind": question.get("kind"),
                    "customer": question["customer"],
                    "question": question["question"],
                    "expect_tools": question.get("expect_tools", []),
                    "expect": question.get("expect"),
                    "answer": asdict(answer),
                    "runner_elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            except Exception as exc:
                entry = {
                    "id": question_id,
                    "kind": question.get("kind"),
                    "customer": question.get("customer"),
                    "question": question.get("question"),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "runner_elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                results["results"].append(entry)
                results["stopped_on_error"] = True
                save_results(output_path, results)
                print(f"Збій на {question_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

            results["results"].append(entry)
            save_results(output_path, results)
            print(
                f"    stopped={answer.stopped}, rounds={answer.rounds}, "
                f"tool_calls={len(answer.calls)}"
            )

            if answer.stopped in {"llm_error", "error"}:
                results["stopped_on_error"] = True
                save_results(output_path, results)
                print(
                    "Зупинено після помилки моделі/сервісу; часткові результати збережено. "
                    "Інші питання не проганялися.",
                    file=sys.stderr,
                )
                return 1

        results["completed"] = True
        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_results(output_path, results)
    finally:
        llm.chat = original_chat
        service.reset()

    print(f"Готово. Фактичні результати збережено у {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
