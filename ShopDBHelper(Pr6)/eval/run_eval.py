"""Run the question set through the full RAG pipeline and save diagnostics."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
sys.path.insert(0, str(ROOT))

from app import index, keyword, llm, rag


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "results.json",
        help="Path for generated results (default: eval/results.json)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next question after an API error or timeout",
    )
    args = parser.parse_args()

    question_set = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["питання"]
    search_index = index.load()
    keyword_index = keyword.build(search_index.chunks)
    previous_results = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        previous_results = {
            result["number"]: result
            for result in existing.get("results", [])
            if isinstance(result, dict) and isinstance(result.get("number"), int)
        }

    results = []
    encountered_error = False
    for number, item in enumerate(question_set, start=1):
        previous = previous_results.get(number)
        if previous is not None and "error" not in previous:
            results.append(previous)
            print(f"[{number}/{len(question_set)}] збережений результат пропущено", flush=True)
            continue

        print(f"[{number}/{len(question_set)}] {item['питання']}", flush=True)
        try:
            response = rag.answer(
                item["питання"],
                search_index,
                keyword_index,
                filters=item.get("фільтри"),
            )
        except llm.LLMError as exc:
            results.append(
                {
                    "number": number,
                    "question": item["питання"],
                    "expected_documents": item["очікувані_документи"],
                    "expected_answer": item["очікувана_відповідь"],
                    "expected_found": item["очікується_відповідь"],
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "manual_review": {
                        "answer_correct": None,
                        "grounded": None,
                        "citations_correct": None,
                        "refusal_correct": None,
                        "failure_type": None,
                    },
                }
            )
            save_results(args.output, sorted(results, key=lambda result: result["number"]))
            print(f"Помилка моделі; часткові результати збережено у {args.output}: {exc}")
            encountered_error = True
            if not args.continue_on_error:
                break
            continue

        retrieved_sources = list(dict.fromkeys(hit.chunk.source for hit in response.retrieved))
        expected_in_context = [
            source for source in item["очікувані_документи"] if source in retrieved_sources
        ]
        results.append(
            {
                "number": number,
                "question": item["питання"],
                "expected_documents": item["очікувані_документи"],
                "expected_documents_in_context": expected_in_context,
                "missing_expected_documents": [
                    source
                    for source in item["очікувані_документи"]
                    if source not in retrieved_sources
                ],
                "retrieved_document_names": retrieved_sources,
                "expected_answer": item["очікувана_відповідь"],
                "expected_found": item["очікується_відповідь"],
                "answer": response.text,
                "found": response.found,
                "sources": [
                    {
                        "ref": source.ref,
                        "source": source.chunk.source,
                        "heading": source.chunk.metadata.get("heading"),
                    }
                    for source in response.sources
                ],
                "retrieved": [
                    {
                        "source": hit.chunk.source,
                        "heading": hit.chunk.metadata.get("heading"),
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in response.retrieved
                ],
                "model": response.model,
                "elapsed": response.elapsed,
                "usage": response.usage,
                "manual_review": {
                    "answer_correct": None,
                    "grounded": None,
                    "citations_correct": None,
                    "refusal_correct": None,
                    "failure_type": None,
                },
            }
        )
        save_results(args.output, sorted(results, key=lambda result: result["number"]))

    print(f"Збережено {len(results)} із {len(question_set)} результатів у {args.output}")
    if encountered_error:
        print("Після усунення проблеми повторний запуск продовжить з першого невдалого питання.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
