"""Run sequential real-provider draft measurements and persist each result."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import gateway, jobs, samples
from app.providers.base import ProviderError

RUNS_JSON = ROOT / "eval" / "provider_runs.json"
SUMMARY_CSV = ROOT / "eval" / "provider_summary.csv"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"
CSV_FIELDS = (
    "run", "sample_id", "kind", "status", "provider", "model", "fallback",
    "attempts", "first_token_seconds", "total_seconds", "total_tokens",
    "warnings_count", "error_kind",
)


def _read_previous_results() -> list[dict[str, Any]]:
    if not RUNS_JSON.exists():
        return []
    try:
        rows = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{RUNS_JSON.name} is not valid JSON") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{RUNS_JSON.name} must contain a JSON list of results")
    return rows


def _write_json(rows: list[dict[str, Any]]) -> None:
    temporary = RUNS_JSON.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(RUNS_JSON)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    elapsed = result.get("elapsed") or {}
    usage = result.get("usage") or {}
    return {
        "run": result["run"],
        "sample_id": result["sample_id"],
        "kind": result["kind"],
        "status": result["status"],
        "provider": result.get("provider") or "",
        "model": result.get("model") or "",
        "fallback": result.get("fallback", False),
        "attempts": result.get("attempts", 0),
        "first_token_seconds": elapsed.get("first_token") or "",
        "total_seconds": elapsed.get("total") or "",
        "total_tokens": usage.get("total_tokens", ""),
        "warnings_count": len(result.get("warnings") or []),
        "error_kind": (result.get("error") or {}).get("kind", ""),
    }


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


def _update_results_markdown(rows: list[dict[str, Any]]) -> None:
    content = RESULTS_MD.read_text(encoding="utf-8")
    generated_heading = "\n## Виміри справжнього провайдера"
    legacy_heading = "\n## Що залишилось перевірити"
    section_starts = [
        position for heading in (generated_heading, legacy_heading)
        if (position := content.find(heading)) >= 0
    ]
    prefix = content[:min(section_starts)].rstrip() if section_starts else content.rstrip()
    sections = ["## Виміри справжнього провайдера"]

    runs = sorted({row["run"] for row in rows})
    for run_number in runs:
        run_rows = [row for row in rows if row["run"] == run_number]
        expected = max((row.get("run_size", 0) for row in run_rows), default=0)
        completed = sum(row["status"] in jobs.TERMINAL for row in run_rows)
        successful = sum(row["status"] == "done" for row in run_rows)
        successes = [
            row for row in run_rows
            if row["status"] == "done" and not row.get("fallback")
        ]
        sections.extend([
            f"\n### Прогін {run_number}: записано {len(run_rows)} із {expected}; "
            f"завершено {completed}, успішно {successful}",
            "",
            "| Звернення | Статус | Провайдер / модель | Спроби | Помилка | Перший фрагмент, с | Разом, с | Токени | Fallback |",
            "|---|---|---|---:|---|---:|---:|---:|---|",
        ])
        for row in run_rows:
            elapsed = row.get("elapsed") or {}
            usage = row.get("usage") or {}
            error_kind = (row.get("error") or {}).get("kind") or "—"
            first = elapsed.get("first_token")
            total = elapsed.get("total")
            provider_model = " / ".join(
                item for item in (row.get("provider"), row.get("model")) if item
            )
            sections.append(
                f"| {row['sample_id']} | {row['status']} | {provider_model or '—'} "
                f"| {row.get('attempts', 0)} | {error_kind} "
                f"| {first if first is not None else '—'} "
                f"| {total if total is not None else '—'} "
                f"| {usage.get('total_tokens', '—')} | {row.get('fallback', False)} |"
            )
        if successes:
            first_token_times = [
                row["elapsed"]["first_token"] for row in successes
                if row.get("elapsed", {}).get("first_token") is not None
            ]
            total_times = [
                row["elapsed"]["total"] for row in successes
                if row.get("elapsed", {}).get("total") is not None
            ]
            averages = []
            if first_token_times:
                averages.append(f"перший фрагмент {mean(first_token_times):.3f} с")
            if total_times:
                averages.append(f"повна відповідь {mean(total_times):.3f} с")
            if averages:
                sections.append(
                    "\nСереднє для успішних відповідей основного провайдера: "
                    + "; ".join(averages)
                    + "."
                )
        elif run_rows:
            failures: dict[str, int] = {}
            for row in run_rows:
                error_kind = (row.get("error") or {}).get("kind")
                if row["status"] != "done" and error_kind:
                    failures[error_kind] = failures.get(error_kind, 0) + 1
            failure_summary = ", ".join(
                f"{kind}: {count}" for kind, count in sorted(failures.items())
            )
            sections.append(
                "\nУспішних відповідей немає, тому час генерації та токени "
                "виміряти не вдалося."
                + (f" Зафіксовані помилки: {failure_summary}." if failure_summary else "")
            )
        if len(run_rows) < expected:
            sections.append("Прогін неповний; запусти скрипт з тим самим `--run`, щоб продовжити.")

    sections.extend([
        "",
        "Виміри зроблені локальним скриптом на зразках із `samples/requests.json`; "
        "див. `provider_runs.json` і `provider_summary.csv` для повних даних.",
    ])
    RESULTS_MD.write_text(prefix + "\n\n" + "\n".join(sections) + "\n", encoding="utf-8")


def _run_one(sample: dict[str, Any], run_number: int, run_size: int) -> dict[str, Any]:
    draft = jobs.start(sample, "eval-real-provider")
    deadline = time.monotonic() + gateway.TOTAL_TIMEOUT + max(gateway.TIMEOUT, 10.0)
    while time.monotonic() < deadline:
        current = jobs.get(draft.draft_id)
        if current is None:
            raise RuntimeError(f"Draft {draft.draft_id} disappeared from the registry")
        if current.status in jobs.TERMINAL:
            break
        time.sleep(0.05)
    else:
        try:
            jobs.cancel(draft.draft_id)
        except jobs.DraftConflict:
            pass
        current = jobs.get(draft.draft_id)

    if current is None:
        raise RuntimeError(f"Draft {draft.draft_id} disappeared from the registry")
    return {
        "run": run_number,
        "run_size": run_size,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample_id": sample["id"],
        "kind": sample.get("kind"),
        "draft_id": draft.draft_id,
        "status": current.status,
        "provider": current.provider,
        "model": current.model,
        "fallback": current.fallback,
        "attempts": current.attempts,
        "elapsed": current.elapsed,
        "usage": current.usage,
        "warnings": current.warnings,
        "error": current.error,
        "text": current.text,
        "status_events": [
            event["data"] for event in current.events if event["event"] == "status"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sequential real-provider drafts and record timing results."
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="number of sample requests to run (default: 5)")
    parser.add_argument("--run", type=int, default=1,
                        help="run number; use another number for a fresh repeat")
    parser.add_argument("--dry-run", action="store_true",
                        help="list selected samples without contacting a provider")
    args = parser.parse_args()

    all_samples = samples.all_samples()
    if args.limit < 1 or args.limit > len(all_samples):
        parser.error(f"--limit must be between 1 and {len(all_samples)}")
    if args.run < 1:
        parser.error("--run must be positive")
    selected = all_samples[:args.limit]

    if args.dry_run:
        print(f"Provider: {gateway.PROVIDER}; samples: {', '.join(s['id'] for s in selected)}")
        return 0
    if gateway.PROVIDER != "openai":
        print("Set LLM_PROVIDER=openai in .env to run real-provider measurements.", flush=True)
        return 2
    try:
        gateway.make_provider("openai")
    except ProviderError as exc:
        print(exc.message, flush=True)
        return 2

    try:
        results = _read_previous_results()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    completed = {
        (row.get("run"), row.get("sample_id"))
        for row in results
    }
    for index, sample in enumerate(selected, start=1):
        key = (args.run, sample["id"])
        if key in completed:
            print(f"[{index}/{args.limit}] {sample['id']} already recorded; skipping.", flush=True)
            continue

        print(f"[{index}/{args.limit}] {sample['id']} · {sample.get('kind', '')}", flush=True)
        result = _run_one(sample, args.run, args.limit)
        results.append(result)
        _write_json(results)
        _write_csv(results)
        _update_results_markdown(results)
        elapsed = result.get("elapsed") or {}
        print(
            f"  {result['status']} · provider={result.get('provider') or '—'}"
            f" · first_token={elapsed.get('first_token', '—')}s"
            f" · total={elapsed.get('total', '—')}s"
            f" · attempts={result['attempts']}",
            flush=True,
        )

    print(f"Raw results: {RUNS_JSON.relative_to(ROOT)}", flush=True)
    print(f"CSV summary: {SUMMARY_CSV.relative_to(ROOT)}", flush=True)
    print(f"Updated report: {RESULTS_MD.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
