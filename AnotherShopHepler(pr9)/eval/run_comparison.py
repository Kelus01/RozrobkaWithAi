import json
import csv
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import assistant, workflow
from shop import service


ROOT = Path(__file__).parent
REQUESTS = json.loads((ROOT / "requests.json").read_text(encoding="utf-8"))
OUT = ROOT / "comparison.json"
SUMMARY_JSON = ROOT / "summary.json"
SUMMARY_CSV = ROOT / "summary.csv"


def snapshot_shop() -> dict:
    snapshot = {
        "returns": service.list_returns(),
        "orders": {},
    }
    for customer in service.list_customers():
        for order in service.list_orders(customer["customer_id"]):
            full = service.get_order(order["order_id"])
            snapshot["orders"][order["order_id"]] = {
                "status": full["status"],
                "status_label": full["status_label"],
            }
    return snapshot


def mutation_calls_from_workflow(run) -> list[dict]:
    return [
        {
            "name": step.name,
            "status": step.status,
            "input": step.input,
            "output": step.output,
        }
        for step in run.steps
        if step.name in {"create_return", "cancel_order"}
    ]


def mutation_calls_from_agent(answer) -> list[dict]:
    return [
        {
            "name": call.name,
            "status": call.status,
            "arguments": call.arguments,
            "result": call.result,
        }
        for call in answer.calls
        if call.name in {"create_return", "cancel_order"}
    ]


def route_correct(request: dict, route: str | None) -> bool:
    expected = request.get("expect_route")
    if expected == "декілька":
        return route in {"multiple", "handed_off"}
    if expected is None:
        return route in {"other", "unknown", "handed_off", "needs_info", "done", "rejected"}
    return route == expected


def workflow_run(request: dict) -> dict:
    run = workflow.start(request["text"], request["customer"])
    if run.status == "awaiting_approval":
        run = workflow.resume(run.run_id, True, "Автоматична перевірка під час порівняння")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "route": run.route,
        "reply": run.reply,
        "steps": [vars(step) for step in run.steps],
        "usage": run.usage,
        "elapsed": run.elapsed,
        "stopped": run.stopped,
        "mutations": mutation_calls_from_workflow(run),
        "model_calls": len([step for step in run.steps if step.kind == "model"]),
    }


def agent_run(request: dict) -> dict:
    answer = assistant.answer(request["text"], request["customer"])
    return {
        "status": answer.stopped,
        "reply": answer.text,
        "calls": [vars(call) for call in answer.calls],
        "rounds": answer.rounds,
        "usage": answer.usage,
        "elapsed": answer.elapsed,
        "stopped": answer.stopped,
        "mutations": mutation_calls_from_agent(answer),
        "model_calls": answer.rounds,
    }


def main() -> None:
    try:
        from app.llm import complete
        complete([{"role": "user", "content": "Відповідь одним словом: тест"}])
    except Exception as exc:
        message = f"Попередня перевірка моделі не пройшла: {exc}"
        print(message, flush=True)
        (ROOT / "abort.txt").write_text(message + "\n", encoding="utf-8")
        return

    results = []
    if OUT.exists():
        try:
            saved = json.loads(OUT.read_text(encoding="utf-8"))
            if isinstance(saved, list):
                results = saved
        except json.JSONDecodeError:
            results = []

    completed_keys = {
        (row.get("mode"), row.get("attempt"), row.get("id"))
        for row in results
    }
    if results:
        print(
            f"Відновлення порівняння: знайдено {len(results)} раніше збережених прогонів.",
            flush=True,
        )
    by_id = {request["id"]: request for request in REQUESTS}
    aborted = None
    total = len(("workflow", "agent")) * 3 * len(REQUESTS)
    completed_count = len(results)
    print(f"Початок порівняння: {len(REQUESTS)} звернень, 2 способи, 3 прогони ({total} прогонів).",
          flush=True)
    for mode in ("workflow", "agent"):
        for attempt in range(1, 4):
            for request in REQUESTS:
                key = (mode, attempt, request["id"])
                if key in completed_keys:
                    continue
                completed_count += 1
                print(
                    f"[{completed_count}/{total}] {mode} · {request['id']} · спроба {attempt}",
                    flush=True,
                )
                if request.get("after") == "r01":
                    first = by_id["r01"]
                    service.reset()
                    workflow_run(first) if mode == "workflow" else agent_run(first)
                else:
                    service.reset()
                data = workflow_run(request) if mode == "workflow" else agent_run(request)
                if data.get("stopped") in {"model_error", "llm_error", "error"}:
                    aborted = (
                        f"{mode}/{request['id']}: модуль моделі повернув контрольовану помилку "
                        f"({data.get('stopped')}). Повний прогін зупинено."
                    )
                    print(aborted, flush=True)
                    break
                route = data.get("route")
                results.append({
                    "id": request["id"],
                    "mode": mode,
                    "attempt": attempt,
                    "expected_route": request.get("expect_route"),
                    "route_correct": route_correct(request, route),
                    "expected_action": request.get("expect_action"),
                    "mutations": data.get("mutations", []),
                    "shop": snapshot_shop(),
                    "data": data,
                })
                OUT.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if aborted:
                break
        if aborted:
            break
    if aborted:
        (ROOT / "abort.txt").write_text(aborted + "\n", encoding="utf-8")
        print("Сирі часткові записи збережено у eval/comparison.json.", flush=True)
        return
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = []
    for request in REQUESTS:
        for mode in ("workflow", "agent"):
            rows = [
                row for row in results
                if row["id"] == request["id"] and row["mode"] == mode
            ]
            usages = [row["data"].get("usage") or {} for row in rows]
            elapsed = [row["data"].get("elapsed") or {} for row in rows]
            distinct = len({
                json.dumps(
                    row["data"].get("steps") or row["data"].get("calls"),
                           sort_keys=True)
                for row in rows
            })
            summary.append({
                "id": request["id"],
                "kind": request.get("kind"),
                "mode": mode,
                "runs": len(rows),
                "route_correct_count": sum(row["route_correct"] for row in rows),
                "route_correct_rate": round(
                    sum(row["route_correct"] for row in rows) / len(rows), 3
                ),
                "mutation_count": sum(
                    len(row["mutations"]) for row in rows
                ),
                "avg_total_tokens": round(mean(
                    float(u.get("total_tokens", 0) or 0) for u in usages
                ), 2),
                "avg_elapsed_seconds": round(mean(
                    float(e.get(
                        "total",
                        sum(v for v in e.values() if isinstance(v, (int, float))),
                    ))
                    for e in elapsed
                ), 3),
                "distinct_paths": distinct,
            })
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
