"""Summarize JSON-lines telemetry without exposing prompt or customer data."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def analyze(events: list[dict[str, Any]], slow_threshold_seconds: float) -> dict[str, Any]:
    attempts = [event for event in events if event.get("event") == "llm.attempt"]
    durations = [
        float(event["duration_ms"]) / 1000 for event in attempts
        if isinstance(event.get("duration_ms"), (int, float))
    ]
    slow = sorted(
        (event for event in attempts
         if isinstance(event.get("duration_ms"), (int, float))
         and event["duration_ms"] >= slow_threshold_seconds * 1000),
        key=lambda event: event["duration_ms"],
        reverse=True,
    )
    failures = Counter(
        event.get("kind", "unknown")
        for event in attempts if event.get("status") != "done"
    )
    attempts_by_draft: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempts_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    costs_by_day: Counter[str] = Counter()
    costs_by_model: Counter[str] = Counter()
    provider_count = 0
    for event in attempts:
        if event.get("draft_id"):
            attempts_by_draft[event["draft_id"]].append(event)
        if event.get("cache_key"):
            attempts_by_key[event["cache_key"]].append(event)
        cost = float(event.get("cost_usd") or 0.0)
        timestamp = str(event.get("ts", ""))
        day = timestamp[:10] if len(timestamp) >= 10 else "unknown"
        costs_by_day[day] += cost
        costs_by_model[str(event.get("model") or "unknown")] += cost
        provider_count += 1

    cache_hits = sum(event.get("event") == "cache.hit" for event in events)
    request_count = sum(event.get("event") == "draft.created" for event in events)
    if not request_count:
        request_count = provider_count + cache_hits
    rescued = sum(
        any(event.get("status") == "done" for event in draft_attempts)
        and any(event.get("attempt", 0) > 1 for event in draft_attempts)
        for draft_attempts in attempts_by_draft.values()
    )
    return {
        "attempts": len(attempts),
        "provider_calls": provider_count,
        "cache_hits": cache_hits,
        "cache_hit_fraction": round(cache_hits / request_count, 4) if request_count else 0.0,
        "duration_seconds": {
            "count": len(durations),
            "median": round(median(durations), 3) if durations else None,
            "p95": _percentile(durations, 0.95),
        },
        "slow_calls": [
            {
                "request_id": event.get("request_id"),
                "draft_id": event.get("draft_id"),
                "attempt": event.get("attempt"),
                "provider": event.get("provider"),
                "model": event.get("model"),
                "queue_ms": event.get("queue_ms"),
                "first_token_ms": event.get("first_token_ms"),
                "duration_ms": event.get("duration_ms"),
                "status": event.get("status"),
            }
            for event in slow[:10]
        ],
        "failures_by_kind": dict(failures),
        "repeated_cache_keys": {
            key: len({row.get("draft_id") for row in rows})
            for key, rows in attempts_by_key.items()
            if len({row.get("draft_id") for row in rows}) > 1
        },
        "draft_attempt_counts": {
            key: len(rows) for key, rows in attempts_by_draft.items() if len(rows) > 1
        },
        "drafts_rescued_by_retry": rescued,
        "cost_usd_by_day": {
            day: round(value, 8) for day, value in sorted(costs_by_day.items())
        },
        "cost_usd_by_model": {
            model: round(value, 8) for model, value in sorted(costs_by_model.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("logs/app.jsonl"))
    parser.add_argument("--slow-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.slow_seconds < 0:
        parser.error("--slow-seconds cannot be negative")
    try:
        report = analyze(load_events(args.log), args.slow_seconds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
