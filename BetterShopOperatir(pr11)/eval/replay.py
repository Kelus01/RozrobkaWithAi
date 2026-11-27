"""Replay a deterministic sequence through the local drafts API."""

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.samples import all_samples


def _json_request(url: str, *, data: dict[str, Any] | None = None,
                  timeout: float = 10.0) -> tuple[dict[str, Any], str | None]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()), response.headers.get("X-Request-ID")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)], 3)


def summarize(results: list[dict[str, Any]], label: str, seed: int,
              count: int) -> dict[str, Any]:
    elapsed = [
        float(row["elapsed"]["total"]) for row in results
        if isinstance(row.get("elapsed"), dict)
        and isinstance(row["elapsed"].get("total"), (int, float))
    ]

    def usage_tokens(row: dict[str, Any]) -> int:
        return int((row.get("usage") or {}).get("total_tokens", 0))

    return {
        "label": label,
        "seed": seed,
        "count": count,
        "requests": len(results),
        "provider_calls": sum(row["attempts"] for row in results),
        "cache_hits": sum(row["cached"] for row in results),
        "total_cost_usd": round(sum(float(row["cost_usd"]) for row in results), 8),
        "total_tokens": sum(usage_tokens(row) for row in results if not row["cached"]),
        "cached_reused_tokens": sum(usage_tokens(row) for row in results if row["cached"]),
        "duration_seconds": {
            "median": round(median(elapsed), 3) if elapsed else None,
            "p95": _percentile(elapsed, 0.95),
        },
        "results": results,
    }


def replay(base_url: str, count: int, seed: int, client: str,
           timeout: float) -> list[dict[str, Any]]:
    sample_ids = [sample["id"] for sample in all_samples()]
    randomizer = random.Random(seed)
    sequence = [randomizer.choice(sample_ids) for _ in range(count)]
    results = []

    for index, sample_id in enumerate(sequence, start=1):
        body, http_request_id = _json_request(
            f"{base_url}/api/drafts",
            data={"request_id": sample_id, "client": client, "fresh": False},
            timeout=timeout,
        )
        draft_id = body["draft_id"]
        request_id = http_request_id or body.get("trace_request_id")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot, _ = _json_request(
                f"{base_url}/api/drafts/{draft_id}", timeout=min(timeout, 10.0)
            )
            if snapshot.get("status") in {"done", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"Draft {draft_id} did not finish in time")

        result = {
            "index": index,
            "sample_id": sample_id,
            "request_id": request_id,
            "draft_id": draft_id,
            "status": snapshot["status"],
            "cached": bool(snapshot.get("cached")),
            "attempts": snapshot.get("attempts", 0),
            "elapsed": snapshot.get("elapsed", {}),
            "cost_usd": snapshot.get("cost_usd", 0.0),
            "cost_estimated": snapshot.get("cost_estimated", False),
            "usage": snapshot.get("usage"),
            "error_kind": (snapshot.get("error") or {}).get("kind"),
        }
        results.append(result)
        print(
            f"[{index}/{count}] {sample_id} {result['status']} "
            f"cached={result['cached']} attempts={result['attempts']} "
            f"cost=${result['cost_usd']:.6f}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--client")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label may contain only letters, numbers, underscores, and hyphens")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    client = args.client or f"eval-replay-{args.label}"

    try:
        results = replay(
            args.base_url.rstrip("/"), args.count, args.seed, client, args.timeout
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.error(str(exc))

    report = summarize(results, args.label, args.seed, args.count)
    output = args.output or Path("eval") / f"replay_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"},
                     ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
