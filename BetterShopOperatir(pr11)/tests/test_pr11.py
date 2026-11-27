import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import budget, cache, costs, gateway, telemetry
from app.drafting import build_request
from app.main import app
from app.providers.base import ProviderError
from app.samples import get as get_sample
from eval.analyze import analyze


class CostTests(unittest.TestCase):
    def test_usage_cost_includes_cached_prompt_tokens(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
        with patch.object(costs, "INPUT_USD_PER_MILLION", 1.0), \
                patch.object(costs, "CACHED_INPUT_USD_PER_MILLION", 0.5), \
                patch.object(costs, "OUTPUT_USD_PER_MILLION", 2.0):
            cost = costs.calculate(usage, [], "")
        self.assertEqual(cost.input_tokens, 100)
        self.assertEqual(cost.output_tokens, 20)
        self.assertEqual(cost.cached_input_tokens, 40)
        self.assertFalse(cost.estimated)
        self.assertAlmostEqual(cost.cost_usd, 0.00012)

    def test_missing_usage_is_explicitly_estimated(self):
        result = costs.calculate(None, [{"role": "user", "content": "hello"}], "world")
        self.assertTrue(result.estimated)
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)


class CacheTests(unittest.TestCase):
    def test_key_ignores_request_identifiers_but_covers_prompt_and_parameters(self):
        first = build_request(get_sample("d01"))
        repeated = build_request(get_sample("d01"))
        first.metadata["request_id"] = "draft-a"
        repeated.metadata["request_id"] = "draft-b"
        key = cache.key_for("mock", "mock-model", "v1", first)
        self.assertEqual(key, cache.key_for("mock", "mock-model", "v1", repeated))
        changed = build_request(get_sample("d01"))
        changed.temperature += 0.1
        self.assertNotEqual(key, cache.key_for("mock", "mock-model", "v1", changed))

    def test_gateway_serves_cache_hit_and_fresh_bypasses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            budget_path = Path(directory) / "budget.json"
            with patch.object(cache, "CACHE_FILE", cache_path), \
                    patch.object(cache, "ENABLED", True), \
                    patch.object(cache, "_entries", None), \
                    patch.object(budget, "STATE_FILE", budget_path), \
                    patch.object(budget, "_state", None), \
                    patch.object(budget, "DAILY_LIMIT_USD", 10.0), \
                    patch.object(gateway, "PROVIDER", "mock"), \
                    patch.object(gateway, "FALLBACK", "none"), \
                    patch.object(gateway, "MAX_ATTEMPTS", 1), \
                    patch.object(gateway, "_slots", threading.BoundedSemaphore(2)), \
                    patch("app.providers.mock.FIRST_DELAY_MS", 0), \
                    patch("app.providers.mock.DELAY_MS", 0), \
                    patch("app.providers.mock.FAILURE_RATE", 0):
                request = build_request(get_sample("d01"))
                first = gateway.generate(
                    request, lambda _: None, lambda _: None, lambda: False,
                    request_id="trace-1", draft_id="draft-1", client="operator",
                )
                chunks = []
                statuses = []
                second = gateway.generate(
                    request, chunks.append, statuses.append, lambda: False,
                    request_id="trace-2", draft_id="draft-2", client="operator",
                )
                self.assertFalse(first.cached)
                self.assertEqual(first.attempts, 1)
                self.assertTrue(second.cached)
                self.assertEqual(second.attempts, 0)
                self.assertEqual(second.cost_usd, 0.0)
                self.assertEqual("".join(chunks), first.text)
                self.assertTrue(any(status.get("cache_hit") for status in statuses))

                fresh = gateway.generate(
                    request, lambda _: None, lambda _: None, lambda: False,
                    request_id="trace-3", draft_id="draft-3", client="operator",
                    fresh=True,
                )
                self.assertFalse(fresh.cached)
                self.assertEqual(fresh.attempts, 1)

    def test_expired_entries_are_not_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cache, "CACHE_FILE", Path(directory) / "cache.json"), \
                    patch.object(cache, "ENABLED", True), \
                    patch.object(cache, "TTL_SECONDS", 0), \
                    patch.object(cache, "_entries", None):
                cache.put("expired", {"text": "old"})
                self.assertIsNone(cache.get("expired"))


class BudgetTests(unittest.TestCase):
    def test_reservations_prevent_concurrent_overcommit_and_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "daily.json"
            with patch.object(budget, "STATE_FILE", state_path), \
                    patch.object(budget, "DAILY_LIMIT_USD", 0.01), \
                    patch.object(budget, "WARNING_FRACTION", 0.8), \
                    patch.object(budget, "_state", None):
                reservation, warning = budget.reserve(0.006)
                self.assertFalse(warning)
                with self.assertRaises(budget.BudgetExceeded):
                    budget.reserve(0.005)
                budget.settle(reservation, 0.006, 0.004)
                status = budget.snapshot()
                self.assertAlmostEqual(status["spent_usd"], 0.004)
                self.assertAlmostEqual(status["remaining_usd"], 0.006)

                budget._state = None
                self.assertAlmostEqual(budget.snapshot()["spent_usd"], 0.004)
                self.assertTrue(json.loads(state_path.read_text())["date"])

    def test_exhausted_budget_stops_before_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(budget, "STATE_FILE", Path(directory) / "daily.json"), \
                    patch.object(budget, "_state", None), \
                    patch.object(budget, "DAILY_LIMIT_USD", 0.00000001), \
                    patch.object(gateway, "PROVIDER", "mock"), \
                    patch.object(gateway, "FALLBACK", "mock"), \
                    patch.object(cache, "ENABLED", False), \
                    patch.object(gateway, "_slots", threading.BoundedSemaphore(1)), \
                    patch.object(gateway, "_stream_attempt") as provider_call:
                with self.assertRaises(ProviderError) as caught:
                    gateway.generate(
                        build_request(get_sample("d01")),
                        lambda _: None, lambda _: None, lambda: False,
                        request_id="budget-trace", draft_id="budget-draft",
                    )
                self.assertEqual(caught.exception.kind, "budget")
                self.assertIsNotNone(caught.exception.reset_at)
                provider_call.assert_not_called()


class AnalysisTests(unittest.TestCase):
    def test_report_splits_latency_failures_retries_and_cross_draft_repeats(self):
        report = analyze([
            {"event": "draft.created", "draft_id": "d1"},
            {"event": "draft.created", "draft_id": "d2"},
            {"event": "draft.created", "draft_id": "d3"},
            {"event": "llm.attempt", "draft_id": "d1", "cache_key": "same",
             "duration_ms": 100, "status": "failed", "kind": "rate_limit",
             "attempt": 1, "cost_usd": 0.01, "model": "m1"},
            {"event": "llm.attempt", "draft_id": "d1", "cache_key": "same",
             "duration_ms": 200, "status": "done", "attempt": 2,
             "cost_usd": 0.02, "model": "m1"},
            {"event": "llm.attempt", "draft_id": "d2", "cache_key": "same",
             "duration_ms": 1000, "status": "done", "attempt": 1,
             "cost_usd": 0.03, "model": "m2"},
            {"event": "cache.hit", "draft_id": "d3"},
        ], slow_threshold_seconds=0.5)
        self.assertEqual(report["duration_seconds"]["median"], 0.2)
        self.assertEqual(report["duration_seconds"]["p95"], 1.0)
        self.assertEqual(report["failures_by_kind"], {"rate_limit": 1})
        self.assertEqual(report["drafts_rescued_by_retry"], 1)
        self.assertEqual(report["repeated_cache_keys"], {"same": 2})
        self.assertAlmostEqual(report["cache_hit_fraction"], 1 / 3, places=4)


class TelemetryTests(unittest.TestCase):
    def test_http_request_id_matches_error_body_and_header(self):
        client = TestClient(app)
        for path in ("/api/drafts/not-a-real-draft", "/not-a-route"):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(
                    response.headers["X-Request-ID"],
                    response.json()["error"]["request_id"],
                )

    def test_jsonl_formatter_does_not_add_prompt_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.jsonl"
            original_path = telemetry._configured_path
            with patch.dict("os.environ", {"LOG_FILE": str(log_path)}):
                telemetry._configured_path = None
                telemetry.configure_logging()
                token = telemetry.set_request_id("trace-safe")
                try:
                    telemetry.log_event(
                        "llm.attempt", draft_id="draft-safe",
                        model="mock", status="done", cost_usd=0.0,
                    )
                finally:
                    telemetry.reset_request_id(token)
                    telemetry._configured_path = None
                    telemetry.configure_logging()
                    telemetry._configured_path = original_path
            row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["request_id"], "trace-safe")
            self.assertEqual(row["event"], "llm.attempt")
            self.assertNotIn("messages", row)
            self.assertNotIn("api_key", row)


if __name__ == "__main__":
    unittest.main()
