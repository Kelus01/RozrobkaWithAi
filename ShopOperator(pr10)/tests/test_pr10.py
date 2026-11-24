from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import gateway, jobs, limits
from app.drafting import build_request, check
from app.main import app
from app.providers.base import Chunk, ProviderError, Request
from app.providers.mock import MockProvider
from app.samples import get as get_sample


class ScriptedProvider:
    name = "scripted"
    model = "scripted-1"

    def __init__(self, scripts):
        self.scripts = iter(scripts)
        self.calls = 0

    def complete(self, request):
        del request
        raise AssertionError("complete is not used by gateway tests")

    def stream(self, request):
        del request
        self.calls += 1
        outcome = next(self.scripts)
        if isinstance(outcome, Exception):
            raise outcome
        yield from outcome


def sample_request():
    return Request(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=50,
        temperature=0,
        metadata={"request_id": "d01"},
    )


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        with limits._lock:
            limits._requests.clear()
            limits._last_cleanup = 0.0

    def test_sliding_window_returns_retry_after_without_counting_rejections(self):
        with patch.object(limits, "PER_MINUTE", 1), patch.object(limits.time, "monotonic", side_effect=[10.0, 11.0, 71.0]):
            limits.acquire("client")
            with self.assertRaises(limits.RateLimited) as caught:
                limits.acquire("client")
            self.assertAlmostEqual(caught.exception.retry_after, 59.0)
            limits.acquire("client")


class MockFailureTests(unittest.TestCase):
    def test_configured_mock_failures_are_classified(self):
        expected = {
            "unavailable": ("unavailable", True),
            "auth": ("auth", False),
            "bad_request": ("bad_request", False),
            "rate_limit": ("rate_limit", True),
        }
        for configured_kind, (expected_kind, retryable) in expected.items():
            with self.subTest(kind=configured_kind), \
                    patch("app.providers.mock.FAILURE_RATE", 1), \
                    patch("app.providers.mock.FAILURE_KIND", configured_kind):
                with self.assertRaises(ProviderError) as caught:
                    list(MockProvider().stream(sample_request()))
                self.assertEqual(caught.exception.kind, expected_kind)
                self.assertEqual(caught.exception.retryable, retryable)
                if expected_kind == "rate_limit":
                    self.assertGreater(caught.exception.retry_after, 0)

    def test_mid_stream_failure_occurs_after_visible_chunks(self):
        chunks = []
        with patch("app.providers.mock.FIRST_DELAY_MS", 0), \
                patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 1), \
                patch("app.providers.mock.FAILURE_KIND", "mid_stream"):
            with self.assertRaises(ProviderError) as caught:
                for chunk in MockProvider().stream(sample_request()):
                    chunks.append(chunk.text)
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertTrue(any(chunks))

    def test_mock_timeout_failure_waits_past_attempt_deadline(self):
        with patch("app.providers.mock.FIRST_DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 1), \
                patch("app.providers.mock.FAILURE_KIND", "timeout"):
            with self.assertRaises(ProviderError) as caught:
                list(MockProvider(timeout=0.01).stream(sample_request()))
        self.assertEqual(caught.exception.kind, "timeout")


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.config = [
            patch.object(gateway, "PROVIDER", "mock"),
            patch.object(gateway, "FALLBACK", "none"),
            patch.object(gateway, "TIMEOUT", 0.5),
            patch.object(gateway, "IDLE_TIMEOUT", 0.2),
            patch.object(gateway, "TOTAL_TIMEOUT", 1.0),
            patch.object(gateway, "MAX_ATTEMPTS", 1),
            patch.object(gateway, "BACKOFF_BASE", 0),
            patch.object(gateway, "BACKOFF_MAX", 0),
            patch.object(gateway, "_slots", threading.BoundedSemaphore(2)),
        ]
        for item in self.config:
            item.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for item in reversed(self.config):
            item.stop()

    def test_mock_provider_streams_sample_without_network(self):
        with patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            received = []
            result = gateway.generate(sample_request(), received.append, lambda _: None, lambda: False)
        self.assertEqual(result.text, get_sample("d01")["reply_example"])
        self.assertEqual("".join(received), result.text)
        self.assertEqual(result.provider, "mock")
        self.assertIsNotNone(result.usage)

    def test_retries_temporary_errors_and_resets_visible_partial_text(self):
        provider = ScriptedProvider([
            [Chunk("partial")],
            [Chunk("complete"), Chunk("", usage={"total_tokens": 1}, finish_reason="stop")],
        ])
        # A generator can yield a chunk and then raise, so define that script explicitly.
        def stream(request):
            del request
            provider.calls += 1
            if provider.calls == 1:
                yield Chunk("partial")
                raise ProviderError("unavailable", "temporary", retryable=True)
            yield Chunk("complete", usage={"total_tokens": 1}, finish_reason="stop")

        provider.stream = stream
        statuses = []
        deltas = []
        with patch.object(gateway, "MAX_ATTEMPTS", 2), patch.object(gateway, "make_provider", return_value=provider):
            result = gateway.generate(sample_request(), deltas.append, statuses.append, lambda: False)
        self.assertEqual(result.text, "complete")
        self.assertEqual(deltas, ["partial", "complete"])
        self.assertTrue(any(status.get("reset") for status in statuses))
        self.assertEqual(provider.calls, 2)

    def test_attempt_timeout_is_bounded(self):
        provider = ScriptedProvider([])

        def slow_stream(request):
            del request
            time.sleep(0.12)
            yield Chunk("late")

        provider.stream = slow_stream
        with patch.object(gateway, "TIMEOUT", 0.03), patch.object(gateway, "IDLE_TIMEOUT", 0.2), \
                patch.object(gateway, "TOTAL_TIMEOUT", 0.4), patch.object(gateway, "make_provider", return_value=provider):
            started = time.monotonic()
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertLess(time.monotonic() - started, 0.1)

    def test_idle_timeout_stops_a_stream_after_partial_text(self):
        provider = ScriptedProvider([])

        def idle_stream(request):
            del request
            yield Chunk("first")
            time.sleep(0.12)
            yield Chunk("late")

        provider.stream = idle_stream
        deltas = []
        with patch.object(gateway, "TIMEOUT", 0.3), patch.object(gateway, "IDLE_TIMEOUT", 0.03), \
                patch.object(gateway, "TOTAL_TIMEOUT", 0.5), patch.object(gateway, "make_provider", return_value=provider):
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), deltas.append, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertEqual(deltas, ["first"])

    def test_total_timeout_applies_across_continuous_chunks(self):
        provider = ScriptedProvider([])

        def endless_stream(request):
            del request
            while True:
                time.sleep(0.02)
                yield Chunk("more")

        provider.stream = endless_stream
        with patch.object(gateway, "TIMEOUT", 0.5), patch.object(gateway, "IDLE_TIMEOUT", 0.2), \
                patch.object(gateway, "TOTAL_TIMEOUT", 0.08), patch.object(gateway, "make_provider", return_value=provider):
            started = time.monotonic()
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertLess(time.monotonic() - started, 0.15)

    def test_concurrency_limit_rejects_without_waiting(self):
        provider = ScriptedProvider([[Chunk("unused")]])
        gateway._slots.acquire()
        gateway._slots.acquire()
        try:
            with patch.object(gateway, "make_provider", return_value=provider):
                started = time.monotonic()
                with self.assertRaises(ProviderError) as caught:
                    gateway.generate(
                        sample_request(), lambda _: None, lambda _: None, lambda: False
                    )
            self.assertEqual(caught.exception.kind, "unavailable")
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(provider.calls, 0)
        finally:
            gateway._slots.release()
            gateway._slots.release()

    def test_auth_is_not_retried(self):
        provider = ScriptedProvider([ProviderError("auth", "denied", retryable=False)])
        with patch.object(gateway, "MAX_ATTEMPTS", 3), patch.object(gateway, "make_provider", return_value=provider):
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "auth")
        self.assertEqual(provider.calls, 1)

    def test_cancellation_interrupts_retry_backoff(self):
        provider = ScriptedProvider([
            ProviderError("unavailable", "temporary", retryable=True)
        ])
        cancellation = threading.Event()
        timer = threading.Timer(0.02, cancellation.set)
        timer.start()
        try:
            with patch.object(gateway, "MAX_ATTEMPTS", 3), \
                    patch.object(gateway, "BACKOFF_BASE", 0.3), \
                    patch.object(gateway, "BACKOFF_MAX", 0.3), \
                    patch.object(gateway, "make_provider", return_value=provider):
                with self.assertRaises(ProviderError) as caught:
                    gateway.generate(
                        sample_request(), lambda _: None, lambda _: None, cancellation.is_set
                    )
        finally:
            timer.join()
        self.assertEqual(caught.exception.kind, "cancelled")
        self.assertEqual(provider.calls, 1)

    def test_retry_after_larger_than_budget_is_not_waited(self):
        provider = ScriptedProvider([
            ProviderError("rate_limit", "slow down", retryable=True, retry_after=1.0)
        ])
        with patch.object(gateway, "TOTAL_TIMEOUT", 0.1), patch.object(gateway, "MAX_ATTEMPTS", 3), \
                patch.object(gateway, "make_provider", return_value=provider):
            started = time.monotonic()
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "rate_limit")
        self.assertEqual(provider.calls, 1)
        self.assertLess(time.monotonic() - started, 0.1)

    def test_retry_after_is_honored_when_it_fits_budget(self):
        provider = ScriptedProvider([
            ProviderError("rate_limit", "slow down", retryable=True, retry_after=0.02),
            [Chunk("ok", usage={"total_tokens": 1}, finish_reason="stop")],
        ])
        with patch.object(gateway, "MAX_ATTEMPTS", 2), patch.object(gateway, "make_provider", return_value=provider):
            started = time.monotonic()
            result = gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(result.text, "ok")
        self.assertGreaterEqual(time.monotonic() - started, 0.018)
        self.assertEqual(provider.calls, 2)

    def test_retry_after_larger_than_budget_does_not_call_fallback(self):
        primary = ScriptedProvider([
            ProviderError("rate_limit", "slow down", retryable=True, retry_after=1.0)
        ])
        fallback = ScriptedProvider([[Chunk("should not run")]])

        def factory(name, **kwargs):
            del kwargs
            return fallback if name == "mock" else primary

        with patch.object(gateway, "PROVIDER", "openai"), patch.object(gateway, "FALLBACK", "mock"), \
                patch.object(gateway, "TOTAL_TIMEOUT", 0.1), patch.object(gateway, "make_provider", side_effect=factory):
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(caught.exception.kind, "rate_limit")
        self.assertEqual(fallback.calls, 0)

    def test_temporary_failure_exhausts_only_configured_attempts(self):
        statuses = []
        with patch("app.providers.mock.FAILURE_RATE", 1), \
                patch("app.providers.mock.FAILURE_KIND", "unavailable"), \
                patch.object(gateway, "MAX_ATTEMPTS", 2), patch.object(gateway, "BACKOFF_BASE", 0):
            with self.assertRaises(ProviderError) as caught:
                gateway.generate(sample_request(), lambda _: None, statuses.append, lambda: False)
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(
            sum("provider_attempt" in status for status in statuses),
            2,
        )

    def test_fallback_is_marked_after_primary_failure(self):
        primary = ScriptedProvider([
            ProviderError("auth", "denied", retryable=False)
        ])
        fallback = ScriptedProvider([
            [Chunk("backup", usage={"total_tokens": 1}, finish_reason="stop")]
        ])

        def factory(name, **kwargs):
            del kwargs
            return fallback if name == "mock" else primary

        statuses = []
        with patch.object(gateway, "PROVIDER", "openai"), patch.object(gateway, "FALLBACK", "mock"), \
                patch.object(gateway, "make_provider", side_effect=factory):
            result = gateway.generate(sample_request(), lambda _: None, statuses.append, lambda: False)
        self.assertEqual(result.text, "backup")
        self.assertTrue(any(status.get("fallback") for status in statuses))


class JobAndRouteTests(unittest.TestCase):
    def setUp(self):
        with jobs._changed:
            jobs._drafts.clear()
            jobs._order.clear()
            jobs._cancellations.clear()

    def _wait_for(self, draft, status, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if draft.status == status:
                return
            time.sleep(0.01)
        self.fail(f"Draft did not reach {status}; current status={draft.status}")

    def test_background_draft_reaches_done_and_replays_events(self):
        with patch.object(gateway, "PROVIDER", "mock"), patch.object(gateway, "FALLBACK", "none"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            draft = jobs.start(get_sample("d01"), "test-client")
            self._wait_for(draft, "done")
        events = list(jobs.subscribe(draft.draft_id))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(draft.text, get_sample("d01")["reply_example"])

    def test_unexpected_background_exception_does_not_leave_running_status(self):
        with self.assertLogs("app.jobs", level="ERROR"):
            with patch("app.gateway.generate", side_effect=RuntimeError("internal detail")):
                draft = jobs.start(get_sample("d01"), "unexpected-test")
                self._wait_for(draft, "failed")
        self.assertEqual(draft.error["kind"], "unknown")
        self.assertNotIn("internal detail", draft.error["message"])

    def test_cancelled_draft_cannot_become_done(self):
        with patch.object(gateway, "PROVIDER", "mock"), patch.object(gateway, "FALLBACK", "none"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 250), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            draft = jobs.start(get_sample("d01"), "test-client")
            self._wait_for(draft, "running")
            self.assertEqual(jobs.cancel(draft.draft_id).status, "cancelled")
            time.sleep(0.3)
        self.assertEqual(draft.status, "cancelled")
        self.assertEqual(list(jobs.subscribe(draft.draft_id))[-1]["event"], "cancelled")
        with self.assertRaises(jobs.DraftConflict):
            jobs.cancel(draft.draft_id)

    def test_http_errors_use_common_shape_and_statuses(self):
        client = TestClient(app)
        unknown_request = client.post(
            "/api/drafts", json={"request_id": "missing", "client": "test"}
        )
        self.assertEqual(unknown_request.status_code, 404)
        self.assertEqual(unknown_request.json()["error"]["kind"], "not_found")

        unknown_draft = client.get("/api/drafts/unknown")
        self.assertEqual(unknown_draft.status_code, 404)
        self.assertIn("request_id", unknown_draft.json()["error"])

        invalid_request = client.post("/api/drafts", json={})
        self.assertEqual(invalid_request.status_code, 400)
        self.assertEqual(invalid_request.json()["error"]["kind"], "bad_request")

    def test_rate_limit_http_response_has_retry_after(self):
        client = TestClient(app)
        with patch.object(limits, "PER_MINUTE", 1), patch.object(gateway, "PROVIDER", "mock"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            with limits._lock:
                limits._requests.clear()
            first = client.post("/api/drafts", json={"request_id": "d01", "client": "limited"})
            second = client.post("/api/drafts", json={"request_id": "d02", "client": "limited"})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)
        self.assertEqual(second.json()["error"]["kind"], "rate_limit")

    def test_five_concurrent_orders_are_limited_per_client(self):
        client = TestClient(app)
        with patch.object(limits, "PER_MINUTE", 3), patch.object(gateway, "PROVIDER", "mock"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            with limits._lock:
                limits._requests.clear()
            with ThreadPoolExecutor(max_workers=5) as pool:
                responses = list(pool.map(
                    lambda _: client.post(
                        "/api/drafts", json={"request_id": "d01", "client": "burst-test"}
                    ),
                    range(5),
                ))
        self.assertEqual(
            sorted(response.status_code for response in responses),
            [202, 202, 202, 429, 429],
        )

    def test_cancelling_completed_draft_returns_conflict(self):
        client = TestClient(app)
        with patch.object(gateway, "PROVIDER", "mock"), patch.object(gateway, "FALLBACK", "none"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            created = client.post(
                "/api/drafts", json={"request_id": "d01", "client": "cancel-done"}
            )
            draft_id = created.json()["draft_id"]
            self._wait_for(jobs.get(draft_id), "done")
            response = client.post(f"/api/drafts/{draft_id}/cancel")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["kind"], "conflict")

    def test_event_endpoint_replays_completed_draft(self):
        client = TestClient(app)
        with patch.object(gateway, "PROVIDER", "mock"), patch.object(gateway, "FALLBACK", "none"), \
                patch("app.providers.mock.FIRST_DELAY_MS", 0), patch("app.providers.mock.DELAY_MS", 0), \
                patch("app.providers.mock.FAILURE_RATE", 0):
            created = client.post(
                "/api/drafts", json={"request_id": "d01", "client": "sse-test"}
            )
            self.assertEqual(created.status_code, 202)
            self._wait_for(jobs.get(created.json()["draft_id"]), "done")
            events = client.get(f"/api/drafts/{created.json()['draft_id']}/events")
        self.assertEqual(events.status_code, 200)
        self.assertIn("event: delta", events.text)
        self.assertIn("event: done", events.text)


class DraftingTests(unittest.TestCase):
    def test_request_separates_quoted_customer_text_from_verified_facts(self):
        sample = get_sample("d01")
        request = build_request(sample)
        self.assertEqual(request.metadata["request_id"], "d01")
        self.assertIn("<customer_message>", request.messages[1]["content"])
        self.assertIn("<verified_facts>", request.messages[1]["content"])
        self.assertIn("Не вигадуй", request.messages[0]["content"])

    def test_draft_checks_warn_about_missing_or_unsupported_references(self):
        sample = get_sample("d01")
        warnings = check("Заявку R-9999 прийнято.", sample)
        self.assertTrue(any("R-5001" in warning for warning in warnings))
        self.assertTrue(any("R-9999" in warning for warning in warnings))

    def test_valid_sample_reply_does_not_warn_about_omitted_years(self):
        sample = get_sample("d01")
        self.assertEqual(check(sample["reply_example"], sample), [])

