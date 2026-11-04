import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import llm
from app.main import app


class LLMTests(unittest.TestCase):
    def test_build_messages_keeps_instruction_context_and_question_separate(self):
        messages = llm.build_messages("Який графік підтримки?", "Правила магазину")

        self.assertEqual([message["role"] for message in messages], ["system", "user", "user"])
        self.assertIn("Правила магазину", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "Який графік підтримки?")

    def test_ask_validates_structured_response_and_round_trips_basic_fields(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"answer":"Відповідь","topic":"support","based_on_rules":"yes",'
                            '"needs_clarification":false,"transfer_to_operator":false,"order_number":"A-123"}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )
        environment = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.4",
            "LLM_MAX_TOKENS": "128",
            "LLM_BASE_URL": "https://example.test",
            "LLM_API_KEY": "secret",
            "LLM_TIMEOUT": "30",
        }

        with (
            patch.dict(os.environ, environment),
            patch.object(llm, "get_client", return_value=client),
            patch.object(llm.time, "perf_counter", side_effect=[10.0, 11.25]),
        ):
            result = llm.ask_structured("  Питання  ", "Правила", history=[{"role": "user", "content": "Попереднє питання"}])

        self.assertEqual(result["answer"], "Відповідь")
        self.assertEqual(result["topic"], "support")
        self.assertEqual(result["order_number"], "A-123")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["usage"]["total_tokens"], 60)
        call = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["temperature"], 0.4)
        self.assertEqual(call["max_tokens"], 128)
        self.assertIn("Попереднє питання", call["messages"][1]["content"])
        self.assertIn("Правила", call["messages"][2]["content"])

    def test_ask_keeps_legacy_minimal_output(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"answer":"Відповідь","topic":"support","based_on_rules":"yes","needs_clarification":false,"transfer_to_operator":false,"order_number":null}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        environment = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "64",
            "LLM_BASE_URL": "https://example.test",
            "LLM_API_KEY": "secret",
            "LLM_TIMEOUT": "30",
        }

        with (
            patch.dict(os.environ, environment),
            patch.object(llm, "get_client", return_value=client),
            patch.object(llm.time, "perf_counter", side_effect=[10.0, 11.25]),
        ):
            result = llm.ask("  Питання  ", "Правила")

        self.assertEqual(
            result,
            {"answer": "Відповідь", "model": "test-model", "elapsed": 1.25},
        )

    def test_ask_retries_once_when_model_returns_invalid_json(self):
        client = Mock()
        valid_json = (
            '{"answer":"Уточніть, будь ласка, номер замовлення.",'
            '"topic":"warranty","based_on_rules":"yes",'
            '"needs_clarification":true,"transfer_to_operator":false,'
            '"order_number":null}'
        )
        client.chat.completions.create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Не можу відповісти."))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))],
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            ),
        ]
        environment = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "128",
            "LLM_BASE_URL": "https://example.test",
            "LLM_API_KEY": "secret",
            "LLM_TIMEOUT": "30",
        }

        with (
            patch.dict(os.environ, environment),
            patch.object(llm, "get_client", return_value=client),
            patch.object(llm.time, "perf_counter", side_effect=[10.0, 11.25]),
        ):
            result = llm.ask_structured("Проблема з навушниками.", "Правила")

        self.assertEqual(result["topic"], "warranty")
        self.assertEqual(result["needs_clarification"], True)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertIn(
            "Попередній результат був невалідним",
            client.chat.completions.create.call_args.kwargs["messages"][0]["content"],
        )

    def test_ask_rejects_invalid_json_after_one_format_retry(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Не JSON"))],
            usage=None,
        )
        environment = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "128",
            "LLM_BASE_URL": "https://example.test",
            "LLM_API_KEY": "secret",
            "LLM_TIMEOUT": "30",
        }

        with (
            patch.dict(os.environ, environment),
            patch.object(llm, "get_client", return_value=client),
        ):
            with self.assertRaises(llm.LLMError) as error:
                llm.ask_structured("Питання", "Правила")

        self.assertEqual(error.exception.code, "invalid_model_response")
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_response_accepts_common_boolean_and_topic_variants(self):
        parsed = llm._parse_model_output(
            '{"answer":"Перевіримо гарантійне звернення.",'
            '"topic":"Гарантія","based_on_rules":true,'
            '"needs_clarification":false,"transfer_to_operator":false,'
            '"order_number":48291}'
        )

        self.assertEqual(parsed.topic, "warranty")
        self.assertEqual(parsed.based_on_rules, "yes")
        self.assertEqual(parsed.order_number, "48291")

    def test_ask_rejects_blank_question_before_calling_model(self):
        with self.assertRaises(llm.LLMError) as error:
            llm.ask(" \t\n", "Правила")

        self.assertEqual(error.exception.code, "empty_question")
        self.assertEqual(error.exception.status_code, 400)


class APITests(unittest.TestCase):
    def setUp(self):
        from app.main import SESSION_STORE

        SESSION_STORE.clear()
        self.client = TestClient(app)

    @patch("app.main.load_context", return_value="Правила магазину")
    @patch(
        "app.main.llm.ask_structured",
        return_value={
            "answer": "Відповідь",
            "topic": "support",
            "based_on_rules": "yes",
            "needs_clarification": False,
            "transfer_to_operator": False,
            "order_number": "A-123",
            "model": "test-model",
            "elapsed": 0.1,
            "usage": {"total_tokens": 42},
        },
    )
    def test_api_returns_structured_response(self, ask_structured, _load_context):
        from app.main import SESSION_STORE

        SESSION_STORE.clear()
        response = self.client.post(
            "/api/ask", json={"session_id": "abc", "question": "Графік підтримки?"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Відповідь")
        self.assertEqual(payload["topic"], "support")
        self.assertEqual(payload["session_id"], "abc")
        self.assertEqual(payload["card"]["order_number"], "A-123")
        ask_structured.assert_called_once()
        self.assertEqual(ask_structured.call_args.args[0], "Графік підтримки?")
        self.assertEqual(ask_structured.call_args.args[1], "Правила магазину")

    def test_api_rejects_whitespace_question(self):
        response = self.client.post("/api/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    @patch("app.main.load_context", return_value="Правила магазину")
    @patch(
        "app.main.llm.ask_structured",
        side_effect=llm.LLMError(
            "Час очікування відповіді минув. Спробуйте ще раз.",
            status_code=504,
            code="upstream_timeout",
        ),
    )
    def test_api_returns_safe_error_message(self, _ask, _load_context):
        response = self.client.post("/api/ask", json={"question": "Питання"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json()["detail"],
            "Час очікування відповіді минув. Спробуйте ще раз.",
        )


if __name__ == "__main__":
    unittest.main()
