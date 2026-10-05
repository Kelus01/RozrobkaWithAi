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

    def test_ask_returns_answer_model_and_elapsed_time(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  Відповідь  "))]
        )
        environment = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.4",
            "LLM_MAX_TOKENS": "128",
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
        call = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["temperature"], 0.4)
        self.assertEqual(call["max_tokens"], 128)
        self.assertEqual(call["messages"][2]["content"], "Питання")

    def test_ask_rejects_blank_question_before_calling_model(self):
        with self.assertRaises(llm.LLMError) as error:
            llm.ask(" \t\n", "Правила")

        self.assertEqual(error.exception.code, "empty_question")
        self.assertEqual(error.exception.status_code, 400)


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.load_context", return_value="Правила магазину")
    @patch(
        "app.main.llm.ask",
        return_value={"answer": "Відповідь", "model": "test-model", "elapsed": 0.1},
    )
    def test_api_returns_answer_model_and_elapsed(self, ask, _load_context):
        response = self.client.post(
            "/api/ask", json={"question": "Графік підтримки?"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"answer": "Відповідь", "model": "test-model", "elapsed": 0.1},
        )
        ask.assert_called_once_with("Графік підтримки?", "Правила магазину")

    def test_api_rejects_whitespace_question(self):
        response = self.client.post("/api/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    @patch("app.main.load_context", return_value="Правила магазину")
    @patch(
        "app.main.llm.ask",
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
