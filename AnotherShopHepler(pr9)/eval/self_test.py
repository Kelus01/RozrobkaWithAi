import json
import re
import unittest

from app import llm, workflow
from shop import service


def fake_complete(messages, structured=None):
    text = messages[-1]["content"]
    if not structured:
        return {"text": "Перевірено за даними магазину.",
                "usage": {"prompt_tokens": 30, "completion_tokens": 10,
                          "total_tokens": 40}}
    lower = text.lower()
    if "скасуйте" in lower and "повернення" in lower:
        data = {"kind": "multiple", "order_id": None, "item": None,
                "reason": None, "quantity": None}
    elif "скасуйте" in lower or "скасувати" in lower:
        match = re.search(r"\b(10\d{3})\b", lower)
        data = {"kind": "cancel", "order_id": match.group(1) if match else None,
                "item": None, "reason": None, "quantity": None}
    elif any(word in lower for word in ("поверни", "повернення", "повернути")):
        match = re.search(r"\b(10\d{3})\b", lower)
        item = ("пилосос" if "пилосос" in lower else
                "фільтр" if "фільтр" in lower else
                "навуш" if "навуш" in lower else
                "роутер" if "роутер" in lower else None)
        defect = any(word in lower for word in ("дефект", "не заряджа", "перестав"))
        data = {"kind": "return", "order_id": match.group(1) if match else None,
                "item": item, "reason": "defect" if defect else "not_suitable",
                "quantity": 1}
    elif any(word in lower for word in ("де моє", "де мій", "посилка",
                                        "лежить", "стан замовлення")):
        match = re.search(r"\b(10\d{3})\b", lower)
        data = {"kind": "status", "order_id": match.group(1) if match else None,
                "item": "годинник" if "годинник" in lower else None,
                "reason": None, "quantity": None}
    else:
        data = {"kind": "other", "order_id": None, "item": None,
                "reason": None, "quantity": None}
    return {"text": json.dumps(data, ensure_ascii=False),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120}}


class WorkflowLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_complete = llm.complete
        llm.complete = fake_complete

    @classmethod
    def tearDownClass(cls):
        llm.complete = cls.original_complete
        service.reset()

    def setUp(self):
        service.reset()

    def test_return_requires_approval(self):
        run = workflow.start(
            "Хочу повернути пилосос із замовлення 10412.", "C-1003")
        self.assertEqual(run.status, "awaiting_approval")
        self.assertEqual(run.pending["tool"], "create_return")

    def test_cancel_requires_approval(self):
        run = workflow.start(
            "Скасуйте, будь ласка, замовлення 10463.", "C-1001")
        self.assertEqual(run.status, "awaiting_approval")

    def test_cancel_forbidden_status(self):
        run = workflow.start("Скасуйте замовлення 10458.", "C-1001")
        self.assertEqual(run.status, "rejected")

    def test_foreign_order(self):
        run = workflow.start(
            "Поверніть пилосос із замовлення 10466.", "C-1001")
        self.assertIn(run.status, {"done", "handed_off", "rejected"})

    def test_status_without_number(self):
        run = workflow.start("Де моє замовлення з годинником?", "C-1001")
        self.assertEqual(run.status, "done")

    def test_expired_return(self):
        run = workflow.start(
            "Хочу повернути роутер, він мені не підійшов.", "C-1002")
        self.assertEqual(run.status, "rejected")

    def test_multiple_actions_handed_off(self):
        run = workflow.start(
            "Скасуйте замовлення 10463 і оформіть повернення навушників із 10421.",
            "C-1001")
        self.assertEqual(run.status, "handed_off")

    def test_double_approval(self):
        run = workflow.start(
            "Хочу повернути пилосос із замовлення 10412.", "C-1003")
        done = workflow.resume(run.run_id, True)
        self.assertEqual(done.status, "done")
        with self.assertRaises(ValueError):
            workflow.resume(run.run_id, True)
        self.assertEqual(len(service.list_returns("10412")), 1)

    def test_rejection(self):
        run = workflow.start(
            "Скасуйте замовлення 10463.", "C-1001")
        rejected = workflow.resume(run.run_id, False, "Клієнт не підтверджує")
        self.assertEqual(rejected.status, "rejected")
        self.assertFalse(rejected.pending)

    def test_repeat_return_has_no_second_application(self):
        run = workflow.start(
            "Хочу повернути пилосос із замовлення 10412.", "C-1003")
        workflow.resume(run.run_id, True)
        repeat = workflow.start(
            "Хочу повернути пилосос із замовлення 10412.", "C-1003")
        self.assertIn("Повторну заявку не створено", repeat.reply or "")
        self.assertEqual(len(service.list_returns("10412")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
