"""Контракт кроку розбору: що модель має витягти зі звернення клієнта
і як це перевіряється.

Окремий модуль, як у ПР4, ПР6 і ПР7: схема — домовленість між моделлю
та рештою workflow. Наступні кроки працюють лише з тим, що пройшло
перевірку тут, і не бачать сирого тексту відповіді моделі.

Що має дати розбір (розділ 2 практичної роботи): вид звернення —
повернення товару, скасування замовлення, питання про стан замовлення
або щось інше; номер замовлення, якщо клієнт його назвав; товар, як
клієнт його назвав; причину повернення.

Рішення — ваші:

* які поля й що з них обовʼязкове; як модель каже «номер не названо» і
  «не розумію, про що звернення» — і чи це одне й те саме;
* причина повернення: категорія сервісу (`not_suitable`, `defect`, …),
  слова клієнта чи обидва? Хто вирішує категорію — модель чи ваш код за
  словами клієнта? Згадайте q11 із ПР8;
* що робити зі зверненням, у якому дві дії, — схема це дозволяє чи ні;
* чи просити від моделі щось понад потрібне наступним крокам — і чи буде
  кому цим скористатися;
* що в цій схемі статичне, а що будується з даних: наприклад, вибір
  товару з позицій конкретного замовлення — це окремий крок із власною
  схемою, `enum` якої складається з результату інструмента.

Описати схему можна JSON Schema вручну або pydantic-моделлю.
"""

import json
from jsonschema import Draft202012Validator


def request_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["return", "cancel", "status", "other", "multiple", "unknown"],
            },
            "order_id": {"type": ["string", "null"]},
            "item": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"], "enum": [
                "not_suitable", "defect", "wrong_item", "damaged", None
            ]},
            "quantity": {"type": ["integer", "null"], "minimum": 1},
        },
        "required": ["kind", "order_id", "item", "reason", "quantity"],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Відповідь моделі має бути JSON-об'єктом")
    errors = sorted(Draft202012Validator(request_schema()).iter_errors(data),
                    key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError("Відповідь моделі не проходить схему: " + "; ".join(
            error.message for error in errors
        ))
    if data["kind"] in {"unknown", "multiple"}:
        data["order_id"] = None
    return data
