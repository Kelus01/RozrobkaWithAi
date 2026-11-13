"""Контракт відповіді моделі: поля, типи, відсутні значення."""

import json
import re
from decimal import Decimal, InvalidOperation

import jsonschema


def _normalize_money(value):
    if value is None or value == "":
        return None
    text = str(value).strip().replace("\u00a0", " ")
    if text in {"-", "—", "null", "None"}:
        return None
    text = text.replace("'", "").replace("`", "")
    if text.lower().endswith("грн"):
        text = text[:-3]
    if text.lower().endswith("uah"):
        text = text[:-3]
    text = text.replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Некоректна сума: {value!r}") from exc
    return format(amount.quantize(Decimal("0.01")), "f")


def _normalize_date(value):
    if value is None or value == "":
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", text):
        d, m, y = text.split(".")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    if re.fullmatch(r"\d{1,2}\/\d{1,2}\/\d{4}", text):
        d, m, y = text.split("/")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    month_map = {
        "січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5,
        "червня": 6, "липня": 7, "серпня": 8, "вересня": 9, "жовтня": 10,
        "листопада": 11, "грудня": 12,
    }
    for month_name, month_num in month_map.items():
        if month_name in text.lower():
            match = re.search(r"(\d{1,2})\s+" + re.escape(month_name) + r"\s+(\d{4})", text.lower())
            if match:
                d, y = match.groups()
                return f"{int(y):04d}-{month_num:02d}-{int(d):02d}"
    raise ValueError(f"Некоректна дата: {value!r}")


def output_schema() -> dict:
    """JSON Schema для структурованої відповіді моделі."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {"type": ["string", "null"]},
            "number": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "valid_until": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "supplier": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                    "iban": {"type": ["string", "null"]},
                },
                "required": ["name", "code", "iban"],
            },
            "buyer": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                },
                "required": ["name", "code"],
            },
            "items": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "quantity": {"type": ["number", "string", "integer", "null"]},
                        "price": {"type": ["string", "number", "null"]},
                        "amount": {"type": ["string", "number", "null"]},
                    },
                    "required": ["name", "unit", "quantity", "price", "amount"],
                },
            },
            "total_without_vat": {"type": ["string", "number", "null"]},
            "vat": {"type": ["string", "number", "null"]},
            "total": {"type": ["string", "number", "null"]},
        },
        "required": [
            "document_type",
            "number",
            "date",
            "supplier",
            "buyer",
            "items",
            "total_without_vat",
            "vat",
            "total",
        ],
    }


def validate(raw: str) -> dict:
    """Перевірити JSON від моделі й повернути очищені дані."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Відповідь моделі має бути об'єктом JSON.")

    validator = jsonschema.Draft202012Validator(output_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        details = []
        for err in errors[:5]:
            details.append(f"{list(err.path) or '<root>'}: {err.message}")
        raise ValueError("Відповідь моделі не проходить схему: " + "; ".join(details))

    document = dict(payload)
    if document.get("date"):
        document["date"] = _normalize_date(document["date"])
    if document.get("valid_until"):
        document["valid_until"] = _normalize_date(document["valid_until"])

    supplier = document.get("supplier")
    if isinstance(supplier, dict):
        supplier = dict(supplier)
        if supplier.get("code") is not None:
            supplier["code"] = str(supplier["code"]).strip()
        if supplier.get("iban") is not None:
            supplier["iban"] = str(supplier["iban"]).replace(" ", "").upper()
        document["supplier"] = supplier

    buyer = document.get("buyer")
    if isinstance(buyer, dict):
        buyer = dict(buyer)
        if buyer.get("code") is not None:
            buyer["code"] = str(buyer["code"]).strip()
        document["buyer"] = buyer

    items = document.get("items")
    if isinstance(items, list):
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_copy = dict(item)
            if item_copy.get("quantity") is not None:
                qty = item_copy["quantity"]
                if isinstance(qty, str):
                    qty = qty.replace(" ", "").replace(",", ".")
                    try:
                        item_copy["quantity"] = int(float(qty)) if float(qty).is_integer() else float(qty)
                    except ValueError:
                        pass
            if item_copy.get("price") is not None:
                item_copy["price"] = _normalize_money(item_copy["price"])
            if item_copy.get("amount") is not None:
                item_copy["amount"] = _normalize_money(item_copy["amount"])
            normalized_items.append(item_copy)
        document["items"] = normalized_items

    for field in ("total_without_vat", "vat", "total"):
        if field in document and document[field] is not None:
            document[field] = _normalize_money(document[field])

    return document
