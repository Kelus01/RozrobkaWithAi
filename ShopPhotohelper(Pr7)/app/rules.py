"""Програмні правила: перевірка змісту вилучених даних."""

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


@dataclass
class Issue:
    """Проблема, знайдена правилом."""

    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def _money(value) -> Decimal | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("\u00a0", " ")
    if text in {"-", "—", "null", "None"}:
        return None
    text = text.replace("'", "").replace("`", "")
    text = text.replace(" ", "")
    if text.lower().endswith("грн"):
        text = text[:-3]
    if text.lower().endswith("uah"):
        text = text[:-3]
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").strip()).lower()


def _iban_checksum(iban: str) -> bool:
    iban = re.sub(r"\s+", "", iban or "").upper()
    if not iban or len(iban) != 29 or not iban.startswith("UA"):
        return False
    rearranged = iban[4:] + iban[:4]
    mapped = ""
    for ch in rearranged:
        mapped += str(ord(ch) - 55) if ch.isalpha() else ch
    remainder = int(mapped) % 97
    return remainder == 1


def _edrpou_checksum(code: str) -> bool:
    code = re.sub(r"\D", "", str(code or ""))
    if len(code) != 8 or not code.isdigit():
        return False
    digits = [int(ch) for ch in code]
    total = sum(d * w for d, w in zip(digits[:-1], [1, 2, 3, 4, 5, 6, 7]))
    remainder = total % 11
    expected = 0 if remainder == 10 else remainder
    return digits[-1] == expected


def _rnokpp_checksum(code: str) -> bool:
    code = re.sub(r"\D", "", str(code or ""))
    if len(code) != 10 or not code.isdigit():
        return False
    digits = [int(ch) for ch in code]
    total = sum(d * w for d, w in zip(digits[:-1], [2, 3, 4, 5, 6, 7, 8, 9, 1]))
    remainder = total % 11
    expected = 0 if remainder in (0, 10) else 11 - remainder
    return digits[-1] == expected


def _parse_date(value):
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        try:
            from datetime import datetime
            return datetime.strptime(str(value), "%d.%m.%Y").date()
        except ValueError:
            return None


def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа."""
    if not isinstance(document, dict):
        return [Issue("document", "missing_required", "Документ не розпізнано або порожній.", "error")]

    issues: list[Issue] = []
    company, suppliers = load_reference()
    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}
    items = document.get("items") or []

    required = [
        ("document_type", "document_type"),
        ("number", "number"),
        ("date", "date"),
        ("supplier.name", "supplier.name"),
        ("supplier.code", "supplier.code"),
        ("supplier.iban", "supplier.iban"),
        ("buyer.name", "buyer.name"),
        ("buyer.code", "buyer.code"),
        ("items", "items"),
        ("total_without_vat", "total_without_vat"),
        ("vat", "vat"),
        ("total", "total"),
    ]

    for value, field in required:
        if value == "supplier.name":
            present = bool((supplier or {}).get("name"))
        elif value == "supplier.code":
            present = bool((supplier or {}).get("code"))
        elif value == "supplier.iban":
            present = bool((supplier or {}).get("iban"))
        elif value == "buyer.name":
            present = bool((buyer or {}).get("name"))
        elif value == "buyer.code":
            present = bool((buyer or {}).get("code"))
        else:
            present = bool(document.get(value))
        if not present:
            issues.append(Issue(field, "missing_required", f"Відсутнє обов'язкове поле: {field}", "error"))

    document_type = str(document.get("document_type") or "").strip().lower()
    if document_type and document_type not in {"рахунок", "рахунок на оплату", "invoice", "invoice/payment", "інший", "видаткова накладна", "накладна"}:
        issues.append(Issue("document_type", "document_type", "Тип документа не є рахунком на оплату.", "error"))

    # Дата
    date_value = document.get("date")
    if date_value:
        parsed_date = _parse_date(date_value)
        if parsed_date is None:
            issues.append(Issue("date", "date_invalid", "Дата має бути в форматі YYYY-MM-DD або відома українською формою.", "error"))
        else:
            today = date.today()
            if parsed_date > today:
                issues.append(Issue("date", "date_future", "Дата рахунку не може бути в майбутньому.", "error"))
    if document.get("valid_until"):
        parsed_valid_until = _parse_date(document.get("valid_until"))
        if parsed_valid_until is None:
            issues.append(Issue("valid_until", "valid_until_invalid", "Строк дії має бути датою.", "error"))
        elif date_value:
            parsed_date = _parse_date(date_value)
            if parsed_valid_until < parsed_date:
                issues.append(Issue("valid_until", "valid_until_before_date", "Строк дії не може бути раніше дати рахунку.", "error"))

    # IBAN and codes
    if supplier.get("iban"):
        iban = re.sub(r"\s+", "", str(supplier["iban"]))
        if not _iban_checksum(iban):
            issues.append(Issue("supplier.iban", "iban_checksum", "IBAN не проходить контрольну суму ISO 13616.", "error"))
        else:
            bank_map = {entry.get("iban"): entry for entry in suppliers}
            if iban not in bank_map:
                issues.append(Issue("supplier.iban", "iban_registry", "IBAN не знайдений у довіднику постачальників.", "error"))

    for field_name, code_value in [("supplier.code", supplier.get("code")), ("buyer.code", buyer.get("code"))]:
        code = re.sub(r"\D", "", str(code_value or ""))
        if not code:
            continue
        if len(code) == 8 and not _edrpou_checksum(code):
            issues.append(Issue(field_name, "code_checksum", "Код не проходить контрольну суму ЄДРПОУ.", "error"))
        elif len(code) == 10 and not _rnokpp_checksum(code):
            issues.append(Issue(field_name, "code_checksum", "Код не проходить контрольну суму РНОКПП.", "error"))
        elif len(code) not in (8, 10):
            issues.append(Issue(field_name, "code_format", "Код постачальника або покупця має бути 8 або 10 цифр.", "error"))

    # Buyer identity 
    buyer_name = (buyer or {}).get("name")
    if buyer_name:
        expected_company_name = company.get("name", "")
        if _normalize_text(buyer_name) != _normalize_text(expected_company_name):
            issues.append(Issue("buyer.name", "buyer", "Покупець має бути компанія «Сузірʼя Рітейл».", "error"))

    # Supplier registry and name match
    supplier_name = (supplier or {}).get("name")
    if supplier_name:
        supplier_match = next((entry for entry in suppliers if _normalize_text(entry.get("name")) == _normalize_text(supplier_name)), None)
        if supplier_match is None:
            issues.append(Issue("supplier.name", "supplier_registry", "Постачальник не знайдений у довіднику.", "error"))
        elif supplier.get("iban"):
            expected_iban = supplier_match.get("iban")
            if _normalize_text(str(supplier.get("iban"))) != _normalize_text(str(expected_iban)):
                issues.append(Issue("supplier.iban", "supplier_registry", "IBAN постачальника не співпадає з довідником.", "error"))

    # Arithmetic
    if isinstance(items, list):
        line_total = Decimal("0")
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            qty = item.get("quantity")
            price = item.get("price")
            amount = item.get("amount")
            qty_dec = _money(qty)
            price_dec = _money(price)
            amount_dec = _money(amount)
            if qty_dec is None or price_dec is None or amount_dec is None:
                continue
            computed = (qty_dec * price_dec).quantize(Decimal("0.01"))
            if abs(computed - amount_dec) > Decimal("0.01"):
                issues.append(Issue(f"items.{idx}.amount", "arithmetic_line", "Сума рядка не дорівнює кількість × ціна.", "error"))
            line_total += amount_dec

        expected_without_vat = _money(document.get("total_without_vat"))
        if expected_without_vat is not None and abs(line_total - expected_without_vat) > Decimal("0.01"):
            issues.append(Issue("total_without_vat", "arithmetic_total", "Сума рядків не збігається з підсумком без ПДВ.", "error"))

        vat_value = _money(document.get("vat"))
        if vat_value is not None:
            vat_expected = Decimal("0")
            supplier_entry = next((entry for entry in suppliers if _normalize_text(entry.get("name")) == _normalize_text(supplier_name)), None)
            if supplier_entry is not None and supplier_entry.get("vat_payer") is True and expected_without_vat is not None:
                vat_expected = (expected_without_vat * Decimal("0.2")).quantize(Decimal("0.01"))
            elif supplier_entry is not None and supplier_entry.get("vat_payer") is False:
                vat_expected = Decimal("0")
            if expected_without_vat is not None and abs(vat_value - vat_expected) > Decimal("0.01"):
                issues.append(Issue("vat", "vat_amount", "ПДВ не відповідає ставці 20% або нулю для неплатника.", "error"))

        total_value = _money(document.get("total"))
        if expected_without_vat is not None and vat_value is not None and total_value is not None:
            expected_total = (expected_without_vat + vat_value).quantize(Decimal("0.01"))
            if abs(total_value - expected_total) > Decimal("0.01"):
                issues.append(Issue("total", "arithmetic_total", "Загальна сума не дорівнює сумі без ПДВ плюс ПДВ.", "error"))

    return issues
