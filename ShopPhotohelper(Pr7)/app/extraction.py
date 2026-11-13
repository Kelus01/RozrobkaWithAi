"""Конвеєр обробки документа: підготовка → вилучення → правила → рішення."""

import time
from dataclasses import dataclass, field

from . import images, llm, rules
from .rules import Issue


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень."""

    decision: str = "reject"
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами."""
    if not isinstance(document, dict):
        return "reject", ["Документ не розпізнано."]

    document_type = str(document.get("document_type") or "").lower().strip()
    if document_type not in {"рахунок", "рахунок на оплату", "invoice", "invoice/payment"}:
        return "reject", ["Це не рахунок на оплату."]

    error_issues = [issue for issue in issues if issue.severity == "error"]
    if error_issues:
        reasons = [issue.message for issue in error_issues[:5]]
        return "review", reasons

    return "auto", ["Жодних критичних проблем не знайдено, документ придатний для реєстру."]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    result = Result()
    started_total = time.perf_counter()

    try:
        prepared = images.prepare(content)
        result.image = {"original": prepared.original, "sent": prepared.sent}
        result.elapsed["prepare"] = round(time.perf_counter() - started_total, 3)
    except images.ImageError as exc:
        result.decision = "reject"
        result.reasons = [str(exc)]
        return result

    try:
        llm_data = llm.extract(prepared)
        result.document = llm_data.get("document")
        result.model = llm_data.get("model")
        result.usage = llm_data.get("usage")
        result.elapsed["extraction"] = llm_data.get("elapsed", {}).get("extraction", 0)
    except llm.LLMError as exc:
        result.decision = "review"
        result.reasons = [str(exc)]
        result.elapsed["extraction"] = round(time.perf_counter() - started_total, 3)
        return result

    try:
        issues = rules.check(result.document or {})
    except Exception as exc:  # pragma: no cover - для захисту від некоректного словника
        result.decision = "review"
        result.reasons = [f"Помилка перевірки документа: {exc}"]
        return result

    result.issues = issues
    result.elapsed["checks"] = round(time.perf_counter() - started_total - result.elapsed.get("prepare", 0) - result.elapsed.get("extraction", 0), 3)
    decision, reasons = decide(result.document, issues)
    result.decision = decision
    result.reasons = reasons
    result.elapsed["total"] = round(time.perf_counter() - started_total, 3)
    return result
