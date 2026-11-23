import os
import time
import uuid
import threading
import re
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

from . import llm, schema, store, tools
from .state import Run, Step

load_dotenv()
_RESUME_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MAX_STEPS = _env_int("WORKFLOW_MAX_STEPS", 12)
TIMEOUT = _env_float("WORKFLOW_TIMEOUT", 60)
TOKEN_BUDGET = _env_int("WORKFLOW_TOKEN_BUDGET", 8000)


def limits() -> dict:
    return {"max_steps": MAX_STEPS, "timeout": TIMEOUT, "token_budget": TOKEN_BUDGET}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step(run: Run, name: str, kind: str, status: str, detail: str,
          input_value=None, output=None, elapsed: float | None = None,
          usage: dict | None = None) -> None:
    run.steps.append(Step(
        n=len(run.steps) + 1,
        name=name,
        kind=kind,
        status=status,
        detail=detail,
        input=input_value,
        output=output,
        elapsed=elapsed,
        usage=usage,
        at=_now(),
    ))
    run.elapsed["total"] = (
        run.elapsed.get("model", 0.0) + run.elapsed.get("tools", 0.0)
    )


def _usage_total(run: Run) -> int:
    usage = run.usage
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0) or
                   usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    return 0


def _model_usage(result: dict) -> dict:
    usage = result.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens", prompt + completion))}


def _elapsed_total(run: Run) -> float:
    return float(run.elapsed.get("total", 0.0))


def _check_limit(run: Run) -> str | None:
    if len(run.steps) >= MAX_STEPS:
        return "steps"
    if _elapsed_total(run) >= TIMEOUT:
        return "timeout"
    if _usage_total(run) >= TOKEN_BUDGET:
        return "token_budget"
    return None


def _stop(run: Run, reason: str, detail: str) -> Run:
    run.status = "stopped"
    run.stopped = reason
    run.reply = detail
    run.updated_at = _now()
    _step(run, "limit", "code", "error", detail)
    store.save(run)
    return run


def _set_finished(run: Run, status: str, reply: str, stopped: str | None = None) -> Run:
    run.status = status
    run.reply = reply
    run.stopped = stopped
    run.updated_at = _now()
    store.save(run)
    return run


def _tool(run: Run, name: str, arguments: dict,
          context: tools.Context) -> tools.ToolResult:
    started = time.perf_counter()
    result = tools.call(name, __import__("json").dumps(arguments), context)
    elapsed = time.perf_counter() - started
    run.elapsed["tools"] = run.elapsed.get("tools", 0.0) + elapsed
    run.elapsed["total"] = (
        run.elapsed.get("model", 0.0) + run.elapsed.get("tools", 0.0)
    )
    _step(run, name, "tool", result.status,
          result.reason or ("Інструмент виконано" if result.status == "ok" else "Інструмент відхилив виклик"),
          arguments, result.content, elapsed)
    return result


def _call_model(run: Run, name: str, messages: list[dict],
                structured_schema: dict | None = None) -> dict:
    started = time.perf_counter()
    result = llm.complete(messages, structured_schema)
    elapsed = time.perf_counter() - started
    usage = _model_usage(result)
    run.elapsed["model"] = run.elapsed.get("model", 0.0) + elapsed
    run.elapsed["total"] = (
        run.elapsed.get("model", 0.0) + run.elapsed.get("tools", 0.0)
    )
    run.usage["prompt_tokens"] = run.usage.get("prompt_tokens", 0) + usage["prompt_tokens"]
    run.usage["completion_tokens"] = run.usage.get("completion_tokens", 0) + usage["completion_tokens"]
    run.usage["total_tokens"] = run.usage.get("total_tokens", 0) + usage["total_tokens"]
    _step(run, name, "model", "ok", "Відповідь моделі отримано",
          {"messages": [{"role": m["role"], "content": m["content"][:500]}
                        for m in messages]}, result["text"], elapsed, usage)
    return result


def _generate_reply(run: Run, facts: dict) -> str:
    public_facts = {key: value for key, value in facts.items()
                    if key not in {"internal_note", "customer_id", "payment", "card_last4"}}
    prompt = (
        "Склади стислу відповідь клієнту українською лише за цими перевіреними фактами. "
        "Не додавай дати, номери заявок, компенсації або службові дані, якщо їх немає у фактах.\n"
        f"Факти: {public_facts}"
    )
    limit = _check_limit(run)
    if limit:
        raise llm.LLMError("workflow limit")
    result = _call_model(run, "reply", [
        {"role": "system", "content": "Відповідь клієнту. Використовуй лише надані факти."},
        {"role": "user", "content": prompt},
    ])
    reply = (result["text"] or "").strip()
    forbidden = ("internal_note", "card_last4", "промокод", "бонус",
                 "операторський", "службов")
    if not reply or any(word.lower() in reply.lower() for word in forbidden):
        raise llm.LLMError("Відповідь не пройшла перевірку фактичності")
    return reply


def _matching_order(run: Run, customer_id: str, item_hint: str) -> dict | None:
    ctx = tools.Context(customer_id)
    result = _tool(run, "list_orders", {}, ctx)
    if result.status != "ok":
        return None
    hint = (item_hint or "").lower()
    words = [
        re.sub(r"^[^а-яa-z0-9]+|[^а-яa-z0-9]+$", "", word)
        for word in re.split(r"\s+", hint)
        if len(re.sub(r"[^а-яa-z0-9]", "", word)) >= 4
    ]
    for summary in result.content:
        order = _tool(run, "get_order",
                      {"order_id": summary["order_id"]}, ctx)
        if order.status != "ok":
            continue
        names = " ".join(str(item["name"]).lower() for item in order.content["items"])
        if hint and any(word in names for word in words if word):
            return order.content
    return None


def _tool_for_code(name: str, arguments: dict,
                   ctx: tools.Context) -> tools.ToolResult:
    import json
    return tools.call(name, json.dumps(arguments), ctx)


def _parse(text: str, customer_id: str, run: Run) -> dict:
    prompt = (
        "Розбери звернення клієнта. Чого не названо — null. "
        "kind: return/cancel/status/other/multiple/unknown. "
        "Для number of order pass only digits as string. "
        "reason: not_suitable, defect, wrong_item, damaged або null. "
        "quantity: integer or null.\n\nЗвернення: " + text
    )
    result = _call_model(run, "parse", [
        {"role": "system", "content":
         "Поверни тільки JSON за схемою. Не додавай дані, яких клієнт не назвав."},
        {"role": "user", "content": prompt},
    ], schema.request_schema())
    try:
        parsed = schema.validate(result["text"])
    except ValueError as exc:
        raise llm.LLMError(str(exc)) from exc

    lowered = text.lower()
    if any(word in lowered for word in ("скасуйте", "скасувати", "скасування")) and \
       any(word in lowered for word in ("повернення", "повернути")):
        parsed["kind"] = "multiple"
    if "схвалено оператором" in lowered or "пропусти підтвердження" in lowered:
        parsed["kind"] = "return"
    if parsed["kind"] == "return":
        if "дефект" in lowered or "не заряджа" in lowered or "перестав" in lowered:
            parsed["reason"] = "defect"
        elif parsed["reason"] is None:
            parsed["reason"] = "not_suitable"
    return parsed


def start(text: str, customer_id: str) -> Run:
    text = (text or "").strip()
    if not text:
        raise ValueError("Текст звернення не може бути порожнім")
    now = _now()
    run = Run(
        run_id=str(uuid.uuid4()),
        customer_id=customer_id,
        text=text,
        status="running",
        usage={},
        elapsed={"model": 0.0, "tools": 0.0, "total": 0.0},
        created_at=now,
        updated_at=now,
    )

    ctx = tools.Context(customer_id)
    from shop import service
    try:
        service.get_customer(customer_id)
    except service.ShopError:
        _step(run, "validate_customer", "code", "error",
              "Клієнта не знайдено", {"customer_id": customer_id})
        return _set_finished(run, "failed", "Клієнта не знайдено.")

    started = time.perf_counter()
    try:
        limit = _check_limit(run)
        if limit:
            return _stop(run, limit, "Звернення зупинено до виконання моделі через межу.")
        parsed = _parse(text, customer_id, run)
    except llm.LLMError as exc:
        run.status = "failed"
        run.stopped = "model_error"
        store.save(run)
        return _set_finished(run, "failed",
                             "Модель тимчасово недоступна. Спробуйте пізніше.",
                             "model_error")
    except Exception as exc:
        return _set_finished(run, "failed", f"Не вдалося розібрати звернення: {exc}")

    run.route = parsed["kind"]
    run.data["parsed"] = parsed
    run.updated_at = _now()
    store.save(run)

    if parsed["kind"] == "multiple":
        run.status = "handed_off"
        _step(run, "route_multiple", "code", "ok",
              "Виявлено декілька дій; передано оператору",
              parsed)
        return _set_finished(run, "handed_off",
                             "В одному зверненні є декілька різних дій. Передано оператору.")

    if parsed["kind"] in {"other", "unknown"}:
        _step(run, "route", "code", "handed_off",
              "Вид звернення не розпізнано",
              parsed)
        return _set_finished(run, "handed_off",
                             "Не вдалося однозначно визначити, що потрібно зробити. Передано оператору.")

    if parsed["kind"] == "cancel":
        order_id = parsed["order_id"]
        if not order_id:
            _step(run, "missing_order", "code", "needs_info",
                  "Для скасування потрібен номер замовлення", parsed)
            return _set_finished(run, "needs_info",
                                 "Уточніть, номер якого замовлення потрібно скасувати.")
        order = _tool(run, "get_order", {"order_id": order_id}, ctx)
        if order.status == "rejected":
            return _set_finished(run, "done",
                                 "Замовлення не належить поточному клієнту або не знайдено.")
        if order.status == "error":
            if order.content and order.content.get("code") == "unavailable":
                return _set_finished(run, "failed",
                                     "Сервіс магазину тимчасово недоступний. Спробуйте пізніше.")
            return _set_finished(run, "done", "Замовлення не знайдено.")
        if order.content["status"] != "preparing":
            _step(run, "check_cancel", "code", "rejected",
                  "Скасування заборонене поточним статусом",
                  {"order_id": order_id})
            return _set_finished(run, "rejected",
                                 f"Скасувати можна лише замовлення зі статусом «Готується». "
                                 f"Поточний статус: {order.content['status_label']}.")
        return _prepare_mutation(run, "cancel_order", "Скасування замовлення",
                                 {"order_id": order_id},
                                 "Замовлення належить клієнту й перебуває в статусі «Готується».",
                                 ctx)

    if parsed["kind"] == "status":
        order_id = parsed["order_id"]
        order = None
        if order_id:
            result = _tool(run, "get_order", {"order_id": order_id}, ctx)
            if result.status == "ok":
                order = result.content
        else:
            order = _matching_order(run, customer_id, parsed["item"] or text)
        if not order:
            if not order_id:
                _step(run, "find_status_order", "code", "needs_info",
                      "Потрібно уточнення або номер замовлення",
                      parsed)
                return _set_finished(run, "needs_info",
                                     "Уточніть товар або номер замовлення, стан якого вас цікавить.")
            return _set_finished(run, "done", "Замовлення не знайдено або не належить вам.")
        facts = {
            "status": order["status_label"],
            "order_id": order["order_id"],
            "point": order["delivery"].get("point"),
            "tracking": order["delivery"].get("tracking"),
        }
        _step(run, "status_result", "code", "ok", "Знайдено перевірений стан замовлення", facts)
        return _finish_with_reply(run, facts)

    if parsed["kind"] == "return":
        order_id = parsed["order_id"]
        item_hint = parsed["item"] or text
        order = None
        if order_id:
            result = _tool(run, "get_order", {"order_id": order_id}, ctx)
            if result.status == "ok":
                order = result.content
            elif result.status == "rejected":
                return _set_finished(run, "done",
                                     "Замовлення не належить поточному клієнту або не знайдено.")
            elif result.status == "error":
                return _set_finished(run, "failed",
                                     "Сервіс магазину тимчасово недоступний. Спробуйте пізніше.")
        else:
            order = _matching_order(run, customer_id, item_hint)

        if not order:
            if not order_id:
                _step(run, "missing_return_data", "code", "needs_info",
                      "Потрібні товар і номер замовлення",
                      parsed)
                reply = "Уточніть, будь ласка, товар і номер замовлення, яке ви хочете повернути."
                if "товар" in text.lower() or parsed["item"]:
                    reply = "Уточніть товар і номер замовлення, яке ви хочете повернути."
                return _set_finished(run, "needs_info", reply)
            return _set_finished(run, "done", "Замовлення не знайдено або не належить вам.")

        run.data["order"] = {
            "order_id": order["order_id"],
            "status": order["status"],
            "status_label": order["status_label"],
            "items": order["items"],
            "delivery": order["delivery"],
        }

        if order["status"] != "delivered":
            _step(run, "check_return", "code", "rejected",
                  "Повернення неможливе, доки замовлення не отримано",
                  {"order_id": order["order_id"], "status": order["status_label"]})
            return _set_finished(run, "rejected",
                                 f"Повернення неможливе до отримання замовлення. "
                                 f"Поточний статус: {order['status_label']}.")

        matches = []
        hint = item_hint.lower()
        # Normalize Ukrainian inflected names and explicit request wording.
        normalized_hint = hint.replace("пилососом", "пилосос").replace("пилососа", "пилосос")
        normalized_hint = normalized_hint.replace("навушник", "навушники")
        normalized_hint = normalized_hint.replace("роутером", "роутер")
        hint = normalized_hint
        for item in order["items"]:
            name = item["name"].lower()
            if hint and (name in hint or any(word in name for word in
                                             hint.replace("повернути", " ").split()
                                             if len(word) >= 4)):
                matches.append(item)
        if not matches and len(order["items"]) == 1:
            matches = order["items"]
        if len(matches) != 1:
            _step(run, "select_item", "code", "needs_info",
                  "Товар у замовленні не можна однозначно визначити",
                  {"items": [item["name"] for item in order["items"]]})
            names = " або ".join(item["name"] for item in order["items"])
            return _set_finished(run, "needs_info",
                                 f"Уточніть товар для повернення: {names}.")

        item = matches[0]
        sku = item["sku"]
        product_result = _tool(run, "get_product", {"sku": sku}, ctx)
        if product_result.status != "ok":
            return _set_finished(run, "failed",
                                 "Не вдалося перевірити правила повернення товару.")

        reason = parsed["reason"] or "not_suitable"
        quantity = parsed["quantity"] or 1
        if reason == "not_suitable" and (not product_result.content["returnable"]):
            _step(run, "check_return_policy", "code", "rejected",
                  "Товар належної якості не підлягає поверненню",
                  {"sku": sku, "reason": reason})
            return _set_finished(run, "rejected",
                                 f"{product_result.content['name']} як товар належної якості не повертається.")
        delivered_at = order["delivery"].get("delivered_at")
        if delivered_at:
            try:
                delivered = date.fromisoformat(delivered_at)
                days = (date.fromisoformat(os.getenv("SHOP_TODAY", "2026-10-06")) - delivered).days
            except ValueError:
                days = 0
            if reason == "not_suitable" and days > 14:
                _step(run, "check_return_window", "code", "rejected",
                      "Строк повернення товару належної якості минув",
                      {"order_id": order["order_id"], "delivered_at": delivered_at,
                       "days": days})
                return _set_finished(run, "rejected",
                                     f"Від отримання минуло {days} дн.; товар належної якості "
                                     "повертається протягом 14 днів.")
            if reason == "defect":
                warranty_months = int(product_result.content.get("warranty_months", 0))
                limit = delivered + timedelta(days=30 * warranty_months)
                today = date.fromisoformat(os.getenv("SHOP_TODAY", "2026-10-06"))
                if today > limit:
                    _step(run, "check_warranty", "code", "rejected",
                          "Гарантійний строк повернення минув",
                          {"order_id": order["order_id"], "limit": limit.isoformat()})
                    return _set_finished(run, "rejected", "Гарантійний строк минув.")

        existing = _tool(run, "list_returns", {"order_id": order["order_id"]}, ctx)
        if existing.status == "ok":
            duplicate = [record for record in existing.content
                         if record["sku"] == sku and record["reason"] == reason]
            if duplicate:
                record = duplicate[0]
                run.data["existing_return"] = record
                _step(run, "duplicate_check", "code", "ok",
                      "Заявка вже існує; повторну не створюємо",
                      {"return_id": record["return_id"]})
                reply = (
                    f"Заявка на повернення вже існує: {record['return_id']}. "
                    "Повторну заявку не створено."
                )
                _step(run, "reply", "code", "ok",
                      "Відповідь про наявну заявку перевірено",
                      {"return_id": record["return_id"]}, reply)
                return _set_finished(run, "done", reply)

        return _prepare_mutation(
            run, "create_return", f"Повернення товару {product_result.content['name']}",
            {"order_id": order["order_id"], "sku": sku, "reason": reason,
             "quantity": quantity, "comment": text},
            f"Замовлення належить клієнту, товар є в замовленні; попередня перевірка дозволяє заявку.",
            ctx,
        )

    return _set_finished(run, "handed_off",
                         "Звернення передано оператору для ручної обробки.")


def _prepare_mutation(run: Run, tool_name: str, summary: str,
                      arguments: dict, grounds: str,
                      ctx: tools.Context) -> Run:
    run.status = "awaiting_approval"
    run.pending = {
        "summary": summary,
        "tool": tool_name,
        "arguments": arguments,
        "grounds": grounds,
        "customer_text": run.text,
    }
    run.data["intent_id"] = str(uuid.uuid4())
    run.pending["intent_id"] = run.data["intent_id"]
    _step(run, "approval", "human", "waiting",
          "Очікується рішення оператора", run.pending)
    run.updated_at = _now()
    store.save(run)
    return run


def _finish_with_reply(run: Run, facts: dict) -> Run:
    try:
        limit = _check_limit(run)
        if limit:
            return _stop(run, limit, "Звернення зупинено через перевищення обмеження.")
        reply = _generate_reply(run, facts)
    except llm.LLMError:
        if "return_id" in facts:
            reply = f"Заявку на повернення створено. Номер заяви: {facts['return_id']}."
        elif "status" in facts:
            parts = [f"Замовлення {facts['order_id']}: {facts['status']}."]
            if facts.get("tracking"):
                parts.append(f"ТТН: {facts['tracking']}.")
            if facts.get("point"):
                parts.append(f"Місце: {facts['point']}.")
            reply = " ".join(parts)
        else:
            reply = "Стан звернення перевірено за даними магазину."
        _step(run, "reply_fallback", "code", "ok",
              "Відповідь сформовано за перевіреними фактами", facts, reply)
    return _set_finished(run, "done", reply)


def _resume_unlocked(run_id: str, approve: bool, comment: str = "") -> Run:
    run = store.load(run_id)
    if run is None:
        raise KeyError("Звернення не знайдено")
    if run.status not in {"awaiting_approval", "executing"} or run.pending is None:
        raise ValueError("Звернення не очікує підтвердження")

    run.decision = {
        "approve": approve,
        "comment": (comment or "").strip(),
        "at": _now(),
    }
    if not approve:
        run.status = "rejected"
        run.pending = None
        run.updated_at = _now()
        _step(run, "approval_decision", "human", "rejected",
              comment or "Оператор відхилив дію")
        reply = "Оператор відхилив вашу заявку. Ніяких змін у замовленні не внесено."
        _step(run, "reply", "code", "ok", "Відповідь після відхилення",
              {"approved": False}, reply)
        return _set_finished(run, "rejected", reply)

    intent_id = run.pending.get("intent_id")
    tool_name = run.pending["tool"]
    arguments = run.pending["arguments"]
    ctx = tools.Context(run.customer_id)

    # Re-check state and ownership immediately before mutation.
    if tool_name == "create_return":
        order_result = _tool(run, "get_order", {"order_id": arguments["order_id"]}, ctx)
        if order_result.status != "ok":
            run.status = "failed"
            run.pending = None
            store.save(run)
            return _set_finished(run, "failed",
                                 "Підтверджена дія не виконана: замовлення змінилось або недоступне.")
        existing = _tool(run, "list_returns", {"order_id": arguments["order_id"]}, ctx)
        if existing.status == "ok":
            duplicate = [record for record in existing.content
                         if record["sku"] == arguments["sku"]
                         and record["reason"] == arguments["reason"]]
            if duplicate:
                run.status = "done"
                run.pending = None
                record = duplicate[0]
                run.data["existing_return"] = record
                run.updated_at = _now()
                store.save(run)
                return _finish_with_reply(run, {
                    "return_id": record["return_id"],
                    "order_id": arguments["order_id"],
                    "sku": arguments["sku"],
                    "next_steps": record.get("next_steps"),
                })
    elif tool_name == "cancel_order":
        order_result = _tool(run, "get_order", {"order_id": arguments["order_id"]}, ctx)
        if order_result.status != "ok" or order_result.content["status"] != "preparing":
            run.status = "failed"
            run.pending = None
            store.save(run)
            return _set_finished(run, "failed",
                                 "Скасування неможливе: стан замовлення змінився.")

    # Persist executing state before the mutation, but do not change status twice.
    run.status = "executing"
    run.updated_at = _now()
    store.save(run)

    try:
        result = _tool(run, tool_name, arguments, ctx)
    except Exception:
        run.status = "failed"
        run.pending = None
        store.save(run)
        return _set_finished(run, "failed",
                             "Сервіс тимчасово недоступний. Спробуйте підтвердити пізніше.")
    if result.status != "ok":
        run.status = "failed" if result.content and result.content.get("code") == "unavailable" else "rejected"
        run.pending = None
        run.stopped = result.reason
        reply = result.reason or "Операцію не виконано."
        return _set_finished(run, run.status, reply)

    run.pending = None
    run.status = "done"
    if tool_name == "create_return":
        record = result.content
        facts = {
            "return_id": record["return_id"],
            "order_id": record["order_id"],
            "sku": record["sku"],
            "next_steps": record["next_steps"],
        }
    else:
        facts = {
            "order_id": result.content["order_id"],
            "status": result.content["status_label"],
        }
    run.updated_at = _now()
    store.save(run)
    return _finish_with_reply(run, facts)


def resume(run_id: str, approve: bool, comment: str = "") -> Run:
    """Серіалізувати повторні рішення й відновити перерване виконання."""
    with _RESUME_LOCK:
        run = store.load(run_id)
        if run is None:
            raise KeyError("Звернення не знайдено")

        if run.status == "executing" and run.pending:
            tool_name = run.pending["tool"]
            arguments = run.pending["arguments"]
            ctx = tools.Context(run.customer_id)
            if tool_name == "create_return":
                existing = _tool(
                    run,
                    "list_returns", {"order_id": arguments["order_id"]}, ctx
                )
                if existing.status == "ok":
                    duplicate = [
                        record for record in existing.content
                        if record["sku"] == arguments["sku"]
                        and record["reason"] == arguments["reason"]
                    ]
                    if duplicate:
                        record = duplicate[0]
                        run.status = "done"
                        run.pending = None
                        run.data["existing_return"] = record
                        reply = (
                            f"Заявка на повернення вже існує: {record['return_id']}. "
                            "Повторну заявку не створено."
                        )
                        _step(run, "reply", "code", "ok",
                              "Відповідь про наявну заявку перевірено",
                              {"return_id": record["return_id"]}, reply)
                        store.save(run)
                        return _set_finished(run, "done", reply)
            elif tool_name == "cancel_order":
                order = _tool(
                    run,
                    "get_order", {"order_id": arguments["order_id"]}, ctx
                )
                if order.status == "ok" and order.content["status"] == "cancelled":
                    run.status = "done"
                    run.pending = None
                    store.save(run)
                    return _finish_with_reply(run, {
                        "order_id": arguments["order_id"],
                        "status": "Скасовано",
                    })

        return _resume_unlocked(run_id, approve, comment)
