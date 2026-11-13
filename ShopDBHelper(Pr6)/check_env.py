"""Check the local dependencies, documents, embeddings, and LLM API setup."""

import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
MIN_DOCS = 10
REQUIRED_PACKAGES = {
    "openai": "openai",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "rank_bm25": "rank-bm25",
}


def report(ok: bool, label: str, message: str = "") -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    print(f"{mark} {label}" + (f" — {message}" if message else ""))


def check_packages() -> bool:
    success = True
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            report(False, f"пакет {package}", "виконайте pip install -r requirements.txt")
            success = False
        else:
            report(True, f"пакет {package}")
    return success


def check_settings() -> bool:
    from dotenv import load_dotenv

    if not (ROOT / ".env").is_file():
        report(False, "файл .env", "скопіюйте .env.example у .env і додайте LLM_API_KEY")
    load_dotenv(ROOT / ".env")

    success = True
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        value = os.getenv(name)
        if not value:
            report(False, f"налаштування {name}", "не задано; перевірте файл .env")
            success = False
        elif name == "LLM_API_KEY":
            report(True, f"налаштування {name}", "ключ задано (значення не показується)")
        else:
            report(True, f"налаштування {name}", value)
    return success


def check_documents() -> bool:
    documents = [path for path in DOCS_DIR.glob("*.md") if path.name.lower() != "readme.md"]
    success = len(documents) >= MIN_DOCS
    report(
        success,
        f"колекція docs/: {len(documents)} документів",
        "" if success else "перевірте, чи повністю розпаковано матеріали ПР6",
    )
    return success


def check_embeddings() -> bool:
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    started = time.perf_counter()
    try:
        model = SentenceTransformer(model_name)
        vector = model.encode(
            ["query: Скільки днів зберігається посилка?"],
            normalize_embeddings=True,
        )
    except Exception as exc:
        report(False, f"модель ембедінгів {model_name}", f"{type(exc).__name__}: {exc}")
        return False
    report(
        True,
        f"модель ембедінгів {model_name}",
        f"розмірність {vector.shape[1]}, перевірка за {time.perf_counter() - started:.1f} с",
    )
    return True


def check_llm() -> bool:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит до мовної моделі", "спершу заповніть налаштування у .env")
        return False

    started = time.perf_counter()
    try:
        response = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        ).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Контекст: посилка зберігається у відділенні 7 днів. "
                        "Питання: скільки днів вона чекає? Відповідай коротко."
                    ),
                }
            ],
            max_tokens=60,
            temperature=0,
        )
    except AuthenticationError:
        report(False, "пробний запит до мовної моделі", "ключ відхилено; перевірте LLM_API_KEY")
        return False
    except RateLimitError:
        report(False, "пробний запит до мовної моделі", "перевищено ліміт API")
        return False
    except APITimeoutError:
        report(False, "пробний запит до мовної моделі", "вичерпано час очікування")
        return False
    except APIConnectionError:
        report(False, "пробний запит до мовної моделі", "не вдалося зʼєднатися з API")
        return False
    except APIStatusError as exc:
        detail = getattr(exc, "message", "") or "провайдер не надав пояснення"
        detail = " ".join(str(detail).split())[:300]
        report(
            False,
            "пробний запит до мовної моделі",
            f"HTTP {exc.status_code}: {detail}",
        )
        return False

    text = response.choices[0].message.content if response.choices else None
    if not text:
        report(False, "пробний запит до мовної моделі", "отримано порожню відповідь")
        return False
    usage = response.usage
    report(
        True,
        "пробний запит до мовної моделі",
        f"{time.perf_counter() - started:.1f} с; "
        f"{usage.total_tokens if usage else 'невідомо'} токенів",
    )
    return True


def main() -> int:
    print("Перевірка середовища\n")
    results = [sys.version_info >= (3, 10, 0), check_packages(), check_documents()]
    report(results[0], f"Python {sys.version.split()[0]}", "потрібен Python 3.10+" if not results[0] else "")
    settings_ready = check_settings()
    results.append(settings_ready)

    if results[1]:
        results.append(check_embeddings())
        if settings_ready:
            results.append(check_llm())
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
