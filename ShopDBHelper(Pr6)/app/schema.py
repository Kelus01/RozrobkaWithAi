"""Structured response contract returned by the language model."""

import json

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: StrictStr = Field(min_length=1)
    source_refs: list[StrictInt]
    found: StrictBool

    @field_validator("answer")
    @classmethod
    def answer_is_not_blank(cls, answer: str) -> str:
        answer = answer.strip()
        if not answer:
            raise ValueError("Текст відповіді не може бути порожнім.")
        return answer

    @field_validator("source_refs")
    @classmethod
    def validate_references(cls, refs: list[int]) -> list[int]:
        if any(ref < 1 for ref in refs):
            raise ValueError("Номери джерел мають бути додатними.")
        if len(set(refs)) != len(refs):
            raise ValueError("Номери джерел не мають повторюватися.")
        return refs


def output_schema() -> dict:
    """Return the JSON Schema sent to the model."""
    return ModelAnswer.model_json_schema()


def validate(raw: str) -> dict:
    """Parse model output and reject malformed or structurally unsafe data."""
    try:
        payload = ModelAnswer.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(f"Відповідь моделі не відповідає схемі: {exc}") from exc

    return payload.model_dump()
