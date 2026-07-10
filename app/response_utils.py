from __future__ import annotations

from typing import Any

from pydantic import BaseModel


FRONTEND_SAFE_FIELDS = [
    "mode",
    "answer",
    "display_diagnosis_ar",
    "display_doctor_ar",
    "urgency_level",
    "suggested_doctor",
    "follow_up_questions",
    "needs_follow_up",
]


def to_frontend_safe_response(data: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Return only patient-facing fields and hide diagnostic/debug support data.

    Full `/chat` responses keep fields such as `extracted_symptoms`, `confidence`,
    `precautions`, and `retrieved_cases` for backend QA and graduation evidence.
    Frontend-safe responses intentionally expose only display labels, urgency,
    routing, answer text, mode, and follow-up prompts.
    """
    if isinstance(data, BaseModel):
        raw = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    else:
        raw = dict(data)

    safe = {field: raw.get(field) for field in FRONTEND_SAFE_FIELDS}
    if raw.get("mode") == "clarification":
        safe["display_diagnosis_ar"] = None
    return safe
