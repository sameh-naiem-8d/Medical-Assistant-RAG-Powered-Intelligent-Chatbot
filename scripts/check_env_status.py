from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.llm_service import LLMService  # noqa: E402


REQUIRED_ARTIFACTS = [
    "faiss.index",
    "maqa_clean_data.pkl",
    "maqa_embeddings.pkl",
    "disease_classifier.pkl",
    "disease_label_encoder.pkl",
    "symptom_columns.pkl",
    "knowledge_base.pkl",
    "medical_knowledge.pkl",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _env_value_present(name: str) -> bool:
    value = os.getenv(name, "").strip()
    if not value:
        return False
    normalized = value.upper()
    return not (
        normalized.startswith("PASTE_")
        or normalized.startswith("YOUR_")
        or normalized in {"KEY", "API_KEY", "GROQ_KEY", "GROQ_API_KEY"}
    )


def main() -> None:
    settings = get_settings()
    llm_service = LLMService(settings)
    env_path = PROJECT_ROOT / ".env"

    print(f"current working directory: {Path.cwd()}")
    print(f".env exists: {_bool_text(env_path.exists())}")
    print(f"GROQ_API_KEYS present: {_bool_text(_env_value_present('GROQ_API_KEYS'))}")
    print(f"GROQ_API_KEY_PRIMARY present: {_bool_text(_env_value_present('GROQ_API_KEY_PRIMARY'))}")
    print(f"GROQ_API_KEY_SECONDARY present: {_bool_text(_env_value_present('GROQ_API_KEY_SECONDARY'))}")
    print(f"legacy GROQ_API_KEY present: {_bool_text(_env_value_present('GROQ_API_KEY'))}")
    print(f"llm_configured: {_bool_text(llm_service.configured)}")
    print(f"llm_key_count: {llm_service.key_count}")
    print(f"artifact directory path: {settings.artifacts_dir}")

    for filename in REQUIRED_ARTIFACTS:
        exists = (settings.artifacts_dir / filename).exists()
        print(f"artifact {filename} exists: {_bool_text(exists)}")


if __name__ == "__main__":
    main()
