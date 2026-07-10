from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _split_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


def _cors_origins_from_env() -> list[str]:
    origins = _split_csv_env(
        "CORS_ORIGINS",
        ["http://127.0.0.1:5173", "http://localhost:5173"],
    )
    if "*" in origins:
        return origins

    for local_origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
        if local_origin not in origins:
            origins.append(local_origin)
    return origins


def _looks_like_placeholder_secret(value: str) -> bool:
    normalized = value.strip().upper()
    return (
        not normalized
        or normalized.startswith("PASTE_")
        or normalized.startswith("YOUR_")
        or normalized in {"KEY", "API_KEY", "GROQ_KEY", "GROQ_API_KEY"}
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item or _looks_like_placeholder_secret(item) or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return cleaned


def _groq_api_keys_from_env() -> list[str]:
    values: list[str] = []

    raw_keys = os.getenv("GROQ_API_KEYS")
    if raw_keys:
        values.extend(item.strip() for item in raw_keys.split(",") if item.strip())

    for name in ("GROQ_API_KEY_PRIMARY", "GROQ_API_KEY_SECONDARY", "GROQ_API_KEY"):
        value = os.getenv(name)
        if value:
            values.append(value)

    return _dedupe_preserve_order(values)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    build_id: str
    frontend_build_id: str
    artifacts_dir: Path
    data_root: Path | None
    groq_api_key: str | None
    groq_api_keys: list[str]
    groq_model: str
    groq_fallback_model: str | None
    cors_origins: list[str]
    embedding_model: str
    reranker_model: str
    rag_top_k: int
    rerank_top_k: int
    min_rag_score_for_evidence: float
    enable_reranker: bool
    enable_llm_judge: bool = False


@lru_cache
def get_settings() -> Settings:
    service_root = Path(__file__).resolve().parents[1]
    load_dotenv(service_root / ".env", override=False)
    data_root_raw = os.getenv("DATA_ROOT")
    groq_api_keys = _groq_api_keys_from_env()

    return Settings(
        app_name=os.getenv("APP_NAME", "MedBridge AI Service"),
        app_version=os.getenv("APP_VERSION", "3.0"),
        build_id=os.getenv("MEDBRIDGE_BUILD_ID", "medbridge-v3-final-20260620"),
        frontend_build_id=os.getenv("MEDBRIDGE_FRONTEND_BUILD_ID", "medbridge-local-v3-final-20260620"),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", service_root / "artifacts")).resolve(),
        data_root=Path(data_root_raw).resolve() if data_root_raw else None,
        groq_api_key=groq_api_keys[0] if groq_api_keys else None,
        groq_api_keys=groq_api_keys,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_fallback_model=os.getenv("GROQ_FALLBACK_MODEL") or None,
        cors_origins=_cors_origins_from_env(),
        embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        reranker_model=os.getenv(
            "RAG_RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
        rag_top_k=_int_env("RAG_TOP_K", 8),
        rerank_top_k=_int_env("RERANK_TOP_K", 4),
        min_rag_score_for_evidence=_float_env("MIN_RAG_SCORE_FOR_EVIDENCE", 0.05),
        enable_reranker=_bool_env("ENABLE_RERANKER", True),
        enable_llm_judge=_bool_env("ENABLE_LLM_JUDGE", False),
    )
