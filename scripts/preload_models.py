from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    embedding_model = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    reranker_model = os.getenv(
        "RAG_RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    enable_reranker = _bool_env("ENABLE_RERANKER", True)

    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Preloading embedding model: {embedding_model}")
    SentenceTransformer(embedding_model)

    if enable_reranker:
        print(f"Preloading reranker model: {reranker_model}")
        CrossEncoder(reranker_model)

    print("Model preload complete.")


if __name__ == "__main__":
    main()
