from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import Settings
from app.rag_service import RAGService


def fake_settings() -> Settings:
    return Settings(
        app_name="MedBridge AI Service",
        app_version="3.0",
        build_id="test-build",
        frontend_build_id="test-frontend-build",
        artifacts_dir=Path("artifacts"),
        data_root=None,
        groq_api_key=None,
        groq_api_keys=[],
        groq_model="fake-model",
        groq_fallback_model=None,
        cors_origins=["*"],
        embedding_model="fake-rag-embedder",
        reranker_model="fake-reranker",
        rag_top_k=8,
        rerank_top_k=4,
        min_rag_score_for_evidence=0.05,
        enable_reranker=True,
    )


class RagOfflineFallbackTests(unittest.TestCase):
    def test_embedder_load_is_offline_only_and_marks_unavailable(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_sentence_transformer(*args: object, **kwargs: object) -> object:
            calls.append({"args": args, "kwargs": kwargs})
            raise RuntimeError("model is not cached")

        service = RAGService.__new__(RAGService)
        service.settings = fake_settings()
        service.embedder = None
        service.reranker = None
        service._embedder_unavailable = False
        service._reranker_unavailable = False

        fake_module = SimpleNamespace(SentenceTransformer=fake_sentence_transformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                service._get_embedder()

            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        self.assertTrue(service._embedder_unavailable)
        self.assertEqual(calls[0]["kwargs"].get("local_files_only"), True)

    def test_retrieve_returns_empty_quickly_when_embedder_is_unavailable(self) -> None:
        service = RAGService.__new__(RAGService)
        service.settings = fake_settings()
        service.index = object()
        service.knowledge_base = [{"q_body": "question", "a_body": "answer", "category": "general"}]
        service.embedder = None
        service.reranker = None
        service._embedder_unavailable = True
        service._reranker_unavailable = False

        started = time.perf_counter()
        cases = service.retrieve("dizzy and off balance")
        elapsed = time.perf_counter() - started

        self.assertEqual(cases, [])
        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
