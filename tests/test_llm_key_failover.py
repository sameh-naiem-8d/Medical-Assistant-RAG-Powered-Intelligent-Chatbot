from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings, _groq_api_keys_from_env
from app.llm_service import LLMService


def fake_settings(keys: list[str]) -> Settings:
    return Settings(
        app_name="MedBridge AI Service",
        app_version="3.0",
        build_id="test-build",
        frontend_build_id="test-frontend-build",
        artifacts_dir=Path("artifacts"),
        data_root=None,
        groq_api_key=keys[0] if keys else None,
        groq_api_keys=keys,
        groq_model="fake-groq-model",
        groq_fallback_model=None,
        cors_origins=["*"],
        embedding_model="fake-embedding",
        reranker_model="fake-reranker",
        rag_top_k=8,
        rerank_top_k=4,
        min_rag_score_for_evidence=0.05,
        enable_reranker=True,
    )


class FakeGroqClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls = 0
        self.models: list[str | None] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        self.models.append(kwargs.get("model") if isinstance(kwargs.get("model"), str) else None)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class SlowGroqClient(FakeGroqClient):
    def __init__(self, delay_seconds: float, content: str | None = None):
        super().__init__(content=content)
        self.delay_seconds = delay_seconds

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        self.models.append(kwargs.get("model") if isinstance(kwargs.get("model"), str) else None)
        time.sleep(self.delay_seconds)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class ModelFallbackGroqClient(FakeGroqClient):
    def create(self, **kwargs: object) -> object:
        model = kwargs.get("model")
        self.calls += 1
        self.models.append(model if isinstance(model, str) else None)
        if model == "primary-model":
            raise RuntimeError("primary model unavailable")
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def service_with_clients(keys: list[str], clients: list[FakeGroqClient]) -> LLMService:
    service = LLMService.__new__(LLMService)
    service.settings = fake_settings(keys)
    service.clients = clients
    service.client = clients[0] if clients else None
    return service


def generate(service: LLMService) -> str:
    return service.generate_answer(
        message="عندي كحة وسخونية وتعب",
        history=[],
        extracted_symptoms=["cough", "high_fever", "fatigue"],
        diagnosis="Common Cold",
        confidence=0.65,
        urgency_level="Medium",
        suggested_doctor="General Practitioner",
        precautions=[],
        diagnosis_description=None,
        follow_up_questions=[],
        retrieved_cases=[],
    )


class LLMKeyFailoverTests(unittest.TestCase):
    def test_single_legacy_key_is_parsed(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "single-test-key"}, clear=True):
            self.assertEqual(_groq_api_keys_from_env(), ["single-test-key"])

    def test_multiple_key_formats_are_parsed_and_deduplicated(self) -> None:
        env = {
            "GROQ_API_KEYS": "primary-test-key, backup-test-key",
            "GROQ_API_KEY_PRIMARY": "primary-test-key",
            "GROQ_API_KEY_SECONDARY": "third-test-key",
            "GROQ_API_KEY": "backup-test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                _groq_api_keys_from_env(),
                ["primary-test-key", "backup-test-key", "third-test-key"],
            )

    def test_placeholder_values_are_not_treated_as_configured_keys(self) -> None:
        env = {
            "GROQ_API_KEYS": "PASTE_PRIMARY_GROQ_KEY_HERE,PASTE_BACKUP_GROQ_KEY_HERE",
            "GROQ_API_KEY": "PASTE_SINGLE_GROQ_KEY_HERE",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_groq_api_keys_from_env(), [])

    def test_first_key_success_does_not_call_second_key(self) -> None:
        first = FakeGroqClient(
            "من الأعراض التي ذكرتها، الاحتمال الأقرب هو: نزلة برد.\n\nالسبب:\nأعراض تنفسية بسيطة."
        )
        second = FakeGroqClient("should not be used")
        service = service_with_clients(["primary-test-key", "backup-test-key"], [first, second])

        answer = generate(service)

        self.assertIn("نزلة برد", answer)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)

    def test_first_key_error_tries_second_key(self) -> None:
        first = FakeGroqClient(error=RuntimeError("rate limit"))
        second = FakeGroqClient(
            "من الأعراض التي ذكرتها، الاحتمال الأقرب هو: نزلة برد.\n\nالسبب:\nتم استخدام المفتاح الاحتياطي."
        )
        service = service_with_clients(["primary-test-key", "backup-test-key"], [first, second])

        answer = generate(service)

        self.assertIn("نزلة برد", answer)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_all_keys_fail_returns_safe_fallback(self) -> None:
        first = FakeGroqClient(error=TimeoutError("timeout"))
        second = FakeGroqClient(error=RuntimeError("quota"))
        service = service_with_clients(["primary-test-key", "backup-test-key"], [first, second])

        answer = generate(service)

        self.assertIn("الاحتمال الأقرب", answer)
        self.assertIn("نزلة برد", answer)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_provider_stall_is_bounded_and_tries_backup_key(self) -> None:
        first = SlowGroqClient(delay_seconds=0.3, content="late answer should not block")
        second = FakeGroqClient(
            "Ù…Ù† Ø§Ù„Ø£Ø¹Ø±Ø§Ø¶ Ø§Ù„ØªÙŠ Ø°ÙƒØ±ØªÙ‡Ø§ØŒ Ø§Ù„Ø§Ø­ØªÙ…Ø§Ù„ Ø§Ù„Ø£Ù‚Ø±Ø¨ Ù‡Ùˆ: Ù†Ø²Ù„Ø© Ø¨Ø±Ø¯.\n\nØ§Ù„Ø³Ø¨Ø¨:\nØªÙ… Ø§Ø³ØªØ®Ø¯Ø§Ù… Ø§Ù„Ù…ÙØªØ§Ø­ Ø§Ù„Ø§Ø­ØªÙŠØ§Ø·ÙŠ Ø¨Ø¹Ø¯ ØªØ¹Ø·Ù„ Ø§Ù„Ù…Ø²ÙˆØ¯."
        )
        service = service_with_clients(["primary-test-key", "backup-test-key"], [first, second])
        service.provider_call_timeout_seconds = 0.05

        started = time.perf_counter()
        answer = generate(service)
        elapsed = time.perf_counter() - started

        self.assertTrue(answer.strip())
        self.assertNotIn("late answer should not block", answer)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertLess(elapsed, 0.5)

    def test_all_provider_stalls_return_safe_fallback(self) -> None:
        first = SlowGroqClient(delay_seconds=0.3)
        second = SlowGroqClient(delay_seconds=0.3)
        service = service_with_clients(["primary-test-key", "backup-test-key"], [first, second])
        service.provider_call_timeout_seconds = 0.05

        started = time.perf_counter()
        answer = generate(service)
        elapsed = time.perf_counter() - started

        self.assertTrue(answer.strip())
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertLess(elapsed, 0.8)

    def test_primary_model_error_tries_configured_fallback_model(self) -> None:
        client = ModelFallbackGroqClient(
            "Ù…Ù† Ø§Ù„Ø£Ø¹Ø±Ø§Ø¶ Ø§Ù„ØªÙŠ Ø°ÙƒØ±ØªÙ‡Ø§ØŒ Ø§Ù„Ø§Ø­ØªÙ…Ø§Ù„ Ø§Ù„Ø£Ù‚Ø±Ø¨ Ù‡Ùˆ: Ù†Ø²Ù„Ø© Ø¨Ø±Ø¯."
        )
        service = service_with_clients(["primary-test-key"], [client])
        service.settings = Settings(
            app_name="MedBridge AI Service",
            app_version="3.0",
            build_id="test-build",
            frontend_build_id="test-frontend-build",
            artifacts_dir=Path("artifacts"),
            data_root=None,
            groq_api_key="primary-test-key",
            groq_api_keys=["primary-test-key"],
            groq_model="primary-model",
            groq_fallback_model="fallback-model",
            cors_origins=["*"],
            embedding_model="fake-embedding",
            reranker_model="fake-reranker",
            rag_top_k=8,
            rerank_top_k=4,
            min_rag_score_for_evidence=0.05,
            enable_reranker=True,
        )

        answer = generate(service)

        self.assertIn("Ù†Ø²Ù„Ø© Ø¨Ø±Ø¯", answer)
        self.assertEqual(client.models[:2], ["primary-model", "fallback-model"])

    def test_health_reports_key_count_without_exposing_key_values(self) -> None:
        class DummyLLM:
            configured = True
            key_count = 2

        original_llm = main_module.llm_service
        main_module.llm_service = DummyLLM()
        try:
            response = TestClient(main_module.app).get("/health")
        finally:
            main_module.llm_service = original_llm

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["llm_configured"])
        self.assertEqual(data["llm_key_count"], 2)
        response_text = response.text
        self.assertNotIn("primary-test-key", response_text)
        self.assertNotIn("backup-test-key", response_text)


if __name__ == "__main__":
    unittest.main()
