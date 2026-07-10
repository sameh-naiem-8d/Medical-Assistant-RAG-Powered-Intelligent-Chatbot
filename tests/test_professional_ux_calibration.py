from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module


class ProfessionalUXCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main_module.app)
        cls.original_llm_client = main_module.llm_service.client
        cls.original_llm_clients = list(getattr(main_module.llm_service, "clients", []))
        cls.original_retrieve = main_module.rag_service.retrieve

        main_module.llm_service.client = None
        main_module.llm_service.clients = []
        main_module.rag_service.retrieve = lambda _: []

    @classmethod
    def tearDownClass(cls) -> None:
        main_module.llm_service.client = cls.original_llm_client
        main_module.llm_service.clients = cls.original_llm_clients
        main_module.rag_service.retrieve = cls.original_retrieve

    def post_chat(self, message: str, history: list[dict] | None = None) -> dict:
        response = self.client.post("/chat", json={"message": message, "history": history or []})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_generic_fever_and_body_aches_does_not_overcommit_to_malaria(self) -> None:
        data = self.post_chat("حاسس اني عندي سخونيه و تكسير في الجسم")

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertLess(data["confidence"], 0.6)
        self.assertLessEqual(len(data["follow_up_questions"]), 2)

        answer = data["answer"]
        self.assertIn("عدوى فيروسية", answer)
        self.assertNotIn("الاحتمال الأقرب هو: ملاريا", answer)
        self.assertNotIn("بكتيريا", answer)
        self.assertNotIn("حمام ملح", answer)
        self.assertNotIn("تأمل", answer)

    def test_previous_malaria_question_is_explanation_not_repeated_template(self) -> None:
        data = self.post_chat(
            "اي الملاريا دي اصلا",
            history=[
                {"role": "assistant", "content": "الاحتمال الأقرب مبدئيًا هو: ملاريا."},
            ],
        )

        self.assertEqual(data["mode"], "diagnosis")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertLessEqual(len(data["follow_up_questions"]), 2)

        answer = data["answer"]
        self.assertIn("طفيل", answer)
        self.assertIn("البعوض", answer)
        self.assertIn("لا تتأكد", answer)
        self.assertNotIn("بكتيريا", answer)
        self.assertNotIn("مستوى الخطورة:", answer)
        self.assertNotIn("نصائح مبدئية:", answer)

    def test_malaria_challenge_without_risk_context_downgrades_diagnosis(self) -> None:
        data = self.post_chat(
            "ليه ملاريا؟ انا معنديش سفر ولا ناموس",
            history=[
                {"role": "assistant", "content": "الاحتمال الأقرب مبدئيًا هو: ملاريا."},
            ],
        )

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertLess(data["confidence"], 0.6)
        self.assertLessEqual(len(data["follow_up_questions"]), 2)
        self.assertIn("طبيب عام", data.get("display_doctor_ar") or "")

        answer = data["answer"]
        self.assertIn("أضعف", answer)
        self.assertIn("عدوى فيروسية", answer)
        self.assertNotIn("الاحتمال الأقرب هو: ملاريا", answer)

    def test_malaria_supported_context_is_possible_but_cautious(self) -> None:
        data = self.post_chat("عندي حرارة ورعشة وتعرق بعد سفر وفي ناموس كتير")

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["possible_diagnosis"], "Malaria")
        self.assertLess(data["confidence"], 0.75)
        self.assertEqual(data["urgency_level"], "Medium")
        self.assertIn("أمراض معدية", data.get("display_doctor_ar") or "")
        self.assertLessEqual(len(data["follow_up_questions"]), 2)

        answer = data["answer"]
        self.assertIn("طفيل", answer)
        self.assertIn("ليست بكتيريا", answer)
        self.assertIn("لا يتأكد", answer)

    def test_emergency_regression_still_overrides_normal_diagnosis(self) -> None:
        data = self.post_chat(
            "من ساعة فجأة نص جسمي الشمال تنمل وبقيت بتكلم بصعوبة ووشي حاسه مايل شوية"
        )

        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertNotEqual(data["possible_diagnosis"], "Hypertension")
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertIn("الطوارئ", data["answer"])


if __name__ == "__main__":
    unittest.main()
