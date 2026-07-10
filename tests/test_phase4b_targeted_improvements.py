from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.knowledge_service import KnowledgeService
from app.safety import choose_urgency


class Phase4BTargetedImprovementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main_module.app)
        cls.knowledge = KnowledgeService(get_settings())
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

    def post_chat(self, message: str) -> dict:
        response = self.client.post("/chat", json={"message": message, "history": []})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_self_harm_phrase_triggers_emergency_without_undefined_diagnosis(self) -> None:
        data = self.post_chat(
            "\u0645\u0634 \u0639\u0627\u064a\u0632 \u0627\u0639\u064a\u0634 "
            "\u0648\u0639\u0627\u064a\u0632 \u0627\u0624\u0630\u064a \u0646\u0641\u0633\u064a"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])
        self.assertNotIn("\u063a\u064a\u0631 \u0645\u062d\u062f\u062f", data["answer"])
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertIn("\u0637\u0648\u0627\u0631\u0626", data["answer"])

    def test_pregnancy_bleeding_triggers_emergency(self) -> None:
        data = self.post_chat(
            "\u0623\u0646\u0627 \u062d\u0627\u0645\u0644 \u0648\u0639\u0646\u062f\u064a "
            "\u0646\u0632\u064a\u0641 \u0645\u0647\u0628\u0644\u064a \u0648\u0623\u0644\u0645 "
            "\u0634\u062f\u064a\u062f \u0623\u0633\u0641\u0644 \u0627\u0644\u0628\u0637\u0646"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])

    def test_child_not_feeding_and_lethargic_triggers_emergency(self) -> None:
        data = self.post_chat(
            "\u0627\u0628\u0646\u064a \u0637\u0641\u0644 \u0645\u0634 "
            "\u0628\u064a\u0631\u0636\u0639 \u0648\u062e\u0627\u0645\u0644"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])

    def test_urinary_retention_triggers_emergency(self) -> None:
        data = self.post_chat(
            "\u0645\u0634 \u0639\u0627\u0631\u0641 \u0627\u062a\u0628\u0648\u0644 "
            "\u0648\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0641\u064a \u0627\u0644\u062e\u0627\u0635\u0631\u0629"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])

    def test_flank_pain_fever_and_urinary_burning_routes_to_urology(self) -> None:
        message = (
            "\u062d\u0631\u0627\u0631\u0629 \u0645\u0639 \u0623\u0644\u0645 "
            "\u062c\u0646\u0628 \u0648\u062d\u0631\u0642\u0627\u0646 \u0628\u0648\u0644"
        )
        symptoms = main_module.classifier_service.extract_symptoms(message)
        urgency = str(main_module.knowledge_service.score_symptoms(symptoms, message)["urgency"])
        self.assertIn("burning_micturition", symptoms)
        self.assertIn("back_pain", symptoms)
        self.assertIn(urgency, {"Medium", "High"})
        self.assertIn(
            self.knowledge.suggest_doctor(None, symptoms, urgency, message=message),
            {"Urologist", "Emergency care"},
        )

    def test_dizziness_with_tinnitus_routes_ent_or_neurology(self) -> None:
        data = self.post_chat(
            "\u0627\u0644\u062f\u0646\u064a\u0627 \u0628\u062a\u0644\u0641 "
            "\u0648\u0648\u062f\u0646\u064a \u0641\u064a\u0647\u0627 \u0637\u0646\u064a\u0646"
        )
        self.assertIn(data["suggested_doctor"], {"ENT specialist", "Neurologist"})
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("\u0637\u0646\u064a\u0646", joined)

    def test_eye_pain_and_vision_change_routes_ophthalmologist(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0639\u064a\u0646 "
            "\u0648\u0632\u063a\u0644\u0644\u0629"
        )
        self.assertEqual(data["suggested_doctor"], "Ophthalmologist")
        self.assertIn(data["mode"], {"clarification", "diagnosis"})

    def test_dental_pain_and_swelling_routes_dentist(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0636\u0631\u0633 "
            "\u0648\u0648\u0631\u0645 \u0641\u064a \u0627\u0644\u0641\u0645"
        )
        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["suggested_doctor"], "Dentist")

    def test_severe_trauma_bleeding_triggers_emergency(self) -> None:
        data = self.post_chat(
            "\u0648\u0642\u0639\u062a \u0639\u0644\u064a\u0627 \u0648\u0641\u064a "
            "\u062c\u0631\u062d \u0639\u0645\u064a\u0642 \u0648\u0646\u0632\u064a\u0641 \u0634\u062f\u064a\u062f"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])

    def test_anxiety_without_self_harm_routes_psychiatrist_not_emergency(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0642\u0644\u0642 \u0648\u062a\u0648\u062a\u0631 "
            "\u0648\u0645\u0634 \u0628\u0646\u0627\u0645"
        )
        self.assertNotEqual(data["urgency_level"], "High")
        self.assertEqual(data["suggested_doctor"], "Psychiatrist")

    def test_vague_fever_asks_infection_focused_questions(self) -> None:
        data = self.post_chat("\u0633\u062e\u0648\u0646\u064a\u0629")
        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["suggested_doctor"], "General Practitioner")
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("\u0627\u0644\u062d\u0631\u0627\u0631\u0629", joined)
        self.assertIn("\u0639\u062f\u0648\u0649", joined)

    def test_context_terms_do_not_make_safe_anxiety_emergency(self) -> None:
        symptoms = main_module.classifier_service.extract_symptoms(
            "\u0642\u0644\u0642 \u0648\u062a\u0648\u062a\u0631 \u0648\u062e\u0648\u0641 \u0634\u062f\u064a\u062f"
        )
        self.assertNotEqual(
            choose_urgency(
                "\u0642\u0644\u0642 \u0648\u062a\u0648\u062a\u0631 \u0648\u062e\u0648\u0641 \u0634\u062f\u064a\u062f",
                symptoms,
                0,
            ),
            "High",
        )


if __name__ == "__main__":
    unittest.main()
