from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.llm_service import HIGH_URGENCY_PREFIX


class Phase3ClarificationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main_module.app)
        cls.original_llm_client = main_module.llm_service.client
        cls.original_llm_clients = list(getattr(main_module.llm_service, "clients", []))
        cls.original_retrieve = main_module.rag_service.retrieve
        cls.retrieve_calls = 0

        main_module.llm_service.client = None
        main_module.llm_service.clients = []

        def fake_retrieve(_: str):
            cls.retrieve_calls += 1
            return [
                {
                    "q_body": "Unrelated weak case",
                    "a_body": "Weak evidence",
                    "category": "general",
                    "score": -0.25,
                },
                {
                    "q_body": "Another weak case",
                    "a_body": "Still weak",
                    "category": "general",
                    "score": 0.01,
                },
            ]

        main_module.rag_service.retrieve = fake_retrieve

    @classmethod
    def tearDownClass(cls) -> None:
        main_module.llm_service.client = cls.original_llm_client
        main_module.llm_service.clients = cls.original_llm_clients
        main_module.rag_service.retrieve = cls.original_retrieve

    def post_chat(self, message: str, history: list[dict] | None = None) -> dict:
        response = self.client.post("/chat", json={"message": message, "history": history or []})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assertClarification(self, data: dict) -> None:
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertTrue(data["needs_follow_up"])
        self.assertGreaterEqual(len(data["follow_up_questions"]), 1)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)
        self.assertEqual(data["retrieved_cases"], [])
        self.assertNotIn("غير محدد", data["answer"])
        self.assertNotIn("Clarifying Questions", data["answer"])
        self.assertNotIn("أسئلة توضيحية", data["answer"])

    def test_unclear_input_returns_clarification(self) -> None:
        data = self.post_chat("مش مرتاح")
        self.assertClarification(data)

    def test_short_vague_input_returns_clarification(self) -> None:
        data = self.post_chat("تعبان")
        self.assertClarification(data)

    def test_throat_body_area_only_questions_are_targeted(self) -> None:
        data = self.post_chat("زوري واجعني")
        self.assertClarification(data)
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("حرارة", joined)
        self.assertIn("بلع", joined)
        self.assertTrue(any(term in joined for term in {"تنفس", "تورم", "رقبة"}))

    def test_abdomen_body_area_only_questions_are_targeted(self) -> None:
        data = self.post_chat("بطني تعبانة")
        self.assertClarification(data)
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("البطن", joined)
        self.assertTrue(any(term in joined for term in {"فين", "قيء", "ترجيع", "إسهال"}))

    def test_object_sensation_in_throat_returns_clarification(self) -> None:
        data = self.post_chat("حاسس بحاجة واقفة في حلقي")
        self.assertClarification(data)
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("بلع", joined)
        self.assertTrue(any(term in joined for term in {"صوت", "تورم", "رقبة"}))

    def test_clear_diagnosis_input_still_returns_diagnosis(self) -> None:
        data = self.post_chat("عندي كحة وسخونية وتعب")
        self.assertEqual(data["mode"], "diagnosis")
        self.assertIsNotNone(data["possible_diagnosis"])
        self.assertGreater(data["confidence"], 0.0)
        self.assertIn(data["urgency_level"], {"Low", "Medium", "High"})
        self.assertNotEqual(data["suggested_doctor"], "Needs more information")

    def test_closing_message_does_not_trigger_diagnosis(self) -> None:
        data = self.post_chat("\u0634\u0643\u0631\u0627")
        self.assertEqual(data["mode"], "closing")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertEqual(data["suggested_doctor"], "Not needed")
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertEqual(data["retrieved_cases"], [])
        self.assertNotIn("\u063a\u064a\u0631 \u0645\u062d\u062f\u062f", data["answer"])

    def test_user_history_can_move_clarification_to_diagnosis(self) -> None:
        history = [
            {"role": "user", "content": "\u0632\u0648\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
            {"role": "assistant", "content": "\u0647\u0644 \u0639\u0646\u062f\u0643 \u062d\u0631\u0627\u0631\u0629 \u0623\u0648 \u0643\u062d\u0629\u061f"},
        ]
        data = self.post_chat(
            "\u0648\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629",
            history=history,
        )
        self.assertEqual(data["mode"], "diagnosis")
        self.assertIsNotNone(data["possible_diagnosis"])
        self.assertIn("cough", data["extracted_symptoms"])
        self.assertIn("throat_irritation", data["extracted_symptoms"])

    def test_assistant_questions_are_not_used_as_symptom_evidence(self) -> None:
        history = [
            {"role": "assistant", "content": "\u0647\u0644 \u0639\u0646\u062f\u0643 \u0643\u062d\u0629 \u0623\u0648 \u062d\u0631\u0627\u0631\u0629\u061f"},
        ]
        data = self.post_chat("\u0644\u0627", history=history)
        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["extracted_symptoms"], [])
        self.assertIsNone(data["possible_diagnosis"])

    def test_repeated_follow_up_question_is_filtered(self) -> None:
        repeated_question = (
            "\u0647\u0644 \u0639\u0646\u062f\u0643 \u0623\u0644\u0645 \u0641\u064a \u0645\u0643\u0627\u0646 "
            "\u0645\u0639\u064a\u0646\u060c \u062d\u0631\u0627\u0631\u0629\u060c \u0643\u062d\u0629\u060c "
            "\u062f\u0648\u062e\u0629\u060c \u0642\u064a\u0621\u060c \u0625\u0633\u0647\u0627\u0644\u060c "
            "\u0637\u0641\u062d \u062c\u0644\u062f\u064a\u060c \u0623\u0648 \u0636\u064a\u0642 \u062a\u0646\u0641\u0633\u061f"
        )
        history = [{"role": "assistant", "content": repeated_question}]
        data = self.post_chat("\u0645\u0634 \u0645\u0631\u062a\u0627\u062d", history=history)
        self.assertEqual(data["mode"], "clarification")
        self.assertNotIn(repeated_question, data["follow_up_questions"])

    def test_emergency_overrides_clarification(self) -> None:
        data = self.post_chat("عندي ألم صدر شديد وضيق تنفس")
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])
        self.assertTrue(data["answer"].startswith(HIGH_URGENCY_PREFIX))

    def test_weak_or_negative_rag_cases_are_not_returned(self) -> None:
        clarification = self.post_chat("مش مرتاح")
        self.assertEqual(clarification["retrieved_cases"], [])

        diagnosis = self.post_chat("عندي كحة وسخونية وتعب")
        self.assertEqual(diagnosis["retrieved_cases"], [])

    def test_phase3_3_digestive_medium_clear_case_does_not_over_clarify(self) -> None:
        data = self.post_chat(
            "\u0648\u062c\u0639 \u0628\u0637\u0646 \u0648\u062d\u0631\u0627\u0631\u0629 \u0639\u0627\u0644\u064a\u0629 \u0648\u0641\u0642\u062f\u0627\u0646 \u0634\u0647\u064a\u0629"
        )
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["suggested_doctor"], "Gastroenterologist")
        self.assertEqual(data["confidence"], 0.0)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)

    def test_phase3_3_vestibular_case_avoids_cardiology(self) -> None:
        data = self.post_chat("\u062f\u0648\u062e\u0629 \u0648\u0639\u062f\u0645 \u0627\u062a\u0632\u0627\u0646 \u0648\u0642\u064a\u0621")
        self.assertEqual(data["mode"], "clarification")
        self.assertNotEqual(data["possible_diagnosis"], "Hypertension")
        self.assertIn(data["suggested_doctor"], {"Neurologist", "ENT specialist"})
        self.assertLessEqual(len(data["follow_up_questions"]), 1)

    def test_phase3_3_stroke_like_case_is_emergency(self) -> None:
        data = self.post_chat("\u062f\u0648\u062e\u0629 \u0648\u062a\u0646\u0645\u064a\u0644 \u0641\u064a \u0646\u0635 \u0627\u0644\u062c\u0633\u0645")
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])

    def test_phase3_3_cervical_safety_output_does_not_leak_raw_english_advice(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0648\u062c\u0639 \u0631\u0642\u0628\u0629 \u0648\u0635\u062f\u0627\u0639 \u0648\u062f\u0648\u062e\u0629"
        )
        answer = data["answer"]
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIn(data["suggested_doctor"], {"Orthopedic doctor", "Neurologist"})
        self.assertTrue(any(term in answer for term in {"الرقبة", "رقبة", "تنميل", "ضعف"}))
        self.assertNotIn("Cervical spondylosis", answer)
        self.assertNotIn("otc", answer.lower())
        self.assertNotIn("pain reliver", answer.lower())

    def test_phase3_3_specialist_doctor_routing_examples(self) -> None:
        skin = self.post_chat("\u0639\u0646\u062f\u064a \u0637\u0641\u062d \u062c\u0644\u062f\u064a \u0648\u062d\u0643\u0629 \u0634\u062f\u064a\u062f\u0629")
        self.assertEqual(skin["suggested_doctor"], "Dermatologist")

        urinary = self.post_chat(
            "\u0639\u0646\u062f\u064a \u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0628\u0648\u0644 \u0648\u0628\u0631\u0648\u062d \u0627\u0644\u062d\u0645\u0627\u0645 \u0643\u062a\u064a\u0631"
        )
        self.assertEqual(urinary["suggested_doctor"], "Urologist")


if __name__ == "__main__":
    unittest.main()
