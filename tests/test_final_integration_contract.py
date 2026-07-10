from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module


class FinalIntegrationContractTests(unittest.TestCase):
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

    def post_chat(self, payload: dict) -> dict:
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_non_diagnostic_boundary(self, data: dict) -> None:
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])

    def test_chat_accepts_backend_contract_and_returns_safe_state_update(self) -> None:
        data = self.post_chat(
            {
                "user_id": "user-123",
                "conversation_id": "conv-abc",
                "language": "en",
                "source": "web",
                "message": "My throat hurts",
                "history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Tell me your symptoms."},
                ],
            }
        )

        self.assertEqual(data["conversation_id"], "conv-abc")
        self.assertIn(data["mode"], {"clarification", "diagnosis", "emergency", "closing"})
        self.assertIsInstance(data["case_state_update"], dict)
        self.assertEqual(data["case_state_update"]["source"], "web")
        self.assertEqual(data["case_state_update"]["language"], "en")
        self.assertIn("known_symptoms", data["case_state_update"])
        self.assertIn("follow_up_questions", data["case_state_update"])
        self.assertNotIn("api_key", str(data).lower())

    def test_legacy_chat_request_still_works_without_new_fields(self) -> None:
        data = self.post_chat({"message": "My throat hurts", "history": []})

        self.assertIsNone(data["conversation_id"])
        self.assertIn("answer", data)
        self.assertIsInstance(data["case_state_update"], dict)

    def test_health_exposes_safe_build_and_model_metadata(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("build_id", data)
        self.assertIn("frontend_build_id", data)
        self.assertIn("llm_model", data)
        self.assertNotIn("api_key", response.text.lower())
        self.assertNotIn("GROQ_API_KEYS", response.text)

    def test_local_demo_conversation_id_uses_in_memory_history(self) -> None:
        conversation_id = "local-demo-contract-test"
        first = self.post_chat(
            {
                "conversation_id": conversation_id,
                "source": "local_demo",
                "message": "My throat hurts",
                "history": [],
            }
        )
        self.assertEqual(first["conversation_id"], conversation_id)
        self.assertTrue(first["case_state_update"]["local_memory"]["enabled"])
        first_count = first["case_state_update"]["local_memory"]["stored_history_messages"]

        second = self.post_chat(
            {
                "conversation_id": conversation_id,
                "source": "local_demo",
                "message": "two days",
                "history": [],
            }
        )
        self.assertEqual(second["conversation_id"], conversation_id)
        self.assertGreater(
            second["case_state_update"]["local_memory"]["stored_history_messages"],
            first_count,
        )

    def test_repeated_letter_profanity_and_family_questions_do_not_diagnose(self) -> None:
        for message in ["كسمكك", "امك عامله اي", "ابوك عامل اي", "fuck you"]:
            with self.subTest(message=message):
                data = self.post_chat(
                    {
                        "message": message,
                        "history": [
                            {"role": "user", "content": "حاسس بسخونية وتكسير"},
                            {"role": "assistant", "content": "الحرارة وصلت كام؟"},
                        ],
                    }
                )
                self.assert_non_diagnostic_boundary(data)

    def test_casual_sequence_is_safe_and_not_identical_static_reply(self) -> None:
        history: list[dict] = []
        answers = []
        for message in ["اهلا", "عامل اي", "اي الدنيا"]:
            with self.subTest(message=message):
                data = self.post_chat({"message": message, "history": history})
                self.assert_non_diagnostic_boundary(data)
                answers.append(data["answer"])
                history.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": data["answer"]},
                    ]
                )

        self.assertGreater(len(set(answers)), 1)

    def test_nonsense_inputs_are_same_language_unclear_responses(self) -> None:
        arabic = self.post_chat({"message": "وووو", "history": []})
        self.assert_non_diagnostic_boundary(arabic)
        self.assertIn("مش", arabic["answer"])
        self.assertRegex(arabic["answer"], r"[\u0600-\u06FF]")

        english = self.post_chat({"message": "aaaaaa", "history": []})
        self.assert_non_diagnostic_boundary(english)
        self.assertIn("clear", english["answer"].lower())
        self.assertNotRegex(english["answer"], r"[\u0600-\u06FF]")

    def test_scope_and_capability_questions_do_not_diagnose(self) -> None:
        for message in ["who are you?", "what can you do?"]:
            with self.subTest(message=message):
                data = self.post_chat({"message": message, "history": []})
                self.assert_non_diagnostic_boundary(data)
                self.assertIn("medical", data["answer"].lower())

    def test_medical_words_are_not_misread_as_abuse(self) -> None:
        for message in ["سخونية", "كحة", "حرقان بول", "كسر في ايدي", "سكر عالي"]:
            with self.subTest(message=message):
                data = self.post_chat({"message": message, "history": []})
                self.assertTrue(
                    data["needs_follow_up"] or data["mode"] in {"diagnosis", "emergency"},
                    msg=data,
                )

    def test_body_ache_and_urinary_routes_are_focused_not_rare_disease(self) -> None:
        body_ache = self.post_chat({"message": "انا حاسس ان جسمي واجعني اوي", "history": []})
        self.assertEqual(body_ache["mode"], "clarification")
        self.assertIsNone(body_ache["possible_diagnosis"])
        self.assertEqual(body_ache["suggested_doctor"], "General Practitioner")
        joined_body_questions = " ".join(body_ache["follow_up_questions"])
        self.assertNotIn("حمل", joined_body_questions)
        self.assertNotIn("النوع", joined_body_questions)
        self.assertNotIn("Malaria", body_ache["answer"])
        self.assertNotIn("AIDS", body_ache["answer"])

        urinary = self.post_chat({"message": "عندي حرقان بول", "history": []})
        self.assertIn(urinary["mode"], {"clarification", "diagnosis"})
        self.assertIn(urinary["suggested_doctor"], {"Urologist", "General Practitioner"})
        self.assertTrue(urinary["needs_follow_up"] or urinary["mode"] == "diagnosis")

    def test_required_domain_routing_examples_remain_specific(self) -> None:
        throat = self.post_chat({"message": "زوري واجعني", "history": []})
        self.assertEqual(throat["mode"], "clarification")
        self.assertIn(throat["suggested_doctor"], {"ENT Specialist", "ENT specialist", "Needs more information"})
        self.assertTrue(throat["follow_up_questions"])

        abdomen = self.post_chat({"message": "بطني بتوجعني", "history": []})
        self.assertEqual(abdomen["mode"], "clarification")
        self.assertIn(abdomen["suggested_doctor"], {"Gastroenterologist", "Needs more information"})
        self.assertTrue(abdomen["follow_up_questions"])

        neuro = self.post_chat({"message": "عندي صداع ودوخة", "history": []})
        self.assertIn(neuro["mode"], {"clarification", "diagnosis"})
        self.assertNotEqual(neuro["suggested_doctor"], "Cardiologist")

        emergency = self.post_chat({"message": "My left side is numb and my speech is slurred", "history": []})
        self.assertEqual(emergency["mode"], "emergency")
        self.assertEqual(emergency["urgency_level"], "High")
        self.assertIn("Emergency", emergency["suggested_doctor"])


if __name__ == "__main__":
    unittest.main()
