from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.llm_service import HIGH_URGENCY_PREFIX
from app.safety import choose_urgency, has_red_flags


class P0FinalValidationRepairTests(unittest.TestCase):
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

    def post_chat(self, message: str, history: list[dict[str, str]] | None = None) -> dict:
        response = self.client.post("/chat", json={"message": message, "history": history or []})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_emergency(self, message: str, history: list[dict[str, str]] | None = None) -> None:
        data = self.post_chat(message, history)
        self.assertEqual(data["mode"], "emergency", message)
        self.assertEqual(data["urgency_level"], "High", message)
        self.assertIn("Emergency", data["suggested_doctor"], message)
        self.assertFalse(data["needs_follow_up"], message)
        self.assertEqual(data["follow_up_questions"], [], message)
        self.assertTrue(data["answer"].startswith(HIGH_URGENCY_PREFIX), message)

    def test_final_validation_emergency_miss_patterns_now_override(self) -> None:
        cases = [
            ("مغص وترجيع ومش قادر اشرب مياه", None),
            ("اسهال بقاله يومين ودوخة وقلة بول", None),
            ("حكة شديدة وتورم في وشي", None),
            ("فقدت الوعي شوية وبعدها صداع شديد", None),
            ("مش وجع عادي في صدري ومعاه عرق بارد", None),
            ("دوخة وتعرق وجوع شديد مع لخبطة", None),
            ("مغص كلوي شديد وغثيان", None),
            ("قلة بول ودوخة بعد اسهال شديد", None),
            ("حرارة عالية وتدهور سريع ودوخة شديدة", None),
            (
                "والألم شديد ومش قادر اقف",
                [{"role": "user", "content": "وقعت على رجلي"}, {"role": "assistant", "content": "هل يوجد نزيف أو ألم شديد؟"}],
            ),
            ("طفل وقع على دماغه وبيستفرغ", None),
            ("pregnant ومعايا bleeding", None),
            ("بفكر أأذي نفسي بس خايف", None),
            ("الم في العين بعد خبطة", None),
        ]
        for message, history in cases:
            with self.subTest(message=message):
                self.assert_emergency(message, history)

    def test_negated_red_flags_do_not_trigger_emergency(self) -> None:
        message = "دوخة بسيطة من غير اغماء ولا الم صدر"
        symptoms = main_module.classifier_service.extract_symptoms(message)
        self.assertFalse(has_red_flags(message, symptoms))
        self.assertNotEqual(choose_urgency(message, symptoms, 0), "High")
        data = self.post_chat(message)
        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["suggested_doctor"], "Needs more information")

    def test_negated_breathlessness_is_not_repeated_in_follow_up_questions(self) -> None:
        data = self.post_chat(
            "لا مفيش ضيق نفس بس في كحة",
            [{"role": "user", "content": "عندي كحة"}, {"role": "assistant", "content": "هل في ضيق تنفس؟"}],
        )
        joined = " ".join(data["follow_up_questions"])
        self.assertNotIn("ضيق تنفس", joined)

    def test_closing_teslam_phrase_stays_closing(self) -> None:
        data = self.post_chat("تسلم يا دكتور")
        self.assertEqual(data["mode"], "closing")
        self.assertEqual(data["suggested_doctor"], "Not needed")
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])

    def test_under_specified_special_contexts_clarify_without_forced_diagnosis(self) -> None:
        for message, doctor in [
            ("عندي قلق وتوتر ومش بنام", "Psychiatrist"),
            ("عندي ألم عين وزغللة", "Ophthalmologist"),
            ("انا حامل وعندي غثيان بسيط وتعب", "Gynecologist"),
            ("ابني عنده حرارة بس", "Pediatrician"),
        ]:
            with self.subTest(message=message):
                data = self.post_chat(message)
                self.assertEqual(data["mode"], "clarification")
                self.assertIsNone(data["possible_diagnosis"])
                self.assertEqual(data["suggested_doctor"], doctor)

    def test_p0_symptom_extraction_expansions(self) -> None:
        symptoms = main_module.classifier_service.extract_symptoms("عندي كحه وسخونه وتكسير فجسمي")
        self.assertIn("cough", symptoms)
        self.assertIn("high_fever", symptoms)
        self.assertIn("muscle_pain", symptoms)

        symptoms = main_module.classifier_service.extract_symptoms("عندي تعرق ورعشه وجوووع")
        self.assertIn("sweating", symptoms)
        self.assertIn("excessive_hunger", symptoms)

        symptoms = main_module.classifier_service.extract_symptoms("سكر عالي وزغللة وعطش")
        self.assertIn("irregular_sugar_level", symptoms)
        self.assertIn("blurred_and_distorted_vision", symptoms)

    def test_top_doctor_routing_repairs(self) -> None:
        for message, expected_doctor in [
            ("عندي رشح واحتقان ووجع حلق بسيط", "General Practitioner"),
            ("ضغطي عالي جدا وزغللة", "Cardiologist"),
            ("سكر عالي وزغللة وعطش", "Endocrinologist"),
            ("افرازات مهبلية وحكة", "Gynecologist"),
            ("صداع نصفي وزغللة وغثيان", "Neurologist"),
        ]:
            with self.subTest(message=message):
                data = self.post_chat(message)
                self.assertEqual(data["suggested_doctor"], expected_doctor)


if __name__ == "__main__":
    unittest.main()
