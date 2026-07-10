from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.llm_service import HIGH_URGENCY_PREFIX


class FinalSafetyPolishTests(unittest.TestCase):
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

    def post_chat(self, message: str) -> dict:
        response = self.client.post("/chat", json={"message": message, "history": []})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assertEmergency(self, message: str, doctor_hint: str | None = None) -> dict:
        data = self.post_chat(message)
        self.assertEqual(data["mode"], "emergency", message)
        self.assertEqual(data["urgency_level"], "High", message)
        self.assertFalse(data["needs_follow_up"], message)
        self.assertEqual(data["follow_up_questions"], [], message)
        self.assertNotEqual(data["possible_diagnosis"], "Hypertension", message)
        doctor_display = data.get("display_doctor_ar") or ""
        self.assertTrue("الطوارئ" in doctor_display or "طوارئ" in doctor_display, message)
        if doctor_hint:
            self.assertIn(doctor_hint, doctor_display, message)
        answer = data["answer"]
        if any("\u0600" <= char <= "\u06FF" for char in message):
            self.assertIn("طوارئ", answer, message)
        else:
            self.assertIn("emergency", answer.lower(), message)
        for blocked in ["الخطورة منخفضة", "استرخاء", "تأمل", "حمام ملح", "راقب فقط"]:
            self.assertNotIn(blocked, answer, message)
        return data

    def test_stroke_emergency_exact_message(self) -> None:
        data = self.assertEmergency(
            "من ساعة فجأة نص جسمي الشمال تنمل وبقيت بتكلم بصعوبة ووشي حاسه مايل شوية، "
            "ومعايا صداع جامد ودوخة. أعمل إيه؟",
            "مخ وأعصاب",
        )
        self.assertEqual(data["display_diagnosis_ar"], "أعراض عصبية خطيرة محتملة")

    def test_stroke_emergency_paraphrases(self) -> None:
        messages = [
            "عندي تنميل في نص جسمي وكلامي تقيل ودوخة شديدة",
            "وشي مايل وبوقي معوج ومش عارف اتكلم كويس",
            "دراعي ورجلي الشمال تقال فجأة ومش متزن",
            "حصلتلي زغللة مفاجئة مع تنميل في ناحية واحدة",
            "my left side is numb and my speech is slurred",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEmergency(message, "مخ وأعصاب")

    def test_mild_headache_and_dizziness_is_not_emergency(self) -> None:
        data = self.post_chat("عندي صداع ودوخة بسيطة من امبارح")
        self.assertNotEqual(data["mode"], "emergency")
        self.assertNotEqual(data["urgency_level"], "High")

    def test_chest_emergency_paraphrases(self) -> None:
        messages = [
            "عندي ألم صدر شديد وضيق تنفس",
            "حاسس بعصرة في صدري وعرق بارد",
            "وجع في صدري وواصل لدراعي الشمال",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEmergency(message, "قلب")

    def test_pregnancy_emergency_paraphrases(self) -> None:
        messages = [
            "أنا حامل في الشهر السابع وعندي نزيف ووجع شديد في بطني والبيبي حركته قلت",
            "حامل وحركة الجنين قلت من الصبح",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEmergency(message, "نساء")

    def test_poisoning_emergency_paraphrases(self) -> None:
        messages = [
            "شربت منظف بالغلط وحاسس بحرقان في زوري وبطني",
            "بلعت كلور بالغلط",
            "خدت جرعة كبيرة من الدوا",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEmergency(message, "سموم")

    def test_allergy_emergency(self) -> None:
        self.assertEmergency("وشي وشفايفي ورموا ومش قادر اتنفس")

    def test_emergency_answer_is_concise_and_non_duplicated(self) -> None:
        data = self.assertEmergency(
            "من ساعة فجأة نص جسمي الشمال تنمل وبقيت بتكلم بصعوبة ووشي مايل",
            "مخ وأعصاب",
        )
        answer = data["answer"]
        self.assertEqual(answer.count(HIGH_URGENCY_PREFIX), 1)
        self.assertNotIn("نصائح مبدئية", answer)
        self.assertNotIn("متى يجب طلب مساعدة عاجلة", answer)
        self.assertLess(len(answer), 750)

    def test_follow_up_question_limits_by_mode(self) -> None:
        clarification = self.post_chat("زوري واجعني")
        self.assertEqual(clarification["mode"], "clarification")
        self.assertLessEqual(len(clarification["follow_up_questions"]), 3)

        diagnosis = self.post_chat("عندي كحة وسخونية وتعب")
        self.assertEqual(diagnosis["mode"], "diagnosis")
        self.assertLessEqual(len(diagnosis["follow_up_questions"]), 2)

        emergency = self.post_chat("عندي ألم صدر شديد وضيق تنفس")
        self.assertEqual(emergency["mode"], "emergency")
        self.assertEqual(emergency["follow_up_questions"], [])
        self.assertFalse(emergency["needs_follow_up"])

        closing = self.post_chat("شكرا")
        self.assertEqual(closing["mode"], "closing")
        self.assertEqual(closing["follow_up_questions"], [])
        self.assertFalse(closing["needs_follow_up"])

    def test_broader_emergency_category_routes(self) -> None:
        cases = [
            ("مش قادر اتكلم من ضيق النفس وشفايفي زرقا", "صدر"),
            ("ابني طفل مش قادر يتنفس وخامل", "أطفال"),
            ("مش عايز اعيش وخدت جرعة كبيرة من الدوا", "نفسي"),
            ("اسهال شديد وترجيع مستمر ومش قادر اشرب ولخبطة", "باطنة"),
            ("انا مريض سكر وعندي لخبطة وجفاف شديد", "غدد"),
            ("وجع بطن شديد والبطن متحجرة", "جراحة"),
            ("مغص كلوي شديد ومعاه حرارة", "مسالك"),
        ]
        for message, doctor_hint in cases:
            with self.subTest(message=message):
                self.assertEmergency(message, doctor_hint)

    def test_clarification_is_friendly_and_specific(self) -> None:
        data = self.post_chat("زوري واجعني")
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertTrue(data["needs_follow_up"])
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertIn("حرارة", joined_questions)
        self.assertIn("البلع", joined_questions)
        self.assertIn("تنفس", joined_questions)
        self.assertIn("الزور", data["answer"])
        self.assertNotIn("أسئلة توضيحية", data["answer"])

    def test_closing_stays_closing(self) -> None:
        data = self.post_chat("شكرا")
        self.assertEqual(data["mode"], "closing")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["urgency_level"], "Low")
        self.assertFalse(data["needs_follow_up"])


if __name__ == "__main__":
    unittest.main()
