from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.chat_engine_v2 import (
    CLARIFICATION_MODE,
    CLOSING_MODE,
    DIAGNOSIS_MODE,
    EMERGENCY_MODE,
    CaseState,
    ChatEngineV2,
    SafePlan,
    _deterministic_answer_review,
)
from app.schemas import ChatRequest


class ConversationBrainTests(unittest.TestCase):
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

    def assert_no_diagnosis_no_questionnaire(self, data: dict) -> None:
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertNotIn("محتاج أعرف كام حاجة", data["answer"])
        self.assertNotIn("هل عندك ألم", data["answer"])
        self.assertNotIn("الاحتمال الأقرب", data["answer"])
        self.assertNotIn("Malaria", data["answer"])
        self.assertNotIn("AIDS", data["answer"])
        self.assertNotIn("ملاريا", data["answer"])
        self.assertNotIn("نقص المناعة", data["answer"])

    def test_temperature_followup_answer_updates_case_without_reasking_temperature(self) -> None:
        data = self.post_chat(
            "الحرارة وصلت 40",
            history=[
                {"role": "user", "content": "حاسس اني عندي سخونيه و تكسير في الجسم"},
                {
                    "role": "assistant",
                    "content": "الاحتمال الأقرب عدوى فيروسية. الحرارة وصلت كام تقريبًا؟ والأعراض بدأت من إمتى؟",
                },
            ],
        )

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["urgency_level"], "Medium")
        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertLessEqual(len(data["follow_up_questions"]), 2)
        self.assertFalse(any("وصلت كام" in question for question in data["follow_up_questions"]))
        self.assertIn("40", data["answer"])
        self.assertIn("عالية", data["answer"])
        self.assertIn("طبيب عام", data.get("display_doctor_ar") or "")
        self.assertNotEqual(data["possible_diagnosis"], "AIDS")
        self.assertNotEqual(data["possible_diagnosis"], "Malaria")
        self.assertNotIn("نقص المناعة", data["answer"])

    def test_casual_message_with_previous_medical_history_does_not_diagnose(self) -> None:
        messages = ["اهلا", "عامل اي", "اي الدنيا", "اي يبررووو", "يعّم انا بسالك اي الدنيا عموما يعني ازيك"]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(
                    message,
                    history=[
                        {"role": "user", "content": "حاسس اني عندي سخونيه و تكسير في الجسم"},
                        {"role": "assistant", "content": "الاحتمال الأقرب مبدئيًا هو: ملاريا."},
                    ],
                )

                self.assertEqual(data["mode"], "clarification")
                self.assertIsNone(data["possible_diagnosis"])
                self.assertEqual(data["confidence"], 0.0)
                self.assertFalse(data["needs_follow_up"])
                self.assertEqual(data["follow_up_questions"], [])
                self.assertNotIn("ملاريا", data["answer"])
                self.assertNotIn("نقص المناعة", data["answer"])

    def test_random_low_information_input_does_not_trigger_questionnaire(self) -> None:
        messages = ["اي يبررووو", "بروو ؟؟", "bro what", "are you okay"]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(message)

                self.assert_no_diagnosis_no_questionnaire(data)

    def test_family_or_offtopic_questions_do_not_trigger_medical_questionnaire(self) -> None:
        messages = [
            "امك عاملة اي",
            "امك عامله اي",
            "ابوك عامل اي طب",
            "خالتك عاملة اي",
            "خالتك عامله اي",
            "اختك عاملة اي",
            "اختك عامله اي",
            "how is your mom",
        ]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(message)

                self.assert_no_diagnosis_no_questionnaire(data)
                if any("a" <= char.lower() <= "z" for char in message):
                    self.assertIn("medical assistant", data["answer"].lower())
                    self.assertNotRegex(data["answer"], r"[\u0600-\u06FF]")
                else:
                    self.assertIn("مساعد طبي", data["answer"])

    def test_arabic_profanity_and_insults_set_boundary_only(self) -> None:
        messages = ["كسمك", "يا غبي", "بتفهم؟", "انت عبيط؟"]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(
                    message,
                    history=[
                        {"role": "user", "content": "حاسس اني عندي سخونيه و تكسير في الجسم"},
                        {"role": "assistant", "content": "الحرارة وصلت كام تقريبًا؟"},
                        {"role": "user", "content": "الحرارة وصلت 40"},
                    ],
                )

                self.assert_no_diagnosis_no_questionnaire(data)
                self.assertTrue("احترام" in data["answer"] or "متضايق" in data["answer"])

    def test_english_casual_and_profanity_stay_english_and_non_diagnostic(self) -> None:
        messages = ["how are you", "what's up bro", "fuck you", "are you stupid?", "this is dumb"]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(message)

                self.assert_no_diagnosis_no_questionnaire(data)
                self.assertNotRegex(data["answer"], r"[\u0600-\u06FF]")
                self.assertTrue(
                    "medical" in data["answer"].lower()
                    or "respectful" in data["answer"].lower()
                    or "frustration" in data["answer"].lower()
                )

    def test_english_casual_message_does_not_diagnose(self) -> None:
        data = self.post_chat(
            "How are you?",
            history=[
                {"role": "assistant", "content": "The closest initial direction is: Malaria."},
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertIn("I’m", data["answer"])
        self.assertNotIn("Malaria", data["answer"])
        self.assertNotIn("AIDS", data["answer"])

    def test_insult_or_frustration_deescalates_without_new_diagnosis(self) -> None:
        for message in ["انت كويس يبني؟", "يبني انت عبيط؟", "this makes no sense", "this is stupid"]:
            with self.subTest(message=message):
                data = self.post_chat(
                    message,
                    history=[
                        {"role": "user", "content": "حاسس اني عندي سخونيه و تكسير في الجسم"},
                        {"role": "assistant", "content": "الحرارة وصلت كام تقريبًا؟"},
                        {"role": "user", "content": "الحرارة وصلت 40"},
                    ],
                )

                self.assertEqual(data["mode"], "clarification")
                self.assertIsNone(data["possible_diagnosis"])
                self.assertEqual(data["confidence"], 0.0)
                self.assertNotIn("نقص المناعة", data["answer"])
                self.assertNotIn("الاحتمال الأقرب", data["answer"])

    def test_body_ache_only_is_broad_clarification_not_rare_disease(self) -> None:
        data = self.post_chat("انا حاسس ان جسمي واجعني اوي")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        self.assertNotIn("ملاريا", data["answer"])
        self.assertNotIn("نقص المناعة", data["answer"])
        self.assertIn("وجع الجسم", data["answer"])

    def test_reproductive_concept_gets_focused_gynecology_clarification(self) -> None:
        data = self.post_chat("انا حاسس اني عندي البريود")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertEqual(data["suggested_doctor"], "Gynecologist")
        self.assertIn("نساء", data.get("display_doctor_ar") or "")
        self.assertTrue(data["needs_follow_up"])
        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertIn("حمل", joined_questions)
        self.assertIn("الدورة", joined_questions)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)
        self.assertNotIn("ألم صدر", joined_questions)
        self.assertNotIn("ضيق تنفس", joined_questions)
        self.assertNotIn("ملاريا", data["answer"])
        self.assertNotIn("نقص المناعة", data["answer"])

    def test_english_period_or_cramps_gets_english_focused_clarification(self) -> None:
        for message in ["I think I have period pain", "I have cramps"]:
            with self.subTest(message=message):
                data = self.post_chat(message)

                self.assertEqual(data["mode"], "clarification")
                self.assertIsNone(data["possible_diagnosis"])
                self.assertEqual(data["confidence"], 0.0)
                self.assertEqual(data["suggested_doctor"], "Gynecologist")
                joined_questions = " ".join(data["follow_up_questions"]).lower()
                self.assertIn("pregnancy", joined_questions)
                self.assertIn("period", joined_questions)
                self.assertIn("menstrual", data["answer"].lower())
                self.assertNotIn("Malaria", data["answer"])
                self.assertNotIn("AIDS", data["answer"])

    def test_throat_pain_stays_throat_focused(self) -> None:
        data = self.post_chat("زوري واجعني")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertIn("حرارة", joined_questions)
        self.assertIn("البلع", joined_questions)
        self.assertIn("تنفس", joined_questions)
        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        self.assertLessEqual(data["answer"].count("- "), 3)

    def test_back_pain_stays_back_focused_not_cervical_or_neuro(self) -> None:
        data = self.post_chat("\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIn("\u0627\u0644\u0638\u0647\u0631", data["answer"])
        self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotIn("Cervical", data["answer"])
        self.assertNotEqual(data["suggested_doctor"], "Neurologist")
        self.assertIn(data["suggested_doctor"], {"Orthopedic doctor", "General Practitioner"})
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertIn("\u0627\u0644\u0638\u0647\u0631", joined_questions)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)
        self.assertEqual(data["case_state_update"]["medical_meaning"]["domain"], "musculoskeletal_back_pain")
        self.assertIn("back", data["case_state_update"]["medical_meaning"]["body_parts"])

    def test_lower_back_pain_asks_back_red_flags(self) -> None:
        data = self.post_chat("\u0648\u062c\u0639 \u0627\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIn("\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631", data["answer"])
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertTrue("بدأ" in joined_questions or "إصابة" in joined_questions or "حمل" in joined_questions)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)

    def test_lower_back_followup_continuity_after_denied_numbness(self) -> None:
        data = self.post_chat(
            "\u0644\u0627 \u0645\u0639\u0647\u0648\u0634 \u062a\u0646\u0645\u064a\u0644",
            history=[
                {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
                {
                    "role": "assistant",
                    "content": (
                        "\u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631 \u0648\u0644\u0627 \u0623\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631\u061f "
                        "\u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u061f "
                        "\u0647\u0644 \u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0631\u062c\u0644 \u0623\u0648 \u0645\u0639\u0627\u0647 \u062a\u0646\u0645\u064a\u0644/\u0636\u0639\u0641 \u0623\u0648 \u0645\u0634\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644 \u0623\u0648 \u0627\u0644\u0628\u0631\u0627\u0632\u061f"
                    ),
                },
                {"role": "user", "content": "\u0627\u0644\u0627\u0644\u0645 \u0641 \u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631"},
                {
                    "role": "assistant",
                    "content": (
                        "\u062a\u0645\u0627\u0645\u060c \u0643\u062f\u0647 \u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631. "
                        "\u0647\u0644 \u0628\u062f\u0623 \u0628\u0639\u062f \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629 \u0623\u0648 \u062d\u0631\u0643\u0629 \u0645\u0641\u0627\u062c\u0626\u0629\u061f"
                    ),
                },
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIn("\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631", data["answer"])
        self.assertIn("\u062a\u0646\u0645\u064a\u0644", data["answer"])
        self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotIn("\u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotEqual(data["suggested_doctor"], "Neurologist")
        self.assertIn(data["suggested_doctor"], {"Orthopedic doctor", "General Practitioner"})
        self.assertNotIn("\u062a\u0646\u0645\u064a\u0644", " ".join(data["follow_up_questions"]))
        self.assertIn("numbness", data["case_state_update"]["denied_concepts"])
        self.assertEqual(data["case_state_update"]["medical_meaning"]["domain"], "musculoskeletal_back_pain")
        self.assertEqual(data["case_state_update"]["active_case"]["active_body_part"], "lower_back")

    def test_lower_back_followup_ignores_naturalizer_body_part_drift(self) -> None:
        original_naturalizer = main_module.llm_service.naturalize_response

        def drifting_naturalizer(**_: object) -> str:
            return "\u062e\u0634\u0648\u0646\u0629 \u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629 \u0648\u064a\u0641\u0636\u0644 \u0637\u0628\u064a\u0628 \u0645\u062e \u0648\u0623\u0639\u0635\u0627\u0628."

        main_module.llm_service.naturalize_response = drifting_naturalizer
        try:
            data = self.post_chat(
                "\u0644\u0627 \u0645\u0639\u0647\u0648\u0634 \u062a\u0646\u0645\u064a\u0644",
                history=[
                    {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
                    {
                        "role": "assistant",
                        "content": (
                            "\u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631 \u0648\u0644\u0627 \u0623\u0639\u0644\u0649\u061f "
                            "\u0647\u0644 \u0645\u0639\u0627\u0647 \u062a\u0646\u0645\u064a\u0644 \u0623\u0648 \u0636\u0639\u0641\u061f"
                        ),
                    },
                    {"role": "user", "content": "\u0627\u0644\u0627\u0644\u0645 \u0641 \u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631"},
                ],
            )
        finally:
            main_module.llm_service.naturalize_response = original_naturalizer

        self.assertEqual(data["mode"], "clarification")
        self.assertIn("\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631", data["answer"])
        self.assertIn("\u062a\u0646\u0645\u064a\u0644", data["answer"])
        self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotIn("\u0645\u062e \u0648\u0623\u0639\u0635\u0627\u0628", data["answer"])
        self.assertEqual(data["suggested_doctor"], "Orthopedic doctor")

    def test_back_pain_questions_are_not_duplicated_in_answer(self) -> None:
        data = self.post_chat("\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a")

        self.assertEqual(len(data["follow_up_questions"]), len(set(data["follow_up_questions"])))
        for question in data["follow_up_questions"]:
            self.assertLessEqual(data["answer"].count(question), 1)
        self.assertNotIn("أسئلة توضيحية", data["answer"])

    def test_inline_naturalized_questions_are_removed_from_answer(self) -> None:
        original_naturalizer = main_module.llm_service.naturalize_response

        def repetitive_naturalizer(**_: object) -> str:
            return (
                "\u0645\u062d\u062a\u0627\u062c \u0623\u0633\u0623\u0644\u0643 \u0639\u0646 \u0627\u0644\u0632\u0648\u0631. "
                "\u0647\u0644 \u0639\u0646\u062f\u0643 \u062d\u0631\u0627\u0631\u0629 \u0623\u0648 \u0643\u062d\u0629\u061f "
                "\u0647\u0644 \u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0632\u064a\u062f \u0645\u0639 \u0627\u0644\u0628\u0644\u0639\u061f"
            )

        main_module.llm_service.naturalize_response = repetitive_naturalizer
        try:
            data = self.post_chat("\u0632\u0648\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a")
        finally:
            main_module.llm_service.naturalize_response = original_naturalizer

        self.assertEqual(data["mode"], "clarification")
        self.assertLessEqual(len(data["follow_up_questions"]), 1)
        for question in data["follow_up_questions"]:
            self.assertLessEqual(data["answer"].count(question), 1)
        self.assertIn("\u0627\u0644\u0632\u0648\u0631", data["answer"])

    def test_short_followup_answers_keep_active_back_case(self) -> None:
        history = [
            {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
            {
                "role": "assistant",
                "content": "\u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631 \u0648\u0644\u0627 \u0623\u0639\u0644\u0649\u061f \u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u061f",
            },
        ]
        for message in [
            "\u0644\u0627",
            "\u0627\u0647",
            "\u0645\u0646 \u064a\u0648\u0645\u064a\u0646",
            "\u0641\u064a \u0627\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
        ]:
            with self.subTest(message=message):
                data = self.post_chat(message, history=history)
                self.assertEqual(data["mode"], "clarification")
                self.assertIsNone(data["possible_diagnosis"])
                self.assertIn("back", data["case_state_update"]["medical_meaning"]["body_parts"])
                self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
                self.assertNotIn("Cervical", data["answer"])

    def test_v2_gynecology_followup_keeps_period_and_discharge_context(self) -> None:
        data = self.post_chat(
            "\u0627\u0646\u0627 21 \u0633\u0646\u0647 \u0648 \u0639\u0646\u062f\u064a \u0627\u0641\u0631\u0627\u0632\u0627\u062a \u0643\u062a\u064a\u0631\u0647",
            history=[
                {"role": "user", "content": "\u0639\u0646\u062f\u064a \u0627\u0644\u0628\u0631\u064a\u0648\u062f"},
                {
                    "role": "assistant",
                    "content": "\u0633\u0646\u0643 \u0643\u0627\u0645\u061f \u0647\u0644 \u0641\u064a \u0625\u0641\u0631\u0627\u0632\u0627\u062a \u0623\u0648 \u0623\u0644\u0645\u061f",
                },
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["suggested_doctor"], "Gynecologist")
        self.assertIn("gynecology", data["case_state_update"]["active_case"]["active_domain"])
        self.assertIn("discharge", data["case_state_update"]["active_case"]["symptoms"])
        self.assertIn("21", data["answer"])
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("\u0644\u0648\u0646", joined)
        self.assertIn("\u062d\u0643\u0629", joined)
        self.assertNotIn("\u0643\u062d\u0629", joined)
        self.assertNotIn("\u0635\u062f\u0631", joined)
        self.assertNotIn("Cervical", data["answer"])

    def test_v2_english_back_pain_does_not_repeat_answered_questions(self) -> None:
        history = [
            {"role": "user", "content": "i have a back-pain"},
            {
                "role": "assistant",
                "content": "Is it upper back or lower back? When did it start? Did it go down the leg?",
            },
            {"role": "user", "content": "upper back"},
            {"role": "assistant", "content": "When did it start, and was there any injury?"},
            {"role": "user", "content": "upper back pain and it started from 3 days and i didn't have any injury"},
            {"role": "assistant", "content": "Does it go down to the leg?"},
            {"role": "user", "content": "upper back pain and it doesnt go down to the leg"},
            {"role": "assistant", "content": "Any numbness, weakness, or bladder/bowel control problems?"},
        ]
        data = self.post_chat("No it doesn't", history=history)

        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["suggested_doctor"], "Orthopedic doctor")
        self.assertNotRegex(data["answer"], r"[\u0600-\u06FF]")
        self.assertIn("upper", data["answer"].lower())
        joined = " ".join(data["follow_up_questions"]).lower()
        self.assertNotIn("upper back or lower back", joined)
        self.assertNotIn("when did it start", joined)
        self.assertNotIn("injury", joined)
        self.assertNotIn("go down to the leg", joined)

    def test_v2_english_worded_duration_does_not_repeat_duration_question(self) -> None:
        data = self.post_chat(
            "It does not go down my leg",
            history=[
                {"role": "user", "content": "I have upper back pain"},
                {
                    "role": "assistant",
                    "content": "When did it start, and was it after injury, heavy lifting, sudden movement, or long sitting?",
                },
                {"role": "user", "content": "It started three days ago and I had no injury"},
                {
                    "role": "assistant",
                    "content": "Does it go down to the leg, or come with numbness, weakness, or bladder/bowel control problems?",
                },
            ],
        )

        joined = " ".join(data["follow_up_questions"]).lower()
        self.assertNotIn("when did it start", joined)
        self.assertNotIn("injury", joined)
        self.assertNotIn("go down to the leg", joined)

    def test_v2_new_headache_complaint_switches_away_from_back_case(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0627\u0644\u0645 \u0641 \u062f\u0645\u0627\u063a\u064a \u0648 \u0635\u062f\u0627\u0639",
            history=[
                {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
                {"role": "assistant", "content": "\u0623\u0644\u0645 \u0627\u0644\u0638\u0647\u0631 \u0641\u064a\u0646\u061f"},
                {"role": "user", "content": "\u0627\u0644\u0627\u0644\u0645 \u0641 \u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631"},
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertIn("\u0635\u062f\u0627\u0639", data["answer"])
        self.assertNotIn("\u0627\u0644\u0636\u0647\u0631", data["answer"])
        self.assertNotIn("\u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotEqual(data["suggested_doctor"], "Orthopedic doctor")

    def test_v2_vague_nerves_is_new_neuro_clarification_not_back_or_cervical(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u062a\u0639\u0628 \u0641\u064a \u0627\u0644\u0627\u0639\u0635\u0627\u0628",
            history=[
                {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
                {"role": "assistant", "content": "\u0623\u0644\u0645 \u0627\u0644\u0638\u0647\u0631 \u0641\u064a\u0646\u061f"},
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertEqual(data["case_state_update"]["active_case"]["active_domain"], "neurology_vague")
        joined = " ".join(data["follow_up_questions"])
        self.assertIn("\u062a\u0646\u0645\u064a\u0644", joined)
        self.assertIn("\u0636\u0639\u0641", joined)
        self.assertNotIn("\u0627\u0644\u0638\u0647\u0631", data["answer"])
        self.assertNotIn("\u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])

    def test_v2_emergency_and_nonmedical_guardrails_still_win(self) -> None:
        emergency = self.post_chat("My left side is numb and my speech is slurred")
        self.assertEqual(emergency["mode"], "emergency")
        self.assertEqual(emergency["urgency_level"], "High")

        abuse = self.post_chat("\u0643\u0633\u0645\u0643\u0643")
        self.assertEqual(abuse["mode"], "clarification")
        self.assertIsNone(abuse["possible_diagnosis"])
        self.assertFalse(abuse["needs_follow_up"])
        self.assertEqual(abuse["follow_up_questions"], [])

    def test_v2_chest_pain_without_red_flags_is_urgent_clarification(self) -> None:
        arabic = self.post_chat("\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0641\u064a \u0635\u062f\u0631\u064a")
        self.assertEqual(arabic["mode"], "clarification")
        self.assertIsNone(arabic["possible_diagnosis"])
        self.assertEqual(arabic["urgency_level"], "Medium")
        self.assertEqual(arabic["suggested_doctor"], "Cardiologist")
        joined_ar = " ".join(arabic["follow_up_questions"])
        self.assertIn("\u0636\u064a\u0642 \u0646\u0641\u0633", joined_ar)
        self.assertIn("\u0639\u0631\u0642 \u0628\u0627\u0631\u062f", joined_ar)
        self.assertIn("\u0630\u0631\u0627\u0639", joined_ar)
        self.assertIn("\u0627\u0644\u0637\u0648\u0627\u0631\u0626", arabic["answer"])
        self.assertNotIn("Heart attack", arabic["answer"])
        self.assertNotIn("\u0623\u0632\u0645\u0629 \u0642\u0644\u0628\u064a\u0629", arabic["answer"])

        english = self.post_chat("I have chest pain")
        self.assertEqual(english["mode"], "clarification")
        self.assertIsNone(english["possible_diagnosis"])
        self.assertEqual(english["urgency_level"], "Medium")
        joined_en = " ".join(english["follow_up_questions"]).lower()
        self.assertIn("shortness of breath", joined_en)
        self.assertIn("cold sweat", joined_en)
        self.assertIn("arm", joined_en)
        self.assertIn("emergency care", english["answer"].lower())
        self.assertNotIn("heart attack", english["answer"].lower())

    def test_v2_chest_pain_with_red_flags_is_emergency(self) -> None:
        data = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0634\u062f\u064a\u062f \u0641\u064a \u0635\u062f\u0631\u064a \u0648\u0636\u064a\u0642 \u0646\u0641\u0633 \u0648\u0639\u0631\u0642 \u0628\u0627\u0631\u062f"
        )
        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("Emergency", data["suggested_doctor"])
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])

    def test_v2_domain_questions_do_not_use_unrelated_generic_emergency_list(self) -> None:
        cases = [
            ("\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a", ["\u0627\u0644\u0638\u0647\u0631"], ["\u0623\u0644\u0645 \u0635\u062f\u0631", "\u0639\u0631\u0642 \u0628\u0627\u0631\u062f"]),
            ("\u0632\u0648\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a", ["\u0627\u0644\u0628\u0644\u0639", "\u062a\u0646\u0641\u0633"], ["\u0627\u0644\u0628\u0648\u0644", "\u0627\u0644\u0630\u0631\u0627\u0639"]),
            ("\u062d\u0631\u0642\u0627\u0646 \u0628\u0648\u0644", ["\u0627\u0644\u062a\u0628\u0648\u0644", "\u062f\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644"], ["\u0623\u0644\u0645 \u0635\u062f\u0631", "\u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629"]),
            ("\u0639\u0646\u062f\u064a \u0627\u0644\u0628\u0631\u064a\u0648\u062f", ["\u0627\u0644\u062f\u0648\u0631\u0629", "\u062d\u0645\u0644"], ["\u0623\u0644\u0645 \u0635\u062f\u0631", "\u0636\u064a\u0642 \u0646\u0641\u0633"]),
            ("\u0639\u0646\u062f\u064a \u0627\u0644\u0645 \u0641 \u062f\u0645\u0627\u063a\u064a \u0648 \u0635\u062f\u0627\u0639", ["\u0627\u0644\u0635\u062f\u0627\u0639", "\u062a\u0646\u0645\u064a\u0644"], ["\u0623\u0644\u0645 \u0635\u062f\u0631", "\u0627\u0644\u0628\u0648\u0644"]),
        ]
        for message, expected_terms, blocked_terms in cases:
            with self.subTest(message=message):
                data = self.post_chat(message)
                joined = " ".join(data["follow_up_questions"]) + " " + data["answer"]
                for term in expected_terms:
                    self.assertIn(term, joined)
                for term in blocked_terms:
                    self.assertNotIn(term, joined)

    def test_v2_deterministic_final_verifier_rejects_wrong_domain_draft(self) -> None:
        plan = SafePlan(
            route="clarification",
            language="ar",
            domain="back_pain",
            answer="\u0623\u0644\u0645 \u0627\u0644\u0638\u0647\u0631 \u0645\u062d\u062a\u0627\u062c \u062a\u0641\u0627\u0635\u064a\u0644 \u0645\u0646 \u063a\u064a\u0631 \u062a\u0634\u062e\u064a\u0635.",
            questions_to_ask=["\u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u061f"],
            doctor_route="Orthopedic doctor",
            case_state=CaseState(active_domain="back_pain", active_body_part="back"),
        )
        bad_review = _deterministic_answer_review(
            "\u064a\u0628\u062f\u0648 \u0623\u0646\u0647\u0627 \u0645\u0634\u0643\u0644\u0629 \u0641\u064a \u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629.",
            plan,
        )
        self.assertFalse(bad_review["is_safe"])
        self.assertTrue(bad_review["has_unrelated_body_part_or_disease"])
        self.assertEqual(bad_review["suggested_fix"], plan.answer)

        chest_plan = SafePlan(
            route="clarification",
            language="en",
            domain="chest_pain",
            answer="Chest pain needs careful checking before assuming an emergency.",
            questions_to_ask=["Any shortness of breath?"],
            doctor_route="Cardiologist",
        )
        chest_review = _deterministic_answer_review("This is probably a heart attack.", chest_plan)
        self.assertFalse(chest_review["is_safe"])
        self.assertEqual(chest_review["suggested_fix"], chest_plan.answer)

    def test_v2_llm_judge_runs_only_for_eligible_medical_routes(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def review_final_answer(self, **_: object) -> dict[str, object]:
                self.calls += 1
                return {
                    "is_safe": True,
                    "is_relevant": True,
                    "is_consistent_with_user": True,
                    "has_unrelated_body_part_or_disease": False,
                    "repeats_answered_questions": False,
                    "uses_wrong_emergency_level": False,
                }

        fake_llm = FakeLLM()
        engine = ChatEngineV2(
            classifier_service=None,
            knowledge_service=None,
            rag_service=None,
            llm_service=fake_llm,
        )
        request = ChatRequest(message="زوري واجعني", history=[])

        eligible_plan = SafePlan(
            route=CLARIFICATION_MODE,
            language="ar",
            domain="throat_ent",
            answer="ألم الزور محتاج شوية تفاصيل.",
            questions_to_ask=["هل عندك حرارة؟"],
        )
        self.assertEqual(engine._judge_answer(eligible_plan.answer, eligible_plan, request), eligible_plan.answer)
        self.assertEqual(fake_llm.calls, 1)

        skipped_plans = [
            SafePlan(route=EMERGENCY_MODE, language="ar", domain="chest_pain", answer="اتجه للطوارئ فورًا."),
            SafePlan(route=CLOSING_MODE, language="ar", domain="closing", answer="تسلم، أتمنى لك الصحة."),
            SafePlan(route=CLARIFICATION_MODE, language="ar", domain="casual", answer="أهلًا بيك."),
            SafePlan(route=CLARIFICATION_MODE, language="ar", domain="abuse", answer="خلينا نتكلم باحترام."),
            SafePlan(route=CLARIFICATION_MODE, language="ar", domain="off_topic", answer="أنا مساعد طبي."),
            SafePlan(route=CLARIFICATION_MODE, language="ar", domain="nonsense", answer="مش قادر أفهم الرسالة."),
        ]
        for plan in skipped_plans:
            with self.subTest(domain=plan.domain, route=plan.route):
                self.assertEqual(engine._judge_answer(plan.answer, plan, request), plan.answer)
        self.assertEqual(fake_llm.calls, 1)

    def test_v2_llm_judge_failure_returns_verified_deterministic_answer(self) -> None:
        class FailingLLM:
            def review_final_answer(self, **_: object) -> dict[str, object]:
                raise TimeoutError("simulated timeout")

        engine = ChatEngineV2(
            classifier_service=None,
            knowledge_service=None,
            rag_service=None,
            llm_service=FailingLLM(),
        )
        plan = SafePlan(
            route=DIAGNOSIS_MODE,
            language="ar",
            domain="throat_ent",
            answer="ألم الزور محتاج متابعة لو مستمر.",
            questions_to_ask=[],
        )
        request = ChatRequest(message="زوري واجعني", history=[])
        self.assertEqual(engine._judge_answer(plan.answer, plan, request), plan.answer)

    def test_v2_llm_judge_invalid_output_returns_deterministic_answer(self) -> None:
        class InvalidOutputLLM:
            def review_final_answer(self, **_: object) -> None:
                return None

        engine = ChatEngineV2(
            classifier_service=None,
            knowledge_service=None,
            rag_service=None,
            llm_service=InvalidOutputLLM(),
        )
        plan = SafePlan(
            route=CLARIFICATION_MODE,
            language="en",
            domain="back_pain",
            answer="Back pain needs a few focused details before guessing a diagnosis.",
            questions_to_ask=["When did it start?"],
        )
        request = ChatRequest(message="I have back pain", history=[])
        self.assertEqual(engine._judge_answer(plan.answer, plan, request), plan.answer)

    def test_v2_llm_judge_unsafe_or_repetitive_rewrite_is_rejected(self) -> None:
        class UnsafeRewriteLLM:
            def review_final_answer(self, **_: object) -> dict[str, object]:
                return {
                    "is_safe": False,
                    "is_relevant": False,
                    "is_consistent_with_user": False,
                    "has_unrelated_body_part_or_disease": True,
                    "repeats_answered_questions": False,
                    "uses_wrong_emergency_level": False,
                    "suggested_fix": "يبدو أنها مشكلة في فقرات الرقبة.",
                }

        engine = ChatEngineV2(
            classifier_service=None,
            knowledge_service=None,
            rag_service=None,
            llm_service=UnsafeRewriteLLM(),
        )
        plan = SafePlan(
            route=CLARIFICATION_MODE,
            language="ar",
            domain="back_pain",
            answer="ألم الظهر محتاج تفاصيل بسيطة من غير تشخيص.",
            questions_to_ask=["بدأ من إمتى؟"],
        )
        request = ChatRequest(message="ضهري واجعني", history=[])
        self.assertEqual(engine._judge_answer(plan.answer, plan, request), plan.answer)

        class RepetitiveRewriteLLM:
            def review_final_answer(self, **_: object) -> dict[str, object]:
                return {
                    "is_safe": False,
                    "is_relevant": True,
                    "is_consistent_with_user": True,
                    "has_unrelated_body_part_or_disease": False,
                    "repeats_answered_questions": True,
                    "uses_wrong_emergency_level": False,
                    "suggested_fix": "بدأ من إمتى؟ بدأ من إمتى؟",
                }

        engine.llm_service = RepetitiveRewriteLLM()
        self.assertEqual(engine._judge_answer(plan.answer, plan, request), plan.answer)

    def test_v2_casual_replies_are_not_identical_static_text(self) -> None:
        first = self.post_chat("\u0627\u0647\u0644\u0627")
        second = self.post_chat("\u0639\u0627\u0645\u0644 \u0627\u064a")
        self.assertIsNone(first["possible_diagnosis"])
        self.assertIsNone(second["possible_diagnosis"])
        self.assertNotEqual(first["answer"], second["answer"])

    def test_neck_pain_stays_neck_focused_not_back_or_abdomen(self) -> None:
        data = self.post_chat("\u0631\u0642\u0628\u062a\u064a \u0648\u0627\u062c\u0639\u0627\u0646\u064a")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIn("\u0627\u0644\u0631\u0642\u0628\u0629", data["answer"])
        self.assertNotIn("\u0627\u0644\u0638\u0647\u0631", data["answer"])
        self.assertNotIn("\u0627\u0644\u0628\u0637\u0646", data["answer"])
        self.assertIn(data["suggested_doctor"], {"Orthopedic doctor", "General Practitioner"})

    def test_exact_arabic_domain_examples_stay_consistent(self) -> None:
        throat = self.post_chat("\u0632\u0648\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a")
        self.assertEqual(throat["mode"], "clarification")
        self.assertIsNone(throat["possible_diagnosis"])
        self.assertNotIn("\u0627\u0644\u0638\u0647\u0631", throat["answer"])
        self.assertNotIn("\u0645\u0639\u062f\u0629", throat["answer"])

        abdomen = self.post_chat("\u0628\u0637\u0646\u064a \u0628\u062a\u0648\u062c\u0639\u0646\u064a")
        self.assertEqual(abdomen["mode"], "clarification")
        self.assertIsNone(abdomen["possible_diagnosis"])
        self.assertIn("\u0627\u0644\u0628\u0637\u0646", " ".join(abdomen["follow_up_questions"]) + abdomen["answer"])
        self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", abdomen["answer"])

        urinary = self.post_chat("\u062d\u0631\u0642\u0627\u0646 \u0628\u0648\u0644")
        self.assertIn(urinary["mode"], {"clarification", "diagnosis"})
        self.assertIn(urinary["suggested_doctor"], {"Urologist", "General Practitioner"})
        self.assertNotIn("\u0627\u0644\u0631\u0642\u0628\u0629", urinary["answer"])

        chest = self.post_chat("\u0635\u062f\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a \u0648\u0636\u064a\u0642 \u0646\u0641\u0633")
        self.assertEqual(chest["mode"], "emergency")
        self.assertEqual(chest["urgency_level"], "High")

    def test_throat_and_fever_history_does_not_overdiagnose_aids(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "message": "My throat still hurts and I also have fever",
                "conversation_id": "test-conv-throat-fever",
                "history": [
                    {"role": "user", "content": "My throat hurts"},
                    {"role": "assistant", "content": "How long has it been hurting?"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertNotEqual(data["possible_diagnosis"], "AIDS")
        self.assertNotIn("AIDS", data["answer"])
        self.assertEqual(data.get("conversation_id"), "test-conv-throat-fever")
        self.assertIsInstance(data.get("case_state_update"), dict)

    def test_abdominal_pain_stays_digestive_focused(self) -> None:
        data = self.post_chat("بطني بتوجعني")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertIn("البطن", joined_questions)
        self.assertLessEqual(len(data["follow_up_questions"]), 1)
        self.assertNotIn("ألم صدر", joined_questions)

    def test_english_body_ache_only_is_english_and_not_rare_disease(self) -> None:
        data = self.post_chat("I have body aches")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        self.assertIn("Body aches", data["answer"])
        self.assertNotIn("Malaria", data["answer"])
        self.assertNotIn("AIDS", data["answer"])

    def test_english_throat_pain_is_english_focused_clarification(self) -> None:
        data = self.post_chat("My throat hurts")

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        self.assertNotRegex(data["answer"], r"[\u0600-\u06FF]")
        self.assertIn("throat", " ".join(data["follow_up_questions"]).lower() + " " + data["answer"].lower())

    def test_english_back_and_neck_pain_are_body_part_consistent(self) -> None:
        back = self.post_chat("my back hurts")
        self.assertEqual(back["mode"], "clarification")
        self.assertIsNone(back["possible_diagnosis"])
        self.assertIn("back", back["answer"].lower())
        self.assertNotIn("neck", back["answer"].lower())
        self.assertNotEqual(back["suggested_doctor"], "Neurologist")
        self.assertEqual(back["case_state_update"]["medical_meaning"]["domain"], "musculoskeletal_back_pain")

        neck = self.post_chat("my neck hurts")
        self.assertEqual(neck["mode"], "clarification")
        self.assertIsNone(neck["possible_diagnosis"])
        self.assertIn("neck", neck["answer"].lower())
        self.assertNotIn("stomach", neck["answer"].lower())
        self.assertEqual(neck["case_state_update"]["medical_meaning"]["domain"], "musculoskeletal_neck_pain")

    def test_correction_negation_removes_fever_assumption(self) -> None:
        data = self.post_chat(
            "انا مقولتش اني عندي حرارة",
            history=[
                {"role": "user", "content": "جسمي واجعني اوي"},
                {
                    "role": "assistant",
                    "content": "السخونية مع تكسير الجسم قد تكون عدوى فيروسية. الحرارة وصلت كام؟",
                },
            ],
        )

        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertNotIn("الحرارة وصلت كام", " ".join(data["follow_up_questions"]))
        self.assertNotIn("ملاريا", data["answer"])
        self.assertIn("نستبعد", data["answer"])

    def test_duration_followup_answer_does_not_repeat_duration_question(self) -> None:
        data = self.post_chat(
            "بقالها يومين",
            history=[
                {"role": "user", "content": "زوري واجعني"},
                {
                    "role": "assistant",
                    "content": "محتاج أعرف كام حاجة بسيطة عشان أوجهك صح. منذ متى بدأ ألم الزور أو الإحساس ده؟",
                },
            ],
        )

        self.assertLessEqual(len(data["follow_up_questions"]), 3)
        joined_questions = " ".join(data["follow_up_questions"])
        self.assertNotIn("منذ متى", joined_questions)
        self.assertNotIn("بدأ من إمتى", joined_questions)

    def test_previous_diagnosis_question_explains_without_reclassifying(self) -> None:
        data = self.post_chat(
            "اي هي الملاريا اصلا",
            history=[
                {"role": "assistant", "content": "الاحتمال الأقرب مبدئيًا هو: ملاريا."},
            ],
        )

        self.assertEqual(data["mode"], "diagnosis")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertEqual(data["confidence"], 0.0)
        self.assertIn("طفيل", data["answer"])
        self.assertIn("البعوض", data["answer"])
        self.assertNotIn("مستوى الخطورة:", data["answer"])

    def test_challenge_without_malaria_risk_downgrades_malaria(self) -> None:
        data = self.post_chat(
            "ليه ملاريا؟ انا معنديش سفر ولا ناموس",
            history=[
                {"role": "assistant", "content": "الاحتمال الأقرب مبدئيًا هو: ملاريا."},
            ],
        )

        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertLess(data["confidence"], 0.6)
        self.assertIn("أضعف", data["answer"])
        self.assertNotIn("الاحتمال الأقرب هو: ملاريا", data["answer"])

    def test_direct_high_fever_message_gets_urgent_same_day_guidance(self) -> None:
        data = self.post_chat("الحرارة 40 وتكسير في جسمي")

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["urgency_level"], "Medium")
        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertLessEqual(len(data["follow_up_questions"]), 2)
        self.assertIn("40", data["answer"])
        self.assertIn("تقييم طبي قريب/اليوم", data["answer"])
        self.assertNotEqual(data["possible_diagnosis"], "AIDS")
        self.assertNotEqual(data["possible_diagnosis"], "Malaria")

    def test_fever_and_body_aches_stays_broad_not_confident_malaria(self) -> None:
        data = self.post_chat("حاسس بسخونية وتكسير")

        self.assertIn(data["mode"], {"diagnosis", "clarification"})
        self.assertNotEqual(data["possible_diagnosis"], "Malaria")
        self.assertNotEqual(data["possible_diagnosis"], "AIDS")
        self.assertNotIn("الاحتمال الأقرب هو: ملاريا", data["answer"])

    def test_english_fever_and_body_aches_stays_broad(self) -> None:
        data = self.post_chat("I have fever and body aches")

        self.assertEqual(data["mode"], "diagnosis")
        self.assertEqual(data["possible_diagnosis"], "Viral or flu-like illness")
        self.assertIn("viral or flu-like", data["answer"])
        self.assertNotIn("Malaria", data["answer"])
        self.assertNotIn("AIDS", data["answer"])

    def test_emergency_regressions_still_override_conversation_logic(self) -> None:
        messages = [
            "من ساعة فجأة نص جسمي الشمال تنمل وبقيت بتكلم بصعوبة ووشي حاسه مايل شوية",
            "عندي ألم صدر شديد وضيق تنفس",
            "بلعت كلور بالغلط",
            "My left side is numb and my speech is slurred",
        ]
        for message in messages:
            with self.subTest(message=message):
                data = self.post_chat(message)
                self.assertEqual(data["mode"], "emergency")
                self.assertEqual(data["urgency_level"], "High")
                self.assertFalse(data["needs_follow_up"])
                self.assertEqual(data["follow_up_questions"], [])

    def test_english_stroke_emergency_answer_is_english(self) -> None:
        data = self.post_chat("My left side is numb and my speech is slurred")

        self.assertEqual(data["mode"], "emergency")
        self.assertEqual(data["urgency_level"], "High")
        self.assertIn("emergency", data["answer"].lower())
        self.assertNotRegex(data["answer"], r"[\u0600-\u06FF]")
        self.assertNotEqual(data["possible_diagnosis"], "Hypertension")


if __name__ == "__main__":
    unittest.main()
