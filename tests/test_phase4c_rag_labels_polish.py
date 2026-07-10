from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import main as main_module
from app.display_labels import display_diagnosis_ar, display_doctor_ar
from app.llm_service import HIGH_URGENCY_PREFIX
from app.rag_service import filter_rag_cases_for_prompt, is_rag_case_safe, sanitize_rag_answer
from app.response_utils import to_frontend_safe_response
from app.schemas import ChatResponse, RetrievedCase


class Phase4CRagLabelsPolishTests(unittest.TestCase):
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

    def test_display_labels_exist_for_internal_diagnosis_and_doctor_labels(self) -> None:
        self.assertIn("\u0646\u0632\u0644\u0629", display_diagnosis_ar("Common Cold") or "")
        self.assertIn("\u0627\u0644\u0645\u0639\u062f\u0629", display_diagnosis_ar("Peptic ulcer diseae") or "")
        self.assertIn("\u0637\u0627\u0631\u0626\u0629", display_diagnosis_ar("Heart attack") or "")
        self.assertEqual(display_doctor_ar("Emergency care"), "\u0627\u0644\u0637\u0648\u0627\u0631\u0626 \u0641\u0648\u0631\u064b\u0627")
        self.assertIn("\u0645\u062e", display_doctor_ar("Neurologist") or "")
        self.assertEqual(display_diagnosis_ar("Unknown English Label"), "\u062d\u0627\u0644\u0629 \u062a\u062d\u062a\u0627\u062c \u062a\u0642\u064a\u064a\u0645 \u0637\u0628\u064a")
        self.assertEqual(display_doctor_ar("Unknown Specialist"), "\u062a\u062e\u0635\u0635 \u0637\u0628\u064a \u0645\u0646\u0627\u0633\u0628")

    def test_internal_misspelled_diagnosis_does_not_appear_in_patient_answer(self) -> None:
        answer = main_module.llm_service.generate_answer(
            message="\u0648\u062c\u0639 \u0645\u0639\u062f\u0629",
            history=[],
            extracted_symptoms=["stomach_pain"],
            diagnosis="Peptic ulcer diseae",
            confidence=0.72,
            urgency_level="Medium",
            suggested_doctor="Gastroenterologist",
            precautions=[],
            diagnosis_description=None,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        self.assertNotIn("Peptic ulcer diseae", answer)
        self.assertIn("\u0627\u0644\u0645\u0639\u062f\u0629", answer)
        self.assertIn("\u0637\u0628\u064a\u0628 \u062c\u0647\u0627\u0632 \u0647\u0636\u0645\u064a", answer)

    def test_emergency_answer_starts_with_urgent_warning(self) -> None:
        answer = main_module.llm_service.generate_answer(
            message="\u0623\u0644\u0645 \u0635\u062f\u0631 \u0634\u062f\u064a\u062f \u0648\u0636\u064a\u0642 \u062a\u0646\u0641\u0633",
            history=[],
            extracted_symptoms=["chest_pain", "breathlessness"],
            diagnosis="Heart attack",
            confidence=0.95,
            urgency_level="High",
            suggested_doctor="Emergency care",
            precautions=[],
            diagnosis_description=None,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        self.assertTrue(answer.startswith(HIGH_URGENCY_PREFIX))
        self.assertNotIn("\u0627\u0646\u062a\u0638\u0631", answer)
        self.assertIn("\u0637\u0648\u0627\u0631\u0626", answer)

    def test_closing_answer_does_not_attempt_diagnosis(self) -> None:
        data = self.post_chat("\u0634\u0643\u0631\u0627")
        self.assertEqual(data["mode"], "closing")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIsNone(data["display_diagnosis_ar"])
        self.assertEqual(data["suggested_doctor"], "Not needed")
        self.assertFalse(data["needs_follow_up"])
        self.assertEqual(data["follow_up_questions"], [])
        self.assertNotIn("\u0627\u0644\u0627\u062d\u062a\u0645\u0627\u0644 \u0627\u0644\u0623\u0642\u0631\u0628", data["answer"])

    def test_unsafe_short_or_unclear_rag_cases_are_filtered(self) -> None:
        unsafe_medication = {
            "q_body": "\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629",
            "a_body": "\u062e\u0630 \u0645\u0636\u0627\u062f \u062d\u064a\u0648\u064a \u0648\u062c\u0631\u0639\u0629 \u064a\u0648\u0645\u064a\u0629 \u0648\u0627\u0633\u062a\u0639\u0645\u0644 \u0628\u062e\u0627\u062e.",
            "category": "respiratory",
            "score": 0.95,
        }
        vague_short = {
            "q_body": "\u062a\u0639\u0628\u0627\u0646",
            "a_body": "\u0631\u0627\u062c\u0639 \u0627\u0644\u0637\u0628\u064a\u0628",
            "category": "general",
            "score": 0.9,
        }
        raw_english = {
            "q_body": "cough fever",
            "a_body": "take medicine now and follow this prescription",
            "category": "general",
            "score": 0.9,
        }
        safe_case = {
            "q_body": "\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629 \u0645\u0646\u0630 \u064a\u0648\u0645\u064a\u0646",
            "a_body": "\u0627\u0644\u0631\u0627\u062d\u0629 \u0648\u0634\u0631\u0628 \u0633\u0648\u0627\u0626\u0644 \u0648\u0645\u062a\u0627\u0628\u0639\u0629 \u0627\u0644\u062d\u0631\u0627\u0631\u0629 \u0645\u0647\u0645\u0629\u060c \u0648\u064a\u0641\u0636\u0644 \u0645\u0631\u0627\u062c\u0639\u0629 \u0637\u0628\u064a\u0628 \u0625\u0630\u0627 \u0632\u0627\u062f\u062a \u0627\u0644\u0623\u0639\u0631\u0627\u0636.",
            "category": "respiratory",
            "score": 0.87,
        }

        self.assertFalse(is_rag_case_safe(unsafe_medication))
        self.assertFalse(is_rag_case_safe(vague_short))
        self.assertFalse(is_rag_case_safe(raw_english))
        self.assertTrue(is_rag_case_safe(safe_case))

        filtered = filter_rag_cases_for_prompt(
            [unsafe_medication, vague_short, raw_english, safe_case],
            urgency_level="Medium",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["category"], "respiratory")
        self.assertEqual(filtered[0]["a_body"], sanitize_rag_answer(safe_case["a_body"]))

    def test_api_does_not_return_unsafe_rag_debug_cases(self) -> None:
        unsafe_case = {
            "q_body": "\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629",
            "a_body": "\u062e\u0630 \u0645\u0636\u0627\u062f \u062d\u064a\u0648\u064a \u0648\u0627\u0633\u062a\u0639\u0645\u0644 \u0628\u062e\u0627\u062e \u0641\u0648\u0631\u0627.",
            "category": "respiratory",
            "score": 0.99,
        }
        safe_case = {
            "q_body": "\u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629 \u0648\u062a\u0639\u0628",
            "a_body": "\u0627\u0644\u0631\u0627\u062d\u0629 \u0648\u0634\u0631\u0628 \u0633\u0648\u0627\u0626\u0644 \u0648\u0645\u062a\u0627\u0628\u0639\u0629 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0645\u0647\u0645\u0629 \u0645\u0639 \u0645\u0631\u0627\u062c\u0639\u0629 \u0637\u0628\u064a\u0628 \u0625\u0630\u0627 \u0632\u0627\u062f\u062a.",
            "category": "respiratory",
            "score": 0.88,
        }
        original_retrieve = main_module.rag_service.retrieve
        main_module.rag_service.retrieve = lambda _: [unsafe_case, safe_case]
        try:
            data = self.post_chat("\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629 \u0648\u062a\u0639\u0628")
        finally:
            main_module.rag_service.retrieve = original_retrieve

        self.assertLessEqual(len(data["retrieved_cases"]), 1)
        for case in data["retrieved_cases"]:
            self.assertNotIn("\u0645\u0636\u0627\u062f", case["a_body"])

    def test_frontend_safe_response_hides_debug_fields(self) -> None:
        response = ChatResponse(
            mode="diagnosis",
            answer="\u0625\u062c\u0627\u0628\u0629 \u0639\u0631\u0628\u064a\u0629",
            extracted_symptoms=["cough"],
            possible_diagnosis="Common Cold",
            display_diagnosis_ar=display_diagnosis_ar("Common Cold"),
            confidence=0.7,
            urgency_level="Medium",
            suggested_doctor="General Practitioner",
            display_doctor_ar=display_doctor_ar("General Practitioner"),
            precautions=[],
            needs_follow_up=True,
            follow_up_questions=["\u0645\u0646\u0630 \u0645\u062a\u0649 \u0628\u062f\u0623\u062a \u0627\u0644\u0623\u0639\u0631\u0627\u0636\u061f"],
            retrieved_cases=[
                RetrievedCase(
                    q_body="\u0633\u0624\u0627\u0644",
                    a_body="\u0625\u062c\u0627\u0628\u0629",
                    category="general",
                    score=0.5,
                )
            ],
        )

        safe = to_frontend_safe_response(response)
        self.assertIn("answer", safe)
        self.assertIn("display_diagnosis_ar", safe)
        self.assertNotIn("retrieved_cases", safe)
        self.assertNotIn("extracted_symptoms", safe)
        self.assertNotIn("confidence", safe)
        self.assertNotIn("possible_diagnosis", safe)

        clarification_safe = to_frontend_safe_response(
            {
                **safe,
                "mode": "clarification",
                "display_diagnosis_ar": display_diagnosis_ar("Common Cold"),
            }
        )
        self.assertIsNone(clarification_safe["display_diagnosis_ar"])

    def test_clarification_answer_has_no_undefined_or_confidence(self) -> None:
        data = self.post_chat("\u0645\u0634 \u0645\u0631\u062a\u0627\u062d")
        self.assertEqual(data["mode"], "clarification")
        self.assertIsNone(data["possible_diagnosis"])
        self.assertIsNone(data["display_diagnosis_ar"])
        self.assertNotIn("\u063a\u064a\u0631 \u0645\u062d\u062f\u062f", data["answer"])
        self.assertNotIn("0.0", data["answer"])


if __name__ == "__main__":
    unittest.main()
