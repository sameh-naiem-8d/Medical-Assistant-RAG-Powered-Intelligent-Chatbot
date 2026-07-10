from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app.chat_engine_v3 import ChatEngineV3
from app.schemas import ChatMessage, ChatRequest


class _DeterministicLLM:
    client = object()

    def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs["deterministic_plan"])  # type: ignore[index]

    def review_v3_answer(self, **_: object) -> dict[str, object]:
        return {"approved": True, "issues": [], "safe_rewrite": None}


class ChatEngineV3TranscriptTests(unittest.TestCase):
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

    def deterministic_engine(self) -> ChatEngineV3:
        return ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=_DeterministicLLM(),
        )

    def test_failed_manual_headache_transcript_now_progresses(self) -> None:
        history: list[dict] = []

        first = self.post_chat(
            "\u062d\u0627\u0633\u0647 \u0627\u0646 \u0627\u0646\u0627 \u062a\u0639\u0628\u0627\u0646\u0647 \u0648 \u0631\u0627\u0633\u064a \u0628\u062a\u0648\u062c\u0639\u0646\u064a",
            history,
        )
        self.assertEqual(first["mode"], "clarification")
        self.assertEqual(first["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertNotIn("\u0623\u0644\u0645 \u0635\u062f\u0631", first["answer"])
        self.assertNotIn("\u0636\u064a\u0642 \u062a\u0646\u0641\u0633", first["answer"])
        self.assertLessEqual(first["answer"].count("\u061f") + first["answer"].count("?"), 2)
        history.extend(
            [
                {"role": "user", "content": "\u062d\u0627\u0633\u0647 \u0627\u0646 \u0627\u0646\u0627 \u062a\u0639\u0628\u0627\u0646\u0647 \u0648 \u0631\u0627\u0633\u064a \u0628\u062a\u0648\u062c\u0639\u0646\u064a"},
                {"role": "assistant", "content": first["answer"]},
            ]
        )

        second = self.post_chat("\u0632\u063a\u0644\u0644\u0629 \u0648 \u062a\u063a\u064a\u0631 \u0641\u064a \u0627\u0644\u0646\u0638\u0631", history)
        self.assertEqual(second["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertTrue(second["case_state_update"]["active_case"]["facts"].get("vision_change"))
        self.assertIn("\u0632\u063a\u0644", second["answer"])
        self.assertNotIn("\u0647\u0644 \u0645\u0639\u0627\u0647 \u0632\u063a\u0644\u0644\u0629", second["answer"])
        history.extend(
            [
                {"role": "user", "content": "\u0632\u063a\u0644\u0644\u0629 \u0648 \u062a\u063a\u064a\u0631 \u0641\u064a \u0627\u0644\u0646\u0638\u0631"},
                {"role": "assistant", "content": second["answer"]},
            ]
        )

        casual = self.post_chat("\u0645\u0627\u0645\u0627 \u0639\u0627\u0645\u0644\u0647 \u0627\u064a", history)
        self.assertEqual(casual["mode"], "clarification")
        self.assertEqual(casual["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertIn("\u0645\u0627\u0645\u062a\u0643", casual["answer"])
        self.assertNotIn("\u0645\u0633\u0627\u0639\u062f \u0637\u0628\u064a", casual["answer"])
        history.extend(
            [
                {"role": "user", "content": "\u0645\u0627\u0645\u0627 \u0639\u0627\u0645\u0644\u0647 \u0627\u064a"},
                {"role": "assistant", "content": casual["answer"]},
            ]
        )

        severe = self.post_chat(
            "\u0645\u0627\u0634\u064a \u0637\u0628 \u0627\u0646\u0627 \u0639\u0646\u062f\u064a \u0635\u062f\u0627\u0639 \u062c\u0627\u0645\u062f",
            history,
        )
        self.assertEqual(severe["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertIn("\u0635\u062f\u0627\u0639", severe["answer"])
        self.assertNotIn("\u0627\u0644\u0638\u0647\u0631", severe["answer"])
        self.assertNotIn("Provide a more detailed response", severe["answer"])
        history.extend(
            [
                {"role": "user", "content": "\u0645\u0627\u0634\u064a \u0637\u0628 \u0627\u0646\u0627 \u0639\u0646\u062f\u064a \u0635\u062f\u0627\u0639 \u062c\u0627\u0645\u062f"},
                {"role": "assistant", "content": severe["answer"]},
            ]
        )

        sudden = self.post_chat("\u0635\u062f\u0627\u0639 \u0641\u062c\u0623\u0647", history)
        self.assertEqual(sudden["mode"], "emergency")
        self.assertEqual(sudden["urgency_level"], "High")
        self.assertIn("Emergency", sudden["suggested_doctor"])
        self.assertFalse(sudden["needs_follow_up"])
        self.assertEqual(sudden["follow_up_questions"], [])
        self.assertNotIn("Provide a more detailed response", sudden["answer"])

    def test_abdominal_transcript_uses_one_visible_question_and_memory_ids(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> None:
                return None

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        history: list[ChatMessage] = []

        def ask(message: str) -> dict:
            response = engine.handle_chat(ChatRequest(message=message, history=history))
            return response.model_dump()

        first_message = "بطني بتوجعني و حاسه اني عايزه ارجع"
        first = ask(first_message)
        self.assertEqual(first["mode"], "clarification")
        self.assertEqual(first["case_state_update"]["active_case"]["active_domain"], "digestive")
        self.assertIn("بطن", first["answer"])
        self.assertIn("ترجيع", first["answer"])
        self.assertNotIn("أسئلة توضيحية", first["answer"])
        self.assertLessEqual(first["answer"].count("؟") + first["answer"].count("?"), 1)
        self.assertLessEqual(len(first["follow_up_questions"]), 1)
        self.assertIn("abdominal_location", first["case_state_update"]["active_case"]["pending_question_ids"])
        history.extend([ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first["answer"])])

        second_message = "الألم في نصف البطن و احساس بالترجيع"
        second = ask(second_message)
        active_case = second["case_state_update"]["active_case"]
        self.assertEqual(active_case["active_domain"], "digestive")
        self.assertEqual(active_case["known_facts"].get("abdominal_location"), "middle_abdomen")
        self.assertTrue(active_case["known_facts"].get("nausea_present"))
        self.assertIn("abdominal_location", active_case["answered_question_ids"])
        self.assertNotIn("الألم فين", second["answer"])
        self.assertNotIn("عايزه ارجع", second["answer"])
        self.assertIn("تاريخ", second["answer"])
        self.assertLessEqual(second["answer"].count("؟") + second["answer"].count("?"), 1)
        history.extend([ChatMessage(role="user", content=second_message), ChatMessage(role="assistant", content=second["answer"])])

        third = ask("لا مفيش تاريخ سابق")
        active_case = third["case_state_update"]["active_case"]
        self.assertEqual(active_case["active_domain"], "digestive")
        self.assertEqual(active_case["known_facts"].get("abdominal_location"), "middle_abdomen")
        self.assertTrue(active_case["known_facts"].get("vomiting_present"))
        self.assertIn("previous_gastric_history", active_case["answered_question_ids"])
        self.assertIn("previous_gastric_history", active_case["denied_facts"])
        self.assertNotIn("الألم فين", third["answer"])
        self.assertLessEqual(third["answer"].count("؟") + third["answer"].count("?"), 1)

    def test_local_demo_renders_answer_only_not_followup_metadata(self) -> None:
        html = Path("local_demo_frontend/index.html").read_text(encoding="utf-8")
        self.assertNotIn("أسئلة توضيحية", html)
        self.assertNotIn("questions-title", html)
        self.assertNotIn("data.follow_up_questions.filter", html)
        self.assertIn("FRONTEND_BUILD_ID", html)
        self.assertIn("CONVERSATION_STORAGE_KEY", html)
        self.assertIn("sessionStorage", html)
        self.assertIn("resetConversationId", html)

    def test_universal_planner_patient_answer_is_used_when_valid(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                return {
                    "intent": "medical_complaint",
                    "case_action": "continue",
                    "domain": "digestive",
                    "clinical_summary": "abdominal pain with nausea",
                    "risk_level": "low",
                    "analysis_for_patient": "Digestive pattern.",
                    "patient_answer": "I understand the stomach pain with nausea. This most often points to a digestive cause, but the exact location matters. Where exactly is the pain?",
                    "next_question": {"id": "abdominal_location", "text": "Where exactly is the pain?"},
                    "doctor_route": "Gastroenterologist",
                    "response_goal": "clarify",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        data = engine.handle_chat(ChatRequest(message="My stomach hurts and I feel nauseous")).model_dump()

        self.assertIn("I understand the stomach pain", data["answer"])
        self.assertEqual(
            data["case_state_update"]["clinical_plan"]["patient_answer_source"],
            "llm_planner",
        )
        self.assertLessEqual(data["answer"].count("?"), 1)

    def test_universal_planner_patient_answer_falls_back_when_invalid(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                return {
                    "intent": "medical_complaint",
                    "case_action": "continue",
                    "domain": "digestive",
                    "clinical_summary": "abdominal pain",
                    "risk_level": "low",
                    "patient_answer": "Clarifying Questions\nWhere is the pain?\nDo you have fever?",
                    "next_question": {"id": "abdominal_location", "text": "Where is the pain?"},
                    "doctor_route": "Gastroenterologist",
                    "response_goal": "clarify",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        data = engine.handle_chat(ChatRequest(message="My stomach hurts")).model_dump()

        self.assertNotIn("Clarifying Questions", data["answer"])
        self.assertLessEqual(data["answer"].count("?"), 1)

    def test_dental_planner_answer_cannot_leak_previous_urinary_case(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                return {
                    "intent": "medical_complaint",
                    "case_action": "start_new",
                    "domain": "urinary",
                    "clinical_summary": "tooth pain",
                    "risk_level": "low",
                    "patient_answer": "أنا أتحدث عن حرقان في البول، ولا أرى دمًا في البول.",
                    "next_question": {
                        "id": "dental_swelling",
                        "text": "هل في تورم في الخد أو اللثة، حرارة، أو صعوبة في فتح الفم؟",
                    },
                    "doctor_route": "Urologist",
                    "response_goal": "clarify",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        data = engine.handle_chat(ChatRequest(message="عندي مشكلة تانية، سني بيوجعني")).model_dump()

        self.assertEqual(data["suggested_doctor"], "Dentist")
        self.assertIn("أسنان", data["answer"])
        self.assertNotIn("بول", data["answer"])
        self.assertEqual(data["case_state_update"]["active_case"]["active_domain"], "dental")

    def test_same_domain_continues_and_true_domain_change_starts_new_case(self) -> None:
        headache = self.post_chat(
            "\u0645\u0627\u0634\u064a \u0637\u0628 \u0635\u062f\u0627\u0639 \u062c\u0627\u0645\u062f",
            [
                {"role": "user", "content": "\u0639\u0646\u062f\u064a \u0635\u062f\u0627\u0639"},
                {"role": "assistant", "content": "\u0627\u0644\u0635\u062f\u0627\u0639 \u0628\u062f\u0623 \u0641\u062c\u0623\u0629 \u0648\u0644\u0627 \u062a\u062f\u0631\u064a\u062c\u064a\u061f"},
            ],
        )
        self.assertEqual(headache["case_state_update"]["active_case"]["active_domain"], "headache")

        changed = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0627\u0644\u0645 \u0641 \u062f\u0645\u0627\u063a\u064a \u0648 \u0635\u062f\u0627\u0639",
            [
                {"role": "user", "content": "\u0636\u0647\u0631\u064a \u0648\u0627\u062c\u0639\u0646\u064a"},
                {"role": "assistant", "content": "\u0623\u0644\u0645 \u0627\u0644\u0638\u0647\u0631 \u0641\u064a\u0646\u061f"},
            ],
        )
        self.assertEqual(changed["case_state_update"]["active_case"]["active_domain"], "headache")
        self.assertNotIn("\u0627\u0644\u0638\u0647\u0631", changed["answer"])

    def test_colloquial_tooth_case_switch_from_urinary_case(self) -> None:
        history = [
            {"role": "user", "content": "\u0639\u0646\u062f\u064a \u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0628\u0648\u0644"},
            {"role": "assistant", "content": "\u0647\u0644 \u0641\u064a \u062f\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644\u061f"},
        ]
        switched = self.post_chat(
            "\u0639\u0646\u062f\u064a \u0645\u0634\u0643\u0644\u0629 \u062a\u0627\u0646\u064a\u0629\u060c \u0633\u0646\u064a \u0628\u064a\u0648\u062c\u0639\u0646\u064a",
            history,
        )
        self.assertEqual(switched["case_state_update"]["active_case"]["active_domain"], "dental")
        self.assertNotIn("\u0627\u0644\u0628\u0648\u0644", switched["answer"])

    def test_side_pain_correction_updates_left_right_without_stale_side(self) -> None:
        first = self.post_chat("\u0639\u0646\u062f\u064a \u0648\u062c\u0639 \u0641\u064a \u062c\u0646\u0628\u064a \u0627\u0644\u064a\u0645\u064a\u0646")
        self.assertEqual(first["case_state_update"]["active_case"]["active_domain"], "side_pain")
        self.assertEqual(first["case_state_update"]["active_case"]["facts"].get("side"), "right")

        second = self.post_chat(
            "\u0644\u0627 \u0645\u0639\u0644\u0634 \u0642\u0635\u062f\u064a \u0627\u0644\u0634\u0645\u0627\u0644",
            history=[
                {"role": "user", "content": "\u0639\u0646\u062f\u064a \u0648\u062c\u0639 \u0641\u064a \u062c\u0646\u0628\u064a \u0627\u0644\u064a\u0645\u064a\u0646"},
                {"role": "assistant", "content": first["answer"]},
            ],
        )
        self.assertEqual(second["case_state_update"]["active_case"]["active_domain"], "side_pain")
        self.assertEqual(second["case_state_update"]["active_case"]["facts"].get("side"), "left")
        self.assertIn("\u0627\u0644\u0634\u0645\u0627\u0644", second["answer"])
        self.assertNotIn("\u0641\u064a\u0631\u0648\u0633", second["answer"])

    def test_invalid_planner_and_judge_outputs_do_not_leak(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.plan_calls = 0
                self.review_calls = 0

            def plan_clinical_turn(self, **_: object) -> object:
                self.plan_calls += 1
                return "not-json"

            def review_v3_answer(self, **_: object) -> object:
                self.review_calls += 1
                return {"approved": False, "safe_rewrite": "Provide a more detailed response or ask a new question to gather more information."}

        fake = FakeLLM()
        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=fake,
        )
        data = engine.handle_chat(ChatRequest(message="\u0639\u0646\u062f\u064a \u0635\u062f\u0627\u0639"))
        self.assertEqual(fake.plan_calls, 1)
        self.assertEqual(fake.review_calls, 1)
        self.assertNotIn("Provide a more detailed response", data.answer)
        self.assertEqual(data.case_state_update["engine_trace"]["planner_used"], False)

    def test_valid_planner_and_approved_judge_are_called_for_medical_turn(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.plan_calls = 0
                self.review_calls = 0

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                self.plan_calls += 1
                return {
                    "intent": "medical_complaint",
                    "case_action": "continue",
                    "domain": "headache",
                    "clinical_summary": "headache without red flags yet",
                    "risk_level": "low",
                    "next_best_question": "\u0627\u0644\u0635\u062f\u0627\u0639 \u0628\u062f\u0623 \u0641\u062c\u0623\u0629 \u0648\u0644\u0627 \u062a\u062f\u0631\u064a\u062c\u064a\u061f",
                    "doctor_route": "Neurologist",
                    "response_goal": "clarify",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                self.review_calls += 1
                return {"approved": True, "issues": [], "safe_rewrite": None}

        fake = FakeLLM()
        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=fake,
        )
        data = engine.handle_chat(ChatRequest(message="\u0639\u0646\u062f\u064a \u0635\u062f\u0627\u0639"))
        self.assertEqual(fake.plan_calls, 1)
        self.assertEqual(fake.review_calls, 1)
        self.assertTrue(data.case_state_update["engine_trace"]["planner_used"])
        self.assertTrue(data.case_state_update["engine_trace"]["judge_used"])
        self.assertIsInstance(data.case_state_update["engine_trace"]["total_latency_ms"], float)
        self.assertLessEqual(len(data.follow_up_questions), 2)

    def test_planner_cannot_erase_recognized_medical_complaint(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                return {
                    "intent": "nonsense",
                    "case_action": "none",
                    "domain": "unknown",
                    "clinical_summary": "incorrectly erased medical meaning",
                    "risk_level": "low",
                    "response_goal": "clarify",
                    "patient_answer": "مش قادر أفهم الرسالة دي كشكوى صحية. اكتب العرض بشكل أوضح.",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        response = engine.handle_chat(ChatRequest(message="عندي صداع بيزيد"))

        self.assertEqual(response.case_state_update["active_case"]["active_domain"], "headache")
        self.assertNotIn("مش قادر أفهم الرسالة دي كشكوى صحية", response.answer)
        self.assertEqual(response.case_state_update["clinical_plan"]["patient_answer_source"], "deterministic_renderer")

    def test_planner_low_value_echo_answer_uses_deterministic_renderer(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **_: object) -> dict[str, object]:
                return {
                    "intent": "medical_complaint",
                    "case_action": "start_new",
                    "domain": "headache",
                    "clinical_summary": "headache needs clarification",
                    "risk_level": "low",
                    "response_goal": "clarify",
                    "patient_answer": "عندي صداع بيزيد",
                }

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        response = engine.handle_chat(ChatRequest(message="عندي صداع بيزيد"))

        self.assertEqual(response.case_state_update["active_case"]["active_domain"], "headache")
        self.assertNotEqual(response.answer.strip(), "عندي صداع بيزيد")
        self.assertEqual(response.case_state_update["clinical_plan"]["patient_answer_source"], "deterministic_renderer")

    def test_side_pain_correction_echo_uses_correction_renderer(self) -> None:
        correction = "\u0644\u0627 \u0645\u0639\u0644\u0634 \u0642\u0635\u062f\u064a \u0627\u0644\u0634\u0645\u0627\u0644"

        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                plan = dict(kwargs["deterministic_plan"])  # type: ignore[index]
                if kwargs.get("message") == correction:
                    plan["patient_answer"] = "\u0642\u062f \u064a\u0643\u0648\u0646 \u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u062c\u0646\u0628\u0643 \u0628\u0633\u0628\u0628 \u0639\u062f\u0648\u0649 \u0623\u0648 \u0625\u0635\u0627\u0628\u0629 \u0641\u064a \u0627\u0644\u0639\u0636\u0644\u0627\u062a."
                return plan

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        history: list[ChatMessage] = []
        first_message = "\u0639\u0646\u062f\u064a \u0648\u062c\u0639 \u0641\u064a \u062c\u0646\u0628\u064a \u0627\u0644\u064a\u0645\u064a\u0646"
        first = engine.handle_chat(ChatRequest(message=first_message, history=history))
        history.extend([ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)])

        response = engine.handle_chat(ChatRequest(message=correction, history=history))

        self.assertNotEqual(response.answer.strip(), correction)
        self.assertIn("\u0627\u0644\u0634\u0645\u0627\u0644", response.answer)
        self.assertEqual(response.case_state_update["active_case"]["known_facts"].get("side"), "left")
        self.assertEqual(response.case_state_update["clinical_plan"]["patient_answer_source"], "deterministic_renderer")

    def test_respiratory_night_worsening_is_acknowledged(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        history: list[ChatMessage] = []
        first_message = "\u0628\u0642\u0627\u0644\u064a \u064a\u0648\u0645\u064a\u0646 \u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0646\u0627\u0634\u0641\u0629"
        first = engine.handle_chat(ChatRequest(message=first_message, history=history))
        history.extend([ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)])
        denial = "\u0645\u0641\u064a\u0634 \u0633\u062e\u0648\u0646\u064a\u0629 \u0648\u0644\u0627 \u0636\u064a\u0642 \u0646\u0641\u0633"
        second = engine.handle_chat(ChatRequest(message=denial, history=history))
        history.extend([ChatMessage(role="user", content=denial), ChatMessage(role="assistant", content=second.answer)])

        response = engine.handle_chat(ChatRequest(message="\u0627\u0644\u0643\u062d\u0629 \u0628\u062a\u0632\u064a\u062f \u0628\u0627\u0644\u0644\u064a\u0644", history=history))

        self.assertTrue(response.case_state_update["active_case"]["known_facts"].get("night_worse"))
        self.assertIn("\u0627\u0644\u0644\u064a\u0644", response.answer)
        self.assertNotIn("\u0645\u0644\u0627\u0631\u064a\u0627", response.answer)

    def test_digestive_post_meal_worsening_continues_case(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        history: list[ChatMessage] = []
        messages = [
            "\u0628\u0637\u0646\u064a \u0628\u062a\u0648\u062c\u0639\u0646\u064a \u0648\u062d\u0627\u0633\u0629 \u0625\u0646\u064a \u0639\u0627\u064a\u0632\u0629 \u0623\u0631\u062c\u0639",
            "\u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0646\u0635 \u0627\u0644\u0628\u0637\u0646 \u0648\u0628\u062f\u0623 \u0645\u0646 \u0627\u0645\u0628\u0627\u0631\u062d",
            "\u0644\u0627 \u0645\u0641\u064a\u0634 \u0625\u0633\u0647\u0627\u0644 \u0628\u0633 \u0631\u062c\u0639\u062a \u0645\u0631\u0629",
        ]
        last = None
        for message in messages:
            last = engine.handle_chat(ChatRequest(message=message, history=history))
            history.extend([ChatMessage(role="user", content=message), ChatMessage(role="assistant", content=last.answer)])
        self.assertIsNotNone(last)

        response = engine.handle_chat(ChatRequest(message="\u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0632\u064a\u062f \u0628\u0639\u062f \u0627\u0644\u0623\u0643\u0644", history=history))

        self.assertEqual(response.case_state_update["active_case"]["active_domain"], "digestive")
        self.assertTrue(response.case_state_update["active_case"]["known_facts"].get("post_meal_worse"))
        self.assertNotIn("\u0645\u0634 \u0642\u0627\u062f\u0631 \u0623\u0641\u0647\u0645", response.answer)
        self.assertIn("\u0627\u0644\u0623\u0643\u0644", response.answer)
        self.assertLessEqual(response.answer.count("\u061f") + response.answer.count("?"), 1)

    def test_unseen_dizziness_and_breathlessness_paraphrases_are_medical(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        dizziness = engine.handle_chat(ChatRequest(message="\u062d\u0627\u0633\u0633 \u0627\u0644\u062f\u0646\u064a\u0627 \u0628\u062a\u062a\u0647\u0632 \u0648\u0645\u0634 \u062b\u0627\u0628\u062a"))
        self.assertIn("dizziness", dizziness.extracted_symptoms)
        self.assertIn("loss_of_balance", dizziness.extracted_symptoms)
        self.assertEqual(dizziness.case_state_update["active_case"]["active_domain"], "vestibular_ent")
        self.assertNotEqual(dizziness.mode, "closing")
        self.assertNotIn("\u0645\u0634 \u0642\u0627\u062f\u0631 \u0623\u0641\u0647\u0645", dizziness.answer)

        balance = engine.handle_chat(ChatRequest(message="\u062d\u0627\u0633\u0633 \u0625\u0646\u064a \u0628\u0641\u0642\u062f \u062a\u0648\u0627\u0632\u0646\u064a"))
        self.assertIn("dizziness", balance.extracted_symptoms)
        self.assertIn("loss_of_balance", balance.extracted_symptoms)
        self.assertEqual(balance.case_state_update["active_case"]["active_domain"], "vestibular_ent")
        self.assertNotIn("\u0645\u0634 \u0642\u0627\u062f\u0631 \u0623\u0641\u0647\u0645", balance.answer)

        breath = engine.handle_chat(ChatRequest(message="\u0646\u0641\u0633\u064a \u0628\u064a\u0642\u0637\u0639 \u0644\u0645\u0627 \u0627\u062a\u062d\u0631\u0643"))
        self.assertIn("breathlessness", breath.extracted_symptoms)
        self.assertEqual(breath.case_state_update["active_case"]["active_domain"], "respiratory")
        self.assertTrue(breath.case_state_update["active_case"]["known_facts"].get("exertional_breathlessness"))
        self.assertNotIn("\u0645\u0634 \u0642\u0627\u062f\u0631 \u0623\u0641\u0647\u0645", breath.answer)

        exertional = engine.handle_chat(ChatRequest(message="\u0628\u062a\u062e\u0646\u0642 \u0645\u0646 \u0623\u0642\u0644 \u0645\u062c\u0647\u0648\u062f"))
        self.assertIn("breathlessness", exertional.extracted_symptoms)
        self.assertEqual(exertional.case_state_update["active_case"]["active_domain"], "respiratory")
        self.assertTrue(exertional.case_state_update["active_case"]["known_facts"].get("exertional_breathlessness"))
        self.assertNotIn("\u0645\u0634 \u0642\u0627\u062f\u0631 \u0623\u0641\u0647\u0645", exertional.answer)

    def test_colloquial_nausea_with_food_context_stays_digestive(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        response = engine.handle_chat(ChatRequest(message="\u062d\u0627\u0633\u0633 \u0628\u0645\u063a\u0635\u0627\u0646 \u0648\u0646\u0641\u0633\u064a \u0645\u0642\u0644\u0648\u0628\u0629 \u0648\u0645\u0634 \u0637\u0627\u064a\u0642 \u0627\u0644\u0623\u0643\u0644"))

        self.assertIn("nausea", response.extracted_symptoms)
        self.assertEqual(response.case_state_update["active_case"]["active_domain"], "digestive")
        self.assertTrue(response.case_state_update["active_case"]["known_facts"].get("nausea_present"))
        self.assertNotIn("\u0625\u064a\u0630\u0627\u0621 \u0646\u0641\u0633\u0643", response.answer)
        self.assertNotEqual(response.suggested_doctor, "Psychiatrist")

    def test_short_closing_phrase_with_goodbye_closes(self) -> None:
        response = self.post_chat("\u0634\u0643\u0631\u0627 \u0633\u0644\u0627\u0645")
        self.assertEqual(response["mode"], "closing")
        self.assertEqual(response["case_state_update"]["engine_route"], "v3_closing")
        self.assertFalse(response["case_state_update"]["engine_trace"]["judge_called"])

    def test_progressive_chest_pressure_and_breathlessness_escalates(self) -> None:
        history: list[dict] = []
        first = self.post_chat(
            "\u0641\u064a \u0636\u063a\u0637 \u0641\u064a \u0635\u062f\u0631\u064a",
            history,
        )
        self.assertEqual(first["case_state_update"]["active_case"]["active_domain"], "chest_pain")
        history.extend(
            [
                {"role": "user", "content": "\u0641\u064a \u0636\u063a\u0637 \u0641\u064a \u0635\u062f\u0631\u064a"},
                {"role": "assistant", "content": first["answer"]},
            ]
        )

        second = self.post_chat(
            "\u0632\u0627\u062f \u0648\u0645\u0639\u0627\u0647 \u0636\u064a\u0642 \u062a\u0646\u0641\u0633 \u0648\u0639\u0631\u0642",
            history,
        )
        self.assertEqual(second["mode"], "emergency")
        self.assertEqual(second["urgency_level"], "High")
        self.assertIn("Emergency", second["suggested_doctor"])
        self.assertTrue(second["case_state_update"]["engine_trace"]["deterministic_override"])

    def test_english_neuro_red_flags_accumulate_to_emergency(self) -> None:
        history: list[dict] = []
        first = self.post_chat("My left side feels numb", history)
        history.extend(
            [
                {"role": "user", "content": "My left side feels numb"},
                {"role": "assistant", "content": first["answer"]},
            ]
        )

        second = self.post_chat("my speech is getting weird", history)
        history.extend(
            [
                {"role": "user", "content": "my speech is getting weird"},
                {"role": "assistant", "content": second["answer"]},
            ]
        )

        third = self.post_chat("I feel dizzy too", history)
        self.assertEqual(third["mode"], "emergency")
        self.assertEqual(third["urgency_level"], "High")
        self.assertIn("Emergency", third["suggested_doctor"])

    def test_arabic_balance_followup_preserves_neuro_emergency_case(self) -> None:
        history: list[dict] = []
        first_message = "\u0639\u0646\u062f\u064a \u062a\u0646\u0645\u064a\u0644 \u0641\u064a \u0646\u0627\u062d\u064a\u0629 \u0645\u0646 \u062c\u0633\u0645\u064a \u0648\u0643\u0644\u0627\u0645\u064a \u0628\u0642\u0649 \u0645\u062a\u0644\u062e\u0628\u0637"
        first = self.post_chat(first_message, history)
        history.extend(
            [
                {"role": "user", "content": first_message},
                {"role": "assistant", "content": first["answer"]},
            ]
        )

        second = self.post_chat("\u0648\u062d\u0627\u0633\u0633 \u0625\u0646\u064a \u0628\u0641\u0642\u062f \u062a\u0648\u0627\u0632\u0646\u064a", history)

        self.assertEqual(second["mode"], "emergency")
        self.assertEqual(second["urgency_level"], "High")
        self.assertIn("Emergency", second["suggested_doctor"])
        self.assertNotEqual(second["case_state_update"]["active_case"]["active_domain"], "vestibular_ent")

    def test_negated_self_harm_phrase_stays_non_emergency(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        response = engine.handle_chat(
            ChatRequest(
                message="\u0642\u0644\u0642\u0627\u0646 \u062c\u062f\u0627 \u0648\u0645\u0634 \u0628\u0646\u0627\u0645 \u0648\u0645\u0634 \u0647\u0627\u0630\u064a \u0646\u0641\u0633\u064a"
            )
        )
        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")

    def test_meta_smalltalk_during_active_case_skips_medical_tools(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                raise AssertionError("casual pause should not call the medical judge")

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        response = engine.handle_chat(
            ChatRequest(
                message="\u0639\u0644\u0649 \u0641\u0643\u0631\u0629 \u0627\u0633\u0645\u0643 \u0627\u064a\u0647",
                history=[
                    ChatMessage(role="user", content="\u0639\u0646\u062f\u064a \u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0632\u0648\u0631 \u0648\u0643\u062d\u0629 \u062e\u0641\u064a\u0641\u0629"),
                    ChatMessage(role="assistant", content="\u0627\u0644\u0632\u0648\u0631 \u0648\u0627\u0644\u0643\u062d\u0629 \u0645\u062d\u062a\u0627\u062c\u064a\u0646 \u0645\u062a\u0627\u0628\u0639\u0629."),
                ],
            )
        )
        trace = response.case_state_update["engine_trace"]
        self.assertEqual(response.case_state_update["engine_route"], "v3_pause_resume")
        self.assertFalse(trace["classifier_used"])
        self.assertFalse(trace["rag_used"])
        self.assertFalse(trace["judge_called"])

    def test_casual_identity_interruption_preserves_abdominal_case_without_repeating_question(self) -> None:
        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": True, "issues": [], "safe_rewrite": None}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )
        history: list[ChatMessage] = []

        first_message = "\u0639\u0646\u062f\u064a \u0648\u062c\u0639 \u0641\u064a \u0628\u0637\u0646\u064a \u0645\u0646 \u0627\u0645\u0628\u0627\u0631\u062d"
        first = engine.handle_chat(ChatRequest(message=first_message, history=history))
        self.assertEqual(first.case_state_update["active_case"]["active_domain"], "digestive")
        self.assertIn("abdominal_location", first.case_state_update["active_case"]["pending_question_ids"])
        history.extend([ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)])

        casual_message = "\u0647\u0648 \u0627\u0646\u062a \u0627\u0633\u0645\u0643 \u0627\u064a\u0647"
        casual = engine.handle_chat(ChatRequest(message=casual_message, history=history))
        self.assertEqual(casual.case_state_update["engine_route"], "v3_pause_resume")
        self.assertIn("MedBridge", casual.answer)
        self.assertNotIn("\u0642\u0648\u0644\u0651\u064a \u0625\u064a\u0647 \u0627\u0644\u062c\u062f\u064a\u062f", casual.answer)
        self.assertLessEqual(casual.answer.count("\u061f") + casual.answer.count("?"), 0)
        history.extend([ChatMessage(role="user", content=casual_message), ChatMessage(role="assistant", content=casual.answer)])

        resumed = engine.handle_chat(ChatRequest(message="\u0627\u0644\u0645\u0647\u0645 \u0627\u0644\u0648\u062c\u0639 \u0628\u064a\u0632\u064a\u062f \u0628\u0639\u062f \u0627\u0644\u0623\u0643\u0644", history=history))
        active_case = resumed.case_state_update["active_case"]
        self.assertEqual(active_case["active_domain"], "digestive")
        self.assertTrue(active_case["known_facts"].get("post_meal_worse"))
        self.assertIn("\u0627\u0644\u0623\u0643\u0644", resumed.answer)
        self.assertNotIn("\u0627\u0644\u0623\u0644\u0645 \u0641\u064a\u0646 \u0628\u0627\u0644\u0638\u0628\u0637", resumed.answer)
        self.assertLessEqual(resumed.answer.count("\u061f") + resumed.answer.count("?"), 1)

    def test_urinary_denial_scope_keeps_only_active_positive_facts(self) -> None:
        engine = self.deterministic_engine()
        first_message = "عندي حرقان وأنا بتبول وبخش الحمام كتير"
        first = engine.handle_chat(ChatRequest(message=first_message))
        history = [ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)]

        response = engine.handle_chat(
            ChatRequest(
                message="مفيش حرارة ولا دم في البول ولا ألم في الجنب أو الظهر، بس الحرقان مستمر وبخش الحمام كتير",
                history=history,
            )
        )
        active_case = response.case_state_update["active_case"]

        self.assertTrue(active_case["known_facts"].get("urinary_burning"))
        self.assertTrue(active_case["known_facts"].get("urinary_frequency"))
        self.assertIn("fever", active_case["denied_facts"])
        self.assertIn("blood_in_urine", active_case["denied_facts"])
        self.assertIn("flank_pain", active_case["denied_facts"])
        self.assertIn("back_pain", active_case["denied_facts"])
        self.assertNotIn("blood_in_urine", active_case["known_facts"])
        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")
        self.assertNotIn("إسعاف", response.answer)
        self.assertNotIn("طوارئ فورًا", response.answer)

    def test_blood_in_urine_correction_clears_stale_urgency_arabic(self) -> None:
        engine = self.deterministic_engine()
        first_message = "في دم في البول"
        first = engine.handle_chat(ChatRequest(message=first_message))
        self.assertIn("blood_in_urine", first.case_state_update["active_case"]["known_facts"])
        history = [ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)]

        response = engine.handle_chat(ChatRequest(message="لا، أنا غلطت، مفيش دم في البول", history=history))
        active_case = response.case_state_update["active_case"]

        self.assertIn("blood_in_urine", active_case["denied_facts"])
        self.assertNotIn("blood_in_urine", active_case["known_facts"])
        self.assertNotIn("spotting_ urination", response.extracted_symptoms)
        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")
        self.assertNotIn("طوارئ فورًا", response.answer)

    def test_repeated_blood_correction_forms_do_not_revive_positive_fact(self) -> None:
        engine = self.deterministic_engine()
        history: list[ChatMessage] = []
        first = engine.handle_chat(ChatRequest(message="في دم في البول", history=history))
        history.extend([ChatMessage(role="user", content="في دم في البول"), ChatMessage(role="assistant", content=first.answer)])

        for correction in ("مفيش دم في البول", "لا يوجد دم في البول"):
            response = engine.handle_chat(ChatRequest(message=correction, history=history))
            active_case = response.case_state_update["active_case"]
            self.assertIn("blood_in_urine", active_case["denied_facts"])
            self.assertNotIn("blood_in_urine", active_case["known_facts"])
            self.assertNotEqual(response.mode, "emergency")
            history.extend([ChatMessage(role="user", content=correction), ChatMessage(role="assistant", content=response.answer)])

    def test_genuine_urinary_red_flags_remain_urgent(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(ChatRequest(message="في دم واضح في البول ومعاه حرارة وألم شديد في الجنب"))
        active_case = response.case_state_update["active_case"]

        self.assertIn("blood_in_urine", active_case["known_facts"])
        self.assertTrue(active_case["known_facts"].get("fever"))
        self.assertTrue(active_case["known_facts"].get("flank_pain"))
        self.assertNotIn("blood_in_urine", active_case["denied_facts"])
        self.assertIn(response.urgency_level, {"Medium", "High"})
        self.assertNotEqual(response.urgency_level, "Low")

    def test_negated_vomiting_does_not_cancel_neurological_red_flags(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(ChatRequest(message="مفيش ترجيع، بس عندي تنميل في نص جسمي وزغللة"))
        active_case = response.case_state_update["active_case"]

        self.assertIn("vomiting_present", active_case["denied_facts"])
        self.assertNotIn("vomiting_present", active_case["known_facts"])
        self.assertTrue(active_case["known_facts"].get("numbness"))
        self.assertTrue(active_case["known_facts"].get("vision_change"))
        self.assertEqual(response.mode, "emergency")
        self.assertEqual(response.urgency_level, "High")

    def test_positional_bilateral_tingling_does_not_invent_weakness_or_emergency(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(
            ChatRequest(message="إيديا الاتنين بينملوا لما بنام عليهم، ولما أتحرك التنميل بيروح، ومفيش ضعف")
        )
        active_case = response.case_state_update["active_case"]

        self.assertIn("weakness", active_case["denied_facts"])
        self.assertNotIn("weakness", active_case["known_facts"])
        self.assertNotIn("weakness", response.extracted_symptoms)
        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")

    def test_general_blood_in_urine_question_is_not_stored_as_confirmed_symptom(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(ChatRequest(message="هل وجود دم في البول خطير؟"))
        active_case = response.case_state_update["active_case"]

        self.assertEqual(active_case["active_domain"], "urinary")
        self.assertNotIn("blood_in_urine", active_case["known_facts"])
        self.assertNotIn("spotting_ urination", response.extracted_symptoms)
        self.assertNotEqual(response.mode, "emergency")

    def test_historical_resolved_vomiting_is_not_active(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(ChatRequest(message="كان عندي ترجيع امبارح لكنه وقف النهارده"))
        active_case = response.case_state_update["active_case"]

        self.assertEqual(active_case["active_domain"], "digestive")
        self.assertIn("vomiting_present", active_case["denied_facts"])
        self.assertNotIn("vomiting_present", active_case["known_facts"])
        self.assertNotIn("vomiting", response.extracted_symptoms)
        self.assertNotEqual(response.mode, "emergency")

    def test_migraine_like_headache_gets_guidance_without_reasking_denied_red_flags(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(
            ChatRequest(
                message="من امبارح عندي صداع نابض في ناحية واحدة من راسي، والنور والصوت بيضايقوني وحاسس بغثيان، بس مفيش ضعف ولا تنميل ولا لخبطة في الكلام"
            )
        )
        active_case = response.case_state_update["active_case"]

        self.assertEqual(active_case["active_domain"], "headache")
        self.assertEqual(response.possible_diagnosis, "Migraine-like headache")
        self.assertTrue(active_case["known_facts"].get("unilateral_headache"))
        self.assertTrue(active_case["known_facts"].get("pulsating_headache"))
        self.assertTrue(active_case["known_facts"].get("photophobia"))
        self.assertTrue(active_case["known_facts"].get("phonophobia"))
        self.assertTrue(active_case["known_facts"].get("nausea_present"))
        self.assertIn("weakness", active_case["denied_facts"])
        self.assertIn("numbness", active_case["denied_facts"])
        self.assertIn("speech_change", active_case["denied_facts"])
        self.assertFalse(response.follow_up_questions)
        self.assertNotIn("تنميل/ضعف", response.answer)
        self.assertIn("صداع نصفي", response.answer)
        self.assertIn("نور", response.answer)
        self.assertNotEqual(response.mode, "emergency")

    def test_respiratory_denials_do_not_revive_chest_or_breathlessness_questions(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(
            ChatRequest(message="بقالي 3 أيام عندي كحة ورشح والتهاب بسيط في الحلق، بس مفيش ضيق نفس ولا ألم في الصدر ولا حرارة عالية")
        )
        active_case = response.case_state_update["active_case"]

        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")
        self.assertIn("breathlessness", active_case["denied_facts"])
        self.assertIn("chest_pain", active_case["denied_facts"])
        self.assertNotIn("breathlessness", active_case["known_facts"])
        self.assertNotIn("chest_pain", active_case["known_facts"])
        self.assertNotIn("Cardiologist", response.suggested_doctor)
        self.assertNotIn("Emergency", response.suggested_doctor)
        self.assertFalse(any("ضيق" in question or "صدر" in question for question in response.follow_up_questions))
        self.assertNotIn("هل مع ألم الصدر", response.answer)
        self.assertIn("كحة", response.answer)
        self.assertIn("رشح", response.answer)

    def test_real_chest_pain_with_breathlessness_still_triggers_emergency(self) -> None:
        engine = self.deterministic_engine()
        response = engine.handle_chat(ChatRequest(message="عندي ألم صدر شديد وضيق تنفس"))

        self.assertEqual(response.mode, "emergency")
        self.assertEqual(response.urgency_level, "High")
        self.assertIn("Emergency", response.suggested_doctor)

    def test_reviewer_rejection_uses_revised_answer_not_original_draft(self) -> None:
        revised = "A headache can have several causes. Since you denied weakness and numbness, focus on rest and monitoring; seek urgent care if neurological symptoms appear."

        class FakeLLM:
            client = object()

            def plan_clinical_turn(self, **kwargs: object) -> dict[str, object]:
                return dict(kwargs["deterministic_plan"])  # type: ignore[index]

            def review_v3_answer(self, **_: object) -> dict[str, object]:
                return {"approved": False, "issues": ["generic"], "revised_answer": revised}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=FakeLLM(),
        )

        response = engine.handle_chat(ChatRequest(message="I have a headache but no numbness or weakness"))

        self.assertEqual(response.answer, revised)
        self.assertNotIn("Did the headache start", response.answer)

    def test_english_blood_in_urine_correction_clears_stale_fact(self) -> None:
        engine = self.deterministic_engine()
        first_message = "I have blood in my urine"
        first = engine.handle_chat(ChatRequest(message=first_message))
        self.assertIn("blood_in_urine", first.case_state_update["active_case"]["known_facts"])
        history = [ChatMessage(role="user", content=first_message), ChatMessage(role="assistant", content=first.answer)]

        response = engine.handle_chat(
            ChatRequest(message="Sorry, I was mistaken. There is no blood in my urine", history=history)
        )
        active_case = response.case_state_update["active_case"]

        self.assertIn("blood_in_urine", active_case["denied_facts"])
        self.assertNotIn("blood_in_urine", active_case["known_facts"])
        self.assertNotEqual(response.mode, "emergency")
        self.assertNotEqual(response.urgency_level, "High")


if __name__ == "__main__":
    unittest.main()
