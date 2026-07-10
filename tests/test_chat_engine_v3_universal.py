from __future__ import annotations

import unittest

from app import main as main_module
from app.chat_engine_v3 import ChatEngineV3
from app.schemas import ChatRequest


class FakeUniversalLLM:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.review_calls = 0

    def plan_clinical_turn(self, **_: object) -> None:
        self.plan_calls += 1
        return None

    def review_v3_answer(self, **_: object) -> dict[str, object]:
        self.review_calls += 1
        return {"approved": True, "issues": [], "safe_rewrite": None}


class ChatEngineV3UniversalMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_retrieve = main_module.rag_service.retrieve
        main_module.rag_service.retrieve = lambda *_args, **_kwargs: []
        cls.fake_llm = FakeUniversalLLM()
        cls.engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=cls.fake_llm,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        main_module.rag_service.retrieve = cls.original_retrieve

    def ask(self, message: str, history: list[dict[str, str]] | None = None) -> dict:
        response = self.engine.handle_chat(ChatRequest(message=message, history=history or []))
        return response.model_dump()

    def assert_universal_contract(self, data: dict, *, emergency: bool = False) -> None:
        update = data["case_state_update"]
        trace = update["engine_trace"]
        self.assertEqual(trace["engine"], "v3_universal")
        self.assertNotEqual(update.get("engine_route"), "v3_unowned_domain")
        if emergency:
            self.assertEqual(data["mode"], "emergency")
            self.assertEqual(data["urgency_level"], "High")
            self.assertFalse(trace["planner_called"])
            self.assertTrue(trace["deterministic_override"])
        else:
            self.assertTrue(trace["planner_called"])
        self.assertLessEqual(data["answer"].count("?") + data["answer"].count("؟"), 2)
        self.assertNotIn("Provide a more detailed response", data["answer"])
        self.assertNotIn("classifier", data["answer"].lower())
        self.assertNotIn("RAG", data["answer"])

    def test_universal_matrix_one_hundred_unseen_turns(self) -> None:
        transcripts = [
            ["عندي صداع", "معاه زغللة", "بدأ فجأة", "شكرا"],
            ["ظهري واجعني", "اسفل الظهر", "بقاله ٣ ايام", "مفيش تنميل"],
            ["عندي ألم صدر", "مفيش ضيق نفس", "بقاله ساعة", "عندي عرق بارد"],
            ["عندي إفرازات وحكة", "لونها ابيض", "مفيش حمل", "في حرقان بسيط"],
            ["حرقان بول", "بتبول كتير", "مفيش دم", "في وجع جنب"],
            ["بطني بتوجعني", "معايا اسهال", "مفيش دم", "مش قادر اشرب مياه"],
            ["عندي كحة وسخونية", "بقالها يومين", "مفيش ضيق نفس", "جسمي مكسر"],
            ["طفح جلدي وحكة", "في ايدي", "مش بينتشر", "في تورم في وشي"],
            ["عيني بتوجعني", "في احمرار", "النظر اتغير شوية", "بعد خبطة"],
            ["ضرسي واجعني", "في تورم في اللثة", "مفيش حرارة", "مش عارف افتح بقي كويس"],
            ["حاسس بقلق جامد", "مفيش افكار اذي نفسي", "مش بنام كويس", "عايز اتكلم"],
            ["تعبان ومش قادر احدد", "جسمي واجعني", "مفيش حرارة", "بقالي اسبوع"],
            ["I have a cough", "two days", "no chest pain", "I also have body aches"],
            ["My throat hurts", "no fever", "with cough", "thanks"],
            ["hello", "what can you do?", "bro what", "fuck you"],
            ["My belly hurts and I feel sick", "middle of my belly", "no diarrhea", "it gets worse after food"],
            ["I have burning when I pee", "no blood", "new problem: my tooth hurts", "two days"],
            ["صداع ودوخه", "gradual مش فجأة", "no vision changes", "بقاله يومين"],
            ["dry cough for two days", "no fever and no breathlessness", "worse at night", "thank you"],
            ["rash with itching", "on my arms", "not spreading", "no lip swelling"],
            ["my eye is red and painful", "vision is a bit blurry", "after dust exposure", "no trauma"],
            ["I feel shaky and sweaty", "very hungry", "no fainting", "I have diabetes"],
            ["lower back pain", "no numbness", "after lifting something", "three days"],
            ["period pain is worse than usual", "not pregnant", "no heavy bleeding", "started yesterday"],
            ["I feel anxious and can't sleep", "no self harm thoughts", "started this week", "I want guidance"],
        ]

        total_turns = 0
        for transcript in transcripts:
            history: list[dict[str, str]] = []
            for message in transcript:
                data = self.ask(message, history)
                emergency = data["mode"] == "emergency"
                self.assert_universal_contract(data, emergency=emergency)
                if message not in {"شكرا", "thanks"}:
                    self.assertIn(data["mode"], {"clarification", "diagnosis", "emergency", "closing"})
                history.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": data["answer"]},
                    ]
                )
                total_turns += 1

        self.assertGreaterEqual(total_turns, 100)
        self.assertGreaterEqual(self.fake_llm.plan_calls, 75)

    def test_provider_failure_and_invalid_outputs_stay_safe(self) -> None:
        class BadLLM:
            def plan_clinical_turn(self, **_: object) -> object:
                return {"domain": "back_pain", "questions_to_ask": ["Provide a more detailed response?"]}

            def review_v3_answer(self, **_: object) -> object:
                return {"approved": False, "safe_rewrite": "Provide a more detailed response or ask a new question."}

        engine = ChatEngineV3(
            classifier_service=main_module.classifier_service,
            knowledge_service=main_module.knowledge_service,
            rag_service=main_module.rag_service,
            llm_service=BadLLM(),
        )
        data = engine.handle_chat(ChatRequest(message="عندي ألم في بطني")).model_dump()
        self.assertEqual(data["case_state_update"]["engine_trace"]["engine"], "v3_universal")
        self.assertNotIn("Provide a more detailed response", data["answer"])
        self.assertLessEqual(data["answer"].count("?") + data["answer"].count("؟"), 2)


if __name__ == "__main__":
    unittest.main()
