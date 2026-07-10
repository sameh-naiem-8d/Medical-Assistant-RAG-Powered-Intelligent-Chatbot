from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
from typing import Any

from .config import Settings
from .display_labels import DIAGNOSIS_DISPLAY_AR, DOCTOR_DISPLAY_AR, display_diagnosis_ar
from .rag_service import filter_rag_cases_for_prompt
from .schemas import ChatMessage

logger = logging.getLogger(__name__)

PROVIDER_CALL_TIMEOUT_SECONDS = 4.0


DIAGNOSIS_ARABIC_NAMES = {
    "Fungal infection": "عدوى فطرية",
    "Allergy": "حساسية",
    "GERD": "ارتجاع المريء",
    "Chronic cholestasis": "ركود صفراوي مزمن",
    "Drug Reaction": "تفاعل دوائي",
    "Peptic ulcer diseae": "قرحة هضمية",
    "AIDS": "نقص المناعة المكتسب",
    "Diabetes": "السكري",
    "Gastroenteritis": "التهاب المعدة والأمعاء",
    "Bronchial Asthma": "ربو شعبي",
    "Hypertension": "ارتفاع ضغط الدم",
    "Migraine": "صداع نصفي",
    "Cervical spondylosis": "خشونة فقرات الرقبة",
    "Paralysis (brain hemorrhage)": "شلل مرتبط بنزيف بالمخ",
    "Jaundice": "يرقان",
    "Malaria": "ملاريا",
    "Chicken pox": "جدري الماء",
    "Dengue": "حمى الضنك",
    "Typhoid": "حمى التيفويد",
    "hepatitis A": "التهاب الكبد أ",
    "Hepatitis B": "التهاب الكبد ب",
    "Hepatitis C": "التهاب الكبد ج",
    "Hepatitis D": "التهاب الكبد د",
    "Hepatitis E": "التهاب الكبد هـ",
    "Alcoholic hepatitis": "التهاب كبد كحولي",
    "Tuberculosis": "درن رئوي",
    "Common Cold": "نزلة برد",
    "Pneumonia": "التهاب رئوي",
    "Dimorphic hemmorhoids(piles)": "بواسير",
    "Heart attack": "اشتباه أزمة قلبية",
    "Varicose veins": "دوالي",
    "Hypothyroidism": "قصور الغدة الدرقية",
    "Hyperthyroidism": "فرط نشاط الغدة الدرقية",
    "Hypoglycemia": "انخفاض سكر الدم",
    "Osteoarthristis": "خشونة المفاصل",
    "Arthritis": "التهاب المفاصل",
    "(vertigo) Paroymsal  Positional Vertigo": "دوار الوضعة الانتيابي الحميد",
    "Acne": "حب الشباب",
    "Urinary tract infection": "التهاب المسالك البولية",
    "Psoriasis": "صدفية",
    "Impetigo": "قوباء",
    "General medical evaluation": "أعراض عامة تحتاج تقييم طبي",
    "Viral or flu-like illness": "عدوى فيروسية / دور برد أو إنفلونزا محتملة",
    "General symptoms needing clarification": "أعراض عامة تحتاج أسئلة توضيحية",
}


DIAGNOSIS_ARABIC_SUMMARIES = {
    "Common Cold": (
        "الكحة مع السخونية والتعب قد تتماشى مع نزلة برد أو عدوى تنفسية فيروسية بسيطة، "
        "خصوصا إذا لم يوجد صفير بالصدر أو ضيق تنفس واضح."
    ),
    "(vertigo) Paroymsal  Positional Vertigo": (
        "الصداع مع الدوخة والغثيان قد يتماشى مع دوار مرتبط بحركة الرأس أو التوازن، "
        "ويحتاج تقييما إذا كان شديدا أو متكررا أو مصحوبا بأعراض عصبية."
    ),
    "Fungal infection": (
        "الطفح الجلدي مع الحكة قد يتماشى مع عدوى فطرية أو التهاب جلدي، "
        "خصوصا إذا كانت المنطقة رطبة أو الحكة مستمرة."
    ),
    "Gastroenteritis": (
        "الإسهال مع الترجيع وألم البطن قد يتماشى مع التهاب المعدة والأمعاء، "
        "وأهم شيء هو الانتباه للجفاف ووجود دم أو حرارة عالية."
    ),
    "Heart attack": (
        "ألم الصدر الشديد مع ضيق التنفس قد يشير إلى مشكلة خطيرة في القلب، "
        "لذلك يحتاج تقييما طبيا عاجلا في الطوارئ."
    ),
    "General medical evaluation": (
        "التعب مع فقدان الشهية ونقص الوزن أعراض عامة لا تكفي وحدها لتحديد مرض معين، "
        "وتحتاج أسئلة إضافية وفحصا طبيا خصوصا إذا استمرت أو زادت."
    ),
    "Cervical spondylosis": (
        "وجع الرقبة مع الصداع والدوخة قد يتماشى مع مشكلة في فقرات الرقبة أو شد عضلي، "
        "ويحتاج تقييم طبي خصوصا إذا كان الألم مستمرا أو معه تنميل أو ضعف."
    ),
    "Peptic ulcer diseae": (
        "ألم البطن مع فقدان الشهية قد يشير إلى تهيج أو قرحة في المعدة أو مشكلة هضمية، "
        "لكن وجود حرارة يحتاج متابعة طبية لاستبعاد عدوى أو التهاب."
    ),
}


DOCTOR_ARABIC_NAMES = {
    "General Practitioner": "طبيب عام",
    "Pulmonologist": "طبيب صدر",
    "Cardiologist": "طبيب قلب",
    "Endocrinologist": "طبيب غدد صماء",
    "Neurologist": "طبيب أعصاب",
    "Gastroenterologist": "طبيب جهاز هضمي",
    "Urologist": "طبيب مسالك بولية",
    "Dermatologist": "طبيب جلدية",
    "Allergist": "طبيب حساسية",
    "Infectious disease specialist": "طبيب أمراض معدية / باطنة",
    "Emergency care": "الطوارئ",
    "ENT specialist": "طبيب أنف وأذن وحنجرة",
    "Pediatrician": "طبيب أطفال",
    "Gynecologist": "طبيب نساء وتوليد",
    "Psychiatrist": "طبيب نفسي",
    "Dentist": "طبيب أسنان",
    "Ophthalmologist": "طبيب عيون",
    "Orthopedic doctor": "طبيب عظام",
}

DIAGNOSIS_ARABIC_NAMES.update(DIAGNOSIS_DISPLAY_AR)
DOCTOR_ARABIC_NAMES.update(DOCTOR_DISPLAY_AR)


URGENCY_ARABIC_NAMES = {
    "Low": "منخفض",
    "Medium": "متوسط",
    "High": "عال",
}


SAFE_GENERIC_PRECAUTIONS = [
    "الراحة وتقليل المجهود مؤقتا",
    "شرب سوائل كافية على فترات صغيرة",
    "متابعة الأعراض وطلب رعاية طبية إذا زادت أو استمرت",
]


BLOCKED_ADVICE_TERMS = {
    "دواء",
    "أدوية",
    "مضاد",
    "مضاد حيوي",
    "حقن",
    "جرعة",
    "حبوب",
    "بخاخ",
    "فيتامين",
    "مكمل",
    "عملية",
    "جراحة",
    "تحليل",
    "أشعة",
    "antibiotic",
    "medicine",
    "drug",
    "dose",
    "tablet",
    "spray",
    "inhaler",
    "vitamin",
    "supplement",
    "surgery",
    "procedure",
    "otc",
    "pain reliver",
    "pain reliever",
    "non-vegetarian",
    "vegetarian",
    "oily food",
    "salt bath",
    "meditation",
    "حمام ملح",
    "تأمل",
}


HIGH_URGENCY_PREFIX = (
    "الأعراض دي ممكن تكون خطيرة وتحتاج طوارئ فورًا. "
    "من الأفضل تروح أقرب طوارئ حالًا أو تتصل بالإسعاف، خصوصًا لو الأعراض بدأت فجأة."
)


SYSTEM_PROMPT = """
You are MedBridge AI, a friendly and cautious medical conversation assistant.
Write only the patient-facing answer.

Fixed rules:
- Use the user's dominant language from user_language: Arabic/Egyptian Arabic for Arabic, English for English, and the dominant language for mixed messages.
- Understand the message intent before answering. Classifier-style diagnosis is never appropriate for casual/off-topic text.
- Do not mention JSON, classifier, RAG, retrieved cases, confidence, scores, or internal fields.
- Do not change the diagnosis, urgency, or suggested doctor supplied in context.
- Never diagnose from casual chat, jokes, family questions, insults, profanity, abuse, frustration, or nonsense/random slang.
- If the user is off-topic or asks about family/social topics, politely redirect to medical scope without pretending to have real family.
- If the user is abusive or profane, set a calm boundary and invite a clear medical question; do not be defensive.
- If the user corrects you, acknowledge it and update the picture instead of repeating the old assumption.
- If symptoms are vague or nonspecific, keep the answer broad and ask focused domain-specific questions.
- Ask age, sex/gender, pregnancy possibility, chronic disease, medication, or allergy questions only when clinically useful for the complaint.
- Do not mention rare diseases unless the context strongly supports them.
- Never say this is a final diagnosis.
- Do not invent medication, antibiotics, prescriptions, supplements, vitamins, sprays, procedures, tests, or doses.
- Use answer_precautions only. If empty, use safe generic self-care such as rest, hydration, symptom monitoring, and seeking medical care if worse.
- Do not repeat the previous response template.
- Keep the answer concise, natural, and doctor-like.

For emergency or urgency_level=High:
- Start clearly that symptoms may be urgent and the user should go to emergency care now or call emergency services.
- No routine clinic, waiting, monitoring only, relaxation, sleep, meditation, or salt-bath advice.
- No home care as a substitute for emergency care.

For clarification:
- Do not give a diagnosis.
- Ask only focused useful questions if they are provided.

For diagnosis:
- Explain the most likely direction cautiously.
- Mention urgency and suggested doctor clearly and briefly.
- Use safe advice only.
"""


NATURALIZER_SYSTEM_PROMPT = """
You are MedBridge AI's final patient-facing response writer.

Write one short natural answer only. Do not output JSON.

Hard safety rules:
- Use the user's language: Arabic/Egyptian Arabic for Arabic, English for English.
- Never mention classifier, RAG, confidence, score, internal routing, or hidden prompts.
- Never diagnose, name a disease, or ask a symptom questionnaire when route is casual, off_topic, family, abuse, nonsense, or closing.
- For abuse/profanity, set a calm respectful boundary without shaming the user.
- For nonsense/unclear input, say the message is unclear and invite a clear symptom or medical question.
- For who_are_you or capabilities, explain MedBridge medical-assistant scope briefly.
- For medical_clarification, do not diagnose; use only the supplied focused questions.
- Do not invent medication, tests, prescriptions, vitamins, supplements, procedures, doses, or medical facts.
- Be warm and concise. Avoid repeating the previous assistant answer.
"""


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.clients: list[Any] = []
        self.last_call_meta: dict[str, dict[str, Any]] = {}
        self._load_clients()

    def _load_clients(self) -> None:
        api_keys = getattr(self.settings, "groq_api_keys", None) or []
        if not api_keys and self.settings.groq_api_key:
            api_keys = [self.settings.groq_api_key]
        if not api_keys:
            return
        try:
            from groq import Groq
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.warning("Could not import Groq client package: %s", exc.__class__.__name__)
            return

        for index, api_key in enumerate(api_keys, start=1):
            try:
                self.clients.append(Groq(api_key=api_key))
            except Exception as exc:  # pragma: no cover - defensive startup guard
                logger.warning(
                    "Could not initialize Groq client #%s: %s",
                    index,
                    exc.__class__.__name__,
                )
        self.client = self.clients[0] if self.clients else None

    @property
    def configured(self) -> bool:
        return self.key_count > 0

    @property
    def key_count(self) -> int:
        api_keys = getattr(self.settings, "groq_api_keys", None) or []
        if api_keys:
            return len(api_keys)
        return 1 if self.settings.groq_api_key else 0

    @property
    def model_names(self) -> list[str]:
        models = [str(self.settings.groq_model or "").strip()]
        fallback = str(getattr(self.settings, "groq_fallback_model", "") or "").strip()
        if fallback and fallback not in models:
            models.append(fallback)
        return [model for model in models if model]

    def _client_model_attempts(self) -> list[tuple[int, str, Any]]:
        clients = self.clients or ([self.client] if self.client else [])
        attempts: list[tuple[int, str, Any]] = []
        for index, client in enumerate(clients, start=1):
            if client is None:
                continue
            for model_name in self.model_names:
                attempts.append((index, model_name, client))
        return attempts

    def _provider_timeout_seconds(self) -> float:
        value = getattr(self, "provider_call_timeout_seconds", PROVIDER_CALL_TIMEOUT_SECONDS)
        try:
            return max(0.1, float(value))
        except (TypeError, ValueError):
            return PROVIDER_CALL_TIMEOUT_SECONDS

    def _create_chat_completion(self, client: Any, *, operation: str, **kwargs: Any) -> Any:
        timeout = self._provider_timeout_seconds()
        kwargs.setdefault("timeout", timeout)
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def run_provider_call() -> None:
            try:
                result_queue.put(("ok", client.chat.completions.create(**kwargs)))
            except BaseException as exc:  # pragma: no cover - defensive provider boundary
                result_queue.put(("error", exc))

        thread = threading.Thread(
            target=run_provider_call,
            name=f"medbridge-llm-{operation}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"{operation} provider call timed out")

        status, payload = result_queue.get_nowait()
        if status == "error":
            raise payload
        return payload

    def _record_llm_success(self, operation: str, key_index: int, model_name: str) -> None:
        if not hasattr(self, "last_call_meta"):
            self.last_call_meta = {}
        primary = str(self.settings.groq_model or "").strip()
        self.last_call_meta[operation] = {
            "selected_model": model_name,
            "key_index": key_index,
            "used_fallback_model": bool(primary and model_name != primary),
            "status": "ok",
        }

    def _record_llm_fallback(self, operation: str) -> None:
        if not hasattr(self, "last_call_meta"):
            self.last_call_meta = {}
        self.last_call_meta[operation] = {
            "selected_model": None,
            "key_index": None,
            "used_fallback_model": False,
            "status": "fallback",
        }

    def generate_answer(
        self,
        message: str,
        history: list[ChatMessage],
        extracted_symptoms: list[str],
        diagnosis: str | None,
        confidence: float,
        urgency_level: str,
        suggested_doctor: str,
        precautions: list[str],
        diagnosis_description: str | None,
        follow_up_questions: list[str],
        retrieved_cases: list[dict[str, Any]],
    ) -> str:
        prompt_retrieved_cases = filter_rag_cases_for_prompt(
            retrieved_cases,
            urgency_level=urgency_level,
        )
        answer_precautions, precaution_sources = self._answer_precautions(
            precautions,
            urgency_level,
            prompt_retrieved_cases,
        )
        user_language = self._detect_language(message)
        display_diagnosis = (
            diagnosis or "condition needing medical assessment"
            if user_language == "English"
            else self._display_diagnosis(diagnosis)
        )
        suggested_doctor_display = (
            suggested_doctor
            if user_language == "English"
            else DOCTOR_ARABIC_NAMES.get(suggested_doctor, suggested_doctor)
        )
        context = {
            "mode": "emergency" if urgency_level == "High" else "diagnosis",
            "user_language": user_language,
            "extracted_symptoms": extracted_symptoms,
            "possible_diagnosis": diagnosis,
            "diagnosis_arabic_name": display_diagnosis,
            "display_diagnosis": display_diagnosis,
            "confidence": confidence,
            "urgency_level": urgency_level,
            "urgency_arabic": URGENCY_ARABIC_NAMES.get(urgency_level, urgency_level),
            "suggested_doctor": suggested_doctor,
            "suggested_doctor_arabic": suggested_doctor_display,
            "answer_precautions": answer_precautions,
            "precaution_sources": precaution_sources,
            "raw_precautions": precautions,
            "diagnosis_summary": DIAGNOSIS_ARABIC_SUMMARIES.get(diagnosis or "") or diagnosis_description,
            "follow_up_questions": follow_up_questions,
            "urgent_warning_signs": self._urgent_warning_signs(extracted_symptoms, urgency_level),
            "retrieved_cases": prompt_retrieved_cases,
        }

        if not self.client:
            return self._fallback_answer(context)

        messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
        for item in history[-6:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"User message:\n{message}\n\n"
                    f"Clinical context JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"Write in this language: {user_language}. "
                    f"Internal diagnosis: {diagnosis!r}. "
                    f"Patient-facing diagnosis: {display_diagnosis!r}. "
                    "Use the patient-facing diagnosis in the answer. "
                    f"Suggested doctor: {context['suggested_doctor_arabic']!r}."
                ),
            }
        )

        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="generation",
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=850,
                )
                self._record_llm_success("generation", index, model_name)
                answer = completion.choices[0].message.content or ""
                answer = answer.strip() or self._fallback_answer(context)
                return self._enforce_answer_contract(answer, context)
            except Exception as exc:
                logger.warning(
                    "Groq generation failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )

        logger.warning("Groq generation failed for all configured keys; using safe fallback answer.")
        self._record_llm_fallback("generation")
        return self._fallback_answer(context)

    def naturalize_response(
        self,
        *,
        route: str,
        message: str,
        language: str,
        history: list[ChatMessage],
        draft_answer: str,
        follow_up_questions: list[str] | None = None,
        suggested_doctor: str | None = None,
        medical_domain: str | None = None,
        diagnosis_allowed: bool = False,
    ) -> str:
        follow_up_questions = follow_up_questions or []
        user_language = "English" if language == "en" else "Arabic/Egyptian Arabic"
        context = {
            "route": route,
            "user_language": user_language,
            "medical_domain": medical_domain or "none",
            "suggested_doctor": suggested_doctor,
            "diagnosis_allowed": diagnosis_allowed,
            "follow_up_questions": follow_up_questions[:3],
            "draft_answer": draft_answer,
            "allowed_content": self._naturalizer_allowed_content(route, diagnosis_allowed),
        }
        fallback = self._naturalized_fallback(
            route=route,
            message=message,
            language=language,
            history=history,
            draft_answer=draft_answer,
            follow_up_questions=follow_up_questions,
            suggested_doctor=suggested_doctor,
        )
        if not self.client:
            return fallback

        messages = [{"role": "system", "content": NATURALIZER_SYSTEM_PROMPT.strip()}]
        for item in history[-6:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Current user message:\n{message}\n\n"
                    f"Safe route context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    "Rewrite the draft into a natural patient-facing response without changing the allowed meaning."
                ),
            }
        )

        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="naturalizer",
                    model=model_name,
                    messages=messages,
                    temperature=0.35,
                    max_tokens=260,
                )
                self._record_llm_success("naturalizer", index, model_name)
                answer = (completion.choices[0].message.content or "").strip()
                if answer and self._naturalized_answer_is_safe(answer, route, diagnosis_allowed):
                    return answer
            except Exception as exc:
                logger.warning(
                    "Groq naturalizer failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )
        self._record_llm_fallback("naturalizer")
        return fallback

    def review_final_answer(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        language: str,
        route: str,
        domain: str,
        doctor_route: str,
        follow_up_questions: list[str],
        draft_answer: str,
        forbidden_items: list[str],
        red_flags: list[str],
        case_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not getattr(self.settings, "enable_llm_judge", False) or not self.client:
            return None

        schema_hint = {
            "is_safe": True,
            "is_relevant": True,
            "is_consistent_with_user": True,
            "has_unrelated_body_part_or_disease": False,
            "repeats_answered_questions": False,
            "uses_wrong_emergency_level": False,
            "uses_static_or_robotic_style": False,
            "suggested_fix": "",
            "reason": "short internal reason",
        }
        context = {
            "message": message,
            "language": language,
            "route": route,
            "domain": domain,
            "doctor_route": doctor_route,
            "follow_up_questions": follow_up_questions[:3],
            "draft_answer": draft_answer,
            "forbidden_items": forbidden_items,
            "domain_red_flags": red_flags,
            "case_state": case_state,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a safety reviewer for a medical chatbot response. "
                    "Return only compact JSON. Do not write patient-facing prose except suggested_fix. "
                    "Do not downgrade true emergency routes. Do not introduce diagnoses or medication. "
                    "Flag unrelated body parts/diseases, repeated questions, wrong urgency, wrong language, or unsafe advice."
                ),
            }
        ]
        for item in history[-4:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Review context JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"Return JSON matching this schema:\n{json.dumps(schema_hint, ensure_ascii=False)}"
                ),
            }
        )

        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="legacy_judge",
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=220,
                )
                raw = (completion.choices[0].message.content or "").strip()
                parsed = self._extract_json_object(raw)
                if isinstance(parsed, dict):
                    self._record_llm_success("legacy_judge", index, model_name)
                    return parsed
            except Exception as exc:
                logger.warning(
                    "Groq final-answer judge failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )
        self._record_llm_fallback("legacy_judge")
        return None

    def plan_clinical_turn(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        language: str,
        active_case: dict[str, Any],
        deterministic_plan: dict[str, Any],
        classifier_evidence: dict[str, Any],
        previous_assistant: str,
    ) -> dict[str, Any] | None:
        if not self.client:
            return None

        schema_hint = {
            "intent": "medical_complaint|medical_question|followup_answer|correction|new_complaint|casual|greeting|off_topic|abuse|nonsense|closing|unclear",
            "case_action": "continue|start_new|pause|resume|close|clarify|none",
            "language": "ar|en|mixed",
            "domain": "normalized medical domain or non_medical",
            "body_parts": [],
            "new_facts": [],
            "new_denials": [],
            "fact_updates": {
                "asserted": [],
                "denied": [],
                "corrected": [],
                "uncertain": [],
                "resolved": [],
            },
            "answered_question_ids": [],
            "clinical_summary": "short accumulated case summary",
            "risk_level": "none|low|moderate|urgent|emergency",
            "risk_reasons": [],
            "candidate_conditions": [{"name": "condition", "reason": "why it fits or why it is uncertain"}],
            "selected_possibilities": [],
            "analysis_for_patient": "one short patient-friendly interpretation, not a final diagnosis",
            "patient_answer": "the single natural patient-facing answer, 2-5 sentences, one question maximum",
            "broad_possibilities": [],
            "next_question": {"id": "stable_semantic_question_id", "text": "one focused question only"},
            "questions_to_ask": [],
            "care_guidance": [],
            "doctor_route": None,
            "response_goal": "reply|clarify|guide|escalate|boundary|close",
            "must_not_repeat": [],
            "must_not_repeat_question_ids": [],
            "forbidden_topics": [],
        }
        context = {
            "current_message": message,
            "language": language,
            "active_case": active_case,
            "deterministic_safety_plan": deterministic_plan,
            "weak_classifier_evidence": classifier_evidence,
            "previous_assistant": previous_assistant,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MedBridge AI's universal structured clinical conversation engine. "
                    "Return only compact JSON, including exactly one patient_answer field for the visible reply. "
                    "Classify any user input, including medical complaints, follow-ups, corrections, "
                    "casual messages, abuse, nonsense, closing, Arabic, English, and mixed language. "
                    "Use accumulated facts, denied facts, answered questions, and previous assistant context. "
                    "First determine whether the newest message answers the last question, adds a fact, denies a symptom, "
                    "corrects previous information, continues the active case, starts a new case, pauses casually, or closes. "
                    "Distinguish asserted, denied, corrected, uncertain, and resolved symptoms in fact_updates. "
                    "Explicit user negation or correction always overrides older positives; never revive a denied symptom "
                    "as a current emergency trigger unless the user reasserts it clearly. "
                    "Do not treat assistant text as patient symptoms. Do not carry old-case facts into a new complaint. "
                    "Use classifier, retrieval, disease labels, and medical knowledge only as weak supporting evidence; "
                    "reject evidence that conflicts with the active body area, facts, denials, corrections, or domain. "
                    "Track semantic question IDs: if the patient answers location, nausea, duration, history, fever, blood, "
                    "or red flags, add the answered IDs and never ask them again. "
                    "Ask at most one primary next_question. Do not add a second question. "
                    "patient_answer must be 2-5 natural sentences normally, acknowledge the newest information, "
                    "briefly analyze what it changes, mention cautious broad possibilities only when supported, "
                    "and include only the approved next_question if one is needed. "
                    "When the available pattern is clinically informative, give useful cautious guidance instead of "
                    "answering only with another question. Do not ask about symptoms already denied or already answered. "
                    "For migraine-like headache patterns, acknowledge unilateral/pulsating pain, light or sound sensitivity, "
                    "nausea, duration, and denied neurological red flags when present. "
                    "For vague input, reason broadly and choose the single highest-value missing detail. "
                    "For abdominal/digestive complaints, do not repeat location or nausea after they are answered. "
                    "Do not force a diagnosis from weak input. Do not invent symptoms. "
                    "Do not expose internal fields, headings, JSON, bullet lists, or labels such as 'أسئلة توضيحية' or 'Clarifying Questions' in patient_answer. "
                    "Do not downgrade red flags from the deterministic safety plan."
                ),
            }
        ]
        for item in history[-8:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Clinical planning context JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"Return JSON matching this schema:\n{json.dumps(schema_hint, ensure_ascii=False)}"
                ),
            }
        )

        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="planner",
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=520,
                )
                raw = (completion.choices[0].message.content or "").strip()
                parsed = self._extract_json_object(raw)
                if isinstance(parsed, dict):
                    self._record_llm_success("planner", index, model_name)
                    return parsed
            except Exception as exc:
                logger.warning(
                    "Groq V3 clinical planner failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )
        self._record_llm_fallback("planner")
        return None

    def review_v3_answer(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        language: str,
        clinical_summary: str,
        risk_level: str,
        risk_reasons: list[str],
        known_facts: dict[str, Any],
        denied_facts: list[str],
        answered_questions: list[str],
        approved_questions: list[str],
        active_case: dict[str, Any] | None = None,
        deterministic_plan: dict[str, Any] | None = None,
        classifier_evidence: dict[str, Any] | None = None,
        draft_answer: str,
        previous_assistant: str,
        domain: str,
    ) -> dict[str, Any] | None:
        if not getattr(self.settings, "enable_llm_judge", False) or not self.client:
            return None

        schema_hint = {
            "approved": True,
            "issues": [],
            "revised_answer": None,
        }
        context = {
            "current_message": message,
            "language": language,
            "domain": domain,
            "active_case": active_case or {},
            "deterministic_plan": deterministic_plan or {},
            "classifier_or_rag_evidence": classifier_evidence or {},
            "clinical_summary": clinical_summary,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "known_facts": known_facts,
            "denied_facts": denied_facts,
            "answered_questions": answered_questions,
            "approved_questions": approved_questions[:1],
            "draft_answer": draft_answer,
            "previous_assistant": previous_assistant,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MedBridge AI's final clinical response reviewer. "
                    "Return only compact JSON. Check relevance, accumulated context, repetition, risk, language, "
                    "unsupported diagnoses, unrelated body parts, internal fallback text, and formulaic style. "
                    "Also check old-case contamination, ignored new facts or denials, wrong doctor route, "
                    "unjustified emergency escalation, missed accumulated emergency, and wrong user language. "
                    "Treat denied, corrected, hypothetical, general-question-only, and resolved symptoms as not active. "
                    "Reject any draft that revives them as confirmed facts, diagnosis reasons, or emergency triggers. "
                    "Reject drafts that ignore important provided symptoms, ask about already denied or answered symptoms, "
                    "give only a generic question when the pattern supports useful cautious guidance, or behave like keyword extraction. "
                    "There must be one patient-visible response in draft_answer only. "
                    "Reject headings like 'أسئلة توضيحية' or 'Clarifying Questions'. "
                    "Reject repeated answered questions and any extra unapproved question. "
                    "If rejected, return revised_answer as the complete final patient-facing answer. "
                    "When rewriting, keep the same clinical meaning, mention useful analysis naturally, "
                    "and use only one approved question if one is provided. "
                    "Never downgrade deterministic emergencies or introduce medication/procedures."
                ),
            }
        ]
        for item in history[-6:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Review context JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"Return JSON matching this schema:\n{json.dumps(schema_hint, ensure_ascii=False)}"
                ),
            }
        )

        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="judge",
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=520,
                )
                raw = (completion.choices[0].message.content or "").strip()
                parsed = self._extract_json_object(raw)
                if isinstance(parsed, dict):
                    self._record_llm_success("judge", index, model_name)
                    return parsed
            except Exception as exc:
                logger.warning(
                    "Groq V3 final judge failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )
        self._record_llm_fallback("judge")
        return None

    def route_intent(
        self,
        *,
        message: str,
        language: str,
        history: list[ChatMessage],
    ) -> dict[str, Any] | None:
        if not self.client:
            return None
        schema_hint = {
            "intent": "emergency|medical_complaint|medical_question|followup_answer|correction|challenge|casual|off_topic|family|abuse|nonsense|closing",
            "medical_domain": "none|general|fever|body_ache|respiratory|throat_ent|digestive|urinary|reproductive_gynecology|pregnancy|skin|eye|dental|neurology|heart_chest|mental_health|allergy|injury|medication|poisoning|child_elderly|chronic",
            "language": "ar|en|mixed",
            "confidence": "low|medium|high",
            "needs_medical_path": False,
            "reason": "short internal reason",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user's intent for a medical chatbot. Do not diagnose. "
                    "Return only compact JSON matching the requested keys. Emergency safety is handled elsewhere."
                ),
            }
        ]
        for item in history[-6:]:
            if item.content:
                role = item.role if item.role in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Message: {message}\nLanguage hint: {language}\n"
                    f"JSON schema example: {json.dumps(schema_hint)}"
                ),
            }
        )
        for index, model_name, client in self._client_model_attempts():
            try:
                completion = self._create_chat_completion(
                    client,
                    operation="intent_router",
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=180,
                )
                raw = (completion.choices[0].message.content or "").strip()
                parsed = self._extract_json_object(raw)
                if isinstance(parsed, dict):
                    self._record_llm_success("intent_router", index, model_name)
                    return parsed
            except Exception as exc:
                logger.warning(
                    "Groq intent router failed for configured key #%s model %s: %s",
                    index,
                    model_name,
                    exc.__class__.__name__,
                )
        self._record_llm_fallback("intent_router")
        return None

    def _naturalizer_allowed_content(self, route: str, diagnosis_allowed: bool) -> str:
        if route in {"casual", "off_topic", "family"}:
            return "friendly medical-scope redirect only; no disease names and no symptom questionnaire"
        if route == "abuse":
            return "calm respectful boundary only; no diagnosis and no questionnaire"
        if route == "nonsense":
            return "say the message is unclear and invite a clear health symptom/question"
        if route in {"who_are_you", "capabilities"}:
            return "briefly explain MedBridge medical assistant scope"
        if route == "closing":
            return "polite closing plus emergency reminder only"
        if route == "medical_clarification" and not diagnosis_allowed:
            return "clarify the complaint using only the provided focused questions; no diagnosis"
        return "rewrite the safe draft without changing its clinical meaning"

    def _naturalized_fallback(
        self,
        *,
        route: str,
        message: str,
        language: str,
        history: list[ChatMessage],
        draft_answer: str,
        follow_up_questions: list[str],
        suggested_doctor: str | None,
    ) -> str:
        english = language == "en"
        variants = self._fallback_variants_en(route, suggested_doctor) if english else self._fallback_variants_ar(route, suggested_doctor)
        if not variants:
            return draft_answer
        previous = [item.content.strip() for item in history[-4:] if item.role == "assistant" and item.content]
        stable_seed = int(hashlib.sha256(f"{route}|{message}".encode("utf-8")).hexdigest(), 16)
        stable_seed += sum((index + 1) * ord(char) for index, char in enumerate(message or ""))
        start_index = stable_seed % len(variants)
        for offset in range(len(variants)):
            candidate = variants[(start_index + offset) % len(variants)]
            if candidate.strip() not in previous:
                return self._append_questions_if_needed(candidate, follow_up_questions)
        return self._append_questions_if_needed(variants[start_index], follow_up_questions)

    def _fallback_variants_ar(self, route: str, suggested_doctor: str | None) -> list[str]:
        if route == "casual":
            return [
                "أهلًا بيك. أنا موجود كمساعد طبي لو عندك عرض أو سؤال صحي تحب نحاول نفهمه بهدوء.",
                "أهلًا. لو في حاجة صحية مضايقاك اكتبها ببساطة، وأنا أساعدك نحدد الخطوة المناسبة.",
                "مرحبًا، أنا معاك في أي سؤال طبي أو أعراض حابب تشرحها.",
            ]
        if route in {"off_topic", "family"}:
            return [
                "أنا هنا كمساعد طبي، فخلّينا في صحتك أو أي سؤال طبي تحب تسأله.",
                "أنا مساعد طبي، ومش هقدر أتكلم عن أمور شخصية أو عائلية. لو عندك عرض صحي اكتبهولي وهساعدك.",
                "خلّينا نركز على صحتك كمساعد طبي. اكتب العرض أو السؤال الطبي اللي محتاج مساعدة فيه.",
            ]
        if route == "abuse":
            return [
                "خلينا نتكلم باحترام عشان أقدر أساعدك. لو عندك عرض صحي أو سؤال طبي اكتبهولي بوضوح.",
                "أنا موجود للمساعدة الطبية. لو أنت متضايق، اكتب المشكلة الصحية بوضوح وأنا هرد بهدوء.",
                "خلينا نحافظ على الاحترام. قولّي العرض أو السؤال الطبي بشكل واضح وهساعدك.",
            ]
        if route == "nonsense":
            return [
                "مش واضح قصدك من الرسالة. لو عندك عرض صحي أو سؤال طبي اكتبهولي بوضوح وهساعدك.",
                "الرسالة مش مفهومة كفاية. اكتبلي العرض اللي حاسس به أو سؤالك الطبي بجملة واضحة.",
                "محتاج رسالة أوضح عشان أساعدك. لو في مشكلة صحية، اشرحها ببساطة.",
            ]
        if route == "who_are_you":
            return [
                "أنا MedBridge AI، مساعد طبي يساعدك تفهم الأعراض، يطرح أسئلة مركزة، وينبهك لو في علامات طوارئ.",
                "أنا مساعد طبي من MedBridge. أقدر أساعد في ترتيب الأعراض وتحديد درجة الخطورة ونوع الطبيب المناسب.",
            ]
        if route == "capabilities":
            return [
                "أقدر أساعدك تفهم الأعراض، أسأل أسئلة مركزة، أقترح درجة الخطورة ونوع الطبيب، وأنبهك لو في علامات طوارئ.",
                "دوري أوجهك طبيًا بشكل مبدئي: نفهم الأعراض، نحدد هل تحتاج طوارئ، ونقترح الطبيب المناسب. ده لا يغني عن الكشف.",
            ]
        if route == "closing":
            return [
                "تمام، أتمنى لك السلامة. لو ظهرت أعراض جديدة أو زادت الأعراض، ابعتلي في أي وقت.",
                "العفو، خلي بالك من نفسك. لو حصل تدهور أو عرض جديد، ارجع اكتبلي أو اطلب رعاية طبية حسب الشدة.",
            ]
        return []

    def _fallback_variants_en(self, route: str, suggested_doctor: str | None) -> list[str]:
        if route == "casual":
            return [
                "I’m here and ready to help with medical questions or symptoms whenever you want to share them.",
                "I’m MedBridge AI. Tell me what symptom or health question you have, and I’ll help you think it through safely.",
                "I’m doing okay. If something medical or health-related is on your mind, describe it and I’ll guide you.",
            ]
        if route in {"off_topic", "family"}:
            return [
                "I’m here as a medical assistant, so let’s keep it to your health. Tell me your symptom or medical question.",
                "I can’t really help with personal or family chat, but I can help with symptoms or health questions.",
                "Let’s keep the conversation medical. If you have a health concern, write it clearly and I’ll help.",
            ]
        if route == "abuse":
            return [
                "Let’s keep it respectful so I can help. If you have symptoms or a medical question, tell me clearly.",
                "I can help with medical concerns, but I need the message to stay respectful and clear.",
                "I hear the frustration. Write the health issue clearly and I’ll do my best to guide you safely.",
            ]
        if route == "nonsense":
            return [
                "I’m not sure what you mean. If you have a symptom or medical question, write it clearly and I’ll help.",
                "That message is unclear to me. Tell me the symptom or health question in a simple sentence.",
                "I need a clearer message to help. If this is about your health, describe what you’re feeling.",
            ]
        if route == "who_are_you":
            return [
                "I’m MedBridge AI, a medical assistant that helps you understand symptoms, urgency, and what type of doctor may fit.",
                "I’m a MedBridge medical assistant. I can ask focused questions, flag urgent symptoms, and guide you on next steps.",
            ]
        if route == "capabilities":
            return [
                "I can help you think through symptoms, ask focused questions, suggest urgency and doctor type, and flag emergencies.",
                "I can support symptom triage, follow-up questions, urgency guidance, and doctor routing. I don’t replace a doctor.",
            ]
        if route == "closing":
            return [
                "You’re welcome. I hope you feel better. If symptoms worsen or new warning signs appear, seek medical care.",
                "Anytime. Take care, and come back if symptoms change or you need help thinking through a health question.",
            ]
        return []

    def _append_questions_if_needed(self, answer: str, follow_up_questions: list[str]) -> str:
        if not follow_up_questions:
            return answer
        if any(question in answer for question in follow_up_questions):
            return answer
        bullet_lines = "\n".join(f"- {question}" for question in follow_up_questions[:3])
        return f"{answer}\n\n{bullet_lines}"

    def _naturalized_answer_is_safe(self, answer: str, route: str, diagnosis_allowed: bool) -> bool:
        if not answer.strip():
            return False
        lowered = answer.lower()
        normalized = answer.strip().lower()
        awkward_markers = {
            "حسناً",
            "حسنا",
            "إنتي عاوزين",
            "انتي عاوزين",
            "إنت عاوزين",
            "انت عاوزين",
            "أنا آسف إننا لا نستطيع",
            "آسف إننا لا نستطيع",
            "أكتبولي",
            "اكتبولي",
            "تبي",
            "لا تفهمت",
            "لماذا تظن",
            "أكون موجود",
            "انت مرحب",
            "أنت مرحب",
            "بلاش وانا موجود",
            "حصلت على هذا الوجع",
            "تعليقات غير محترمة",
            "شئ",
            "هناك على إثرها",
            "هل يمكننا التحدث باحترام",
            "قمت بكتابة",
            "بكل سرور",
            "\u0643\u062a\u0627\u0628\u0629\u0648\u0644\u064a",
            "\u0643\u062a\u0627\u0628\u0647\u0648\u0644\u064a",
            "\u0648 \u0633\u0623\u062d\u0627\u0648\u0644",
            "\u0644\u0648\u062d\u0638 \u0623\u0646\u0643",
            "\u0644\u0648\u062d\u0638 \u0627\u0646\u0643",
            "\u0647\u0630\u0627 \u0627\u0644\u0648\u062e\u0632",
            "\u0623\u0633\u0628\u0627\u0628 \u0647\u0630\u0627 \u0627\u0644\u0648\u062e\u0632",
            "\u0645\u0645\u0643\u0646 \u062a\u0633\u0627\u0639\u062f\u0646\u0627",
            "\u0646\u0633\u062a\u0637\u064a\u0639 \u0641\u0647\u0645 \u0623\u0643\u062b\u0631",
            "\u0645\u0627 \u0634\u0627\u0621 \u0627\u0644\u0644\u0647",
            "\u0623\u0646\u0627 \u062d\u0627\u0648\u0644",
            "\u0627\u0646\u0627 \u062d\u0627\u0648\u0644",
        }
        if any(marker.lower() in normalized for marker in awkward_markers):
            return False
        if route == "medical_clarification" and answer.count("?") + answer.count("\u061f") > 4:
            return False
        if any(term in lowered for term in ("classifier", "rag", "confidence", "retrieved cases", "api key")):
            return False
        if route in {"off_topic", "family"} and re.search(r"[\u0600-\u06FF]", answer):
            if "مساعد طبي" not in answer and "مساعدة طبية" not in answer:
                return False
        if route == "abuse" and re.search(r"[\u0600-\u06FF]", answer):
            if "احترام" not in answer and "متضايق" not in answer:
                return False
            if not ("خلينا" in answer or "خلّينا" in answer or "متضايق" in answer):
                return False
        if route == "nonsense" and re.search(r"[\u0600-\u06FF]", answer):
            if not any(phrase in answer for phrase in ("مش واضح", "غير واضحة", "مش مفهومة", "غير مفهومة")):
                return False
        if route in {"casual", "off_topic", "family", "abuse", "nonsense", "who_are_you", "capabilities", "closing"}:
            blocked = {
                "malaria",
                "aids",
                "cancer",
                "hypertension",
                "pneumonia",
                "asthma",
                "hepatitis",
                "diagnosis:",
                "the closest",
                "ملاريا",
                "نقص المناعة",
                "سرطان",
                "التهاب رئوي",
                "ربو",
                "الاحتمال الأقرب",
                "تشخيصك",
            }
            if any(term in lowered for term in blocked):
                return False
        if not diagnosis_allowed and route == "medical_clarification":
            blocked = {"malaria", "aids", "cancer", "hypertension", "pneumonia", "ملاريا", "نقص المناعة", "سرطان"}
            if any(term in lowered for term in blocked):
                return False
        return not self._contains_blocked_advice(answer)

    def _extract_json_object(self, raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _display_diagnosis(self, diagnosis: str | None) -> str:
        if not diagnosis:
            return "حالة تحتاج تقييم طبي"
        return display_diagnosis_ar(diagnosis) or diagnosis

    def _detect_language(self, message: str) -> str:
        arabic = len(re.findall(r"[\u0600-\u06FF]", message or ""))
        latin = len(re.findall(r"[A-Za-z]", message or ""))
        if latin > 0 and arabic == 0:
            return "English"
        if latin > 0 and arabic > 0:
            return "dominant mixed Arabic/English"
        return "Arabic/Egyptian Arabic"

    def _fallback_answer(self, context: dict[str, Any]) -> str:
        if context.get("user_language") == "English":
            return self._fallback_answer_en(context)
        diagnosis = context.get("display_diagnosis") or "حالة تحتاج تقييم طبي"
        urgency = context.get("urgency_arabic") or context.get("urgency_level", "منخفض")
        doctor = context.get("suggested_doctor_arabic") or context.get("suggested_doctor", "طبيب عام")
        precautions = context.get("answer_precautions") or SAFE_GENERIC_PRECAUTIONS
        description = context.get("diagnosis_summary")

        advice = "، ".join(precautions[:4])
        reason = description or "الأعراض المذكورة تتماشى مع هذا الاحتمال حسب نموذج الأعراض وطبقة المعرفة الطبية."
        urgent = "، ".join(context.get("urgent_warning_signs") or self._urgent_warning_signs([], "Low"))

        if context.get("urgency_level") == "High":
            return (
                f"{HIGH_URGENCY_PREFIX}\n\n"
                f"الاحتمال الأقرب: {diagnosis}.\n\n"
                f"السبب:\n{reason}\n\n"
                "مستوى الخطورة:\nعال.\n\n"
                f"الطبيب المناسب:\n{doctor}.\n\n"
                "نصائح مبدئية:\nلا تنتظر ولا تعتمد على نصائح منزلية كبديل عن الطوارئ. حاول تخلي شخص قريب يساعدك في الوصول للرعاية العاجلة.\n\n"
                f"متى يجب طلب مساعدة عاجلة:\nالآن، خاصة إذا كانت الأعراض بدأت فجأة أو بتزيد بسرعة. {urgent}\n\n"
                "تنبيه:\nهذا ليس تشخيصًا نهائيًا، ويجب مراجعة الطبيب للتأكيد."
            )

        return (
            f"فاهمك. بناءً على الأعراض اللي ذكرتها، الاحتمال الأقرب مبدئيًا هو: {diagnosis}.\n\n"
            f"{reason}\n\n"
            f"مستوى الخطورة: {urgency}. الطبيب المناسب: {doctor}.\n\n"
            f"حاليًا التزم بنصائح آمنة: {advice}.\n\n"
            f"اطلب مساعدة طبية بسرعة لو ظهر أو زاد أي من الآتي: {urgent}.\n\n"
            "تنبيه: هذا ليس تشخيصًا نهائيًا، ومراجعة الطبيب مهمة للتأكيد."
        )

    def _fallback_answer_en(self, context: dict[str, Any]) -> str:
        diagnosis = context.get("possible_diagnosis") or context.get("display_diagnosis") or "a condition that needs medical assessment"
        urgency = context.get("urgency_level", "Low")
        doctor = context.get("suggested_doctor", "General Practitioner")
        precautions = context.get("answer_precautions") or ["rest", "hydrate", "monitor symptoms", "seek medical care if symptoms worsen"]
        description = context.get("diagnosis_summary") or "The symptoms point in this direction, but they are not enough for a final diagnosis."
        advice = ", ".join(precautions[:4])
        urgent = ", ".join(context.get("urgent_warning_signs") or ["chest pain, breathing trouble, fainting, bleeding, or rapid worsening"])

        if urgency == "High":
            return (
                "These symptoms may be urgent. Please go to the emergency department now or call emergency services.\n\n"
                f"Main concern: {diagnosis}.\n\n"
                f"Why: {description}\n\n"
                f"Suggested care: {doctor}.\n\n"
                "Do not wait or rely on home care instead of emergency assessment.\n\n"
                "Note: this is not a final diagnosis."
            )

        return (
            f"Based on what you described, the closest initial direction is: {diagnosis}.\n\n"
            f"{description}\n\n"
            f"Urgency: {urgency}. Suggested doctor: {doctor}.\n\n"
            f"Safe first steps: {advice}.\n\n"
            f"Seek medical care quickly if any warning sign appears: {urgent}.\n\n"
            "Note: this is not a final diagnosis."
        )

    def _enforce_answer_contract(self, answer: str, context: dict[str, Any]) -> str:
        internal_diagnosis = context.get("possible_diagnosis")
        display_diagnosis = context.get("display_diagnosis") or self._display_diagnosis(internal_diagnosis)
        urgency = context.get("urgency_level")

        if internal_diagnosis and internal_diagnosis != display_diagnosis:
            answer = answer.replace(f" ({internal_diagnosis})", "")
            answer = answer.replace(f"({internal_diagnosis})", "")
            answer = answer.replace(internal_diagnosis, display_diagnosis)
            answer = answer.replace(f"{display_diagnosis} ({display_diagnosis})", display_diagnosis)
            answer = answer.replace(f"{display_diagnosis} - {display_diagnosis}", display_diagnosis)

        if display_diagnosis and display_diagnosis not in answer:
            reason_marker = "\n\nالسبب:"
            first_line = f"أفهم قلقك. بناءً على الأعراض المذكورة، الاحتمال الأقرب هو: {display_diagnosis}."
            if reason_marker in answer:
                answer = first_line + answer[answer.index(reason_marker) :]
            else:
                answer = f"{first_line}\n\n{answer}"

        first_line = f"أفهم قلقك. بناءً على الأعراض المذكورة، الاحتمال الأقرب هو: {display_diagnosis}."
        duplicate_marker = "\n\nأفهم قلقك. بناءً على الأعراض المذكورة، الاحتمال الأقرب هو:"
        reason_marker = "\n\nالسبب:"
        if answer.startswith(first_line) and duplicate_marker in answer and reason_marker in answer:
            answer = answer[: answer.index(duplicate_marker)] + answer[answer.index(reason_marker) :]

        answer = self._fill_placeholder_sections(answer, context)

        if urgency == "High":
            answer = answer.replace("عيادة الصحة العامة", "الطوارئ")
            answer = answer.replace("انتظر", "لا تنتظر")
            answer = answer.replace("راقب فقط", "اطلب مساعدة طبية عاجلة")
            answer = answer.replace("لا لا تنتظر", "لا تنتظر")
            if not answer.startswith(HIGH_URGENCY_PREFIX):
                answer = f"{HIGH_URGENCY_PREFIX}\n\n{answer}"

        return answer

    def _fill_placeholder_sections(self, answer: str, context: dict[str, Any]) -> str:
        defaults = {
            "السبب": (
                context.get("diagnosis_summary")
                or "الأعراض المذكورة تتماشى مع هذا الاحتمال حسب نموذج الأعراض وطبقة المعرفة الطبية."
            ),
            "مستوى الخطورة": f"{context.get('urgency_arabic') or context.get('urgency_level', 'منخفض')}.",
            "الطبيب المناسب": f"{context.get('suggested_doctor_arabic') or context.get('suggested_doctor', 'طبيب عام')}.",
            "نصائح مبدئية": "، ".join(context.get("answer_precautions") or SAFE_GENERIC_PRECAUTIONS),
            "متى يجب طلب مساعدة عاجلة": "، ".join(
                context.get("urgent_warning_signs") or self._urgent_warning_signs([], "Low")
            ),
        }
        for heading, default in defaults.items():
            escaped = re.escape(heading)
            answer = re.sub(
                rf"({escaped}:\s*\n)\s*(?:\.\.\.|…)\s*(?=\n\n|$)",
                rf"\1{default}",
                answer,
            )
            answer = re.sub(
                rf"({escaped}:\s*)(?:\.\.\.|…)\s*(?=\n\n|$)",
                rf"\1{default}",
                answer,
            )
        return answer.replace("السبب: غير محدد.", f"السبب: {defaults['السبب']}")

    def _answer_precautions(
        self,
        precautions: list[str],
        urgency_level: str,
        retrieved_cases: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        if urgency_level == "High":
            return (
                [
                    "التوجه للطوارئ فورًا أو الاتصال بالإسعاف",
                    "تجنب المجهود وعدم قيادة السيارة بنفسك",
                    "البقاء مع شخص قريب لحين وصول المساعدة",
                ],
                ["safety_policy"],
            )

        translated = self._translate_precautions(precautions)
        sources: list[str] = []
        if translated:
            sources.append("medical_knowledge_base")

        retrieved_advice = self._retrieved_case_precautions(retrieved_cases)
        if retrieved_advice:
            sources.append("retrieved_medical_cases")

        advice = self._dedupe(translated + retrieved_advice)
        if advice:
            return advice, sources

        return SAFE_GENERIC_PRECAUTIONS.copy(), ["safe_generic_policy"]

    def _translate_precautions(self, precautions: list[str]) -> list[str]:
        translations = {
            "drink vitamin c rich drinks": "شرب سوائل كافية ومتابعة الترطيب",
            "take vapour": "استنشاق بخار دافئ بحذر إذا كان مريحا",
            "avoid cold food": "تجنب الأطعمة أو المشروبات الباردة مؤقتا إذا كانت تزيد الأعراض",
            "keep fever in check": "متابعة درجة الحرارة وطلب رعاية طبية إذا ارتفعت أو استمرت",
            "lie down": "الاستلقاء في مكان آمن وهادئ",
            "avoid sudden change in body": "تجنب تغيير وضع الجسم بشكل مفاجئ",
            "avoid abrupt head movment": "تجنب حركة الرأس المفاجئة",
            "relax": "الراحة مؤقتا وتقليل المجهود",
            "bath twice": "غسل المنطقة بلطف والحفاظ على النظافة",
            "use detol or neem in bathing water": "الحفاظ على نظافة الجلد دون استخدام مواد مهيجة",
            "keep infected area dry": "الحفاظ على المنطقة المصابة جافة",
            "use clean cloths": "استخدام ملابس ومناشف نظيفة",
            "stop eating solid food for while": "تخفيف الأكل الصلب مؤقتا إذا كان القيء مستمرا",
            "try taking small sips of water": "شرب رشفات صغيرة ومتكررة من الماء",
            "rest": "الراحة",
            "ease back into eating": "العودة للأكل تدريجيا عند تحسن القيء",
            "follow up": "المتابعة مع طبيب",
            "use heating pad or cold pack": "استخدام كمادات دافئة أو باردة على الرقبة إذا كانت مريحة",
            "exercise": "تجنب الإجهاد والحركة العنيفة، وممارسة حركة خفيفة فقط إذا كانت لا تزيد الألم",
            "consult doctor": "مراجعة طبيب إذا استمر الألم أو ظهر تنميل أو ضعف",
            "avoid fatty spicy food": "تجنب الأكل الدسم أو الحار إذا كان يزيد ألم المعدة",
            "consume probiotic food": "اختيار أكل خفيف وسهل الهضم ومتابعة الأعراض",
            "eliminate milk": "تجنب الأطعمة أو المشروبات التي تلاحظ أنها تزيد الأعراض مؤقتا",
            "limit alcohol": "تجنب الكحوليات والمهيجات",
        }

        translated = []
        for item in precautions:
            normalized = item.strip().lower()
            value = translations.get(normalized, item.strip())
            if value and not self._contains_blocked_advice(value):
                translated.append(value)
        return self._dedupe(translated)

    def _retrieved_case_precautions(self, retrieved_cases: list[dict[str, Any]]) -> list[str]:
        advice: list[str] = []
        safe_phrases = [
            "الراحة",
            "شرب سوائل",
            "الإكثار من السوائل",
            "متابعة الأعراض",
            "مراجعة الطبيب",
            "استشارة الطبيب",
            "التوجه للطوارئ",
            "تجنب المهيجات",
            "الحفاظ على النظافة",
            "الحفاظ على المنطقة جافة",
        ]
        for case in retrieved_cases[:3]:
            text = " ".join(str(case.get(key, "")) for key in ("a_body", "q_body"))
            if not text or self._contains_blocked_advice(text):
                continue
            for phrase in safe_phrases:
                if phrase in text:
                    advice.append(phrase)
        return self._dedupe(advice)

    def _contains_blocked_advice(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in BLOCKED_ADVICE_TERMS)

    def _dedupe(self, items: list[str]) -> list[str]:
        cleaned = []
        seen = set()
        for item in items:
            value = item.strip()
            if value and value not in seen:
                cleaned.append(value)
                seen.add(value)
        return cleaned

    def _urgent_warning_signs(self, symptoms: list[str], urgency_level: str) -> list[str]:
        symptom_set = set(symptoms)

        if urgency_level == "High":
            return [
                "استمرار ألم الصدر أو زيادة ضيق التنفس",
                "إغماء، تعرق شديد، زرقة في الشفاه، أو صعوبة في الكلام",
                "تدهور سريع في الحالة",
            ]

        if "cough" in symptom_set:
            return [
                "ضيق تنفس أو ألم في الصدر",
                "حرارة مرتفعة لا تتحسن",
                "صفير شديد في الصدر أو تدهور سريع",
            ]
        if {"headache", "dizziness"}.issubset(symptom_set):
            return [
                "صداع شديد مفاجئ أو ضعف/تنميل في طرف من الجسم",
                "قيء متكرر أو عدم اتزان شديد",
                "زغللة شديدة أو فقدان وعي",
            ]
        if "skin_rash" in symptom_set or "itching" in symptom_set:
            return [
                "تورم في الوجه أو الشفاه أو صعوبة تنفس",
                "انتشار سريع للطفح أو ظهور إفرازات",
                "حرارة مرتفعة مع الطفح",
            ]
        if {"diarrhoea", "vomiting"}.intersection(symptom_set):
            return [
                "علامات جفاف مثل قلة البول أو دوخة شديدة",
                "دم في البراز أو القيء",
                "ألم شديد في البطن أو حرارة مرتفعة",
            ]
        return [
            "ألم صدر، ضيق تنفس، إغماء، نزيف، أو تدهور سريع",
        ]
