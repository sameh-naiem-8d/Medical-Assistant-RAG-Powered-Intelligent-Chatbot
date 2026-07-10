from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .display_labels import display_diagnosis_ar, display_doctor_ar
from .llm_service import HIGH_URGENCY_PREFIX
from .safety import EmergencySignal, detect_emergency_signal, normalize_text
from .schemas import ChatMessage, ChatRequest, ChatResponse


CLARIFICATION_MODE = "clarification"
DIAGNOSIS_MODE = "diagnosis"
EMERGENCY_MODE = "emergency"
CLOSING_MODE = "closing"

MAX_QUESTIONS = 3
V2_OWNED_DOMAINS = {"back_pain", "gynecology", "headache", "neurology_vague", "chest_pain"}


@dataclass
class MessageMeaning:
    language: str
    intent: str = "medical_complaint"
    domain: str = "unknown"
    body_parts: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    denials: set[str] = field(default_factory=set)
    severity: str | None = None
    duration: str | None = None
    red_flags: list[str] = field(default_factory=list)
    is_new_complaint: bool = True
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "domain": self.domain,
            "body_parts": self.body_parts,
            "symptoms": self.symptoms,
            "facts": self.facts,
            "denials": sorted(self.denials),
            "severity": self.severity,
            "duration": self.duration,
            "red_flags": self.red_flags,
            "is_new_complaint": self.is_new_complaint,
            "confidence": self.confidence,
        }


@dataclass
class CaseState:
    active_domain: str | None = None
    active_body_part: str | None = None
    symptoms: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    denials: set[str] = field(default_factory=set)
    answered_questions: set[str] = field(default_factory=set)
    pending_questions: list[str] = field(default_factory=list)
    duration: str | None = None
    severity: str | None = None
    doctor_route: str | None = None
    language: str = "ar"
    case_closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_domain": self.active_domain,
            "active_body_part": self.active_body_part,
            "symptoms": self.symptoms,
            "facts": self.facts,
            "denials": sorted(self.denials),
            "answered_questions": sorted(self.answered_questions),
            "pending_questions": self.pending_questions,
            "duration": self.duration,
            "severity": self.severity,
            "doctor_route": self.doctor_route,
            "language": self.language,
            "case_closed": self.case_closed,
        }


@dataclass
class SafePlan:
    route: str
    language: str
    domain: str
    answer: str
    questions_to_ask: list[str] = field(default_factory=list)
    doctor_route: str = "Needs more information"
    urgency_level: str = "Low"
    diagnosis: str | None = None
    confidence: float = 0.0
    symptoms: list[str] = field(default_factory=list)
    precautions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    case_state: CaseState | None = None
    current_meaning: MessageMeaning | None = None
    used_legacy: bool = False


class ChatEngineV2:
    """Stateless conversation engine used before the legacy classifier-heavy path.

    V2 owns conversation intent, active-case switching, missing-field selection, and
    final safety verification for the domains where manual testing found drift.
    The legacy path remains available only as a compatibility fallback for domains
    not yet owned by V2.
    """

    def __init__(self, *, classifier_service: Any, knowledge_service: Any, rag_service: Any, llm_service: Any):
        self.classifier_service = classifier_service
        self.knowledge_service = knowledge_service
        self.rag_service = rag_service
        self.llm_service = llm_service

    def handle_chat(
        self,
        request: ChatRequest,
        *,
        legacy_handler: Callable[[ChatRequest], ChatResponse] | None = None,
    ) -> ChatResponse:
        if _looks_like_mojibake(request.message):
            return self._fallback_to_legacy_or_generic(request, legacy_handler)

        recent_history = request.history[-10:]
        language = _detect_language(request.message, request.language)
        current_symptoms = self._extract_symptoms_safe(request.message)
        current_meaning = understand_message(request.message, language=language, symptoms=current_symptoms)

        if current_meaning.intent == "closing":
            return self._response_from_plan(
                SafePlan(
                    route=CLOSING_MODE,
                    language=language,
                    domain="closing",
                    answer=_closing_answer(language),
                    doctor_route="Not needed",
                    current_meaning=current_meaning,
                ),
                request,
            )

        emergency_signal = detect_emergency_signal(request.message, current_symptoms)
        if emergency_signal:
            return self._emergency_response(
                request=request,
                signal=emergency_signal,
                language=language,
                symptoms=current_symptoms,
                current_meaning=current_meaning,
            )

        gate_plan = self._non_medical_gate(request, current_meaning)
        if gate_plan:
            if legacy_handler and current_meaning.intent != "off_topic":
                response = legacy_handler(request)
                response.case_state_update.setdefault("engine", "legacy_via_v2_gate")
                return response
            return self._response_from_plan(gate_plan, request)

        history_state = reconstruct_case_state(recent_history, default_language=language)
        previous_domain = history_state.active_domain
        active_case = decide_and_update_case(history_state, current_meaning, recent_history)
        current_meaning.is_new_complaint = bool(
            active_case.active_domain and active_case.active_domain == current_meaning.domain and not _is_followup_answer(current_meaning)
        )

        if not _v2_should_own(current_meaning, active_case, previous_domain):
            if legacy_handler:
                response = legacy_handler(request)
                response.case_state_update.setdefault("engine", "legacy_via_v2_unowned")
                return response

        current_plus_history_symptoms = _dedupe(active_case.symptoms + current_symptoms)
        history_message_text = " ".join(item.content for item in recent_history if item.role == "user")
        combined_message = f"{history_message_text} {request.message}".strip()
        combined_emergency = detect_emergency_signal(combined_message, current_plus_history_symptoms)
        if combined_emergency and current_meaning.domain != "unknown":
            return self._emergency_response(
                request=request,
                signal=combined_emergency,
                language=language,
                symptoms=current_plus_history_symptoms,
                current_meaning=current_meaning,
                case_state=active_case,
            )

        plan = self._build_plan(request, current_meaning, active_case, current_plus_history_symptoms)
        if not plan and legacy_handler:
            legacy_response = legacy_handler(request)
            legacy_response.case_state_update.setdefault("engine", "legacy_via_v2")
            return legacy_response
        if not plan:
            plan = self._generic_medical_clarification(current_meaning, active_case, current_plus_history_symptoms)

        return self._response_from_plan(plan, request)

    def _fallback_to_legacy_or_generic(
        self,
        request: ChatRequest,
        legacy_handler: Callable[[ChatRequest], ChatResponse] | None,
    ) -> ChatResponse:
        if legacy_handler:
            response = legacy_handler(request)
            response.case_state_update.setdefault("engine", "legacy_via_v2_mojibake")
            return response
        language = _detect_language(_repair_mojibake(request.message), request.language)
        meaning = understand_message(_repair_mojibake(request.message), language=language, symptoms=[])
        return self._response_from_plan(
            self._generic_medical_clarification(meaning, CaseState(language=language), []),
            request,
        )

    def _extract_symptoms_safe(self, message: str) -> list[str]:
        try:
            return list(self.classifier_service.extract_symptoms(message, history_text=""))
        except Exception:
            return []

    def _non_medical_gate(self, request: ChatRequest, meaning: MessageMeaning) -> SafePlan | None:
        if meaning.intent not in {"abuse", "off_topic", "casual", "nonsense"}:
            return None
        answer = {
            "abuse": (
                "خلينا نتكلم باحترام عشان أقدر أساعدك. لو عندك عرض صحي اكتبه بوضوح."
                if meaning.language != "en"
                else "I can help with health symptoms, but I need us to keep it respectful. Tell me the medical concern clearly."
            ),
            "off_topic": (
                "أنا مساعد طبي هنا للمساعدة في الأعراض والأسئلة الصحية فقط. لو عندك شكوى طبية اكتبها لي بوضوح."
                if meaning.language != "en"
                else "I’m a medical assistant for symptoms and health questions. Tell me the health concern if you want help."
            ),
            "casual": (
                "أهلًا بيك. لو عندك عرض صحي أو سؤال طبي، اكتبه لي وهساعدك خطوة بخطوة."
                if meaning.language != "en"
                else "Hi. If you have a symptom or health question, tell me and I’ll help you step by step."
            ),
            "nonsense": (
                "مش قادر أفهم الرسالة دي كشكوى صحية. اكتب العرض أو السؤال الطبي بشكل أوضح."
                if meaning.language != "en"
                else "I can’t understand that as a health concern. Please write the symptom or medical question more clearly."
            ),
        }[meaning.intent]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=meaning.language,
            domain=meaning.intent,
            answer=answer,
            doctor_route="Not needed",
            current_meaning=meaning,
        )

    def _emergency_response(
        self,
        *,
        request: ChatRequest,
        signal: EmergencySignal,
        language: str,
        symptoms: list[str],
        current_meaning: MessageMeaning,
        case_state: CaseState | None = None,
    ) -> ChatResponse:
        if language == "en":
            answer = (
                "These symptoms may be urgent. Please go to the emergency department now or call emergency services.\n\n"
                f"Main concern: {signal.diagnosis}.\n\n"
                f"Why: {_english_emergency_reason(signal.reason)}\n\n"
                "Do not wait, monitor only, or drive yourself if symptoms are severe or started suddenly."
            )
        else:
            answer = (
                f"{HIGH_URGENCY_PREFIX}\n\n"
                "اتجه إلى الطوارئ الآن ولا تؤجل التقييم.\n\n"
                f"القلق الطبي الأقرب: {signal.display_diagnosis_ar}.\n\n"
                f"السبب: {signal.reason}\n\n"
                "لا تنتظر ولا تكتفي بالمراقبة في البيت، وخلي شخص قريب يفضل معاك لحد ما توصل للرعاية المناسبة."
            )
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=EMERGENCY_MODE,
            answer=answer,
            extracted_symptoms=symptoms,
            possible_diagnosis=signal.diagnosis,
            display_diagnosis_ar=signal.display_diagnosis_ar,
            confidence=0.99,
            urgency_level="High",
            suggested_doctor=signal.doctor,
            display_doctor_ar=signal.display_doctor_ar,
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            meaning=current_meaning,
            case_state=case_state,
            engine_route="v2_emergency",
        )
        return response

    def _build_plan(
        self,
        request: ChatRequest,
        meaning: MessageMeaning,
        case_state: CaseState,
        symptoms: list[str],
    ) -> SafePlan | None:
        domain = case_state.active_domain or meaning.domain
        if domain in {"back_pain", "neck_pain"}:
            return self._back_or_neck_plan(domain, meaning, case_state, symptoms)
        if domain == "gynecology":
            return self._gynecology_plan(meaning, case_state, symptoms)
        if domain == "headache":
            return self._headache_plan(meaning, case_state, symptoms)
        if domain == "neurology_vague":
            return self._vague_neuro_plan(meaning, case_state, symptoms)
        if domain == "chest_pain":
            return self._chest_plan(meaning, case_state, symptoms)
        if domain == "throat_ent":
            return self._throat_plan(meaning, case_state, symptoms)
        if domain == "digestive":
            return self._digestive_plan(meaning, case_state, symptoms)
        if domain == "urinary":
            return self._urinary_plan(meaning, case_state, symptoms)
        if domain == "body_ache":
            return self._body_ache_plan(meaning, case_state, symptoms)
        if domain in {"unknown", None}:
            return self._generic_medical_clarification(meaning, case_state, symptoms)
        return None

    def _back_or_neck_plan(
        self,
        domain: str,
        meaning: MessageMeaning,
        case_state: CaseState,
        symptoms: list[str],
    ) -> SafePlan:
        is_back = domain == "back_pain"
        language = meaning.language or case_state.language
        doctor = "Orthopedic doctor"
        questions = _missing_questions_for_back_case(case_state, language) if is_back else _missing_questions_for_neck_case(case_state, language)
        enough_context = (
            (len(questions) <= 1 and (case_state.facts.get("location") or case_state.duration or case_state.denials))
            or (is_back and case_state.facts.get("location") and "numbness" in case_state.denials)
        )
        if language == "en":
            location = str(case_state.facts.get("location") or ("back" if is_back else "neck")).replace("_", " ")
            if enough_context:
                answer = (
                    f"Thanks, I’ll keep this focused on {location} pain. Broad common causes include muscle strain, posture, lifting, or irritation around the spine/joints. "
                    "Avoid heavy lifting for now, keep gentle movement instead of staying in bed all day, and consider a warm compress if it feels comfortable. "
                    "Seek urgent care if weakness, numbness, bladder/bowel control problems, fever with severe pain, trauma, or rapid worsening appears."
                )
            else:
                answer = (
                    f"{'Back' if is_back else 'Neck'} pain can come from muscle strain, posture, lifting, or spine/joint irritation. "
                    "I’ll ask only the missing details so we do not repeat the same questions."
                )
        else:
            location_ar = _arabic_location(case_state.facts.get("location"), is_back)
            if enough_context:
                no_numb = " عدم وجود تنميل مطمن نسبيًا." if "numbness" in case_state.denials else ""
                answer = (
                    f"تمام، هفضل مركز على {location_ar}.{no_numb} الأسباب الشائعة ممكن تكون شد عضلي، وضعية غلط، حمل حاجة تقيلة، أو تهيج في الفقرات/المفاصل. "
                    "حاول تتجنب الحمل التقيل، اتحرك حركة خفيفة من غير عنف، وما تفضلش في السرير طول اليوم. "
                    "اطلب رعاية عاجلة لو ظهر ضعف، تنميل شديد، فقدان تحكم في البول أو البراز، حرارة مع ألم شديد، إصابة قوية، أو تدهور سريع."
                )
            else:
                answer = (
                    f"{location_ar} غالبًا يحتاج شوية تفاصيل عن المكان والمدة وطريقة بداية الألم. "
                    "هسأل فقط عن النقط الناقصة من غير ما أكرر اللي جاوبت عليه."
                )
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain=domain,
            answer=answer,
            questions_to_ask=questions,
            doctor_route=doctor,
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _gynecology_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        questions = _missing_questions_for_gynecology(case_state, language)
        if language == "en":
            answer = (
                "I’ll keep this focused on the menstrual period/discharge concern. "
                "Discharge details matter because color, smell, itching, burning, pelvic pain, fever, or pregnancy possibility can change the next step."
            )
        else:
            age = f" وسنك {case_state.facts.get('age')} سنة" if case_state.facts.get("age") else ""
            discharge = " ومع وجود إفرازات كثيرة" if "discharge" in case_state.symptoms else ""
            answer = (
                f"تمام، هكمل على نفس موضوع البريود/النساء{age}{discharge}. "
                "تفاصيل الإفرازات مهمة عشان نعرف هل الموضوع بسيط ولا محتاج كشف نساء قريب."
            )
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="gynecology",
            answer=answer,
            questions_to_ask=questions,
            doctor_route="Gynecologist",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _headache_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        questions = _missing_questions_for_headache(case_state, language)
        if language == "en":
            answer = (
                "I’ll treat this as a new headache complaint, not as the previous case. "
                "Headache assessment depends on onset, severity, vision symptoms, vomiting, fever, and any weakness or numbness."
            )
        else:
            answer = (
                "هتعامل مع ده كشكوى صداع/ألم في الدماغ جديدة، مش امتداد للحالة السابقة. "
                "مهم نعرف بداية الصداع وشدته وهل معاه علامات عصبية أو قيء أو زغللة."
            )
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="headache",
            answer=answer,
            questions_to_ask=questions,
            doctor_route="General Practitioner",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _vague_neuro_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if language == "en":
            answer = "When you say nerve tiredness, I need to understand what sensation you mean before routing it."
            questions = [
                "Do you mean numbness, weakness, tremor, pain, burning, or tingling?",
                "Where exactly do you feel it?",
                "Did it start suddenly or is it gradual?",
            ]
        else:
            answer = "لما تقول تعب في الأعصاب، محتاج أوضح قصدك إيه بالضبط بدل ما أفترض تشخيص غلط."
            questions = [
                "تقصد تنميل، ضعف، رعشة، ألم، حرقان، ولا إحساس تاني؟",
                "الإحساس مكانه فين بالضبط؟",
                "بدأ فجأة ولا تدريجي؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="neurology_vague",
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="General Practitioner",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _chest_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if _has_chest_emergency_context(meaning):
            if language == "en":
                answer = (
                    "Chest pain with those warning signs can be urgent. Please go to the emergency department now or call emergency services.\n\n"
                    "Main concern: possible cardiac emergency.\n\n"
                    "Do not wait at home, and avoid driving yourself if the pain is severe, pressure-like, or comes with breathing trouble, sweating, fainting, or spreading pain."
                )
            else:
                answer = (
                    f"{HIGH_URGENCY_PREFIX}\n\n"
                    "ألم الصدر مع العلامات اللي ذكرتها ممكن يشير لمشكلة قلبية طارئة. اتجه للطوارئ فورًا أو اتصل بالإسعاف.\n\n"
                    "لا تنتظر في البيت، ويفضل ما تسوقش بنفسك لو الألم شديد أو ضاغط أو معاه ضيق نفس/عرق بارد/إغماء/امتداد للذراع أو الفك أو الظهر."
                )
            return SafePlan(
                route=EMERGENCY_MODE,
                language=language,
                domain="chest_pain",
                answer=answer,
                questions_to_ask=[],
                doctor_route="Emergency Department / Cardiologist",
                urgency_level="High",
                diagnosis="Cardiac emergency concern",
                confidence=0.99,
                symptoms=symptoms,
                case_state=case_state,
                current_meaning=meaning,
            )

        questions = _missing_questions_for_chest_case(case_state, language)
        if language == "en":
            answer = (
                "Chest pain needs careful checking, but I need a few details before assuming an emergency. "
                "If the pain is severe, pressure-like, comes with shortness of breath, cold sweat, fainting, or spreads to the arm, jaw, or back, seek emergency care now."
            )
        else:
            answer = (
                "ألم الصدر لازم ناخده بجدية، لكن محتاجين نعرف تفاصيله قبل ما نفترض إنه طارئ. "
                "لو الألم شديد أو ضاغط، أو معاه ضيق نفس، عرق بارد، إغماء، أو بيمتد للذراع أو الفك أو الظهر، اتجه إلى الطوارئ فورًا."
            )
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="chest_pain",
            answer=answer,
            questions_to_ask=questions,
            doctor_route="Cardiologist",
            urgency_level="Medium",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _throat_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if _has_respiratory_cluster(case_state):
            return None  # type: ignore[return-value]
        if language == "en":
            answer = (
                "I’ll keep this focused on throat pain and swallowing before guessing a diagnosis. "
                "For now, fluids, rest, and avoiding smoke/irritants are reasonable; seek urgent care if breathing or swallowing saliva becomes difficult."
            )
            questions = [
                "Do you also have fever or cough?",
                "Does it get worse when swallowing?",
                "Any breathing difficulty, choking feeling, voice change, or trouble opening your mouth?",
            ]
        else:
            answer = (
                "ألم الزور أو الحلق محتاج أعرف كام حاجة بسيطة عن البلع والحرارة والكحة قبل أي استنتاج. "
                "مبدئيًا السوائل والراحة وتجنب الدخان أو المهيجات ممكن يساعدوا، لكن صعوبة التنفس أو عدم القدرة على بلع الريق تحتاج رعاية عاجلة."
            )
            questions = [
                "هل عندك حرارة أو كحة؟",
                "هل الألم بيزيد مع البلع؟",
                "هل في صعوبة تنفس، إحساس باختناق، تغير في الصوت، أو صعوبة فتح الفم؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="throat_ent",
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="Needs more information",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _digestive_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if language == "en":
            answer = (
                "I’ll keep this focused on abdominal/digestive symptoms before guessing a diagnosis. "
                "If symptoms are mild, hydration and light food may help; seek urgent care if pain is severe/worsening, vomiting persists, blood appears, or fainting occurs."
            )
            questions = [
                "Where exactly is the abdominal pain: upper, lower, right, left, or around the belly button?",
                "Any vomiting or nausea?",
                "Any diarrhea or constipation?",
            ]
        else:
            answer = (
                "ألم البطن محتاج أسئلة مركزة عن مكان الألم وأعراض المعدة والهضم قبل أي استنتاج. "
                "لو الألم بسيط، اشرب سوائل وكل أكل خفيف، لكن الألم الشديد أو المتزايد، القيء المستمر، الدم، أو الإغماء يحتاج رعاية عاجلة."
            )
            questions = [
                "الألم فين بالضبط في البطن: فوق، تحت، يمين، شمال، ولا حوالين السرة؟",
                "هل في ترجيع أو غثيان؟",
                "هل في إسهال أو إمساك؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="digestive",
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="Needs more information",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _urinary_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if language == "en":
            answer = (
                "Burning with urination keeps this focused on urinary symptoms and warning signs. "
                "Drink fluids if you can, and arrange medical review if burning persists or if fever, flank pain, blood in urine, or pregnancy is possible."
            )
            questions = [
                "Do you feel burning while urinating?",
                "Are you urinating more often or urgently?",
                "Have you noticed blood in the urine?",
            ]
        else:
            answer = (
                "حرقان البول يخلينا نركز على أعراض التبول والمسالك البولية. "
                "اشرب سوائل لو تقدر، وراجع طبيب لو الحرقان مستمر أو ظهر حرارة، ألم جنب، دم في البول، أو في احتمال حمل."
            )
            questions = [
                "هل في حرقان أثناء التبول؟",
                "هل بتدخل الحمام كتير أو حاسس إنك محتاج تتبول باستمرار؟",
                "هل لاحظت دم في البول؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="urinary",
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="Urologist",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _body_ache_plan(self, meaning: MessageMeaning, case_state: CaseState, symptoms: list[str]) -> SafePlan:
        language = meaning.language or case_state.language
        if language == "en":
            answer = (
                "Body aches alone are broad, so I should not jump to a rare diagnosis from that alone. "
                "Rest, fluids, and monitoring temperature are reasonable while we clarify the pattern."
            )
            questions = [
                "When did the body aches start?",
                "Do you also have fever, sore throat, cough, stomach symptoms, urinary burning, or recent heavy effort?",
                "Is the pain all over or focused in one area?",
            ]
        else:
            answer = (
                "وجع الجسم لوحده عرض عام جدًا، ومش كفاية لتشخيص محدد أو مرض نادر من غير تفاصيل. "
                "مبدئيًا الراحة، السوائل، ومتابعة الحرارة مفيدين لحد ما نوضح الصورة."
            )
            questions = [
                "وجع الجسم بدأ من إمتى؟",
                "هل فيه حرارة، التهاب حلق، كحة، إسهال، حرقان بول، أو كان بعد مجهود؟",
                "الوجع في الجسم كله ولا في مكان معين؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain="body_ache",
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="General Practitioner",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _generic_medical_clarification(
        self,
        meaning: MessageMeaning,
        case_state: CaseState,
        symptoms: list[str],
    ) -> SafePlan:
        language = meaning.language or case_state.language
        if language == "en":
            answer = "I need a little more detail before giving medical guidance."
            questions = [
                "What symptom is bothering you most?",
                "When did it start?",
                "Is it getting worse quickly, very severe, or associated with confusion, fainting, repeated vomiting, or trouble breathing?",
            ]
        else:
            answer = "محتاج تفاصيل أوضح قبل أي توجيه طبي."
            questions = [
                "أكتر عرض مضايقك إيه بالضبط؟",
                "بدأ من إمتى؟",
                "هل الأعراض بتزيد بسرعة، شديدة جدًا، أو معاها لخبطة، إغماء، قيء متكرر، أو صعوبة تنفس؟",
            ]
        return SafePlan(
            route=CLARIFICATION_MODE,
            language=language,
            domain=meaning.domain,
            answer=answer,
            questions_to_ask=_filter_answered_questions(questions, case_state),
            doctor_route="Needs more information",
            symptoms=symptoms,
            case_state=case_state,
            current_meaning=meaning,
        )

    def _response_from_plan(self, plan: SafePlan, request: ChatRequest) -> ChatResponse:
        plan.questions_to_ask = _dedupe(plan.questions_to_ask)[:MAX_QUESTIONS]
        answer = self._maybe_naturalize(plan, request)
        answer = self._verify_answer(answer, plan)
        answer = self._judge_answer(answer, plan, request)
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=plan.route,
            answer=answer,
            extracted_symptoms=plan.symptoms,
            possible_diagnosis=plan.diagnosis,
            display_diagnosis_ar=display_diagnosis_ar(plan.diagnosis),
            confidence=plan.confidence,
            urgency_level=plan.urgency_level,
            suggested_doctor=plan.doctor_route,
            display_doctor_ar=display_doctor_ar(plan.doctor_route),
            precautions=plan.precautions,
            needs_follow_up=plan.route == CLARIFICATION_MODE and bool(plan.questions_to_ask),
            follow_up_questions=plan.questions_to_ask if plan.route == CLARIFICATION_MODE else [],
            retrieved_cases=[],
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            meaning=plan.current_meaning,
            case_state=plan.case_state,
            engine_route=f"v2_{plan.route}",
        )
        return response

    def _maybe_naturalize(self, plan: SafePlan, request: ChatRequest) -> str:
        # V2 keeps body-area medical guidance deterministic because these were the
        # exact routes where LLM phrasing previously repeated questions or drifted.
        if plan.route != CLARIFICATION_MODE or plan.domain in {
            "back_pain",
            "neck_pain",
            "chest_pain",
            "throat_ent",
            "digestive",
            "urinary",
            "gynecology",
            "headache",
            "neurology_vague",
            "body_ache",
        }:
            return plan.answer
        try:
            return self.llm_service.naturalize_response(
                route=plan.route,
                message=request.message,
                language=plan.language,
                history=request.history[-6:],
                draft_answer=plan.answer,
                follow_up_questions=plan.questions_to_ask,
                suggested_doctor=plan.doctor_route,
                medical_domain=plan.domain,
                diagnosis_allowed=bool(plan.diagnosis),
            )
        except Exception:
            return plan.answer

    def _verify_answer(self, answer: str, plan: SafePlan) -> str:
        cleaned = _strip_duplicate_questions(answer, plan.questions_to_ask)
        forbidden_by_domain = {
            "back_pain": {"cervical", "neck", "الرقبة", "رقبة", "ملاريا", "الإيدز", "AIDS", "Malaria"},
            "headache": {"cervical", "خشونة", "فقرات الرقبة", "back pain", "ألم الظهر", "ضهر"},
            "neurology_vague": {"cervical", "فقرات الرقبة", "ألم الظهر", "ضهر"},
            "gynecology": {"كحة", "صدر", "chest", "cough", "cervical", "ملاريا", "AIDS"},
            "digestive": {"cervical", "فقرات الرقبة"},
            "urinary": {"cervical", "فقرات الرقبة"},
            "chest_pain": {"cervical", "فقرات الرقبة", "ألم الظهر", "معدة", "البطن", "throat", "urinary"},
        }
        if plan.domain == "chest_pain" and plan.route != EMERGENCY_MODE:
            heart_attack_terms = {"heart attack", "أزمة قلبية", "جلطة قلبية", "احتشاء"}
            if any(_contains_term(cleaned, term) for term in heart_attack_terms):
                return plan.answer
        if any(_contains_term(cleaned, term) for term in forbidden_by_domain.get(plan.domain, set())):
            return plan.answer
        if len([part for part in re.split(r"[؟?]", cleaned) if part.strip()]) > MAX_QUESTIONS + 1:
            return plan.answer
        return cleaned or plan.answer

    def _judge_answer(self, answer: str, plan: SafePlan, request: ChatRequest) -> str:
        deterministic = _deterministic_answer_review(answer, plan)
        if not _review_passed(deterministic):
            return deterministic.get("suggested_fix") or plan.answer

        if not _should_run_llm_judge(plan):
            return answer

        review_fn = getattr(self.llm_service, "review_final_answer", None)
        if not callable(review_fn):
            return answer
        try:
            llm_review = review_fn(
                message=request.message,
                history=request.history[-6:],
                language=plan.language,
                route=plan.route,
                domain=plan.domain,
                doctor_route=plan.doctor_route,
                follow_up_questions=plan.questions_to_ask,
                draft_answer=answer,
                forbidden_items=sorted(_forbidden_terms_for_domain(plan.domain)),
                red_flags=_red_flags_for_domain(plan.domain, plan.language),
                case_state=plan.case_state.to_dict() if plan.case_state else {},
            )
        except Exception:
            return answer
        if not isinstance(llm_review, dict):
            return answer
        if _review_passed(llm_review):
            return answer
        suggested = str(llm_review.get("suggested_fix") or "").strip()
        if suggested and _review_passed(_deterministic_answer_review(suggested, plan)):
            return self._verify_answer(suggested, plan)
        return plan.answer

    def _case_update(
        self,
        *,
        request: ChatRequest,
        response: ChatResponse,
        meaning: MessageMeaning | None,
        case_state: CaseState | None,
        engine_route: str,
    ) -> dict[str, Any]:
        return {
            "engine": "v2",
            "engine_route": engine_route,
            "source": request.source,
            "language": (meaning.language if meaning else request.language) or request.language or "ar",
            "known_symptoms": response.extracted_symptoms,
            "follow_up_questions": response.follow_up_questions,
            "current_meaning": meaning.to_dict() if meaning else {},
            "active_case": case_state.to_dict() if case_state else {},
            "medical_meaning": _legacy_medical_meaning_compat(meaning, case_state),
            "denied_concepts": sorted((case_state.denials if case_state else set()) - {"generic_no"}),
            "conversation_id": request.conversation_id or request.session_id,
            "mode": response.mode,
            "urgency_level": response.urgency_level,
        }


def understand_message(message: str, *, language: str | None = None, symptoms: list[str] | None = None) -> MessageMeaning:
    repaired = _repair_mojibake(message)
    lang = language or _detect_language(repaired)
    text = normalize_text(repaired)
    meaning = MessageMeaning(language=lang)
    symptoms = symptoms or []

    if _is_closing(repaired):
        meaning.intent = "closing"
        meaning.domain = "closing"
        return meaning
    if _is_abuse(text):
        meaning.intent = "abuse"
        meaning.domain = "abuse"
        return meaning
    if _is_offtopic(text):
        meaning.intent = "off_topic"
        meaning.domain = "off_topic"
        return meaning
    if _is_casual(text):
        meaning.intent = "casual"
        meaning.domain = "casual"
        return meaning
    if _is_nonsense(repaired):
        meaning.intent = "nonsense"
        meaning.domain = "nonsense"
        meaning.confidence = "low"
        return meaning

    meaning.facts.update(_extract_facts(repaired, lang))
    meaning.denials.update(_extract_denials(repaired))
    meaning.duration = meaning.facts.get("duration")
    meaning.severity = meaning.facts.get("severity")

    domain, body_parts, concept_symptoms = _detect_domain_and_symptoms(repaired, symptoms)
    meaning.domain = domain
    meaning.body_parts = body_parts
    meaning.symptoms = _dedupe(concept_symptoms + symptoms)
    meaning.intent = "followup_answer" if _is_followup_answer(meaning) else "medical_complaint"
    meaning.confidence = "high" if domain != "unknown" else "low"
    return meaning


def reconstruct_case_state(history: list[ChatMessage], *, default_language: str) -> CaseState:
    state = CaseState(language=default_language)
    for item in history[-10:]:
        if item.role == "user" and item.content:
            if _looks_like_mojibake(item.content):
                continue
            meaning = understand_message(item.content, language=_detect_language(item.content))
            if meaning.intent == "closing":
                state = CaseState(language=default_language, case_closed=True)
                continue
            if meaning.intent in {"abuse", "off_topic", "casual", "nonsense"}:
                continue
            if not state.active_domain or _should_start_new_case(state, meaning):
                state = CaseState(language=meaning.language, case_closed=False)
            _merge_meaning_into_case(state, meaning)
        elif item.role == "assistant" and item.content:
            _remember_assistant_questions(state, item.content)
    return state


def decide_and_update_case(
    history_state: CaseState,
    current: MessageMeaning,
    history: list[ChatMessage],
) -> CaseState:
    if current.intent in {"abuse", "off_topic", "casual", "nonsense", "closing"}:
        return history_state
    if not history_state.active_domain or history_state.case_closed:
        state = CaseState(language=current.language)
    elif _should_start_new_case(history_state, current):
        state = CaseState(language=current.language)
    else:
        state = history_state
    _merge_meaning_into_case(state, current)
    if _is_bare_negative(current) and state.active_domain:
        _apply_contextual_negative_answer(state)
    return state


def _merge_meaning_into_case(state: CaseState, meaning: MessageMeaning) -> None:
    if meaning.domain not in {"unknown", "closing", "abuse", "off_topic", "casual", "nonsense"}:
        state.active_domain = meaning.domain
    if meaning.body_parts:
        state.active_body_part = meaning.body_parts[0]
    state.language = meaning.language or state.language
    state.symptoms = _dedupe(state.symptoms + meaning.symptoms)
    state.denials.update(meaning.denials)
    state.facts.update({key: value for key, value in meaning.facts.items() if value not in (None, "", [])})
    if meaning.duration:
        state.duration = meaning.duration
        state.answered_questions.add("duration")
    if meaning.severity:
        state.severity = meaning.severity
    for fact_key in meaning.facts:
        state.answered_questions.add(fact_key)
    for denial in meaning.denials:
        state.answered_questions.add(denial)
    state.doctor_route = _doctor_for_domain(state.active_domain)


def _should_start_new_case(state: CaseState, meaning: MessageMeaning) -> bool:
    if meaning.domain in {"unknown", "closing", "abuse", "off_topic", "casual", "nonsense"}:
        return False
    if not state.active_domain:
        return True
    if meaning.domain == state.active_domain:
        return False
    if _is_followup_answer(meaning) and _followup_compatible_with_case(state, meaning):
        return False
    return True


def _v2_should_own(
    meaning: MessageMeaning,
    active_case: CaseState,
    previous_domain: str | None,
) -> bool:
    domain = active_case.active_domain or meaning.domain
    symptom_set = set(meaning.symptoms)
    if meaning.domain == "chest_pain" and "chest_pain" in meaning.denials:
        return False
    if meaning.domain == "urinary" and "dehydration" in symptom_set and {"diarrhoea", "vomiting", "dizziness"}.intersection(symptom_set):
        return False
    if domain in {"back_pain", "gynecology", "chest_pain", "throat_ent", "urinary"}:
        return True
    if meaning.domain == "neurology_vague":
        return True
    if meaning.domain == "headache" and set(meaning.symptoms).issubset({"headache"}):
        return True
    if meaning.domain == "headache" and previous_domain in {"back_pain", "neck_pain", "neurology_vague"}:
        return True
    if meaning.domain == "unknown" and previous_domain == "back_pain":
        return True
    return False


def _followup_compatible_with_case(state: CaseState, meaning: MessageMeaning) -> bool:
    if not state.active_domain:
        return False
    if state.active_domain == "gynecology" and (
        "age" in meaning.facts or "discharge" in meaning.symptoms or meaning.domain == "gynecology"
    ):
        return True
    if state.active_domain == "back_pain" and (
        meaning.facts.keys() & {"location", "duration", "injury"}
        or meaning.denials
        or meaning.domain in {"back_pain", "unknown"}
    ):
        return True
    if state.active_domain == "chest_pain" and (
        meaning.facts.keys() & {"duration", "severity", "chest_pressure", "chest_radiation", "cold_sweat", "breathlessness", "fainting"}
        or meaning.denials
        or meaning.domain in {"chest_pain", "unknown"}
    ):
        return True
    return bool(meaning.denials or meaning.duration or meaning.severity)


def _detect_domain_and_symptoms(message: str, classifier_symptoms: list[str]) -> tuple[str, list[str], list[str]]:
    text = normalize_text(_repair_mojibake(message))
    symptoms: list[str] = []
    body_parts: list[str] = []

    if _has_any(
        text,
        {
            "افرازات",
            "إفرازات",
            "البريود",
            "الدورة",
            "دوره",
            "period",
            "period pain",
            "menstrual",
            "cramps",
            "vaginal discharge",
            "discharge",
        },
    ):
        symptoms.extend(["gynecology_context"])
        if _has_any(text, {"افرازات", "إفرازات", "discharge"}):
            symptoms.append("discharge")
        body_parts.append("gynecology")
        return "gynecology", body_parts, symptoms
    if _has_any(text, {"رقبتي", "الرقبه", "الرقبة", "neck pain", "my neck hurts"}):
        symptoms.append("neck_pain")
        body_parts.append("neck")
        return "neck_pain", body_parts, symptoms
    if "chest_pain" in classifier_symptoms or _has_any(
        text,
        {
            "الم صدر",
            "ألم صدر",
            "الم في صدري",
            "ألم في صدري",
            "الم في الصدر",
            "ألم في الصدر",
            "صدري واجعني",
            "وجع صدري",
            "وجع في صدري",
            "chest pain",
            "my chest hurts",
            "chest pressure",
        },
    ):
        symptoms.append("chest_pain")
        if "breathlessness" in classifier_symptoms:
            symptoms.append("breathlessness")
        body_parts.append("chest")
        return "chest_pain", body_parts, symptoms
    if _has_any(text, {"صداع", "دماغي", "راسي", "رأسي", "headache", "head pain"}):
        symptoms.append("headache")
        body_parts.append("head")
        return "headache", body_parts, symptoms
    if _has_any(text, {"الاعصاب", "الأعصاب", "اعصابي", "أعصابي", "nerve", "nerves"}):
        symptoms.append("neuro_unclear")
        body_parts.append("neurology")
        return "neurology_vague", body_parts, symptoms
    if _has_any(text, {"ضهري", "ظهري", "الضهر", "الظهر", "back pain", "back-pain", "my back hurts", "lower back", "upper back"}):
        symptoms.append("back_pain")
        location = _extract_back_location(text)
        body_parts.append(location or "back")
        return "back_pain", body_parts, symptoms
    if _has_any(text, {"زوري", "زورى", "حلقي", "حلق", "throat hurts", "sore throat", "my throat hurts"}):
        symptoms.append("throat_irritation")
        body_parts.append("throat")
        return "throat_ent", body_parts, symptoms
    if _has_any(text, {"بطني", "بطن", "معدتي", "معدة", "stomach", "abdomen", "abdominal pain"}):
        symptoms.append("abdominal_pain")
        body_parts.append("abdomen")
        return "digestive", body_parts, symptoms
    if _has_any(text, {"حرقان بول", "حرقان في البول", "التبول", "بول", "urinary", "urine", "burning urination"}):
        symptoms.append("burning_micturition")
        body_parts.append("urinary")
        return "urinary", body_parts, symptoms
    if _has_any(text, {"جسمي واجعني", "تكسير", "body aches", "body ache", "my body aches"}):
        symptoms.append("muscle_pain")
        return "body_ache", ["body"], symptoms
    if {"cough", "mild_fever"}.intersection(set(classifier_symptoms)) or _has_any(text, {"كحة", "سخونية", "fever", "cough"}):
        symptoms.extend(item for item in classifier_symptoms if item not in symptoms)
        return "respiratory", ["respiratory"], symptoms
    return "unknown", [], classifier_symptoms


def _extract_facts(message: str, language: str) -> dict[str, Any]:
    text = normalize_text(_repair_mojibake(message))
    facts: dict[str, Any] = {}
    age_match = re.search(r"(\d{1,3})\s*(?:سنة|سنه|years?|yo)", text)
    if age_match:
        facts["age"] = int(age_match.group(1))
    duration_match = re.search(
        r"(\d+)\s*(?:ايام|أيام|يوم|days?|اسابيع|أسابيع|weeks?|ساعات|hours?)",
        text,
    )
    if duration_match:
        facts["duration"] = duration_match.group(0)
    elif word_duration_match := re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|hours?)\s+ago\b",
        text,
    ):
        facts["duration"] = word_duration_match.group(0)
    elif _has_any(text, {"من امبارح", "من أمس", "from yesterday"}):
        facts["duration"] = "من امبارح" if language != "en" else "from yesterday"
    location = _extract_back_location(text)
    if location:
        facts["location"] = location
    if _has_any(text, {"severe", "شديد", "جامد", "اوي", "قوي"}):
        facts["severity"] = "severe"
    if _has_any(text, {"ضغط في صدري", "ضغط على الصدر", "عصرة في صدري", "عصره في صدري", "pressure", "crushing"}):
        facts["chest_pressure"] = True
    if _has_any(text, {"ضيق تنفس", "ضيق نفس", "صعوبة تنفس", "مش قادر اتنفس", "shortness of breath", "difficulty breathing"}):
        facts["breathlessness"] = True
    if _has_any(text, {"عرق بارد", "تعرق بارد", "cold sweat", "sweating"}):
        facts["cold_sweat"] = True
    if _has_any(text, {"اغماء", "إغماء", "فقدان الوعي", "دوخة شديدة", "fainting", "near-fainting"}):
        facts["fainting"] = True
    if _has_any(text, {"دراعي الشمال", "يمتد للذراع", "واصل لدراعي", "الفك", "jaw", "arm", "radiating"}):
        facts["chest_radiation"] = True
    if _has_any(text, {"مريض قلب", "عندي مرض قلب", "جلطة قبل كده", "heart disease", "known heart disease"}):
        facts["known_heart_disease"] = True
    if _has_any(text, {"بعد مجهود", "بعد طلوع السلم", "after exertion", "with exertion"}):
        facts["exertional_chest_pain"] = True
    if _has_any(text, {"افرازات", "إفرازات", "discharge"}):
        facts["discharge_present"] = True
    if _has_any(text, {"حمل تقيل", "حاجة تقيلة", "heavy lifting", "injury", "اصابة", "إصابة"}):
        facts["injury_or_lifting_mentioned"] = True
    return facts


def _extract_denials(message: str) -> set[str]:
    text = normalize_text(_repair_mojibake(message))
    denials: set[str] = set()
    if not _has_negation_marker(text):
        return denials
    denials.add("generic_no")
    if _has_any(text, {"تنميل", "numb", "numbness", "tingling"}):
        denials.add("numbness")
    if _has_any(text, {"ضعف", "weakness", "weak"}):
        denials.add("weakness")
    if _has_any(text, {"بول", "براز", "bladder", "bowel"}):
        denials.add("bladder_bowel")
    if _has_any(text, {"الرجل", "leg", "go down", "radiat", "ينزل"}):
        denials.add("radiation")
    if _has_any(text, {"اصابة", "إصابة", "injury", "حمل", "lifting"}):
        denials.add("injury")
    if _has_any(text, {"حرارة", "fever"}):
        denials.add("fever")
    if _has_any(text, {"ضيق تنفس", "ضيق نفس", "صعوبة تنفس", "shortness of breath", "difficulty breathing"}):
        denials.add("breathlessness")
    if _has_any(text, {"ألم صدر", "الم صدر", "وجع صدر", "وجع في صدري", "صدري", "chest pain", "chest hurts"}):
        denials.add("chest_pain")
    if _has_any(text, {"عرق بارد", "cold sweat", "sweating"}):
        denials.add("cold_sweat")
    if _has_any(text, {"اغماء", "إغماء", "فقدان الوعي", "fainting"}):
        denials.add("fainting")
    if _has_any(text, {"دراعي", "الذراع", "الفك", "jaw", "arm", "radiat"}):
        denials.add("chest_radiation")
    return denials


def _has_negation_marker(text: str) -> bool:
    normalized = normalize_text(_repair_mojibake(text))
    phrase_markers = {
        "مفيش",
        "مافيش",
        "مش",
        "معهوش",
        "معهاش",
        "من غير",
        "بدون",
        "no",
        "not",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "without",
    }
    if any(marker in normalized for marker in phrase_markers):
        return True
    return bool(re.search(r"(?<![\w\u0600-\u06FF])(?:لا|لأ)(?![\w\u0600-\u06FF])", normalized))


def _missing_questions_for_back_case(case: CaseState, language: str) -> list[str]:
    facts = case.facts
    denials = case.denials
    questions_en = {
        "location": "Is it upper back or lower back?",
        "duration": "When did it start?",
        "duration_injury": "When did it start, and was it after injury, heavy lifting, sudden movement, or long sitting?",
        "injury": "Did it start after injury, heavy lifting, sudden movement, or long sitting?",
        "radiation_red_flags": "Does it go down to the leg, or come with numbness, weakness, or bladder/bowel control problems?",
        "red_flags": "Any numbness, weakness, or bladder/bowel control problems?",
    }
    questions_ar = {
        "location": "الألم في أسفل الظهر ولا أعلى الظهر؟",
        "duration": "بدأ من إمتى؟",
        "duration_injury": "بدأ من إمتى، وهل كان بعد إصابة أو حمل حاجة تقيلة أو قعدة طويلة؟",
        "injury": "هل بدأ بعد إصابة، حمل حاجة تقيلة، حركة مفاجئة، أو قعدة طويلة؟",
        "radiation_red_flags": "هل الألم بينزل على الرجل أو معاه تنميل/ضعف أو مشكلة في التحكم في البول أو البراز؟",
        "red_flags": "هل معاه ضعف أو مشكلة في التحكم في البول أو البراز؟" if "numbness" in denials else "هل معاه تنميل، ضعف، أو مشكلة في التحكم في البول أو البراز؟",
    }
    q = questions_en if language == "en" else questions_ar
    missing: list[str] = []
    if not facts.get("location"):
        missing.append(q["location"])
    injury_unknown = "injury" not in denials and not facts.get("injury_or_lifting_mentioned")
    if not case.duration and injury_unknown:
        missing.append(q["duration_injury"])
    elif not case.duration:
        missing.append(q["duration"])
    if case.duration and injury_unknown:
        missing.append(q["injury"])
    if "radiation" not in denials and not {"numbness", "weakness", "bladder_bowel"}.intersection(denials):
        missing.append(q["radiation_red_flags"])
    elif not {"weakness", "bladder_bowel"}.intersection(denials):
        missing.append(q["red_flags"])
    return _filter_answered_questions(missing, case)


def _missing_questions_for_neck_case(case: CaseState, language: str) -> list[str]:
    if language == "en":
        questions = [
            "When did the neck pain start?",
            "Does it spread to the shoulder or arm?",
            "Any numbness, weakness, fever, severe headache, or stiffness?",
        ]
    else:
        questions = [
            "ألم الرقبة بدأ من إمتى؟",
            "هل بينزل على الكتف أو الذراع؟",
            "هل معاه تنميل، ضعف، حرارة، صداع شديد، أو تيبس شديد؟",
        ]
    return _filter_answered_questions(questions, case)


def _missing_questions_for_chest_case(case: CaseState, language: str) -> list[str]:
    answered = case.denials | set(case.facts)
    if language == "en":
        questions: list[str] = []
        if "severity" not in case.facts and "chest_pressure" not in case.facts:
            questions.append("Is the chest pain severe, pressure-like, burning, sharp, or mild?")
        missing_red_flags = not {
            "breathlessness",
            "cold_sweat",
            "fainting",
            "chest_radiation",
        }.intersection(answered)
        if missing_red_flags:
            questions.append("Any shortness of breath, cold sweat, fainting, or pain spreading to the arm, jaw, or back?")
        if not case.duration:
            questions.append("When did it start, and did it start suddenly or after exertion?")
        return _filter_answered_questions(questions, case)

    questions = []
    if "severity" not in case.facts and "chest_pressure" not in case.facts:
        questions.append("الألم شديد أو ضاغط، ولا حارق/حاد/خفيف؟")
    missing_red_flags = not {
        "breathlessness",
        "cold_sweat",
        "fainting",
        "chest_radiation",
    }.intersection(answered)
    if missing_red_flags:
        questions.append("هل معاه ضيق نفس، عرق بارد، إغماء، أو بيمتد للذراع أو الفك أو الظهر؟")
    if not case.duration:
        questions.append("بدأ من إمتى، وهل بدأ فجأة أو بعد مجهود؟")
    return _filter_answered_questions(questions, case)


def _missing_questions_for_gynecology(case: CaseState, language: str) -> list[str]:
    if language == "en":
        questions = []
        if "discharge" in case.symptoms or case.facts.get("discharge_present"):
            questions.append("What is the discharge color, and does it have a bad smell, itching, or burning?")
            questions.append("Do you have pelvic/lower abdominal pain or fever?")
            if "pregnancy" not in case.denials and "pregnancy" not in case.facts:
                questions.append("Is there any possibility of pregnancy?")
        else:
            questions.append("Is the period pain unusual for you, and is your menstrual cycle late, heavy, or irregular?")
            questions.append("Any unusual discharge, bleeding, pelvic pain, fever, or itching?")
            if "pregnancy" not in case.denials and "pregnancy" not in case.facts:
                questions.append("Is there any possibility of pregnancy?")
        return _filter_answered_questions(questions, case)
    questions = []
    if "discharge" in case.symptoms or case.facts.get("discharge_present"):
        questions.append("لون الإفرازات إيه، وهل لها ريحة وحشة أو معاها حكة/حرقان؟")
        questions.append("هل في ألم أسفل البطن أو حرارة؟")
        if "pregnancy" not in case.denials and "pregnancy" not in case.facts:
            questions.append("هل في احتمال حمل؟")
    else:
        questions.append("هل الدورة متأخرة، غزيرة، أو مختلفة عن المعتاد؟")
        questions.append("هل في إفرازات غير معتادة، نزيف، ألم أسفل البطن، حرارة، أو حكة؟")
        if "pregnancy" not in case.denials and "pregnancy" not in case.facts:
            questions.append("هل في احتمال حمل؟")
    return _filter_answered_questions(questions, case)


def _missing_questions_for_headache(case: CaseState, language: str) -> list[str]:
    if language == "en":
        questions = [
            "Did the headache start suddenly or is it the worst headache you have felt?",
            "Any vision changes, vomiting, weakness, numbness, or confusion?",
            "When did it start and how severe is it?",
        ]
    else:
        questions = [
            "الصداع بدأ فجأة أو هو أسوأ صداع حسيت به؟",
            "هل معاه زغللة، قيء، ضعف، تنميل، أو لخبطة؟",
            "بدأ من إمتى وشدته قد إيه؟",
        ]
    return _filter_answered_questions(questions, case)


def _filter_answered_questions(questions: list[str], case: CaseState) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    answered_text = normalize_text(" ".join(case.answered_questions) + " " + " ".join(case.pending_questions))
    for question in questions:
        normalized = normalize_text(question)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized and normalized in answered_text:
            continue
        filtered.append(question)
    return filtered[:MAX_QUESTIONS]


def _remember_assistant_questions(case: CaseState, text: str) -> None:
    for part in re.split(r"[؟?]\s*", _repair_mojibake(text)):
        cleaned = part.strip()
        if not cleaned:
            continue
        case.pending_questions.append(cleaned)
        normalized = normalize_text(cleaned)
        if _has_any(normalized, {"امتى", "when", "duration", "started"}):
            case.answered_questions.add("duration_question")
        if _has_any(normalized, {"اسفل", "اعلى", "lower", "upper"}):
            case.answered_questions.add("location_question")
        if _has_any(normalized, {"تنميل", "numb", "weak", "ضعف", "بول", "براز", "bladder", "bowel"}):
            case.answered_questions.add("red_flags_question")
        if _has_any(normalized, {"ضيق نفس", "عرق بارد", "إغماء", "الذراع", "الفك", "الصدر", "shortness of breath", "cold sweat", "fainting", "jaw", "arm", "chest"}):
            case.answered_questions.add("chest_red_flags_question")


def _apply_contextual_negative_answer(case: CaseState) -> None:
    if case.active_domain == "back_pain":
        if "radiation" not in case.denials and case.facts.get("location"):
            case.denials.add("radiation")
            case.answered_questions.add("radiation")
        else:
            case.denials.update({"numbness", "weakness", "bladder_bowel"})
            case.answered_questions.update({"numbness", "weakness", "bladder_bowel"})


def _is_followup_answer(meaning: MessageMeaning) -> bool:
    if meaning.intent == "followup_answer":
        return True
    if meaning.domain == "unknown" and (meaning.facts or meaning.denials or _is_short_yes_no_text(meaning)):
        return True
    if meaning.facts.keys() & {"age", "duration", "location", "severity", "discharge_present"}:
        return True
    return False


def _is_bare_negative(meaning: MessageMeaning) -> bool:
    text_values = set(meaning.denials)
    return meaning.domain == "unknown" and not meaning.facts and not meaning.symptoms and bool(text_values)


def _is_short_yes_no_text(meaning: MessageMeaning) -> bool:
    return meaning.confidence == "low" and bool(meaning.denials)


def _has_respiratory_cluster(case: CaseState) -> bool:
    symptom_set = set(case.symptoms)
    return bool({"cough", "mild_fever", "high_fever", "phlegm", "runny_nose", "congestion"}.intersection(symptom_set))


def _has_chest_emergency_context(meaning: MessageMeaning) -> bool:
    facts = meaning.facts
    if meaning.domain != "chest_pain":
        return False
    def present(key: str) -> bool:
        return bool(facts.get(key)) and key not in meaning.denials

    if present("known_heart_disease"):
        return True
    if present("breathlessness") or present("cold_sweat") or present("fainting") or present("chest_radiation"):
        return True
    if facts.get("severity") == "severe" and (present("chest_pressure") or present("exertional_chest_pain")):
        return True
    if facts.get("severity") == "severe" and present("breathlessness"):
        return True
    return False


def _doctor_for_domain(domain: str | None) -> str:
    return {
        "back_pain": "Orthopedic doctor",
        "neck_pain": "Orthopedic doctor",
        "chest_pain": "Cardiologist",
        "gynecology": "Gynecologist",
        "headache": "General Practitioner",
        "neurology_vague": "General Practitioner",
        "throat_ent": "Needs more information",
        "digestive": "Needs more information",
        "urinary": "Urologist",
        "body_ache": "General Practitioner",
    }.get(domain or "", "Needs more information")


def _arabic_location(location: Any, is_back: bool) -> str:
    if location == "lower_back":
        return "ألم أسفل الظهر"
    if location == "upper_back":
        return "ألم أعلى الظهر"
    return "ألم الظهر" if is_back else "ألم الرقبة"


def _extract_back_location(text: str) -> str | None:
    if _has_any(text, {"lower back", "اسفل الظهر", "أسفل الظهر", "اسفل الضهر", "أسفل الضهر"}):
        return "lower_back"
    if _has_any(text, {"upper back", "اعلى الظهر", "أعلى الظهر", "اعلى الضهر", "أعلى الضهر"}):
        return "upper_back"
    return None


def _detect_language(message: str, hint: str | None = None) -> str:
    if hint in {"ar", "en", "mixed"}:
        return hint
    repaired = _repair_mojibake(message or "")
    arabic = len(re.findall(r"[\u0600-\u06FF]", repaired))
    latin = len(re.findall(r"[A-Za-z]", repaired))
    if latin > 0 and arabic == 0:
        return "en"
    if latin > 0 and arabic > 0:
        return "mixed"
    return "ar"


def _closing_answer(language: str) -> str:
    return "You’re welcome. Wishing you good health." if language == "en" else "العفو، أتمنى لك الصحة والسلامة."


def _is_closing(text: str) -> bool:
    normalized = normalize_text(_repair_mojibake(text))
    return normalized in {"شكرا", "شكرًا", "متشكر", "تسلم", "thanks", "thank you", "bye", "goodbye"}


def _is_abuse(text: str) -> bool:
    return _has_any(text, {"كسم", "fuck", "shit", "asshole", "bitch"})


def _is_offtopic(text: str) -> bool:
    return _has_any(
        text,
        {
            "who are you",
            "what are you",
            "what can you do",
            "what can u do",
            "what do you do",
            "امك عاملة اي",
            "امك عامله اي",
            "ابوك عامل اي",
            "how is your mom",
            "how is your mother",
            "your mom",
        },
    )


def _is_casual(text: str) -> bool:
    return normalize_text(text) in {"اهلا", "هاي", "hi", "hello", "عامل اي", "ازيك", "اي الدنيا"}


def _is_nonsense(text: str) -> bool:
    stripped = re.sub(r"\s+", "", _repair_mojibake(text or ""))
    if len(stripped) <= 1:
        return True
    return len(set(stripped.lower())) <= 2 and len(stripped) >= 4


def _has_any(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(_repair_mojibake(text))
    return any(normalize_text(_repair_mojibake(term)) in normalized for term in terms)


def _contains_term(text: str, term: str) -> bool:
    return normalize_text(term) in normalize_text(text)


def _forbidden_terms_for_domain(domain: str) -> set[str]:
    return {
        "back_pain": {"cervical", "neck", "الرقبة", "فقرات الرقبة", "ملاريا", "الإيدز", "AIDS", "Malaria"},
        "headache": {"cervical", "فقرات الرقبة", "ألم الظهر", "back pain"},
        "neurology_vague": {"cervical", "فقرات الرقبة", "ألم الظهر", "back pain"},
        "gynecology": {"كحة", "chest", "cough", "cervical", "ملاريا", "AIDS"},
        "digestive": {"cervical", "فقرات الرقبة"},
        "urinary": {"cervical", "فقرات الرقبة", "ألم الصدر", "chest pain"},
        "chest_pain": {"cervical", "فقرات الرقبة", "ألم الظهر", "معدة", "البطن", "throat", "urinary"},
    }.get(domain, set())


def _red_flags_for_domain(domain: str, language: str) -> list[str]:
    red_flags_en = {
        "back_pain": [
            "leg weakness",
            "saddle/private-area numbness",
            "loss of bladder or bowel control",
            "severe trauma",
            "fever with severe back pain",
            "rapidly worsening pain",
        ],
        "throat_ent": [
            "breathing difficulty",
            "inability to swallow saliva",
            "severe throat/neck swelling",
            "high persistent fever",
            "stiff neck",
        ],
        "digestive": [
            "severe or worsening abdominal pain",
            "persistent vomiting",
            "blood in vomit or stool",
            "rigid abdomen",
            "fainting",
            "pregnancy possibility with severe pain or bleeding",
        ],
        "urinary": [
            "fever with flank/back pain",
            "blood in urine",
            "inability to urinate",
            "pregnancy with urinary symptoms",
            "severe side/flank pain",
        ],
        "gynecology": [
            "heavy bleeding",
            "severe pelvic pain",
            "fever",
            "foul-smelling discharge",
            "fainting or severe dizziness",
            "pregnancy possibility with pain or bleeding",
        ],
        "headache": [
            "sudden worst headache",
            "weakness or numbness",
            "speech or vision changes",
            "fever with neck stiffness",
            "confusion",
            "repeated vomiting",
            "after head injury",
        ],
        "body_ache": [
            "very high fever",
            "difficulty breathing",
            "confusion",
            "stiff neck",
            "severe weakness",
            "dehydration",
            "persistent worsening symptoms",
        ],
        "chest_pain": [
            "shortness of breath",
            "cold sweat",
            "fainting",
            "pain spreading to arm, jaw, or back",
            "severe pressure-like pain",
            "known heart disease",
        ],
    }
    if language == "en":
        return red_flags_en.get(domain, [])
    red_flags_ar = {
        "back_pain": ["ضعف في الرجل", "تنميل في منطقة حساسة", "فقدان التحكم في البول أو البراز", "إصابة قوية", "حرارة مع ألم ظهر شديد", "تدهور سريع"],
        "throat_ent": ["صعوبة تنفس", "عدم القدرة على بلع الريق", "تورم شديد", "حرارة عالية مستمرة", "تيبس الرقبة"],
        "digestive": ["ألم شديد أو متزايد", "قيء مستمر", "دم في القيء أو البراز", "تحجر البطن", "إغماء", "احتمال حمل مع ألم شديد أو نزيف"],
        "urinary": ["حرارة مع ألم جنب/ظهر", "دم في البول", "عدم القدرة على التبول", "حمل مع أعراض بولية", "ألم جنب شديد"],
        "gynecology": ["نزيف غزير", "ألم حوض شديد", "حرارة", "إفرازات برائحة كريهة", "إغماء أو دوخة شديدة", "احتمال حمل مع ألم أو نزيف"],
        "headache": ["صداع مفاجئ شديد", "ضعف أو تنميل", "تغير الكلام أو النظر", "حرارة مع تيبس الرقبة", "لخبطة", "قيء متكرر", "بعد إصابة رأس"],
        "body_ache": ["حرارة عالية جدًا", "صعوبة تنفس", "لخبطة", "تيبس رقبة", "ضعف شديد", "جفاف", "تدهور مستمر"],
        "chest_pain": ["ضيق نفس", "عرق بارد", "إغماء", "امتداد الألم للذراع أو الفك أو الظهر", "ألم ضاغط شديد", "مرض قلب معروف"],
    }
    return red_flags_ar.get(domain, [])


def _deterministic_answer_review(answer: str, plan: SafePlan) -> dict[str, Any]:
    cleaned = answer or ""
    forbidden = _forbidden_terms_for_domain(plan.domain)
    has_forbidden = any(_contains_term(cleaned, term) for term in forbidden)
    repeats_questions = any(question and cleaned.count(question) > 1 for question in plan.questions_to_ask)
    too_many_questions = len([part for part in re.split(r"[؟?]", cleaned) if part.strip()]) > MAX_QUESTIONS + 1
    wrong_emergency = plan.route == EMERGENCY_MODE and "emergency" not in cleaned.lower() and "طوارئ" not in cleaned
    chest_overdiagnosis = (
        plan.domain == "chest_pain"
        and plan.route != EMERGENCY_MODE
        and any(_contains_term(cleaned, term) for term in {"heart attack", "أزمة قلبية", "جلطة قلبية", "احتشاء"})
    )
    is_safe = not (has_forbidden or wrong_emergency or chest_overdiagnosis)
    is_relevant = not has_forbidden
    is_consistent = not (has_forbidden or chest_overdiagnosis)
    return {
        "is_safe": is_safe,
        "is_relevant": is_relevant,
        "is_consistent_with_user": is_consistent,
        "has_unrelated_body_part_or_disease": has_forbidden,
        "repeats_answered_questions": repeats_questions or too_many_questions,
        "uses_wrong_emergency_level": wrong_emergency,
        "uses_static_or_robotic_style": False,
        "suggested_fix": plan.answer if not (is_safe and is_relevant and is_consistent and not repeats_questions and not too_many_questions) else "",
        "reason": "deterministic V2 final answer review",
    }


def _review_passed(review: dict[str, Any]) -> bool:
    return (
        bool(review.get("is_safe", True))
        and bool(review.get("is_relevant", True))
        and bool(review.get("is_consistent_with_user", True))
        and not bool(review.get("has_unrelated_body_part_or_disease", False))
        and not bool(review.get("repeats_answered_questions", False))
        and not bool(review.get("uses_wrong_emergency_level", False))
    )


def _should_run_llm_judge(plan: SafePlan) -> bool:
    if plan.route in {EMERGENCY_MODE, CLOSING_MODE}:
        return False
    if plan.domain in {"abuse", "off_topic", "casual", "nonsense", "closing"}:
        return False
    return plan.route in {CLARIFICATION_MODE, DIAGNOSIS_MODE}


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _strip_duplicate_questions(answer: str, questions: list[str]) -> str:
    cleaned = answer or ""
    for question in questions:
        if question:
            cleaned = cleaned.replace(question, "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([؟?.,،])", r"\1", cleaned).strip()
    return cleaned


def _repair_mojibake(text: str) -> str:
    if not _looks_like_mojibake(text):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    return ("Ø" in text or "Ù" in text) and not any("\u0600" <= char <= "\u06FF" for char in text)


def _english_emergency_reason(reason: str) -> str:
    if not any("\u0600" <= char <= "\u06FF" for char in reason):
        return reason
    return "The message includes red-flag symptoms that need urgent assessment."


def _legacy_medical_meaning_compat(
    meaning: MessageMeaning | None,
    case_state: CaseState | None,
) -> dict[str, Any]:
    domain = (case_state.active_domain if case_state else None) or (meaning.domain if meaning else None)
    body_part = case_state.active_body_part if case_state else None
    body_parts: list[str] = []
    if body_part == "lower_back":
        body_parts = ["back", "lower_back"]
    elif body_part == "upper_back":
        body_parts = ["back", "upper_back"]
    elif body_part:
        body_parts = [body_part]
    elif meaning:
        body_parts = meaning.body_parts
    domain_map = {
        "back_pain": "musculoskeletal_back_pain",
        "neck_pain": "musculoskeletal_neck_pain",
        "chest_pain": "heart_chest",
        "gynecology": "reproductive",
        "digestive": "digestive_abdominal",
        "throat_ent": "throat_ent",
        "urinary": "urinary",
        "headache": "headache",
        "neurology_vague": "neurology_vague",
        "body_ache": "body_ache",
    }
    return {
        "language": (meaning.language if meaning else None) or (case_state.language if case_state else "ar"),
        "domain": domain_map.get(domain or "", domain),
        "body_parts": body_parts,
        "symptoms": (case_state.symptoms if case_state else []) or (meaning.symptoms if meaning else []),
        "red_flags": meaning.red_flags if meaning else [],
        "denied": sorted(((case_state.denials if case_state else set()) | (meaning.denials if meaning else set())) - {"generic_no"}),
    }
