from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

from .chat_engine_v2 import (
    CLARIFICATION_MODE,
    CLOSING_MODE,
    DIAGNOSIS_MODE,
    EMERGENCY_MODE,
    CaseState,
    MessageMeaning,
    _dedupe,
    _detect_language,
    _has_any,
    _is_abuse,
    _is_casual,
    _is_closing,
    _is_nonsense,
    _is_offtopic,
    _legacy_medical_meaning_compat,
    _looks_like_mojibake,
    _repair_mojibake,
    understand_message,
)
from .display_labels import display_diagnosis_ar, display_doctor_ar
from .llm_service import HIGH_URGENCY_PREFIX
from .safety import EmergencySignal, choose_urgency, detect_emergency_signal, has_red_flags, normalize_text
from .schemas import ChatMessage, ChatRequest, ChatResponse


logger = logging.getLogger(__name__)

V3_OWNED_DOMAINS = {"universal"}
NON_MEDICAL_INTENTS = {"abuse", "off_topic", "casual", "greeting", "nonsense"}
PLANNER_SKIP_INTENTS = set()
JUDGE_SKIP_INTENTS = {"casual", "greeting", "off_topic", "abuse", "closing", "nonsense"}
MAX_V3_QUESTIONS = 1
V2_COMPAT_DOMAINS = {"back_pain", "gynecology", "headache", "neurology_vague", "chest_pain"}

INTERNAL_LEAK_PATTERNS = (
    "provide a more detailed response",
    "ask a new question to gather more information",
    "classifier",
    "retrieved cases",
    "rag",
    "json",
    "internal",
    "prompt",
)


DOMAIN_PROFILES: dict[str, dict[str, str]] = {
    "throat_ent": {
        "doctor": "ENT Specialist",
        "ar_name": "ألم الحلق أو الزور",
        "en_name": "throat symptoms",
        "ar_base": "فاهم إن المشكلة في الزور. نحتاج نحدد هل الموضوع التهاب بسيط ولا فيه علامات محتاجة كشف أسرع.",
        "en_base": "I understand this is focused on your throat. We need to clarify whether it looks like a simple irritation or needs faster medical care.",
        "ar_question": "هل مع وجع الزور حرارة، كحة، صعوبة في البلع أو التنفس، أو تورم في الرقبة؟",
        "en_question": "Do you also have fever, cough, trouble swallowing or breathing, or neck swelling?",
    },
    "digestive": {
        "doctor": "Gastroenterologist",
        "ar_name": "أعراض هضمية أو ألم بالبطن",
        "en_name": "digestive or abdominal symptoms",
        "ar_base": "ألم البطن محتاج نعرف مكانه وطبيعته قبل أي استنتاج.",
        "en_base": "Abdominal pain needs a bit more context before guessing a cause.",
        "ar_question": "الألم فين بالظبط، وهل معاه قيء، إسهال، حرارة، أو دم؟",
        "en_question": "Where exactly is the pain, and is there vomiting, diarrhea, fever, or blood?",
    },
    "side_pain": {
        "doctor": "General Practitioner",
        "ar_name": "ألم في الجنب",
        "en_name": "side pain",
        "ar_base": "ألم الجنب محتاج نحدد مكانه وطبيعته وهل مرتبط بالبول أو البطن أو الحركة.",
        "en_base": "Side pain needs context about exact location, urinary symptoms, abdominal symptoms, and movement.",
        "ar_question": "الألم في الجنب مستمر ولا بييجي ويروح، وهل معاه حرقان بول، حرارة، قيء، أو ألم بطن؟",
        "en_question": "Is the side pain constant or coming and going, and is there urinary burning, fever, vomiting, or abdominal pain?",
    },
    "urinary": {
        "doctor": "Urologist",
        "ar_name": "أعراض بولية",
        "en_name": "urinary symptoms",
        "ar_base": "أعراض البول زي الحرقان أو التكرار محتاجة تفاصيل عن الحرارة والألم ووجود دم.",
        "en_base": "Urinary symptoms need context about fever, pain, and blood in the urine.",
        "ar_question": "هل في حرارة، ألم في الجنب أو الظهر، دم في البول، أو تكرار شديد في التبول؟",
        "en_question": "Is there fever, side/flank pain, blood in urine, or very frequent urination?",
    },
    "respiratory": {
        "doctor": "Pulmonologist",
        "ar_name": "أعراض تنفسية",
        "en_name": "respiratory symptoms",
        "ar_base": "الأعراض التنفسية محتاجة نعرف مدتها وهل فيها ضيق نفس أو ألم صدر.",
        "en_base": "Respiratory symptoms need duration and red-flag clarification.",
        "ar_question": "بقالها قد إيه، وهل في ضيق نفس، ألم صدر، صفير، أو حرارة عالية؟",
        "en_question": "How long has this been going on, and is there shortness of breath, chest pain, wheezing, or high fever?",
    },
    "skin": {
        "doctor": "Dermatologist",
        "ar_name": "مشكلة جلدية",
        "en_name": "skin symptoms",
        "ar_base": "الطفح أو الحكة محتاجين نعرف المكان والانتشار وأي حساسية أو تورم.",
        "en_base": "Skin rash or itching needs location, spread, and allergy context.",
        "ar_question": "الطفح فين، وهل بينتشر أو معاه تورم في الشفاه/الوجه أو ضيق نفس؟",
        "en_question": "Where is the rash, and is it spreading or linked with face/lip swelling or breathing trouble?",
    },
    "eye": {
        "doctor": "Ophthalmologist",
        "ar_name": "أعراض بالعين أو النظر",
        "en_name": "eye or vision symptoms",
        "ar_base": "تغير النظر أو ألم العين محتاج توضيح سريع لأن بعض الحالات تحتاج كشف عيون.",
        "en_base": "Eye pain or vision change needs careful clarification because some cases need eye assessment.",
        "ar_question": "هل التغير في عين واحدة ولا الاتنين، وهل في ألم شديد، احمرار، أو فقدان مفاجئ للنظر؟",
        "en_question": "Is it one eye or both, and is there severe pain, redness, or sudden vision loss?",
    },
    "dental": {
        "doctor": "Dentist",
        "ar_name": "ألم أسنان أو لثة",
        "en_name": "dental symptoms",
        "ar_base": "ألم الأسنان غالبا يحتاج تقييم أسنان، خصوصا لو فيه تورم أو حرارة.",
        "en_base": "Tooth pain often needs dental assessment, especially with swelling or fever.",
        "ar_question": "هل في تورم في الخد أو اللثة، حرارة، أو صعوبة في فتح الفم؟",
        "en_question": "Is there cheek/gum swelling, fever, or trouble opening your mouth?",
    },
    "mental_health": {
        "doctor": "Psychiatrist",
        "ar_name": "صحة نفسية",
        "en_name": "mental health concern",
        "ar_base": "حاسس بيك. نقدر نتكلم عن القلق أو الضيق خطوة خطوة، مع الانتباه لأي خطر على سلامتك.",
        "en_base": "I hear you. We can talk through anxiety or distress step by step while checking safety.",
        "ar_question": "هل عندك أفكار لإيذاء نفسك أو إنك مش قادر تكون آمن دلوقتي؟",
        "en_question": "Are you having thoughts of harming yourself or feeling unable to stay safe right now?",
    },
    "endocrine": {
        "doctor": "Endocrinologist",
        "ar_name": "أعراض سكر أو غدد",
        "en_name": "endocrine or sugar-related symptoms",
        "ar_base": "أعراض زي العطش الشديد أو الجوع أو الرعشة محتاجة نربطها بالأكل وقياس السكر إن وجد.",
        "en_base": "Symptoms like thirst, hunger, sweating, or shaking need context around meals and glucose readings if available.",
        "ar_question": "هل عندك قياس سكر قريب، وهل الأعراض بتحصل مع جوع شديد أو تعرق أو رعشة؟",
        "en_question": "Do you have a recent glucose reading, and do symptoms happen with strong hunger, sweating, or shaking?",
    },
    "cardiology": {
        "doctor": "Cardiologist",
        "ar_name": "أعراض ضغط أو قلب",
        "en_name": "blood pressure or cardiovascular symptoms",
        "ar_base": "ارتفاع الضغط مع زغللة أو صداع يحتاج متابعة حذرة، خصوصا لو القراءة عالية أو الأعراض جديدة.",
        "en_base": "High blood pressure with vision change or headache needs careful assessment, especially if readings are high or symptoms are new.",
        "ar_question": "هل قست الضغط؟ ولو آه، القراءة كانت كام؟ وهل في ألم صدر، ضيق نفس، صداع شديد، أو تنميل؟",
        "en_question": "Did you measure your blood pressure, and if so what was the reading? Any chest pain, breathlessness, severe headache, or numbness?",
    },
    "pediatrics": {
        "doctor": "Pediatrician",
        "ar_name": "أعراض عند طفل",
        "en_name": "child symptoms",
        "ar_base": "لأن الشكوى تخص طفل، الأفضل نقيّم السن ودرجة الحرارة والنشاط العام بحذر.",
        "en_base": "Because this is about a child, age, temperature, feeding, and activity level matter.",
        "ar_question": "سن الطفل كام، والحرارة كام بالقياس، وهل نشاطه وأكله/شربه طبيعي؟",
        "en_question": "How old is the child, what is the measured temperature, and are activity and drinking/feeding normal?",
    },
    "vestibular_ent": {
        "doctor": "ENT specialist",
        "ar_name": "دوخة أو طنين بالأذن",
        "en_name": "dizziness or ear ringing",
        "ar_base": "الدوخة مع طنين أو إحساس إن الدنيا بتلف ممكن تكون مرتبطة بالأذن الداخلية أو الاتزان، وتحتاج توضيح للعلامات العصبية.",
        "en_base": "Dizziness with ringing or spinning can relate to the inner ear or balance system, while neurological warning signs still matter.",
        "ar_question": "هل الطنين في ودن واحدة ولا الاتنين، وهل في ضعف سمع، قيء متكرر، تنميل/ضعف، أو زغللة؟",
        "en_question": "Is the ringing in one ear or both, and is there hearing loss, repeated vomiting, numbness/weakness, or vision change?",
    },
    "infectious": {
        "doctor": "General Practitioner",
        "ar_name": "حرارة أو عدوى محتملة",
        "en_name": "fever or possible infection",
        "ar_base": "السخونية لوحدها عرض عام، ومحتاجين نعرف درجتها ومدتها وباقي الأعراض قبل أي استنتاج.",
        "en_base": "Fever by itself is broad, so the temperature, duration, and associated symptoms matter before guessing a cause.",
        "ar_question": "الحرارة كام بالقياس، وبقالها قد إيه؟ وهل في أي أعراض عدوى معها زي كحة، ألم حلق، ألم بطن، حرقان بول، طفح، أو تدهور عام؟",
        "en_question": "What is the measured temperature and duration? Any cough, sore throat, abdominal pain, urinary burning, rash, or worsening general condition?",
    },
    "body_ache": {
        "doctor": "General Practitioner",
        "ar_name": "تكسير أو ألم عام بالجسم",
        "en_name": "body aches",
        "ar_base": "وجع الجسم أو تكسير الجسم عرض عام وممكن يحصل مع إجهاد أو عدوى بسيطة، فالأهم نعرف الحرارة والمدة.",
        "en_base": "Body aches are general and can happen with fatigue or infection, so duration and fever matter.",
        "ar_question": "هل في سخونية أو كحة أو التهاب حلق، وبقال التكسير قد إيه؟",
        "en_question": "Is there fever, cough, or sore throat, and how long have the body aches been present?",
    },
    "neck_pain": {
        "doctor": "Orthopedic doctor",
        "ar_name": "ألم رقبة",
        "en_name": "neck pain",
        "ar_base": "ألم الرقبة يحتاج نعرف هل هو شد عضلي بسيط ولا معاه علامات عصبية.",
        "en_base": "Neck pain needs context to separate muscle strain from neurological warning signs.",
        "ar_question": "هل الألم نازل للكتف أو الذراع، أو معاه تنميل، ضعف، حرارة، أو تيبس شديد؟",
        "en_question": "Does it spread to the shoulder or arm, or come with numbness, weakness, fever, or severe stiffness?",
    },
    "orthopedics": {
        "doctor": "Orthopedic doctor",
        "ar_name": "إصابة أو ألم بالعظام/المفاصل",
        "en_name": "bone, joint, or injury symptoms",
        "ar_base": "لو في كسر أو إصابة، مهم نعرف مكانها وشدة الألم وهل في تورم أو عدم قدرة على تحريك الطرف.",
        "en_base": "For a possible fracture or injury, location, swelling, deformity, and ability to move or bear weight matter.",
        "ar_question": "الإصابة فين بالظبط، وهل في تورم شديد، شكل غير طبيعي، نزيف، أو عدم قدرة على تحريك المكان؟",
        "en_question": "Where exactly is the injury, and is there severe swelling, deformity, bleeding, or inability to move the area?",
    },
    "unknown": {
        "doctor": "General Practitioner",
        "ar_name": "عرض غير واضح",
        "en_name": "unclear symptom",
        "ar_base": "محتاج أفهم العرض الأساسي بشكل أوضح عشان أساعدك بأمان.",
        "en_base": "I need to understand the main symptom more clearly so I can help safely.",
        "ar_question": "إيه أكتر عرض مضايقك، وفين مكانه؟",
        "en_question": "What is the main symptom bothering you, and where is it located?",
    },
}


@dataclass
class ClinicalCaseState:
    case_id: str | None = None
    active_domain: str | None = None
    active_body_part: str | None = None
    symptoms: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    denials: set[str] = field(default_factory=set)
    answered_questions: set[str] = field(default_factory=set)
    asked_question_slots: set[str] = field(default_factory=set)
    pending_questions: list[str] = field(default_factory=list)
    asked_question_ids: set[str] = field(default_factory=set)
    answered_question_ids: set[str] = field(default_factory=set)
    last_question_id: str | None = None
    pending_question_ids: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)
    risk_level: str = "none"
    doctor_route: str | None = None
    last_assistant_answer: str = ""
    language: str = "ar"
    case_closed: bool = False
    paused: bool = False
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "active_domain": self.active_domain,
            "domain": self.active_domain,
            "active_body_part": self.active_body_part,
            "body_parts": [self.active_body_part] if self.active_body_part else [],
            "symptoms": self.symptoms,
            "known_facts": self.facts,
            "facts": self.facts,
            "denials": sorted(self.denials),
            "denied_facts": sorted(self.denials),
            "answered_questions": sorted(self.answered_questions),
            "asked_question_slots": sorted(self.asked_question_slots),
            "pending_questions": self.pending_questions,
            "asked_question_ids": sorted(self.asked_question_ids),
            "answered_question_ids": sorted(self.answered_question_ids),
            "last_question_id": self.last_question_id,
            "pending_question_ids": self.pending_question_ids,
            "contradictions": self.contradictions,
            "risk_signals": self.risk_signals,
            "risk_level": self.risk_level,
            "doctor_route": self.doctor_route,
            "language": self.language,
            "case_closed": self.case_closed,
            "paused": self.paused,
            "status": self.status,
        }


@dataclass
class ClinicalPlan:
    intent: str
    case_action: str
    domain: str
    clinical_summary: str
    new_facts: list[str] = field(default_factory=list)
    new_denials: list[str] = field(default_factory=list)
    risk_level: str = "low"
    risk_reasons: list[str] = field(default_factory=list)
    broad_possibilities: list[str] = field(default_factory=list)
    candidate_conditions: list[dict[str, Any]] = field(default_factory=list)
    selected_possibilities: list[str] = field(default_factory=list)
    analysis_for_patient: str = ""
    patient_answer: str = ""
    diagnosis: str | None = None
    confidence: float = 0.0
    next_question_id: str | None = None
    next_best_question: str | None = None
    optional_second_question: str | None = None
    care_guidance: list[str] = field(default_factory=list)
    doctor_route: str | None = None
    response_goal: str = "clarify"
    must_not_repeat: list[str] = field(default_factory=list)
    forbidden_topics: list[str] = field(default_factory=list)
    deterministic_override: bool = False
    source: str = "deterministic"

    @property
    def questions(self) -> list[str]:
        return [q for q in [self.next_best_question, self.optional_second_question] if q]


class ChatEngineV3:
    """Conversation-first clinical reasoning engine.

    V3 establishes the active clinical case before classifier/RAG evidence is
    considered. It keeps deterministic emergency handling in front, lets the LLM
    propose a structured turn plan when available, validates that plan, then
    performs a final deterministic safety pass before returning the API schema.
    """

    def __init__(self, *, classifier_service: Any, knowledge_service: Any, rag_service: Any, llm_service: Any):
        self.classifier_service = classifier_service
        self.knowledge_service = knowledge_service
        self.rag_service = rag_service
        self.llm_service = llm_service

    def _new_trace(self) -> dict[str, Any]:
        return {
            "_started_at": time.perf_counter(),
            "engine": "v3_universal",
            "planner_called": False,
            "planner_valid": False,
            "planner_used": False,
            "planner_fallback": False,
            "intent": None,
            "case_action": None,
            "classifier_used": False,
            "rag_used": False,
            "judge_called": False,
            "judge_approved": False,
            "judge_used": False,
            "judge_failed": False,
            "judge_fallback": False,
            "deterministic_override": False,
            "planner_latency_ms": None,
            "response_generation_latency_ms": None,
            "judge_latency_ms": None,
            "total_latency_ms": None,
        }

    def _finalize_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(trace)
        started_at = finalized.pop("_started_at", None)
        if finalized.get("total_latency_ms") is None and started_at:
            finalized["total_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return finalized

    def _empty_response(self, request: ChatRequest) -> ChatResponse:
        language = _detect_language(request.message, request.language)
        answer = (
            "Please write a symptom or medical question so I can help safely."
            if language == "en"
            else "اكتب العرض أو السؤال الطبي بوضوح عشان أقدر أساعدك بأمان."
        )
        doctor = "Not needed"
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=CLARIFICATION_MODE,
            answer=answer,
            extracted_symptoms=[],
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        trace = self._new_trace()
        trace["_started_at"] = time.perf_counter()
        trace.update({"intent": "empty", "case_action": "none", "planner_fallback": True})
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=None,
            case_state=ClinicalCaseState(language=language),
            plan=ClinicalPlan("empty", "none", "non_medical", "Empty message.", doctor_route=doctor),
            trace=trace,
            engine_route="v3_empty",
        )
        return response

    def _run_universal_planner(
        self,
        *,
        request: ChatRequest,
        recent_history: list[ChatMessage],
        language: str,
        active_case: ClinicalCaseState,
        deterministic_plan: ClinicalPlan,
        symptoms: list[str],
        trace: dict[str, Any],
        support_evidence: dict[str, Any] | None = None,
    ) -> ClinicalPlan:
        trace["intent"] = deterministic_plan.intent
        trace["case_action"] = deterministic_plan.case_action
        plan = deterministic_plan
        planner = getattr(self.llm_service, "plan_clinical_turn", None)
        if not callable(planner) or deterministic_plan.intent in PLANNER_SKIP_INTENTS:
            trace["planner_fallback"] = True
            return plan

        trace["planner_called"] = True
        try:
            planner_started = time.perf_counter()
            llm_plan = planner(
                message=request.message,
                history=recent_history,
                language=language,
                active_case=active_case.to_dict(),
                deterministic_plan=self._plan_to_dict(deterministic_plan),
                classifier_evidence=support_evidence or {"extracted_symptoms": symptoms},
                previous_assistant=active_case.last_assistant_answer,
            )
            trace["planner_latency_ms"] = round((time.perf_counter() - planner_started) * 1000, 2)
            planner_meta = getattr(self.llm_service, "last_call_meta", {}).get("planner", {})
            if planner_meta:
                trace["planner_selected_model"] = planner_meta.get("selected_model")
                trace["planner_key_index"] = planner_meta.get("key_index")
                trace["planner_model_fallback_used"] = bool(planner_meta.get("used_fallback_model"))
                trace["planner_provider_status"] = planner_meta.get("status")
        except Exception as exc:
            logger.warning("V3 universal planner failed safely: %s", exc.__class__.__name__)
            llm_plan = None

        validated = self._validate_llm_plan(llm_plan, deterministic_plan, active_case)
        if validated:
            plan = validated
            trace["planner_valid"] = True
            trace["planner_used"] = True
        else:
            trace["planner_fallback"] = True
        trace["intent"] = plan.intent
        trace["case_action"] = plan.case_action
        return plan

    def _collect_supporting_evidence(
        self,
        message: str,
        symptoms: list[str],
        case_state: ClinicalCaseState,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {"extracted_symptoms": symptoms, "classifier_top": [], "rag_cases": []}
        domain = case_state.active_domain or "unknown"
        if domain in {"unknown", "casual", "off_topic", "abuse", "nonsense", "closing"}:
            return evidence

        if symptoms:
            try:
                top = self.classifier_service.predict_top_k(symptoms, k=5)
                evidence["classifier_top"] = [
                    {"diagnosis": item.diagnosis, "confidence": item.confidence}
                    for item in top[:5]
                ]
                trace["classifier_used"] = bool(evidence["classifier_top"])
            except Exception:
                evidence["classifier_top"] = []

        if symptoms and len(normalize_text(message).split()) >= 2:
            try:
                retrieved = self.rag_service.retrieve(message)
                filtered = self.rag_service.filter_evidence(retrieved, urgency_level="Low")
                evidence["rag_cases"] = [
                    {
                        "category": case.get("category"),
                        "score": case.get("score", 0.0),
                    }
                    for case in filtered[:3]
                ]
                trace["rag_used"] = bool(evidence["rag_cases"])
            except Exception:
                evidence["rag_cases"] = []
        return evidence

    def _offline_compat_response(
        self,
        *,
        request: ChatRequest,
        fallback_handler: Callable[[ChatRequest], ChatResponse] | None,
        current_meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        plan: ClinicalPlan,
        trace: dict[str, Any],
    ) -> ChatResponse | None:
        """Use the mature deterministic V2/legacy path only when external LLM is disabled.

        This keeps offline regression tests stable without making V2 the live domain
        router. The response is still wrapped with V3 universal trace metadata.
        """
        if not fallback_handler:
            return None
        if getattr(self.llm_service, "client", None):
            return None
        if plan.intent in NON_MEDICAL_INTENTS or plan.domain in NON_MEDICAL_INTENTS:
            return None
        if plan.domain not in V2_COMPAT_DOMAINS:
            return None
        if plan.case_action == "start_new":
            return None
        if plan.risk_level == "emergency" or trace.get("deterministic_override"):
            return None
        try:
            response = fallback_handler(request)
        except Exception as exc:
            logger.warning("V3 offline compatibility fallback failed safely: %s", exc.__class__.__name__)
            return None
        if response.mode != EMERGENCY_MODE:
            response.answer = self._verify_answer(response.answer, plan, case_state, current_message=request.message)
            response.follow_up_questions = _safe_follow_up_questions(response.follow_up_questions, case_state)
            response.needs_follow_up = bool(response.follow_up_questions)
            if _doctor_route_conflicts_with_denials(response.suggested_doctor, case_state):
                response.suggested_doctor = plan.doctor_route or _doctor_for_domain(case_state.active_domain)
                response.display_doctor_ar = display_doctor_ar(response.suggested_doctor)
        original_update = dict(response.case_state_update or {})
        update = self._case_update(
            request=request,
            response=response,
            current_meaning=current_meaning,
            case_state=case_state,
            plan=plan,
            trace=trace,
            engine_route="v3_offline_compat_support",
        )
        update["compat_support_engine"] = original_update.get("engine")
        update["compat_support_route"] = original_update.get("engine_route")
        for key, value in original_update.items():
            update.setdefault(key, value)
        response.case_state_update = update
        return response

    def handle_chat(
        self,
        request: ChatRequest,
        *,
        fallback_handler: Callable[[ChatRequest], ChatResponse] | None = None,
    ) -> ChatResponse:
        recent_history = request.history[-12:]
        raw_message = request.message or ""
        if not raw_message.strip():
            return self._empty_response(request)

        language = _detect_language(request.message, request.language)
        current_symptoms = self._extract_symptoms_safe(request.message)
        current_meaning = understand_message(request.message, language=language, symptoms=current_symptoms)
        if _is_v3_closing_turn(raw_message, current_symptoms):
            current_meaning.intent = "closing"
            current_meaning.domain = "closing"
            current_meaning.is_new_complaint = False
        elif _is_v3_meta_smalltalk_turn(raw_message, current_symptoms):
            current_meaning.intent = "casual"
            current_meaning.domain = "casual"
            current_meaning.is_new_complaint = False
        self._add_v3_meaning(current_meaning, request.message, current_symptoms, None)
        history_state = self._reconstruct_case_state(recent_history, default_language=language)
        self._add_v3_meaning(current_meaning, request.message, current_symptoms, history_state)
        trace = self._new_trace()

        current_effective_symptoms = _symptoms_without_denied_facts(_dedupe(current_meaning.symptoms), current_meaning.denials)
        current_turn_emergency = None
        if current_meaning.intent != "closing":
            current_turn_emergency = detect_emergency_signal(_message_without_negated_scopes(request.message), current_effective_symptoms)
        if current_turn_emergency:
            emergency_case_action = "continue" if history_state.active_domain else "start_new"
            emergency_case = self._merge_current_turn(history_state, current_meaning, emergency_case_action, request.message)
            if not emergency_case.active_domain:
                emergency_case.active_domain = current_turn_emergency.category
            trace["deterministic_override"] = True
            trace["intent"] = current_meaning.intent
            trace["case_action"] = emergency_case_action
            return self._emergency_response(
                request=request,
                signal=current_turn_emergency,
                language=language,
                symptoms=_symptoms_without_denied_facts(_dedupe(emergency_case.symptoms + current_effective_symptoms), emergency_case.denials),
                case_state=emergency_case,
                current_meaning=current_meaning,
                trace=trace,
            )

        if current_meaning.intent == "closing":
            plan = ClinicalPlan(
                intent="closing",
                case_action="close",
                domain="closing",
                clinical_summary="User closed the conversation.",
                doctor_route="Not needed",
                response_goal="close",
            )
            self._run_universal_planner(
                request=request,
                recent_history=recent_history,
                language=language,
                active_case=history_state,
                deterministic_plan=plan,
                symptoms=current_symptoms,
                trace=trace,
            )
            return self._closing_response(request, language, history_state, trace)

        if current_meaning.intent in NON_MEDICAL_INTENTS:
            if history_state.active_domain and current_meaning.intent in {"casual", "off_topic", "greeting"}:
                plan = ClinicalPlan(
                    intent=current_meaning.intent,
                    case_action="pause",
                    domain=history_state.active_domain or current_meaning.domain,
                    clinical_summary=_summary_for_case(history_state, language),
                    doctor_route=history_state.doctor_route or _doctor_for_domain(history_state.active_domain),
                    response_goal="reply",
                )
                self._run_universal_planner(
                    request=request,
                    recent_history=recent_history,
                    language=language,
                    active_case=history_state,
                    deterministic_plan=plan,
                    symptoms=current_symptoms,
                    trace=trace,
                )
                compat = self._offline_compat_response(
                    request=request,
                    fallback_handler=fallback_handler,
                    current_meaning=current_meaning,
                    case_state=history_state,
                    plan=plan,
                    trace=trace,
                )
                if compat and history_state.active_domain not in {"headache", "neurology_vague"}:
                    return compat
                return self._casual_pause_response(request, language, history_state, current_meaning, trace)
            plan = ClinicalPlan(
                intent=current_meaning.intent,
                case_action="none",
                domain=current_meaning.domain,
                clinical_summary="Non-medical turn.",
                doctor_route="Not needed",
                response_goal="boundary" if current_meaning.intent in {"abuse", "off_topic", "nonsense"} else "reply",
            )
            self._run_universal_planner(
                request=request,
                recent_history=recent_history,
                language=language,
                active_case=history_state,
                deterministic_plan=plan,
                symptoms=current_symptoms,
                trace=trace,
            )
            compat = self._offline_compat_response(
                request=request,
                fallback_handler=fallback_handler,
                current_meaning=current_meaning,
                case_state=history_state,
                plan=plan,
                trace=trace,
            )
            if compat:
                return compat
            return self._non_medical_or_fallback(request, None, current_meaning, history_state, trace)

        case_action = self._decide_case_action(history_state, current_meaning, request.message)
        active_case = self._merge_current_turn(history_state, current_meaning, case_action, request.message)

        combined_symptoms = _symptoms_without_denied_facts(_dedupe(active_case.symptoms), active_case.denials)
        accumulated_emergency = self._accumulated_emergency_signal(active_case, request.message, combined_symptoms)
        if accumulated_emergency:
            trace["deterministic_override"] = True
            trace["intent"] = current_meaning.intent
            trace["case_action"] = case_action
            return self._emergency_response(
                request=request,
                signal=accumulated_emergency,
                language=language,
                symptoms=combined_symptoms,
                case_state=active_case,
                current_meaning=current_meaning,
                trace=trace,
            )

        support_evidence = self._collect_supporting_evidence(request.message, combined_symptoms, active_case, trace)

        deterministic_plan = self._deterministic_plan(
            request=request,
            language=language,
            current_meaning=current_meaning,
            case_state=active_case,
            case_action=case_action,
        )
        plan = self._run_universal_planner(
            request=request,
            recent_history=recent_history,
            language=language,
            active_case=active_case,
            deterministic_plan=deterministic_plan,
            symptoms=combined_symptoms,
            trace=trace,
            support_evidence=support_evidence,
        )

        plan = self._apply_domain_safety(plan, active_case, request.message, combined_symptoms)
        if plan.deterministic_override:
            trace["deterministic_override"] = True

        compat = self._offline_compat_response(
            request=request,
            fallback_handler=fallback_handler,
            current_meaning=current_meaning,
            case_state=active_case,
            plan=plan,
            trace=trace,
        )
        if compat and (
            active_case.active_domain not in {"headache", "neurology_vague"}
            or _has_any(request.message, {"رقبة", "رقبتي", "neck"})
        ):
            return compat

        response_started = time.perf_counter()
        draft_answer = self._render_answer(plan, active_case, current_meaning, request)
        trace["response_generation_latency_ms"] = round((time.perf_counter() - response_started) * 1000, 2)
        verification_plan = plan if plan.source == "llm_planner" or plan.risk_level == "emergency" else deterministic_plan
        verified_answer = self._verify_answer(draft_answer, verification_plan, active_case, current_message=request.message)
        if verified_answer != draft_answer and plan.patient_answer:
            plan = replace(plan, patient_answer="", source="deterministic_fallback")

        final_answer = verified_answer
        if self._should_run_final_judge(plan):
            reviewer = getattr(self.llm_service, "review_v3_answer", None)
            if callable(reviewer):
                trace["judge_called"] = True
                try:
                    judge_started = time.perf_counter()
                    review = reviewer(
                        message=request.message,
                        history=recent_history,
                        language=language,
                        clinical_summary=plan.clinical_summary,
                        risk_level=plan.risk_level,
                        risk_reasons=plan.risk_reasons,
                        known_facts=active_case.facts,
                        denied_facts=sorted(active_case.denials),
                        answered_questions=sorted(active_case.answered_questions),
                        approved_questions=plan.questions,
                        active_case=active_case.to_dict(),
                        deterministic_plan=self._plan_to_dict(deterministic_plan),
                        classifier_evidence=support_evidence,
                        draft_answer=verified_answer,
                        previous_assistant=active_case.last_assistant_answer,
                        domain=plan.domain,
                    )
                    trace["judge_latency_ms"] = round((time.perf_counter() - judge_started) * 1000, 2)
                    judge_meta = getattr(self.llm_service, "last_call_meta", {}).get("judge", {})
                    if judge_meta:
                        trace["judge_selected_model"] = judge_meta.get("selected_model")
                        trace["judge_key_index"] = judge_meta.get("key_index")
                        trace["judge_model_fallback_used"] = bool(judge_meta.get("used_fallback_model"))
                        trace["judge_provider_status"] = judge_meta.get("status")
                except Exception as exc:
                    logger.warning("V3 final judge failed safely: %s", exc.__class__.__name__)
                    review = None
                    trace["judge_failed"] = True
                if isinstance(review, dict):
                    approved = bool(review.get("approved"))
                    rewrite = str(review.get("revised_answer") or review.get("safe_rewrite") or "").strip()
                    if approved:
                        trace["judge_approved"] = True
                        trace["judge_used"] = True
                    elif rewrite:
                        checked = self._verify_answer(rewrite, verification_plan, active_case, current_message=request.message)
                        if checked != verification_plan.clinical_summary and checked == rewrite:
                            final_answer = checked
                            trace["judge_used"] = True
                        else:
                            trace["judge_fallback"] = True
                    else:
                        trace["judge_fallback"] = True
                        final_answer = self._verify_answer("", deterministic_plan, active_case, current_message=request.message)
                else:
                    trace["judge_failed"] = True
                    trace["judge_fallback"] = True
                    final_answer = self._verify_answer("", deterministic_plan, active_case, current_message=request.message)

        final_answer = self._verify_answer(final_answer, verification_plan, active_case, current_message=request.message)
        logger.info(
            "MedBridge V3 turn: planner_called=%s planner_used=%s judge_called=%s judge_used=%s override=%s",
            trace["planner_called"],
            trace["planner_used"],
            trace["judge_called"],
            trace["judge_used"],
            trace["deterministic_override"],
        )
        return self._response_from_plan(
            request=request,
            plan=plan,
            deterministic_plan=deterministic_plan,
            answer=final_answer,
            symptoms=combined_symptoms,
            case_state=active_case,
            current_meaning=current_meaning,
            trace=trace,
        )

    def _extract_symptoms_safe(self, message: str) -> list[str]:
        try:
            return list(self.classifier_service.extract_symptoms(message, history_text=""))
        except Exception:
            return []

    def _reconstruct_case_state(self, history: list[ChatMessage], *, default_language: str) -> ClinicalCaseState:
        state = ClinicalCaseState(language=default_language)
        for item in history[-12:]:
            content = item.content or ""
            if item.role == "assistant":
                state.last_assistant_answer = content
                extracted_questions = _extract_questions(content)
                if not extracted_questions and state.paused:
                    continue
                state.pending_questions = extracted_questions
                pending_ids: list[str] = []
                for question in state.pending_questions:
                    pending_ids.extend(_question_ids_for_text(question, state.active_domain))
                state.pending_question_ids = _dedupe(pending_ids)
                state.last_question_id = state.pending_question_ids[0] if state.pending_question_ids else None
                state.asked_question_ids.update(state.pending_question_ids)
                for question in state.pending_questions:
                    slot = _question_slot(question)
                    if slot:
                        state.asked_question_slots.add(slot)
                continue
            if item.role != "user" or not content or _looks_like_mojibake(content):
                continue
            language = _detect_language(content, default_language)
            symptoms = self._extract_symptoms_safe(content)
            meaning = understand_message(content, language=language, symptoms=symptoms)
            self._add_v3_meaning(meaning, content, symptoms, state)
            if meaning.intent == "closing":
                state = ClinicalCaseState(language=default_language, case_closed=True)
                continue
            if meaning.intent in NON_MEDICAL_INTENTS:
                if state.active_domain:
                    state.paused = True
                    state.status = "paused"
                continue
            action = self._decide_case_action(state, meaning, content)
            state = self._merge_current_turn(state, meaning, action, content)
        return state

    def _add_v3_meaning(
        self,
        meaning: MessageMeaning,
        message: str,
        symptoms: list[str],
        active_case: ClinicalCaseState | None,
    ) -> None:
        facts, extra_symptoms, body_part, domain_hint, explicit_denials = _extract_v3_concepts(message, symptoms)
        if _looks_like_malaria_explanation_followup(message, active_case):
            meaning.intent = "medical_question"
            meaning.domain = "infectious"
            meaning.is_new_complaint = False
            return
        if _is_family_or_personal_chat(message):
            meaning.intent = "off_topic"
            meaning.domain = "off_topic"
            return
        if meaning.intent == "closing":
            return
        if _denies_fever(message):
            facts.pop("fever", None)
            facts.pop("temperature", None)
            extra_symptoms = [symptom for symptom in extra_symptoms if symptom not in {"mild_fever", "high_fever"}]
            meaning.denials.add("fever")
            if active_case and active_case.active_domain:
                domain_hint = active_case.active_domain
        meaning.denials.update(explicit_denials)
        inactive_mentions = _inactive_fact_mentions(message)
        meaning.facts = _facts_without_denials(meaning.facts, inactive_mentions)
        meaning.symptoms = _symptoms_without_denied_facts(meaning.symptoms, inactive_mentions)
        non_medical_intent = _v3_zero_evidence_non_medical_intent(message) if not (extra_symptoms or facts or body_part) else None
        if active_case and active_case.active_domain and not (extra_symptoms or facts or body_part) and non_medical_intent != "abuse":
            if _is_negative_short_answer(message) or _is_positive_short_answer(message):
                meaning.intent = "followup_answer"
                meaning.domain = active_case.active_domain
                meaning.is_new_complaint = False
                return
        if not (extra_symptoms or facts or body_part):
            if non_medical_intent:
                meaning.intent = non_medical_intent
                meaning.domain = "off_topic" if non_medical_intent == "casual" else non_medical_intent
                meaning.is_new_complaint = False
                return
        if (extra_symptoms or facts or body_part) and meaning.intent in {"nonsense", "casual", "greeting", "off_topic", "unclear"}:
            meaning.intent = "medical"
            meaning.domain = domain_hint or "unknown"
            meaning.is_new_complaint = True
        if (
            active_case
            and active_case.active_domain in {"headache", "neurology_vague"}
            and (facts.get("vision_change") or {"dizziness", "numbness", "weakness", "slurred_speech", "loss_of_balance"}.intersection(extra_symptoms))
        ):
            domain_hint = active_case.active_domain
            body_part = body_part or ("head" if active_case.active_domain == "headache" else "neurology")
        if (
            active_case
            and active_case.active_domain == "back_pain"
            and meaning.denials.intersection({"numbness", "weakness", "radiation", "bladder_bowel", "generic_no"})
        ):
            meaning.domain = "back_pain"
        meaning.facts.update({key: value for key, value in facts.items() if value not in (None, "", [])})
        meaning.denials.difference_update(_expanded_fact_keys(facts.keys()))
        blocked_facts = _expanded_fact_keys(set(meaning.denials) | set(inactive_mentions))
        meaning.facts = _facts_without_denials(meaning.facts, blocked_facts)
        meaning.symptoms = _without_explicitly_negated_red_flags(message, _dedupe(meaning.symptoms + extra_symptoms))
        meaning.symptoms = _symptoms_without_denied_facts(meaning.symptoms, blocked_facts)
        if "chest_pain" in meaning.denials and "chest_pain" not in meaning.symptoms:
            meaning.facts.pop("chest_pain", None)
            meaning.body_parts = [part for part in meaning.body_parts if part != "chest"]
            if meaning.domain == "chest_pain":
                meaning.domain = "unknown"
        for key in facts:
            meaning.facts.setdefault(key, facts[key])
        if "fainting" in meaning.denials:
            meaning.facts.pop("fainting", None)
        if body_part and body_part not in meaning.body_parts:
            meaning.body_parts.append(body_part)
        priority_domains = {
            "cardiology",
            "endocrine",
            "gynecology",
            "pediatrics",
            "vestibular_ent",
            "infectious",
            "neck_pain",
        }
        if domain_hint in priority_domains and meaning.domain not in {domain_hint, "chest_pain"}:
            meaning.domain = domain_hint
        if domain_hint and (meaning.domain == "unknown" or (active_case and active_case.active_domain == domain_hint)):
            meaning.domain = domain_hint
        if meaning.domain == "unknown" and active_case and active_case.active_domain:
            if facts or meaning.denials or symptoms:
                meaning.domain = active_case.active_domain
        if "severity" in facts:
            meaning.severity = str(facts["severity"])
        if "duration" in facts:
            meaning.duration = str(facts["duration"])

    def _decide_case_action(self, state: ClinicalCaseState, meaning: MessageMeaning, message: str) -> str:
        if meaning.intent in NON_MEDICAL_INTENTS:
            return "pause"
        if meaning.intent == "closing":
            return "close"
        if not state.active_domain or state.case_closed:
            return "start_new"
        clear_domain_change = (
            meaning.domain not in {"unknown", "closing", "abuse", "off_topic", "casual", "nonsense"}
            and meaning.domain != state.active_domain
            and not ({state.active_domain, meaning.domain} <= {"headache", "neurology_vague"})
            and not ({state.active_domain, meaning.domain} <= {"throat_ent", "respiratory", "infectious"})
        )
        if (_is_negative_short_answer(message) or _question_ids_answered_by_message(message, meaning, state)) and not clear_domain_change:
            return "continue"
        explicit_new = _has_any(
            message,
            {"مشكلة تانية", "حاجة تانية", "موضوع تاني", "شكوى تانية", "new problem", "different issue", "another issue"},
        )
        if explicit_new:
            return "start_new"
        if meaning.domain == state.active_domain:
            return "continue"
        if meaning.domain == "unknown" and (meaning.facts or meaning.denials or meaning.duration or meaning.severity):
            return "continue"
        if meaning.domain == "unknown" and len(normalize_text(message).split()) <= 5:
            return "continue"
        if {state.active_domain, meaning.domain} <= {"headache", "neurology_vague"}:
            return "continue"
        return "start_new"

    def _merge_current_turn(
        self,
        state: ClinicalCaseState,
        meaning: MessageMeaning,
        case_action: str,
        message: str,
    ) -> ClinicalCaseState:
        if case_action == "start_new":
            state = ClinicalCaseState(language=meaning.language)
        elif case_action == "close":
            return ClinicalCaseState(language=meaning.language, case_closed=True, status="closed")
        elif case_action == "pause":
            state.paused = True
            state.status = "paused"
            return state

        if case_action == "continue" and state.active_domain and (
            _is_negative_short_answer(message) or _question_ids_answered_by_message(message, meaning, state)
        ):
            domain = state.active_domain
        else:
            domain = meaning.domain if meaning.domain not in {"unknown", "closing", "abuse", "off_topic", "casual", "nonsense"} else state.active_domain
        if domain:
            state.active_domain = domain
        if meaning.body_parts:
            state.active_body_part = meaning.body_parts[0]
        elif state.active_domain == "headache":
            state.active_body_part = "head"
        elif state.active_domain == "chest_pain":
            state.active_body_part = "chest"
        state.language = meaning.language or state.language
        state.symptoms = _dedupe(state.symptoms + meaning.symptoms)
        if _is_negative_short_answer(message):
            pending_ids = list(state.pending_question_ids or ([state.last_question_id] if state.last_question_id else []))
            pending_fact_keys = {
                fact_key
                for fact_key in (_fact_key_for_question_id(question_id) for question_id in pending_ids)
                if fact_key
            }
            if pending_fact_keys:
                meaning.denials = {denial for denial in meaning.denials if denial in pending_fact_keys or denial == "generic_no"}
                meaning.facts = {
                    key: value
                    for key, value in meaning.facts.items()
                    if key in pending_fact_keys or value not in (False, None, "")
                }
        asserted_fact_keys = _expanded_fact_keys(meaning.facts.keys())
        for fact_key in asserted_fact_keys:
            state.denials.discard(fact_key)
        state.denials.update(_expanded_fact_keys(meaning.denials))
        state.facts.update({key: value for key, value in meaning.facts.items() if value not in (None, "", [])})
        for denial in _expanded_fact_keys(meaning.denials):
            if denial in state.facts:
                state.facts.pop(denial, None)
        state.symptoms = _symptoms_without_denied_facts(state.symptoms, state.denials)
        if "fever" in meaning.denials:
            state.facts.pop("temperature", None)
            state.symptoms = [symptom for symptom in state.symptoms if symptom not in {"mild_fever", "high_fever"}]
        self._apply_question_memory_from_turn(state, meaning, message)
        for key in meaning.facts:
            state.answered_questions.add(str(key))
        for denial in meaning.denials:
            state.answered_questions.add(str(denial))
        if meaning.duration:
            state.facts["duration"] = meaning.duration
            state.answered_questions.add("duration")
        if meaning.severity:
            state.facts["severity"] = meaning.severity
            state.answered_questions.add("severity")
        state.doctor_route = _doctor_for_domain(state.active_domain)
        state.status = "active"
        return state

    def _apply_question_memory_from_turn(
        self,
        state: ClinicalCaseState,
        meaning: MessageMeaning,
        message: str,
    ) -> None:
        answered_ids = set(_question_ids_answered_by_message(message, meaning, state))
        pending_ids = list(state.pending_question_ids or ([state.last_question_id] if state.last_question_id else []))
        if _is_negative_short_answer(message) and pending_ids:
            for question_id in pending_ids:
                answered_ids.add(question_id)
                fact_key = _fact_key_for_question_id(question_id)
                if fact_key:
                    state.facts[fact_key] = False
                    state.denials.add(fact_key)
        elif _is_positive_short_answer(message) and pending_ids:
            for question_id in pending_ids:
                answered_ids.add(question_id)
                fact_key = _fact_key_for_question_id(question_id)
                if fact_key:
                    state.facts[fact_key] = True

        for question_id in sorted(answered_ids):
            state.answered_question_ids.add(question_id)
            state.answered_questions.add(question_id)
            if question_id in state.pending_question_ids:
                state.pending_question_ids = [qid for qid in state.pending_question_ids if qid != question_id]
        if state.last_question_id in state.answered_question_ids:
            state.last_question_id = state.pending_question_ids[0] if state.pending_question_ids else None
        if not state.pending_question_ids and answered_ids:
            state.pending_questions = []

    def _deterministic_plan(
        self,
        *,
        request: ChatRequest,
        language: str,
        current_meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        domain = case_state.active_domain or current_meaning.domain or "unknown"
        if domain == "headache":
            return self._headache_plan(language, current_meaning, case_state, case_action)
        if domain == "neurology_vague":
            return self._neurology_vague_plan(language, current_meaning, case_state, case_action)
        if domain == "chest_pain":
            return self._chest_plan(language, current_meaning, case_state, case_action)
        if domain == "back_pain":
            return self._back_plan(language, current_meaning, case_state, case_action)
        if domain == "gynecology":
            return self._gynecology_plan(language, current_meaning, case_state, case_action)
        if domain == "digestive":
            return self._digestive_plan(language, current_meaning, case_state, case_action)
        if domain == "infectious" and case_state.facts.get("temperature") and "high_fever" in set(case_state.symptoms):
            return self._viral_plan(language, current_meaning, case_state, case_action)
        if _is_previous_malaria_question(request):
            return self._malaria_explanation_plan(language, current_meaning, case_state, case_action, request.message)
        if _has_malaria_support(request.message, case_state):
            return self._malaria_supported_plan(language, current_meaning, case_state, case_action)
        if _has_generic_viral_cluster(request.message, case_state.symptoms):
            return self._viral_plan(language, current_meaning, case_state, case_action)
        if domain in DOMAIN_PROFILES:
            return self._profile_plan(language, current_meaning, case_state, case_action, domain)
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain=domain,
            clinical_summary="Medical concern needs clarification.",
            next_best_question="اكتب العرض الأساسي اللي مضايقك أكتر؟" if language != "en" else "What is the main symptom bothering you most?",
            doctor_route="General Practitioner",
        )

    def _digestive_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        question_id, question = _select_next_question("digestive", language, case_state)
        facts = case_state.facts
        possibilities = _digestive_possibilities(case_state)
        if (
            facts.get("post_meal_worse")
            and facts.get("abdominal_location")
            and (facts.get("nausea_present") or facts.get("vomiting_present"))
        ):
            question_id = None
            question = None
        enough_for_guidance = bool(
            facts.get("abdominal_location")
            or facts.get("duration")
            or facts.get("nausea_present")
            or facts.get("vomiting_present")
            or facts.get("diarrhea_present")
            or facts.get("post_meal_worse")
        )
        response_goal = "guide" if enough_for_guidance and not question else "clarify"
        if enough_for_guidance and question_id in {"fever_present", "blood_present"}:
            response_goal = "guide"
        return ClinicalPlan(
            intent=meaning.intent,
            case_action=case_action,
            domain="digestive",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            new_denials=sorted(meaning.denials),
            risk_level="urgent" if {"blood_vomit_or_stool", "dehydration", "severe_abdominal_pain"}.intersection(facts) else "low",
            risk_reasons=[],
            broad_possibilities=possibilities,
            selected_possibilities=possibilities[:3],
            analysis_for_patient=_digestive_analysis_for_patient(case_state, language),
            next_question_id=question_id,
            next_best_question=_avoid_repeated_question(question, case_state),
            care_guidance=["hydration", "monitor symptoms", "seek care if worsening"],
            doctor_route="Gastroenterologist",
            response_goal=response_goal,
            must_not_repeat=case_state.pending_questions,
        )

    def _profile_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
        domain: str,
    ) -> ClinicalPlan:
        profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["unknown"])
        question_id, question = _select_next_question(domain, language, case_state)
        if not question:
            question = profile["en_question"] if language == "en" else profile["ar_question"]
            question_id = _question_slot(question)
        if domain == "body_ache" and "fever" in case_state.denials:
            question = (
                "بقال وجع الجسم قد إيه، وهل معاه كحة أو التهاب حلق أو إجهاد شديد؟"
                if language != "en"
                else "How long have the body aches been present, and is there cough, sore throat, or severe fatigue?"
            )
            question_id = "duration"
        risk = "moderate" if domain in {"chest_pain", "mental_health"} else "low"
        if domain == "urinary" and {"blood_in_urine", "flank_pain", "fever"}.intersection(case_state.facts):
            risk = "urgent"
        if domain == "digestive" and {"blood_vomit_or_stool", "dehydration", "severe_abdominal_pain"}.intersection(case_state.facts):
            risk = "urgent"
        if domain == "skin" and {"face_lip_swelling", "breathlessness"}.intersection(case_state.facts):
            risk = "urgent"
        return ClinicalPlan(
            intent=meaning.intent,
            case_action=case_action,
            domain=domain,
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            new_denials=sorted(meaning.denials),
            risk_level=risk,
            risk_reasons=[] if risk == "low" else [f"{domain} red flag needs clarification"],
            next_question_id=question_id,
            next_best_question=_avoid_repeated_question(question, case_state),
            doctor_route=profile["doctor"],
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _viral_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        temperature = case_state.facts.get("temperature")
        risk_reasons = [f"temperature:{temperature}"] if temperature else []
        question = (
            "هل الحرارة عالية أو مستمرة، أو في ضيق نفس/ألم صدر، وبقال الأعراض قد إيه؟"
            if language != "en"
            else "Is the fever high or persistent, or is there shortness of breath/chest pain, and how long has this been going on?"
        )
        return ClinicalPlan(
            intent=meaning.intent,
            case_action=case_action,
            domain=case_state.active_domain or "infectious",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            risk_level="moderate" if "high_fever" in set(case_state.symptoms) else "low",
            risk_reasons=risk_reasons,
            broad_possibilities=["Viral or flu-like illness"],
            diagnosis="Viral or flu-like illness",
            confidence=0.45,
            next_best_question=_avoid_repeated_question(question, case_state),
            care_guidance=["rest", "hydration", "monitor symptoms"],
            doctor_route="General Practitioner",
            response_goal="diagnose",
            must_not_repeat=case_state.pending_questions,
        )

    def _malaria_supported_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        question = (
            "هل الحرارة بتيجي في نوبات مع رعشة شديدة، وسافرت أو اتعرضت لناموس في منطقة ينتشر فيها المرض؟"
            if language != "en"
            else "Does the fever come in episodes with chills, and was there travel or mosquito exposure in a malaria area?"
        )
        return ClinicalPlan(
            intent=meaning.intent,
            case_action=case_action,
            domain="infectious",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            risk_level="moderate",
            risk_reasons=["fever/chills/sweating with travel or mosquito exposure"],
            broad_possibilities=["Malaria"],
            diagnosis="Malaria",
            confidence=0.62,
            next_best_question=_avoid_repeated_question(question, case_state),
            care_guidance=["seek medical assessment", "hydrate"],
            doctor_route="Infectious disease specialist",
            response_goal="diagnose",
            must_not_repeat=case_state.pending_questions,
        )

    def _malaria_explanation_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
        message: str,
    ) -> ClinicalPlan:
        challenged = _has_any(message, {"ليه", "لماذا", "مش", "معنديش", "مفيش", "no travel", "no mosquito", "why"})
        diagnosis = "Viral or flu-like illness" if challenged else None
        return ClinicalPlan(
            intent="medical_question",
            case_action=case_action,
            domain="infectious",
            clinical_summary="User is asking about a previous malaria suggestion.",
            new_facts=_new_fact_names(meaning),
            new_denials=sorted(meaning.denials),
            risk_level="low",
            broad_possibilities=[diagnosis] if diagnosis else [],
            diagnosis=diagnosis,
            confidence=0.45 if diagnosis else 0.0,
            next_best_question=None,
            doctor_route="General Practitioner",
            response_goal="diagnose",
        )

    def _headache_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        facts = case_state.facts
        new_facts = _new_fact_names(meaning)
        doctor = "Neurologist"
        summary = _summary_for_case(case_state, language)
        if facts.get("vision_change") and facts.get("sudden_onset"):
            return ClinicalPlan(
                intent="medical",
                case_action=case_action,
                domain="headache",
                clinical_summary=summary,
                new_facts=new_facts,
                risk_level="emergency",
                risk_reasons=["sudden headache with visual change"],
                doctor_route="Emergency Department / Neurologist",
                response_goal="escalate",
                deterministic_override=True,
            )
        question = None
        risk = "low"
        reasons: list[str] = []
        guidance = []
        if facts.get("sudden_onset") and not facts.get("vision_change"):
            risk = "urgent"
            reasons.append("sudden headache")
            question = (
                "هل مع الصداع زغللة، ضعف أو تنميل في ناحية من الجسم، صعوبة كلام، قيء متكرر، أو تيبس رقبة؟"
                if language != "en"
                else "Do you also have vision changes, weakness or numbness on one side, slurred speech, repeated vomiting, or neck stiffness?"
            )
            guidance.append("Same-day urgent medical assessment is safer if this is new, severe, or unusual.")
        elif facts.get("vision_change"):
            risk = "moderate"
            reasons.append("headache with visual change")
            question = (
                "الصداع بدأ فجأة ولا تدريجي؟"
                if language != "en"
                else "Did the headache start suddenly or build gradually?"
            )
        elif not facts.get("sudden_onset") and "sudden_onset" not in case_state.answered_questions:
            question = (
                "الصداع بدأ فجأة ولا تدريجي، وهل معاه تنميل/ضعف أو زغللة؟"
                if language != "en"
                else "Did the headache start suddenly or gradually, and is there numbness, weakness, or vision change?"
            )
        elif not facts.get("vision_change") and "vision_change" not in case_state.answered_questions:
            question = (
                "هل معاه زغللة أو تغير في النظر؟"
                if language != "en"
                else "Is there any blurred vision or change in your vision?"
            )
        else:
            question = (
                "بقاله قد إيه وبيزيد ولا ثابت؟"
                if language != "en"
                else "How long has it been going on, and is it worsening or staying the same?"
            )
        if _is_migraine_like_headache(case_state):
            return ClinicalPlan(
                intent="medical",
                case_action=case_action,
                domain="headache",
                clinical_summary=summary,
                new_facts=new_facts,
                new_denials=sorted(meaning.denials),
                risk_level="low",
                risk_reasons=[],
                broad_possibilities=["Migraine-like headache"],
                candidate_conditions=[
                    {
                        "name": "Migraine-like headache",
                        "reason": "unilateral pulsating headache with light/sound sensitivity and nausea",
                    }
                ],
                selected_possibilities=["Migraine-like headache"],
                diagnosis="Migraine-like headache",
                confidence=0.62,
                next_best_question=None,
                care_guidance=["rest in a quiet dark place", "hydrate", "seek care if new, severe, persistent, or red flags appear"],
                doctor_route=doctor,
                response_goal="diagnose",
                must_not_repeat=case_state.pending_questions,
            )
        question = _avoid_repeated_question(question, case_state)
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain="headache",
            clinical_summary=summary,
            new_facts=new_facts,
            risk_level=risk,
            risk_reasons=reasons,
            broad_possibilities=["headache causes need clinical context"],
            next_best_question=question,
            care_guidance=guidance,
            doctor_route=doctor,
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _neurology_vague_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        question = (
            "هل في تنميل، ضعف، دوخة شديدة، تغير في الكلام أو النظر؟"
            if language != "en"
            else "Is there numbness, weakness, severe dizziness, speech change, or vision change?"
        )
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain="neurology_vague",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            risk_level="moderate",
            next_best_question=_avoid_repeated_question(question, case_state),
            doctor_route="Neurologist",
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _chest_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        facts = case_state.facts
        question = None
        risk = "moderate"
        reasons = ["chest pain needs red-flag clarification"]
        if not {"breathlessness", "cold_sweat", "chest_radiation", "fainting"}.intersection(facts):
            question = (
                "هل مع ألم الصدر ضيق نفس، عرق بارد، إغماء، أو ألم ممتد للذراع أو الفك؟"
                if language != "en"
                else "Any shortness of breath, cold sweat, fainting, or pain spreading to the arm or jaw?"
            )
        elif not facts.get("duration"):
            question = (
                "بدأ من إمتى، وهل ظهر فجأة أو مع مجهود؟"
                if language != "en"
                else "When did it start, and did it start suddenly or with exertion?"
            )
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain="chest_pain",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            risk_level=risk,
            risk_reasons=reasons,
            next_best_question=_avoid_repeated_question(question, case_state),
            doctor_route="Cardiologist",
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _back_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        facts = case_state.facts
        denials = case_state.denials
        question = None
        if not facts.get("location"):
            question = "الألم في أسفل الظهر ولا أعلى الظهر؟" if language != "en" else "Is it upper back or lower back?"
        elif not facts.get("duration"):
            question = "بدأ من إمتى، وهل كان بعد إصابة أو حمل حاجة تقيلة؟" if language != "en" else "When did it start, and was it after injury or heavy lifting?"
        elif not {"numbness", "weakness", "bladder_bowel", "radiation"}.intersection(denials):
            question = (
                "هل الألم بينزل على الرجل أو معاه تنميل/ضعف أو مشكلة في التحكم في البول أو البراز؟"
                if language != "en"
                else "Does it go down the leg, or come with numbness, weakness, or bladder/bowel control problems?"
            )
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain="back_pain",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            new_denials=sorted(meaning.denials),
            risk_level="low",
            next_best_question=_avoid_repeated_question(question, case_state),
            doctor_route="Orthopedic doctor",
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _gynecology_plan(
        self,
        language: str,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        case_action: str,
    ) -> ClinicalPlan:
        facts = case_state.facts
        symptoms = set(case_state.symptoms)
        if "discharge" in symptoms or facts.get("discharge_present"):
            question = (
                "لون الإفرازات إيه، وهل في ريحة وحشة أو حكة/حرقان؟"
                if language != "en"
                else "What color is the discharge, and is there a bad smell, itching, or burning?"
            )
        elif "pregnancy" not in facts and "pregnancy" not in case_state.denials:
            question = (
                "هل في احتمال حمل أو الدورة متأخرة/مختلفة عن المعتاد؟"
                if language != "en"
                else "Is there any possibility of pregnancy, or is your period late or unusual?"
            )
        else:
            question = (
                "هل في ألم أسفل البطن، حرارة، نزيف، أو إفرازات غير معتادة؟"
                if language != "en"
                else "Any lower abdominal pain, fever, bleeding, or unusual discharge?"
            )
        return ClinicalPlan(
            intent="medical",
            case_action=case_action,
            domain="gynecology",
            clinical_summary=_summary_for_case(case_state, language),
            new_facts=_new_fact_names(meaning),
            risk_level="low",
            next_best_question=_avoid_repeated_question(question, case_state),
            doctor_route="Gynecologist",
            response_goal="clarify",
            must_not_repeat=case_state.pending_questions,
        )

    def _accumulated_emergency_signal(
        self,
        case_state: ClinicalCaseState,
        message: str,
        symptoms: list[str],
    ) -> EmergencySignal | None:
        positive_message = _message_without_negated_scopes(message)
        effective_symptoms = _symptoms_without_denied_facts(symptoms, case_state.denials)
        direct = detect_emergency_signal(positive_message, effective_symptoms)
        if direct:
            return direct
        if not direct and _is_high_risk_by_existing_safety(positive_message, effective_symptoms):
            return _generic_red_flag_signal(case_state.active_domain, case_state.language)
        facts = case_state.facts
        if case_state.active_domain in {"headache", "neurology_vague"} and facts.get("vision_change") and facts.get("sudden_onset"):
            return EmergencySignal(
                "neurological",
                "Neurological emergency concern",
                "أعراض عصبية خطيرة محتملة",
                "Emergency Department / Neurologist",
                "الطوارئ فورًا / طبيب مخ وأعصاب",
                "اجتماع صداع بدأ فجأة مع زغللة أو تغير في النظر قد يكون علامة عصبية خطيرة ويحتاج تقييمًا عاجلًا.",
            )
        if case_state.active_domain in {"headache", "neurology_vague"} and facts.get("fainting") and facts.get("severity") == "severe":
            return EmergencySignal(
                "neurological",
                "Neurological emergency concern",
                "أعراض عصبية خطيرة محتملة",
                "Emergency Department / Neurologist",
                "الطوارئ فورًا / طبيب مخ وأعصاب",
                "فقدان الوعي مع صداع شديد قد يكون علامة خطورة عصبية ويحتاج تقييمًا عاجلًا.",
            )
        if case_state.active_domain == "chest_pain" and (
            facts.get("breathlessness") or facts.get("cold_sweat") or facts.get("fainting") or facts.get("chest_radiation")
        ):
            return EmergencySignal(
                "cardiac",
                "Cardiac emergency concern",
                "اشتباه مشكلة قلبية طارئة",
                "Emergency Department / Cardiologist",
                "الطوارئ فورًا / طبيب قلب",
                "ألم الصدر مع ضيق نفس أو عرق بارد أو امتداد الألم يحتاج طوارئ.",
            )
        if case_state.active_domain == "back_pain" and facts.get("bladder_bowel_or_leg_weakness"):
            return EmergencySignal(
                "neurological",
                "Spinal emergency concern",
                "اشتباه ضغط عصبي أو مشكلة خطيرة بالعمود الفقري",
                "Emergency Department / Orthopedic doctor",
                "الطوارئ فورًا / طبيب عظام",
                "ألم الظهر مع ضعف بالرجل أو فقدان التحكم في البول/البراز علامة خطورة وتحتاج طوارئ.",
            )
        if case_state.active_domain == "digestive" and (
            facts.get("blood_vomit_or_stool") or facts.get("dehydration") or facts.get("severe_abdominal_pain")
        ):
            return EmergencySignal(
                "digestive",
                "Digestive emergency concern",
                "اشتباه مشكلة هضمية طارئة",
                "Emergency Department / Gastroenterologist",
                "الطوارئ فورًا / طبيب جهاز هضمي",
                "وجود دم أو جفاف شديد أو ألم بطن شديد يحتاج تقييمًا عاجلًا.",
            )
        if case_state.active_domain == "urinary" and facts.get("blood_in_urine"):
            return EmergencySignal(
                "urinary",
                "Urinary emergency concern",
                "اشتباه مشكلة بولية تحتاج تقييم عاجل",
                "Emergency Department / Urologist",
                "الطوارئ أو طبيب مسالك بشكل عاجل",
                "وجود دم في البول يحتاج تقييمًا عاجلًا خصوصًا لو معه ألم شديد أو حرارة.",
            )
        if case_state.active_domain == "skin" and facts.get("face_lip_swelling") and facts.get("breathlessness"):
            return EmergencySignal(
                "allergy",
                "Severe allergic reaction concern",
                "اشتباه حساسية شديدة",
                "Emergency Department / Allergist",
                "الطوارئ فورًا / طبيب حساسية",
                "الطفح أو الحكة مع تورم الشفاه/الوجه وضيق التنفس قد يكون حساسية شديدة ويحتاج طوارئ.",
            )
        if case_state.active_domain == "eye" and facts.get("sudden_vision_loss"):
            return EmergencySignal(
                "eye",
                "Urgent vision concern",
                "تغير مفاجئ خطير في النظر",
                "Emergency Department / Ophthalmologist",
                "الطوارئ فورًا / طبيب عيون",
                "فقدان أو تغير النظر المفاجئ يحتاج تقييمًا عاجلًا.",
            )
        return None

    def _apply_domain_safety(
        self,
        plan: ClinicalPlan,
        case_state: ClinicalCaseState,
        message: str,
        symptoms: list[str],
    ) -> ClinicalPlan:
        emergency = self._accumulated_emergency_signal(case_state, message, symptoms)
        if emergency:
            plan.risk_level = "emergency"
            plan.risk_reasons = [emergency.reason]
            plan.doctor_route = emergency.doctor
            plan.response_goal = "escalate"
            plan.deterministic_override = True
            return plan
        if plan.risk_level == "emergency" and not _raw_text_has_emergency_signal(message, case_state):
            plan.risk_level = "urgent" if plan.domain in {"chest_pain", "neurology_vague"} else "moderate"
            plan.response_goal = "clarify"
        return plan

    def _validate_llm_plan(
        self,
        raw: Any,
        deterministic: ClinicalPlan,
        case_state: ClinicalCaseState,
    ) -> ClinicalPlan | None:
        if not isinstance(raw, dict):
            return None
        domain = str(raw.get("domain") or deterministic.domain).strip() or deterministic.domain
        domain_corrected = False
        if domain in {"non_medical", "none", "unknown", ""} and deterministic.domain not in {"unknown", "casual", "off_topic", "abuse", "nonsense", "closing"}:
            domain = deterministic.domain
            domain_corrected = True
        elif (
            deterministic.domain not in {"unknown", "casual", "off_topic", "abuse", "nonsense", "closing"}
            and domain != deterministic.domain
        ):
            domain = deterministic.domain
            domain_corrected = True
        intent = str(raw.get("intent") or deterministic.intent).strip() or deterministic.intent
        case_action = str(raw.get("case_action") or deterministic.case_action).strip() or deterministic.case_action
        response_goal = str(raw.get("response_goal") or deterministic.response_goal).strip() or deterministic.response_goal
        if deterministic.intent == "medical":
            if intent in NON_MEDICAL_INTENTS or intent in {"nonsense", "unclear", "none"}:
                intent = "medical"
            if case_action in {"none", "pause", "close"}:
                case_action = deterministic.case_action if deterministic.case_action not in {"none", "pause", "close"} else "continue"
            if response_goal in {"none", "close"}:
                response_goal = deterministic.response_goal if deterministic.response_goal not in {"none", "close"} else "clarify"
        risk = _safe_risk_level(raw.get("risk_level"), deterministic.risk_level)
        if deterministic.risk_level == "emergency" and risk != "emergency":
            return None
        if risk == "emergency" and deterministic.risk_level != "emergency":
            if not _raw_plan_has_emergency_reason(raw):
                risk = deterministic.risk_level if deterministic.risk_level in {"urgent", "moderate"} else "moderate"
        next_question_obj = raw.get("next_question") if isinstance(raw.get("next_question"), dict) else {}
        raw_questions = raw.get("questions_to_ask")
        question_values = _strings(raw_questions) if raw_questions is not None else []
        if not question_values:
            question_values = [
                next_question_obj.get("text"),
                raw.get("next_best_question"),
                raw.get("optional_second_question"),
            ]
        q1 = _clean_question(question_values[0]) if len(question_values) >= 1 else None
        q2 = _clean_question(question_values[1]) if len(question_values) >= 2 else None
        questions = [q for q in [q1, q2] if q]
        filtered: list[str] = []
        for question in questions:
            slot = _question_slot(question)
            if slot and slot in case_state.answered_questions:
                continue
            if _question_is_repeated(question, case_state):
                continue
            if _question_mentions_denied_fact(question, case_state):
                continue
            filtered.append(question)
        if len(filtered) > MAX_V3_QUESTIONS:
            filtered = filtered[:MAX_V3_QUESTIONS]
        next_question_text = filtered[0] if filtered else deterministic.next_best_question
        next_question_id = str(next_question_obj.get("id") or "").strip() or (
            _question_slot(next_question_text) if next_question_text else deterministic.next_question_id
        )
        if deterministic.next_question_id and next_question_id == deterministic.next_question_id:
            next_question_text = deterministic.next_best_question
        diagnosis = str(raw.get("diagnosis") or "").strip() or deterministic.diagnosis
        confidence_raw = raw.get("confidence", deterministic.confidence)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw or 0.0)))
        except (TypeError, ValueError):
            confidence = deterministic.confidence
        patient_answer = str(raw.get("patient_answer") or deterministic.patient_answer or "").strip()
        if deterministic.intent == "medical" and _patient_answer_erases_medical_meaning(patient_answer):
            patient_answer = ""
        return ClinicalPlan(
            intent=intent,
            case_action=case_action,
            domain=domain,
            clinical_summary=str(raw.get("clinical_summary") or deterministic.clinical_summary),
            new_facts=_strings(raw.get("new_facts")) or deterministic.new_facts,
            new_denials=_strings(raw.get("new_denials")) or deterministic.new_denials,
            risk_level=risk,
            risk_reasons=_strings(raw.get("risk_reasons")) or deterministic.risk_reasons,
            broad_possibilities=_strings(raw.get("broad_possibilities"))[:3] or deterministic.broad_possibilities,
            candidate_conditions=_candidate_condition_list(raw.get("candidate_conditions")) or deterministic.candidate_conditions,
            selected_possibilities=_strings(raw.get("selected_possibilities"))[:3] or deterministic.selected_possibilities,
            analysis_for_patient=str(raw.get("analysis_for_patient") or deterministic.analysis_for_patient or "").strip(),
            patient_answer=patient_answer,
            diagnosis=diagnosis,
            confidence=confidence,
            next_question_id=next_question_id,
            next_best_question=next_question_text,
            optional_second_question=None,
            care_guidance=_strings(raw.get("care_guidance"))[:3] or deterministic.care_guidance,
            doctor_route=deterministic.doctor_route if domain_corrected else str(raw.get("doctor_route") or deterministic.doctor_route or ""),
            response_goal=response_goal,
            must_not_repeat=_strings(raw.get("must_not_repeat")) or deterministic.must_not_repeat,
            forbidden_topics=_strings(raw.get("forbidden_topics")),
            source="llm_planner",
        )

    def _render_answer(
        self,
        plan: ClinicalPlan,
        case_state: ClinicalCaseState,
        current_meaning: MessageMeaning,
        request: ChatRequest,
    ) -> str:
        language = current_meaning.language or case_state.language
        question = plan.next_best_question
        if plan.risk_level == "emergency":
            signal = self._accumulated_emergency_signal(case_state, request.message, case_state.symptoms)
            reason = signal.reason if signal else "الأعراض المذكورة تحتاج تقييمًا عاجلًا."
            if language == "en":
                return (
                    "These symptoms may be urgent. Please go to the emergency department now or call emergency services.\n\n"
                    f"Why: {reason if not _contains_arabic(reason) else 'The combination of symptoms can be a neurological or cardiac red flag.'}"
                )
            return f"{HIGH_URGENCY_PREFIX}\n\n{reason}\n\nلا تنتظر ولا تكتفي بالمراقبة في البيت."

        if plan.patient_answer and plan.source == "llm_planner":
            return plan.patient_answer

        if language == "en":
            return self._render_answer_en(plan, case_state, current_meaning, question)
        return self._render_answer_ar(plan, case_state, current_meaning, question)

    def _render_answer_ar(
        self,
        plan: ClinicalPlan,
        case_state: ClinicalCaseState,
        current_meaning: MessageMeaning,
        question: str | None,
    ) -> str:
        domain = plan.domain
        facts = case_state.facts
        if plan.response_goal == "diagnose" or plan.diagnosis:
            return self._render_diagnosis_ar(plan, question)
        if domain == "headache":
            if current_meaning.facts.get("vision_change"):
                base = "الزغللة أو تغير النظر مع الصداع معلومة مهمة، وبتخلينا نركز على بداية الصداع وشدته بدل ما نكرر نفس الأسئلة."
            elif current_meaning.facts.get("sudden_onset"):
                base = "كون الصداع بدأ فجأة معلومة مهمة وبتزود درجة القلق، خصوصًا لو ده صداع جديد أو غير معتاد."
            elif current_meaning.facts.get("severity") == "severe":
                base = "فهمت إن الصداع شديد. بما إننا بنتكلم عن نفس الصداع، الأهم نربطه بأي علامات عصبية أو تغير في النظر."
            else:
                if case_state.facts.get("fatigue"):
                    base = "فهمت إن عندك تعب مع صداع. ده ممكن يحصل لأسباب بسيطة زي إجهاد أو عدوى خفيفة، لكن محتاجين نحدد شكل الصداع الأول."
                else:
                    base = "فهمت إن عندك صداع. الصداع ممكن يكون من أسباب بسيطة، لكن محتاجين نحدد بدايته وهل معاه أي علامات عصبية."
            if facts.get("vision_change") and not facts.get("sudden_onset"):
                base += " وجود زغللة قبل كده يخلي سؤال بداية الصداع مهم."
            if question:
                return f"{base}\n\n{question}"
            return f"{base}\n\nلو الصداع جديد أو بيزيد بسرعة، الأفضل تكشف عند طبيب مخ وأعصاب أو طوارئ حسب الشدة."
        if domain == "neurology_vague":
            base = "خلينا نركز على الأعراض العصبية نفسها من غير ما نفترض تشخيص."
            return f"{base}\n\n{question}" if question else base
        if domain == "chest_pain":
            base = (
                "ألم الصدر محتاج تعامل بحذر. مش كل ألم صدر يعني سبب خطير، لكن وجود ضيق نفس أو عرق بارد أو امتداد الألم يغير درجة الخطورة فورًا."
                " لو أي علامة من دول موجودة، الطوارئ فورًا هي الاختيار الآمن."
            )
            return f"{base}\n\n{question}" if question else base
        if domain == "back_pain":
            location = "أسفل الظهر" if case_state.facts.get("location") == "lower_back" else "الظهر"
            if current_meaning.denials:
                base = f"تمام، نفي التنميل معلومة مطمئنة جزئيًا، وهنفضل مركزين على ألم {location}."
            else:
                base = f"ألم {location} غالبًا يحتاج تحديد المكان والامتداد قبل أي استنتاج."
            return f"{base}\n\n{question}" if question else f"{base}\n\nلو الألم مستمر أو بيزيد، طبيب عظام أو طبيب عام مناسب كبداية."
        if domain == "gynecology":
            if current_meaning.facts.get("age"):
                base = f"تمام، السن {current_meaning.facts.get('age')} سنة معلومة مفيدة. هنركز على أعراض النساء المذكورة من غير ما ندخل في أعراض غير مرتبطة."
            elif "discharge" in case_state.symptoms:
                base = "الإفرازات المهبلية تحتاج تفاصيل عن اللون والرائحة والحكة لأنها بتفرق في التوجيه."
            else:
                base = "أعراض الدورة أو النساء تحتاج شوية تفاصيل محددة قبل أي استنتاج."
            return f"{base}\n\n{question}" if question else base
        if domain == "digestive":
            base = plan.analysis_for_patient or _digestive_analysis_for_patient(case_state, "ar")
            if question:
                return f"{base}\n\n{question}"
            return (
                f"{base}\n\n"
                "النصيحة المبدئية: اشرب سوائل بكميات صغيرة وراقب الأعراض. "
                "لو الألم شديد، في دم، جفاف، حرارة عالية، أو ترجيع مستمر، الكشف يبقى أهم."
            )
        if domain in DOMAIN_PROFILES:
            profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["unknown"])
            base = profile["ar_base"]
            if domain == "side_pain" and (current_meaning.facts.get("side_correction") or case_state.facts.get("side_correction")):
                side = "الشمال" if case_state.facts.get("side") == "left" else "اليمين" if case_state.facts.get("side") == "right" else "الجنب"
                base = f"تمام، نصححها: الألم في الجنب {side}. هنكمل على المعلومة الجديدة ونسيب الناحية القديمة."
            if domain == "respiratory" and case_state.facts.get("night_worse"):
                base = "فهمت إن الكحة بتزيد بالليل. ده ممكن يحصل مع تهيج في الشعب أو ارتجاع أو حساسية، ومع نفي السخونية وضيق النفس نفضل مركزين على نمط الكحة ومدتها."
            if domain == "body_ache" and "fever" in current_meaning.denials:
                base = "تمام، نستبعد السخونية من الكلام الحالي ونركز على وجع الجسم نفسه والمدة وباقي الأعراض."
            return f"{base}\n\n{question}" if question else f"{base}\n\nلو الأعراض مستمرة أو بتزيد، الكشف عند {display_doctor_ar(profile['doctor'])} مناسب."
        return f"فاهمك. محتاج أحدد العرض الأساسي بدقة.\n\n{question}" if question else "فاهمك. محتاج تفاصيل أكثر عن العرض الأساسي."

    def _render_diagnosis_ar(self, plan: ClinicalPlan, question: str | None) -> str:
        diagnosis = plan.diagnosis or (plan.broad_possibilities[0] if plan.broad_possibilities else "حالة تحتاج تقييم طبي")
        display = display_diagnosis_ar(diagnosis)
        doctor = display_doctor_ar(plan.doctor_route or _doctor_for_domain(plan.domain))
        if plan.intent == "medical_question" and not plan.diagnosis:
            base = (
                "الملاريا عدوى يسببها طفيل وينتقل غالبا عن طريق البعوض. "
                "لكن لا تتأكد من الكلام أو من أي تشخيص بمجرد الأعراض؛ لازم سياق زي سفر أو تعرض لناموس وفحص طبي عند اللزوم."
            )
        elif diagnosis == "Malaria":
            base = (
                "وجود حرارة مع رعشة وتعرق بعد سفر أو تعرض لناموس يخلي الملاريا احتمال محتاج تقييم، "
                "لأنها عدوى يسببها طفيل وينقلها البعوض، وليست بكتيريا. لا يتأكد الكلام ده من الأعراض وحدها."
            )
        elif diagnosis == "Migraine-like headache":
            base = (
                "الوصف ده يتماشى أكثر مع صداع نصفي محتمل: صداع نابض في ناحية واحدة، ومعاه حساسية للنور والصوت وغثيان. "
                "نفي الضعف أو التنميل أو لخبطة الكلام مطمئن جزئيًا، لكن لا يمكن تأكيد التشخيص من المحادثة وحدها."
            )
            guidance = (
                "حاول ترتاح في مكان هادئ ومظلم، اشرب سوائل، وقلل الشاشات والمجهود مؤقتًا. "
                "يفضل مراجعة طبيب لو الصداع جديد عليك، شديد، متكرر، أو لا يتحسن. "
                "اتجه للطوارئ لو ظهر ضعف أو تنميل في ناحية واحدة، لخبطة كلام، فقدان وعي، قيء مستمر، أو صداع مفاجئ شديد جدًا."
            )
            if question:
                return f"{base}\n\nالطبيب المناسب كبداية: {doctor}.\n\n{guidance}\n\n{question}"
            return f"{base}\n\nالطبيب المناسب كبداية: {doctor}.\n\n{guidance}"
        elif diagnosis == "Viral or flu-like illness":
            if plan.intent == "medical_question":
                base = (
                    "لو مفيش سفر أو تعرض واضح لناموس، احتمال الملاريا يبقى أضعف، "
                    "والاتجاه الأقرب غالبا عدوى فيروسية أو دور برد/إنفلونزا. لا تتأكد من الكلام ده إلا بالكشف لو الأعراض مستمرة أو شديدة."
                )
            else:
                base = (
                    f"من الأعراض اللي ذكرتها، الاتجاه الأقرب مبدئيا هو: {display}. "
                    "السخونية مع تكسير الجسم ممكن تحصل مع عدوى فيروسية أو دور برد/إنفلونزا، خصوصا لو مفيش سفر أو تعرض لناموس."
                )
                temperature_reason = next((reason for reason in plan.risk_reasons if reason.startswith("temperature:")), "")
                if temperature_reason:
                    temperature = temperature_reason.split(":", 1)[1]
                    base += f" بما إن الحرارة وصلت {temperature}، الأفضل تقييم طبي قريب/اليوم خصوصا لو مستمرة أو معها تدهور عام."
        else:
            base = f"من الأعراض اللي ذكرتها، الاتجاه الأقرب مبدئيا هو: {display}. ده احتمال أولي وليس تشخيصا نهائيا."
        guidance = "الراحة، شرب سوائل كفاية، ومتابعة الحرارة والأعراض. اطلب كشف أسرع لو الأعراض بتزيد أو ظهرت ضيق نفس، ألم صدر، تشوش، جفاف، أو حرارة عالية مستمرة."
        if question:
            return f"{base}\n\nالطبيب المناسب كبداية: {doctor}.\n\n{guidance}\n\n{question}"
        return f"{base}\n\nالطبيب المناسب كبداية: {doctor}.\n\n{guidance}"

    def _render_answer_en(
        self,
        plan: ClinicalPlan,
        case_state: ClinicalCaseState,
        current_meaning: MessageMeaning,
        question: str | None,
    ) -> str:
        domain = plan.domain
        facts = case_state.facts
        if plan.response_goal == "diagnose" or plan.diagnosis:
            return self._render_diagnosis_en(plan, question)
        if domain == "headache":
            if current_meaning.facts.get("vision_change"):
                base = "Blurred or changed vision with a headache matters, so I’m focusing on the headache onset and severity rather than repeating the same question."
            elif current_meaning.facts.get("sudden_onset"):
                base = "A headache that starts suddenly is more concerning, especially if it is new, severe, or unusual for you."
            else:
                if case_state.facts.get("fatigue"):
                    base = "I understand you have fatigue with a headache. It can have simple causes, but we need to clarify the headache pattern first."
                else:
                    base = "I understand you have a headache. It can have simple causes, but we need to clarify its onset and any neurological warning signs first."
            if facts.get("vision_change") and not facts.get("sudden_onset"):
                base += " Since you mentioned vision change earlier, the onset is important."
            return f"{base}\n\n{question}" if question else base
        if domain == "chest_pain":
            base = "Chest pain needs careful triage. Not every chest pain is an emergency, but shortness of breath, cold sweat, fainting, or spreading pain changes the risk immediately. If any of these are present, emergency care is the safest choice."
            return f"{base}\n\n{question}" if question else base
        if domain == "back_pain":
            location = "upper back" if case_state.facts.get("location") == "upper_back" else "back"
            base = f"Let’s keep this focused on your {location} pain and avoid guessing a diagnosis from one symptom."
            return f"{base}\n\n{question}" if question else f"{base}\n\nIf it persists or worsens, a general practitioner or orthopedic doctor is a reasonable next step."
        if domain == "gynecology":
            base = "Let’s keep this focused on the menstrual/gynecology symptoms you mentioned and ask only the next useful detail."
            return f"{base}\n\n{question}" if question else base
        if domain == "digestive":
            base = plan.analysis_for_patient or _digestive_analysis_for_patient(case_state, "en")
            if question:
                return f"{base}\n\n{question}"
            return (
                f"{base}\n\n"
                "For now, small sips of fluids and monitoring are reasonable. Seek care faster if pain is severe, "
                "there is blood, dehydration, high fever, or persistent vomiting."
            )
        if domain in DOMAIN_PROFILES:
            profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["unknown"])
            base = profile["en_base"]
            return f"{base}\n\n{question}" if question else f"{base}\n\nIf it persists or worsens, {profile['doctor']} is a reasonable next step."
        return f"I understand. I need one clearer detail first.\n\n{question}" if question else "I understand. Tell me the main symptom more clearly."

    def _render_diagnosis_en(self, plan: ClinicalPlan, question: str | None) -> str:
        diagnosis = plan.diagnosis or (plan.broad_possibilities[0] if plan.broad_possibilities else "a condition needing medical assessment")
        doctor = plan.doctor_route or _doctor_for_domain(plan.domain)
        if plan.intent == "medical_question" and not plan.diagnosis:
            base = (
                "Malaria is an infection caused by a parasite and usually spread by mosquitoes. "
                "Symptoms alone do not confirm it; travel or mosquito exposure and medical assessment matter."
            )
        elif diagnosis == "Malaria":
            base = (
                "Fever with chills and sweating after travel or mosquito exposure can make malaria a possibility. "
                "Malaria is caused by a parasite spread by mosquitoes, not bacteria, and symptoms alone cannot confirm it."
            )
        elif diagnosis == "Viral or flu-like illness":
            base = (
                "Based on what you described, the closest initial direction is a viral or flu-like illness. "
                "Body aches with fever commonly fit that pattern, especially without travel or mosquito exposure."
            )
        elif diagnosis == "Migraine-like headache":
            base = (
                "What you described fits best with a possible migraine-like headache: one-sided pulsating pain with "
                "light sensitivity, sound sensitivity, and nausea. The denied weakness, numbness, and speech trouble "
                "are partly reassuring, but chat alone cannot confirm a diagnosis."
            )
            guidance = (
                "Rest in a quiet dark place, drink fluids, and reduce screens and exertion for now. "
                "Seek medical care if this headache is new for you, severe, recurring, or not improving. "
                "Go to emergency care if one-sided weakness or numbness, speech trouble, loss of consciousness, "
                "persistent vomiting, or a sudden worst headache appears."
            )
            if question:
                return f"{base}\n\nSuggested doctor: {doctor}.\n\n{guidance}\n\n{question}"
            return f"{base}\n\nSuggested doctor: {doctor}.\n\n{guidance}"
        else:
            base = f"Based on what you described, the closest initial direction is: {diagnosis}. This is not a final diagnosis."
        guidance = "Safe first steps are rest, hydration, and monitoring. Seek medical care faster if symptoms worsen or warning signs appear."
        if question:
            return f"{base}\n\nSuggested doctor: {doctor}.\n\n{guidance}\n\n{question}"
        return f"{base}\n\nSuggested doctor: {doctor}.\n\n{guidance}"

    def _verify_answer(
        self,
        answer: str,
        deterministic_plan: ClinicalPlan,
        case_state: ClinicalCaseState,
        *,
        current_message: str = "",
    ) -> str:
        candidate = (answer or "").strip()
        lowered = candidate.lower()
        guard_state = _final_guard_state(case_state, current_message)
        fallback_plan = replace(deterministic_plan, patient_answer="", source="deterministic_fallback")
        fallback = self._render_answer(
            fallback_plan,
            guard_state,
            MessageMeaning(language=guard_state.language, domain=fallback_plan.domain),
            ChatRequest(message="fallback"),
        )
        fallback = _limit_to_one_question(fallback)
        if candidate.lstrip().startswith(("{", "[")):
            return fallback
        if not candidate or any(pattern in lowered for pattern in INTERNAL_LEAK_PATTERNS):
            return fallback
        if _is_low_value_medical_answer(candidate, deterministic_plan):
            return fallback
        if "أسئلة توضيحية" in candidate or "Clarifying Questions" in candidate:
            return fallback
        if deterministic_plan.next_best_question and _question_frequency(candidate, deterministic_plan.next_best_question) > 1:
            return fallback
        if _question_count(candidate) > MAX_V3_QUESTIONS:
            return fallback
        if _has_unrelated_body_part(candidate, deterministic_plan.domain, guard_state):
            return fallback
        if _has_answered_question_contamination(candidate, guard_state):
            return fallback
        if _answer_conflicts_with_denied_facts(candidate, guard_state):
            return _denial_safe_fallback(deterministic_plan, guard_state)
        if deterministic_plan.domain == "side_pain" and case_state.facts.get("side_correction"):
            corrected_side = case_state.facts.get("side")
            expected_terms = {"left": {"الشمال", "left"}, "right": {"اليمين", "right"}}.get(str(corrected_side), set())
            if expected_terms and not any(term in lowered or term in candidate for term in expected_terms):
                return fallback
        if not candidate or any(pattern in lowered for pattern in INTERNAL_LEAK_PATTERNS):
            return fallback
        if deterministic_plan.risk_level == "emergency":
            if case_state.language == "en":
                if "emergency" not in lowered:
                    return "These symptoms may be urgent. Please go to the emergency department now or call emergency services."
            elif "الطوارئ" not in candidate and "طوارئ" not in candidate:
                return f"{HIGH_URGENCY_PREFIX}\n\nاتجه للطوارئ فورًا ولا تنتظر."
        if deterministic_plan.domain == "headache":
            blocked = {"back pain", "cervical", "neck", "الظهر", "فقرات الرقبة", "الرقبة"}
            if any(term in lowered or term in candidate for term in blocked):
                return self._render_answer(fallback_plan, guard_state, MessageMeaning(language=guard_state.language, domain="headache"), ChatRequest(message="fallback"))
        if deterministic_plan.domain == "back_pain":
            blocked = {"cervical", "neck pain", "malaria", "aids", "فقرات الرقبة", "الرقبة", "ملاريا", "الإيدز"}
            if any(term in lowered or term in candidate for term in blocked):
                return self._render_answer(fallback_plan, guard_state, MessageMeaning(language=guard_state.language, domain="back_pain"), ChatRequest(message="fallback"))
        if deterministic_plan.domain == "chest_pain" and deterministic_plan.risk_level != "emergency":
            blocked = {"heart attack", "أزمة قلبية", "جلطة قلبية", "اشتباه مشكلة قلبية طارئة"}
            if any(term in lowered or term in candidate for term in blocked):
                return self._render_answer(fallback_plan, guard_state, MessageMeaning(language=guard_state.language, domain="chest_pain"), ChatRequest(message="fallback"))
        return _limit_to_one_question(candidate)

    def _should_run_final_judge(self, plan: ClinicalPlan) -> bool:
        if plan.risk_level == "emergency":
            return False
        if plan.intent in JUDGE_SKIP_INTENTS:
            return False
        if plan.domain in {"casual", "off_topic", "abuse", "closing", "nonsense", "non_medical"}:
            return False
        return plan.response_goal in {"clarify", "guide", "acknowledge", "diagnose", "reply"}

    def _response_from_plan(
        self,
        *,
        request: ChatRequest,
        plan: ClinicalPlan,
        deterministic_plan: ClinicalPlan,
        answer: str,
        symptoms: list[str],
        case_state: ClinicalCaseState,
        current_meaning: MessageMeaning,
        trace: dict[str, Any],
    ) -> ChatResponse:
        if plan.risk_level == "emergency":
            mode = EMERGENCY_MODE
        elif plan.response_goal == "diagnose" or plan.diagnosis:
            mode = DIAGNOSIS_MODE
        else:
            mode = CLARIFICATION_MODE
        urgency = _urgency_from_risk(plan.risk_level)
        questions = _safe_follow_up_questions(plan.questions, case_state)[:MAX_V3_QUESTIONS] if urgency != "High" else []
        doctor = plan.doctor_route or _doctor_for_domain(plan.domain)
        if _doctor_route_conflicts_with_denials(doctor, case_state):
            doctor = _doctor_for_domain(case_state.active_domain)
        diagnosis = plan.diagnosis or (plan.broad_possibilities[0] if mode == DIAGNOSIS_MODE and plan.broad_possibilities else None)
        if mode == CLARIFICATION_MODE and not diagnosis and plan.domain in {"unknown", "non_medical", ""}:
            doctor = "Needs more information"
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=mode,
            answer=_limit_to_one_question(answer),
            extracted_symptoms=symptoms,
            possible_diagnosis=diagnosis,
            display_diagnosis_ar=display_diagnosis_ar(diagnosis) if diagnosis else None,
            confidence=plan.confidence if diagnosis else 0.0,
            urgency_level=urgency,
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=bool(questions),
            follow_up_questions=questions,
            retrieved_cases=[],
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=current_meaning,
            case_state=case_state,
            plan=plan,
            trace=trace,
            engine_route=f"v3_{mode}",
        )
        return response

    def _closing_response(
        self,
        request: ChatRequest,
        language: str,
        case_state: ClinicalCaseState,
        trace: dict[str, Any],
    ) -> ChatResponse:
        answer = "You’re welcome. Wishing you good health." if language == "en" else "العفو، أتمنى لك الصحة والسلامة."
        doctor = "Not needed"
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=CLOSING_MODE,
            answer=answer,
            extracted_symptoms=[],
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=None,
            case_state=ClinicalCaseState(language=language, case_closed=True),
            plan=ClinicalPlan("closing", "close", "closing", "Case closed.", doctor_route=doctor, response_goal="close"),
            trace=trace,
            engine_route="v3_closing",
        )
        return response

    def _casual_pause_response(
        self,
        request: ChatRequest,
        language: str,
        case_state: ClinicalCaseState,
        meaning: MessageMeaning,
        trace: dict[str, Any],
    ) -> ChatResponse:
        case_state.paused = True
        is_family_chat = _is_family_or_personal_chat(request.message)
        is_identity_chat = _is_identity_or_name_chat(request.message)
        if language == "en":
            if is_family_chat:
                answer = "I can’t know your family personally, but I hope they’re well."
                questions: list[str] = []
            elif is_identity_chat:
                answer = "I’m MedBridge AI, here to help you think through symptoms safely."
                questions = []
            elif case_state.active_domain == "headache":
                answer = "Got it. Let’s keep the medical thread in view: "
                answer += "since you mentioned headache, did it start suddenly or gradually?"
                questions = ["Did the headache start suddenly or gradually?"]
            else:
                answer = "Got it. Let’s keep the medical thread in view: "
                answer += "tell me the next detail about the symptom we were discussing."
                questions = ["What changed or what detail can you add about the symptom?"]
        else:
            if is_family_chat:
                answer = "ماعرفش مامتك أو أهلك شخصيًا، بس أتمنى يكونوا بخير."
                questions = []
            elif is_identity_chat:
                answer = "أنا MedBridge AI، موجود عشان أساعدك تفهم الأعراض بشكل آمن ومختصر."
                questions = []
            elif case_state.active_domain == "headache":
                answer = "تمام، نرجع للموضوع الصحي اللي كنا بنتكلم فيه: "
                if case_state.facts.get("vision_change"):
                    answer += "إنت قلت إن في زغللة مع الصداع، فالصداع بدأ فجأة ولا تدريجي؟"
                    questions = ["الصداع بدأ فجأة ولا تدريجي؟"]
                else:
                    answer += "الصداع بدأ فجأة ولا تدريجي؟"
                    questions = ["الصداع بدأ فجأة ولا تدريجي؟"]
            else:
                answer = "تمام، نرجع للموضوع الصحي اللي كنا بنتكلم فيه: "
                answer += "قولّي إيه الجديد في العرض اللي كنا بنتكلم عنه؟"
                questions = ["إيه الجديد في العرض اللي كنا بنتكلم عنه؟"]
        doctor = _doctor_for_domain(case_state.active_domain)
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=CLARIFICATION_MODE,
            answer=answer,
            extracted_symptoms=case_state.symptoms,
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        plan = ClinicalPlan(
            intent=meaning.intent,
            case_action="pause",
            domain=case_state.active_domain or "casual",
            clinical_summary=_summary_for_case(case_state, language),
            next_best_question=questions[0] if questions else None,
            doctor_route=doctor,
            response_goal="acknowledge",
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=meaning,
            case_state=case_state,
            plan=plan,
            trace=trace,
            engine_route="v3_pause_resume",
        )
        return response

    def _non_medical_or_fallback(
        self,
        request: ChatRequest,
        fallback_handler: Callable[[ChatRequest], ChatResponse] | None,
        meaning: MessageMeaning,
        case_state: ClinicalCaseState,
        trace: dict[str, Any],
    ) -> ChatResponse:
        if fallback_handler:
            response = fallback_handler(request)
            response.case_state_update.setdefault("engine", "v2")
            response.case_state_update.setdefault("engine_route", "v3_nonmedical_fallback")
            return response
        language = meaning.language
        if meaning.intent == "abuse":
            answer = (
                "I can help with health symptoms, but we need to keep it respectful."
                if language == "en"
                else "خلينا نتكلم باحترام عشان أقدر أساعدك. لو عندك عرض صحي اكتبه بوضوح."
            )
        elif meaning.intent == "nonsense":
            answer = (
                "I could not understand that as a health concern. Write the symptom clearly."
                if language == "en"
                else "مش قادر أفهم الرسالة دي كشكوى صحية. اكتب العرض بشكل أوضح."
            )
        elif meaning.intent in {"casual", "greeting"}:
            normalized = normalize_text(request.message)
            if language == "en":
                answer = (
                    "Hi, I’m here as a medical assistant. Tell me any health symptom or medical question when you’re ready."
                    if "hi" in normalized or "hello" in normalized
                    else "I’m here as a medical assistant. Send me any health symptom or medical question when you’re ready."
                )
            else:
                if _has_any(normalized, {"اهلا", "هاي", "هلا"}):
                    answer = "أهلا بيك، أنا مساعد طبي. ابعتلي أي عرض صحي أو سؤال طبي لما تحب."
                elif _has_any(normalized, {"عامل اي", "ازيك"}):
                    answer = "أنا تمام، وموجود كمساعد طبي لو عندك عرض صحي أو سؤال طبي."
                else:
                    answer = "تمام، موجود معاك كمساعد طبي لو عندك عرض صحي أو سؤال طبي."
        else:
            answer = (
                "I’m a medical assistant, so tell me your symptom or health question."
                if language == "en"
                else "أنا مساعد طبي، فخلينا في صحتك أو أي سؤال طبي تحب تسأله."
            )
        doctor = "Not needed"
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=CLARIFICATION_MODE,
            answer=answer,
            extracted_symptoms=[],
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=meaning,
            case_state=case_state,
            plan=ClinicalPlan(meaning.intent, "pause", meaning.intent, "Non-medical turn.", doctor_route=doctor),
            trace=trace,
            engine_route="v3_nonmedical",
        )
        return response

    def _emergency_response(
        self,
        *,
        request: ChatRequest,
        signal: EmergencySignal,
        language: str,
        symptoms: list[str],
        case_state: ClinicalCaseState,
        current_meaning: MessageMeaning,
        trace: dict[str, Any],
    ) -> ChatResponse:
        if language == "en":
            reason = signal.reason if not _contains_arabic(signal.reason) else "The symptom combination includes a serious red flag."
            answer = (
                "These symptoms may be urgent. Please go to the emergency department now or call emergency services.\n\n"
                f"Main concern: {signal.diagnosis}.\n\n"
                f"Why: {reason}\n\n"
                "Do not wait, monitor only, or drive yourself if symptoms are severe or started suddenly."
            )
        else:
            answer = (
                f"{HIGH_URGENCY_PREFIX}\n\n"
                f"السبب: {signal.reason}\n\n"
                f"التوجيه الأنسب: {signal.display_doctor_ar}.\n\n"
                "لا تنتظر ولا تكتفي بالمراقبة في البيت، وخلي شخص قريب يساعدك في الوصول للرعاية العاجلة."
            )
        response = ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=EMERGENCY_MODE,
            answer=answer,
            extracted_symptoms=symptoms,
            possible_diagnosis=signal.diagnosis,
            display_diagnosis_ar=signal.display_diagnosis_ar or display_diagnosis_ar(signal.diagnosis),
            confidence=0.99,
            urgency_level="High",
            suggested_doctor=signal.doctor,
            display_doctor_ar=signal.display_doctor_ar or display_doctor_ar(signal.doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        plan = ClinicalPlan(
            intent="medical",
            case_action="continue",
            domain=case_state.active_domain or current_meaning.domain or signal.category,
            clinical_summary=_summary_for_case(case_state, language),
            risk_level="emergency",
            risk_reasons=[signal.reason],
            doctor_route=signal.doctor,
            response_goal="escalate",
            deterministic_override=True,
        )
        response.case_state_update = self._case_update(
            request=request,
            response=response,
            current_meaning=current_meaning,
            case_state=case_state,
            plan=plan,
            trace=trace,
            engine_route="v3_emergency",
        )
        return response

    def _case_update(
        self,
        *,
        request: ChatRequest,
        response: ChatResponse,
        current_meaning: MessageMeaning | None,
        case_state: ClinicalCaseState,
        plan: ClinicalPlan,
        trace: dict[str, Any],
        engine_route: str,
    ) -> dict[str, Any]:
        self._apply_response_question_state(case_state, response.follow_up_questions)
        started_at = trace.pop("_started_at", None)
        if started_at and trace.get("total_latency_ms") is None:
            trace["total_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        legacy_state = CaseState(
            active_domain=case_state.active_domain,
            active_body_part=case_state.active_body_part,
            symptoms=case_state.symptoms,
            facts=case_state.facts,
            denials=case_state.denials,
            answered_questions=case_state.answered_questions,
            pending_questions=response.follow_up_questions,
            doctor_route=response.suggested_doctor,
            language=case_state.language,
            case_closed=case_state.case_closed,
        )
        active_case_dict = case_state.to_dict()
        active_case_id = active_case_dict.get("case_id") or f"{case_state.active_domain or 'unknown'}-active"
        active_case_dict["case_id"] = active_case_id
        return {
            "engine": "v3",
            "engine_route": engine_route,
            "source": request.source,
            "language": case_state.language or request.language or "ar",
            "conversation_summary": _summary_for_case(case_state, case_state.language or request.language or "ar"),
            "active_case_id": active_case_id,
            "paused_case_ids": [],
            "cases": [active_case_dict],
            "known_symptoms": response.extracted_symptoms,
            "follow_up_questions": response.follow_up_questions,
            "follow_up_question_ids": case_state.pending_question_ids,
            "current_meaning": current_meaning.to_dict() if current_meaning else {},
            "active_case": active_case_dict,
            "medical_meaning": _legacy_medical_meaning_compat(current_meaning, legacy_state),
            "denied_concepts": sorted(case_state.denials - {"generic_no"}),
            "conversation_id": request.conversation_id or request.session_id,
            "mode": response.mode,
            "urgency_level": response.urgency_level,
            "clinical_plan": {
                "intent": plan.intent,
                "case_action": plan.case_action,
                "domain": plan.domain,
                "risk_level": plan.risk_level,
                "risk_reasons": plan.risk_reasons,
                "candidate_conditions": plan.candidate_conditions,
                "selected_possibilities": plan.selected_possibilities,
                "analysis_for_patient": plan.analysis_for_patient,
                "patient_answer_source": "llm_planner" if plan.patient_answer and plan.source == "llm_planner" else "deterministic_renderer",
                "next_question_id": plan.next_question_id,
                "response_goal": plan.response_goal,
                "source": plan.source,
            },
            "engine_trace": self._finalize_trace(trace),
        }

    def _apply_response_question_state(self, case_state: ClinicalCaseState, questions: list[str]) -> None:
        ids: list[str] = []
        for question in questions:
            for question_id in _question_ids_for_text(question, case_state.active_domain):
                if question_id not in ids:
                    ids.append(question_id)
        case_state.pending_questions = list(questions)
        case_state.pending_question_ids = ids
        case_state.last_question_id = ids[0] if ids else None
        case_state.asked_question_ids.update(ids)
        case_state.asked_question_slots.update(ids)

    def _fallback(
        self,
        request: ChatRequest,
        fallback_handler: Callable[[ChatRequest], ChatResponse] | None,
        *,
        route: str,
    ) -> ChatResponse:
        if fallback_handler:
            response = fallback_handler(request)
            response.case_state_update.setdefault("engine_route", route)
            return response
        language = _detect_language(request.message, request.language)
        doctor = "Needs more information"
        answer = "I need one clearer medical detail first." if language == "en" else "محتاج تفاصيل طبية أوضح عشان أساعدك بأمان."
        return ChatResponse(
            conversation_id=request.conversation_id or request.session_id,
            mode=CLARIFICATION_MODE,
            answer=answer,
            extracted_symptoms=[],
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=doctor,
            display_doctor_ar=display_doctor_ar(doctor),
            precautions=[],
            needs_follow_up=True,
            follow_up_questions=["What symptom is bothering you most?"] if language == "en" else ["إيه أكتر عرض مضايقك؟"],
            retrieved_cases=[],
            case_state_update={
                "engine": "v3",
                "engine_route": route,
                "source": request.source,
                "language": language,
                "known_symptoms": [],
                "follow_up_questions": [],
                "conversation_id": request.conversation_id or request.session_id,
            },
        )

    def _plan_to_dict(self, plan: ClinicalPlan) -> dict[str, Any]:
        return {
            "intent": plan.intent,
            "case_action": plan.case_action,
            "domain": plan.domain,
            "clinical_summary": plan.clinical_summary,
            "new_facts": plan.new_facts,
            "new_denials": plan.new_denials,
            "risk_level": plan.risk_level,
            "risk_reasons": plan.risk_reasons,
            "broad_possibilities": plan.broad_possibilities,
            "candidate_conditions": plan.candidate_conditions,
            "selected_possibilities": plan.selected_possibilities,
            "analysis_for_patient": plan.analysis_for_patient,
            "patient_answer": plan.patient_answer,
            "diagnosis": plan.diagnosis,
            "confidence": plan.confidence,
            "next_question_id": plan.next_question_id,
            "questions_to_ask": plan.questions,
            "next_best_question": plan.next_best_question,
            "optional_second_question": plan.optional_second_question,
            "care_guidance": plan.care_guidance,
            "doctor_route": plan.doctor_route,
            "response_goal": plan.response_goal,
            "must_not_repeat": plan.must_not_repeat,
            "forbidden_topics": plan.forbidden_topics,
        }


FACT_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "fever": {"patterns": ("حرارة", "سخونية", "سخونيه", "حمى", "fever", "temperature"), "facts": ("fever", "temperature"), "symptoms": ("mild_fever", "high_fever")},
    "blood_in_urine": {"patterns": ("دم في البول", "دم واضح في البول", "دم ظاهر في البول", "بول بدم", "البول فيه دم", "blood in urine", "blood in my urine", "blood in the urine", "urine has blood", "bloody urine"), "facts": ("blood_in_urine",), "symptoms": ("spotting_ urination",)},
    "flank_pain": {"patterns": ("ألم في الجنب", "الم في الجنب", "ألم شديد في الجنب", "الم شديد في الجنب", "وجع في الجنب", "ألم جنب", "الم جنب", "وجع جنب", "الخاصرة", "الخاصره", "flank pain", "side pain"), "facts": ("flank_pain",), "symptoms": ("side_pain",)},
    "back_pain": {"patterns": ("ألم في الظهر", "الم في الظهر", "وجع في الظهر", "ألم في الضهر", "الم في الضهر", "وجع في الضهر", "الظهر", "الضهر", "back pain", "back hurts"), "facts": ("back_pain",), "symptoms": ("back_pain",)},
    "vomiting": {"patterns": ("ترجيع", "قيء", "استفراغ", "vomiting", "throwing up"), "facts": ("vomiting_present",), "symptoms": ("vomiting",)},
    "weakness": {"patterns": ("ضعف", "مش قادر احرك", "weakness", "weak"), "facts": ("weakness", "bladder_bowel_or_leg_weakness"), "symptoms": ("weakness", "weakness_of_one_body_side")},
    "numbness": {"patterns": ("تنميل", "خدر", "numb", "numbness"), "facts": ("numbness",), "symptoms": ("numbness",)},
    "speech_change": {"patterns": ("كلام بصعوبة", "تقل في الكلام", "لخبطة كلام", "لخبطة في الكلام", "speech is slurred", "slurred speech"), "facts": ("speech_change",), "symptoms": ("slurred_speech",)},
    "breathlessness": {"patterns": ("ضيق تنفس", "صعوبة تنفس", "مش قادر اتنفس", "نفسي بيقطع", "نهجان", "shortness of breath", "difficulty breathing", "breathless"), "facts": ("breathlessness", "exertional_breathlessness"), "symptoms": ("breathlessness",)},
    "chest_pain": {"patterns": ("ألم صدر", "ألم في الصدر", "ألم في صدري", "الم صدر", "الم في الصدر", "الم في صدري", "وجع صدر", "وجع في الصدر", "وجع في صدري", "ضغط في صدري", "chest pain", "chest pressure"), "facts": ("chest_pain", "chest_radiation"), "symptoms": ("chest_pain",)},
    "bleeding": {"patterns": ("نزيف", "دم في القيء", "دم في البراز", "vomiting blood", "blood in stool", "bleeding"), "facts": ("blood_vomit_or_stool",), "symptoms": ("stomach_bleeding",)},
    "fainting": {"patterns": ("اغماء", "إغماء", "فقدت الوعي", "فقدان الوعي", "fainting", "fainted", "passed out"), "facts": ("fainting",), "symptoms": ("coma",)},
    "severe_pain": {"patterns": ("ألم شديد", "الم شديد", "وجع شديد", "severe pain"), "facts": ("severity", "severe_abdominal_pain"), "symptoms": ()},
    "vision_change": {"patterns": ("زغللة", "زغلله", "تشوش", "تغير في النظر", "blurred vision", "vision change"), "facts": ("vision_change", "sudden_vision_loss"), "symptoms": ("blurred_and_distorted_vision",)},
    "pregnancy": {"patterns": ("حامل", "حمل", "pregnant", "pregnancy"), "facts": ("pregnancy",), "symptoms": ("gynecology_context",)},
    "face_lip_swelling": {"patterns": ("تورم الشفاه", "تورم الوجه", "تورم الوش", "face swelling", "lip swelling"), "facts": ("face_lip_swelling",), "symptoms": ()},
    "urinary_burning": {"patterns": ("حرقان بول", "حرقان في البول", "حرقان وانا بتبول", "حرقان لما بتبول", "burning urination"), "facts": ("urinary_burning",), "symptoms": ("burning_micturition",)},
    "urinary_frequency": {"patterns": ("كثرة التبول", "تبول كتير", "بول كتير", "بخش الحمام كتير", "بدخل الحمام كتير", "بروح الحمام كتير", "frequent urination"), "facts": ("urinary_frequency",), "symptoms": ("continuous_feel_of_urine",)},
}

NEGATION_SCOPE_MARKERS = (
    "ومش حاسس ب",
    "ومش حاسه ب",
    "ومش عندي",
    "وما عنديش",
    "وماعنديش",
    "ومافيش",
    "وما فيش",
    "ومفيش",
    "ولا يوجد",
    "وبدون",
    "ومن غير",
    "مش حاسس ب",
    "مش حاسه ب",
    "مش عندي",
    "ما عنديش",
    "ماعنديش",
    "مافيش",
    "ما فيش",
    "مفيش",
    "لا يوجد",
    "لا اعاني من",
    "لا أعاني من",
    "من غير",
    "بدون",
    "ولا",
    "لا",
    "i do not have",
    "i don't have",
    "not experiencing",
    "there is no",
    "without",
    "not",
    "no",
)
CONTRAST_SCOPE_MARKERS = (" بس ", " لكن ", " ولكن ", " انما ", " بينما ", " but ", " however ", " except ")

GENERAL_FACT_QUESTION_MARKERS = (
    "هل وجود",
    "هل ظهور",
    "هل الدم",
    "هل ممكن",
    "ممكن يكون",
    "خايف يكون",
    "خايفة يكون",
    "الدكتور سالني لو",
    "الدكتور سألني لو",
    "doctor asked",
    "asked me if",
    "is blood",
    "can blood",
    "could it be",
    "worried it is",
)
PERSONAL_ASSERTION_MARKERS = (
    "عندي",
    "عندى",
    "حاسس",
    "حاسه",
    "حاسة",
    "بعاني",
    "باعاني",
    "اشعر",
    "أشعر",
    "i have",
    "i am having",
    "i feel",
)
RESOLVED_FACT_MARKERS = (
    "وقف",
    "وقفت",
    "راح",
    "اختفى",
    "اختفت",
    "انتهى",
    "انتهت",
    "مش موجود دلوقتي",
    "مش موجود حاليا",
    "stopped",
    "resolved",
    "gone",
    "went away",
    "not happening now",
)


def _extract_v3_concepts(message: str, classifier_symptoms: list[str]) -> tuple[dict[str, Any], list[str], str | None, str | None, set[str]]:
    text = normalize_text(_repair_mojibake(message))
    facts: dict[str, Any] = {}
    symptoms: list[str] = []
    body_part: str | None = None
    domain_hint: str | None = None
    explicit_denials = _explicitly_denied_fact_keys(message)
    inactive_mentions = _inactive_fact_mentions(message)
    resolved_mentions = _resolved_fact_keys(message)
    explicit_denials.update(resolved_mentions)

    if _has_any(text, {"صداع", "وجع راس", "وجع دماغ", "راسي", "دماغي", "headache", "head pain"}):
        symptoms.append("headache")
        body_part = "head"
        domain_hint = "headache"
    if domain_hint == "headache":
        if _has_any(text, {"ناحية واحدة", "جهه واحدة", "جهة واحدة", "جانب واحد", "نص راسي", "نص دماغي", "one side", "one-sided", "unilateral"}):
            facts["unilateral_headache"] = True
        if _has_any(text, {"نابض", "بينبض", "نبض", "pulsating", "throbbing"}):
            facts["pulsating_headache"] = True
        if _has_any(text, {"النور بيضايق", "الضوء بيضايق", "النور", "الضوء", "حساسية للنور", "حساسية للضوء", "light bothers", "light sensitivity", "photophobia"}):
            facts["photophobia"] = True
        if _has_any(text, {"الصوت بيضايق", "الاصوات بتضايق", "الأصوات بتضايق", "الصوت", "الأصوات", "الاصوات", "حساسية للصوت", "sound bothers", "sound sensitivity", "phonophobia"}):
            facts["phonophobia"] = True
    if _has_any(text, {"زغللة", "زغلله", "تشوش", "تغير في النظر", "النظر اتغير", "blurred vision", "vision change", "visual change"}):
        facts["vision_change"] = True
        symptoms.append("blurred_and_distorted_vision")
        body_part = body_part or "head"
    if _has_any(text, {"فجأة", "فجاه", "مفاجئ", "بدأ فجأة", "sudden", "suddenly"}):
        facts["sudden_onset"] = True
    if _has_any(text, {"تدريجي", "بالتدريج", "مش فجأة", "gradual", "gradually", "not sudden"}):
        facts["sudden_onset"] = False
    if _has_any(text, {"شديد", "جامد", "اوي", "قوي", "اسوء", "أسوأ", "severe", "worst"}):
        facts["severity"] = "severe"
    temperature_match = re.search(r"(?:الحرارة|حرارتي|temperature|temp)?\s*(3[89]|4[0-2])", text)
    if temperature_match:
        facts["temperature"] = temperature_match.group(1)
        facts["fever"] = True
        symptoms.append("high_fever")
        domain_hint = domain_hint or "infectious"
        body_part = body_part or "general"
    if _has_any(text, {"تعب", "تعبانة", "ارهاق", "مرهق", "fatigue", "tired", "body aches", "body ache"}):
        facts["fatigue"] = True
        symptoms.append("fatigue")
        if _has_any(text, {"body aches", "body ache"}):
            domain_hint = domain_hint or "body_ache"
    if _has_any(text, {"جسمي مكسر", "تكسير", "جسمي واجعني", "جسمي واجعنى", "وجع الجسم", "وجع في الجسم"}):
        facts["fatigue"] = True
        symptoms.append("fatigue")
        domain_hint = domain_hint or "body_ache"
    if _has_any(text, {"مش مرتاح", "مش مرتاحة", "مش مرتاحه", "مش كويس", "حاسس مش كويس", "not feeling well", "unwell"}):
        facts["vague_discomfort"] = True
        symptoms.append("general_discomfort")
        domain_hint = domain_hint or "unknown"
    if _has_any(text, {"تنميل", "نص جسمي", "ناحية واحدة", "one side", "numb", "numbness"}):
        facts["numbness"] = True
        symptoms.append("numbness")
        domain_hint = domain_hint or "neurology_vague"
    if _has_any(text, {"ضعف", "مش قادر احرك", "weakness", "weak"}):
        facts["weakness"] = True
        symptoms.append("weakness")
        domain_hint = domain_hint or "neurology_vague"
    if _has_any(text, {"دوخة", "دوار", "الدنيا بتلف", "الدنيا بتتهز", "الدنيا بتلف بيا", "حاسس الدنيا بتتهز", "عدم اتزان", "مش متزن", "مش ثابت", "مش قادر اثبت", "حاسس مش ثابت", "بفقد توازني", "فقد توازن", "فاقد توازني", "توازني", "dizzy", "dizziness", "vertigo", "spinning", "loss of balance", "unsteady"}):
        symptoms.append("dizziness")
        if _has_any(text, {"الدنيا بتلف", "دوار", "vertigo", "spinning"}):
            symptoms.append("spinning_movements")
        if _has_any(text, {"عدم اتزان", "مش متزن", "مش ثابت", "مش قادر اثبت", "الدنيا بتتهز", "بفقد توازني", "فقد توازن", "فاقد توازني", "توازني", "loss of balance", "unsteady"}):
            symptoms.append("loss_of_balance")
        domain_hint = domain_hint or "vestibular_ent"
        body_part = body_part or "head"
    if _has_any(text, {"طنين", "ودني", "أذني", "اذني", "ear ringing", "tinnitus", "ringing in my ear"}):
        facts["tinnitus"] = True
        symptoms.append("tinnitus")
        domain_hint = "vestibular_ent"
        body_part = "ear"
    if _has_any(text, {"كلام بصعوبة", "تقل في الكلام", "speech is slurred", "slurred speech", "speech is weird", "speech feels weird", "speech getting weird", "speech is getting weird", "my speech is getting weird", "talking weird", "talk weird"}):
        facts["speech_change"] = True
        symptoms.append("slurred_speech")
        domain_hint = domain_hint or "neurology_vague"
    if _has_any(
        text,
        {
            "ألم صدر",
            "ألم في صدري",
            "ألم في الصدر",
            "الم صدر",
            "الم في صدري",
            "الم في الصدر",
            "وجع صدر",
            "وجع في صدري",
            "وجع في الصدر",
            "صدري واجعني",
            "ضغط في صدري",
            "ضغط على صدري",
            "ضغط على الصدر",
            "chest pain",
            "chest hurts",
            "chest pressure",
            "pressure in my chest",
        },
    ):
        facts["chest_pain"] = True
        symptoms.append("chest_pain")
        body_part = "chest"
        domain_hint = "chest_pain"
    if _has_any(text, {"ضيق تنفس", "صعوبة تنفس", "مش قادر اتنفس", "مش قادر أتنفس", "نفسي بيقطع", "نفسي بيقطع لما اتحرك", "نهجان", "بنهج", "نفس قصير", "بتخنق من أقل مجهود", "بتخنق من اقل مجهود", "بخنق من أقل مجهود", "بخنق من اقل مجهود", "shortness of breath", "difficulty breathing", "breathless", "short of breath"}):
        facts["breathlessness"] = True
        symptoms.append("breathlessness")
        if domain_hint != "chest_pain":
            domain_hint = "respiratory"
            body_part = "respiratory"
        if _has_any(text, {"لما اتحرك", "مع الحركة", "مع المجهود", "أقل مجهود", "اقل مجهود", "أطلع السلم", "اطلع السلم", "السلم", "on exertion", "when moving", "stairs"}):
            facts["exertional_breathlessness"] = True
    if _has_any(text, {"عرق بارد", "تعرق بارد", "cold sweat"}):
        facts["cold_sweat"] = True
        symptoms.append("sweating")
    if _has_any(text, {"بيمتد للذراع", "دراعي الشمال", "الفك", "jaw", "arm", "radiating"}):
        facts["chest_radiation"] = True
    if _has_any(text, {"اغماء", "إغماء", "fainting", "fainted"}):
        facts["fainting"] = True
    if _has_any(text, {"فقدت الوعي", "فقدان الوعي", "loss of consciousness", "passed out"}):
        facts["fainting"] = True
    if _has_any(text, {"جنبي", "جنبى", "الجنب", "وجع جنب", "ألم جنب", "side pain", "my side hurts"}):
        symptoms.append("side_pain")
        domain_hint = "side_pain"
        body_part = "side"
        if _has_any(text, {"اليمين", "يمين", "right"}):
            facts["side"] = "right"
            body_part = "right_side"
        if _has_any(text, {"الشمال", "شمال", "left"}):
            facts["side"] = "left"
            body_part = "left_side"
        if _has_any(text, {"قصدي", "معلش", "تصحيح", "sorry", "I mean"}):
            facts["side_correction"] = True
    elif _has_any(text, {"قصدي الشمال", "معلش قصدي الشمال", "قصدي اليمين", "معلش قصدي اليمين", "I mean left", "I mean right"}):
        symptoms.append("side_pain")
        domain_hint = "side_pain"
        facts["side_correction"] = True
        if _has_any(text, {"الشمال", "left"}):
            facts["side"] = "left"
            body_part = "left_side"
        elif _has_any(text, {"اليمين", "right"}):
            facts["side"] = "right"
            body_part = "right_side"
    if _has_any(text, {"ضهري", "ظهري", "الظهر", "الضهر", "back pain", "back hurts", "upper back", "lower back"}):
        symptoms.append("back_pain")
        body_part = "back"
        domain_hint = "back_pain"
        if _has_any(text, {"اسفل الظهر", "أسفل الظهر", "اسفل الضهر", "أسفل الضهر", "lower back"}):
            facts["location"] = "lower_back"
            body_part = "lower_back"
        elif _has_any(text, {"اعلى الظهر", "أعلى الظهر", "اعلى الضهر", "أعلى الضهر", "upper back"}):
            facts["location"] = "upper_back"
            body_part = "upper_back"
    if _has_any(text, {"كسر", "مكسور", "اتكسرت", "إصابة", "اصابة", "وقعت", "fracture", "broken", "injury"}):
        facts["injury_context"] = True
        symptoms.append("injury")
        domain_hint = "orthopedics"
        if _has_any(text, {"ايدي", "إيدي", "ايد", "يد", "hand", "arm"}):
            body_part = "arm_or_hand"
    if _has_any(text, {"رقبة", "رقبتي", "وجع رقبة", "ألم رقبة", "neck pain", "neck hurts"}):
        symptoms.append("neck_pain")
        body_part = "neck"
        if domain_hint not in {"neurology_vague", "chest_pain"}:
            domain_hint = "neck_pain"
    if _has_any(text, {"ضغط", "ضغطي", "ضغط الدم", "blood pressure", "hypertension"}):
        facts["blood_pressure_context"] = True
        symptoms.append("high_blood_pressure")
        if domain_hint != "chest_pain":
            domain_hint = "cardiology"
            body_part = "cardiology"
    if _has_any(text, {"سكر", "سكري", "قياس السكر", "sugar", "diabetes", "glucose"}):
        facts["sugar_context"] = True
        symptoms.append("irregular_sugar_level")
        domain_hint = "endocrine"
        body_part = "endocrine"
    if _has_any(text, {"حامل", "حمل", "pregnant", "pregnancy"}):
        facts["pregnancy"] = True
        symptoms.append("gynecology_context")
        domain_hint = "gynecology"
        body_part = "gynecology"
    if _has_any(text, {"ابني", "بنتي", "طفلي", "طفل", "رضيع", "baby", "child", "my son", "my daughter"}):
        facts["pediatric_context"] = True
        if domain_hint not in {"chest_pain", "neurology_vague", "digestive", "urinary", "skin"}:
            domain_hint = "pediatrics"
            body_part = body_part or "pediatrics"
    if _has_any(text, {"افرازات", "إفرازات", "البريود", "الدورة", "period", "menstrual", "cramps", "discharge", "vaginal"}):
        symptoms.append("gynecology_context")
        body_part = "gynecology"
        domain_hint = "gynecology"
        if _has_any(text, {"افرازات", "إفرازات", "discharge"}):
            symptoms.append("discharge")
            facts["discharge_present"] = True
        age_match = re.search(r"(\d{1,3})\s*(?:سنة|سنه|years?|yo)", text)
        if age_match:
            facts["age"] = int(age_match.group(1))
    if _has_any(text, {"زوري", "زورى", "الزور", "وجع الزور", "حرقان في الزور", "حلقي", "الحلق", "sore throat", "throat hurts", "my throat hurts"}):
        symptoms.append("throat_irritation")
        body_part = "throat"
        domain_hint = "throat_ent"
    if _has_any(text, {"كحة", "كحه", "سعال", "رشح", "احتقان", "سخونية", "سخونيه", "حرارة", "حمى", "fever", "cough", "runny nose", "congestion"}):
        has_airway_symptom = _has_any(text, {"كحة", "كحه", "سعال", "رشح", "احتقان", "cough", "runny nose", "congestion"})
        has_cold_congestion = _has_any(text, {"رشح", "احتقان", "runny nose", "congestion"})
        if has_cold_congestion and domain_hint in {None, "unknown", "throat_ent", "respiratory"}:
            domain_hint = "infectious"
        if domain_hint in {None, "unknown"}:
            domain_hint = "respiratory" if has_airway_symptom else "infectious"
        body_part = body_part or ("respiratory" if has_airway_symptom else "general")
        if _has_any(text, {"كحة", "كحه", "سعال", "cough"}):
            symptoms.append("cough")
        if _has_any(text, {"سخونية", "سخونيه", "حمى", "حرارة", "fever"}):
            symptoms.append("mild_fever")
            facts["fever"] = True
    if _has_any(text, {"بالليل", "بالليل اكتر", "بتزيد بالليل", "ليل", "at night", "nighttime", "worse at night"}):
        facts["night_worse"] = True
    if _has_any(text, {"بطني", "بطن", "معدتي", "معدة", "مغصان", "اسهال", "إسهال", "ترجيع", "قيء", "غثيان", "نفسي مقلوبة", "نفسي مقلوبه", "نفسى مقلوبة", "نفسى مقلوبه", "عايز ارجع", "عايزه ارجع", "عايزة ارجع", "احساس بالترجيع", "إحساس بالترجيع", "قرفان", "مش طايق الأكل", "مش طايق الاكل", "مش طايقة الأكل", "مش طايقة الاكل", "مش طايقه الأكل", "مش طايقه الاكل", "stomach", "abdomen", "diarrhea", "diarrhoea", "vomiting", "nausea", "nauseated", "feel like vomiting"}):
        has_core_digestive = _has_any(text, {"بطني", "بطن", "معدتي", "معدة", "اسهال", "إسهال", "stomach", "abdomen", "diarrhea", "diarrhoea"})
        if domain_hint not in {"gynecology", "vestibular_ent", "neurology_vague", "headache"} or has_core_digestive:
            domain_hint = "digestive"
            body_part = "abdomen"
        if _has_any(text, {"بطني", "بطن", "معدتي", "معدة", "stomach", "abdomen"}):
            symptoms.append("abdominal_pain")
        if _has_any(text, {"نص البطن", "نصف البطن", "وسط البطن", "منتصف البطن", "middle abdomen", "center abdomen", "centre abdomen"}):
            facts["abdominal_location"] = "middle_abdomen"
        elif _has_any(text, {"يمين البطن", "الناحية اليمين", "right abdomen", "right side"}):
            facts["abdominal_location"] = "right_abdomen"
        elif _has_any(text, {"شمال البطن", "الناحية الشمال", "left abdomen", "left side"}):
            facts["abdominal_location"] = "left_abdomen"
        elif _has_any(text, {"فوق البطن", "upper abdomen"}):
            facts["abdominal_location"] = "upper_abdomen"
        elif _has_any(text, {"تحت البطن", "lower abdomen"}):
            facts["abdominal_location"] = "lower_abdomen"
        if _has_any(text, {"اسهال", "إسهال", "diarrhea", "diarrhoea"}):
            symptoms.append("diarrhoea")
            facts["diarrhea_present"] = True
        if _has_any(text, {"ترجيع", "قيء", "استفراغ", "vomiting", "throwing up"}):
            symptoms.append("vomiting")
            facts["vomiting_present"] = True
        if _has_any(text, {"غثيان", "نفسي مقلوبة", "نفسي مقلوبه", "نفسى مقلوبة", "نفسى مقلوبه", "عايز ارجع", "عايزه ارجع", "عايزة ارجع", "احساس بالترجيع", "إحساس بالترجيع", "قرفان", "مش طايق الأكل", "مش طايق الاكل", "مش طايقة الأكل", "مش طايقة الاكل", "مش طايقه الأكل", "مش طايقه الاكل", "nausea", "nauseated", "feel like vomiting"}):
            symptoms.append("nausea")
            facts["nausea_present"] = True
        if _has_any(text, {"تاريخ سابق", "حصل قبل كده", "قرحة", "قولون", "previous history", "happened before", "ulcer"}):
            facts["previous_gastric_history"] = not _is_negative_short_answer(text)
    if _has_any(text, {"بعد الأكل", "بعد الاكل", "بيزيد بعد الأكل", "بيزيد بعد الاكل", "مع الأكل", "مع الاكل", "after eating", "after food", "post meal", "post-meal"}):
        facts["post_meal_worse"] = True
        if domain_hint in {None, "unknown", "digestive"}:
            domain_hint = "digestive"
            body_part = body_part or "abdomen"
    urinary_context = _has_any(
        text,
        {
            "حرقان بول",
            "حرقان في البول",
            "حرقان وانا بتبول",
            "حرقان لما بتبول",
            "كثرة التبول",
            "تبول كتير",
            "بول كتير",
            "بخش الحمام كتير",
            "بدخل الحمام كتير",
            "بروح الحمام كتير",
            "دم في البول",
            "دم واضح في البول",
            "دم ظاهر في البول",
            "بول بدم",
            "البول",
            "بتبول",
            "التبول",
            "urine",
            "urinary",
            "urination",
            "burning urination",
            "blood in urine",
            "blood in my urine",
            "blood in the urine",
        },
    )
    if urinary_context:
        domain_hint = "urinary"
        body_part = "urinary"
        has_urinary_burning = _has_any(
            text,
            {"حرقان بول", "حرقان في البول", "حرقان وانا بتبول", "حرقان لما بتبول", "burning urination"},
        ) or (_has_any(text, {"حرقان"}) and _has_any(text, {"بول", "بتبول", "التبول", "urine", "urination"}))
        has_urinary_frequency = _has_any(
            text,
            {"كثرة التبول", "تبول كتير", "بول كتير", "بخش الحمام كتير", "بدخل الحمام كتير", "بروح الحمام كتير", "frequent urination"},
        )
        if has_urinary_burning:
            symptoms.append("burning_micturition")
            facts["urinary_burning"] = True
        if has_urinary_frequency:
            symptoms.append("continuous_feel_of_urine")
            facts["urinary_frequency"] = True
        if _has_any(text, {"دم في البول", "دم واضح في البول", "دم ظاهر في البول", "بول بدم", "blood in urine", "blood in my urine", "blood in the urine", "urine has blood"}):
            facts["blood_in_urine"] = True
            symptoms.append("spotting_ urination")
        if _has_any(text, {"ألم جنب", "الم جنب", "ألم شديد في الجنب", "الم شديد في الجنب", "وجع جنب", "flank pain"}):
            facts["flank_pain"] = True
    if _has_any(text, {"طفح", "حكة", "هرش", "حبوب", "احمرار", "rash", "itching", "itchy", "acne", "pimples"}):
        domain_hint = "skin"
        body_part = "skin"
        if _has_any(text, {"طفح", "rash"}):
            symptoms.append("skin_rash")
        if _has_any(text, {"حكة", "هرش", "itching", "itchy"}):
            symptoms.append("itching")
        if _has_any(text, {"حبوب", "acne", "pimples"}):
            symptoms.append("pus_filled_pimples")
    if _has_any(text, {"تورم الشفاه", "تورم الوجه", "face swelling", "lip swelling"}):
        facts["face_lip_swelling"] = True
    if _has_any(text, {"عيني", "العين", "ألم عين", "وجع عين", "تغير النظر", "زغللة", "vision", "eye pain", "red eye"}):
        if domain_hint not in {"headache", "neurology_vague", "vestibular_ent", "cardiology", "endocrine"}:
            domain_hint = "eye"
            body_part = "eye"
        symptoms.append("blurred_and_distorted_vision" if _has_any(text, {"زغللة", "vision change", "blurred vision"}) else "eye_pain")
        if _has_any(text, {"فقدان النظر", "مش شايف", "sudden vision loss"}):
            facts["sudden_vision_loss"] = True
    if _has_any(text, {"سني", "سنى", "سنان", "أسنان", "ضرس", "لثة", "tooth", "toothache", "gum"}):
        domain_hint = "dental"
        body_part = "mouth"
        symptoms.append("tooth_pain")
    food_or_nausea_context = _has_any(text, {"الأكل", "الاكل", "أكل", "اكل", "طعام", "نفسي مقلوبة", "نفسي مقلوبه", "غثيان", "قرفان", "food", "nausea", "nauseated"}) or facts.get("nausea_present")
    if _has_any(text, {"قلق", "اكتئاب", "مخنوق", "مش طايق", "anxiety", "panic", "depressed", "depression"}) and not food_or_nausea_context:
        domain_hint = "mental_health"
        body_part = "mental_health"
        symptoms.append("anxiety")
    if _has_any(text, {"عطش", "جوع شديد", "رعشة", "تعرق", "سكر", "sugar", "diabetes", "thirst", "shaking", "sweating", "hungry"}):
        if domain_hint not in {"chest_pain", "cardiology"}:
            domain_hint = "endocrine"
            body_part = "endocrine"
        if _has_any(text, {"عطش", "thirst"}):
            symptoms.append("excessive_thirst")
        if _has_any(text, {"جوع شديد", "hungry"}):
            symptoms.append("excessive_hunger")
        if _has_any(text, {"رعشة", "shaking"}):
            symptoms.append("shivering")
        if _has_any(text, {"تعرق", "sweating"}):
            symptoms.append("sweating")
    if _has_any(text, {"نزيف", "دم في القيء", "دم في البراز", "vomiting blood", "blood in stool"}):
        facts["blood_vomit_or_stool"] = True
    if _has_any(text, {"جفاف", "مش بتبول", "دوخة شديدة", "dehydration"}):
        facts["dehydration"] = True
    if _has_any(text, {"ألم شديد في البطن", "وجع بطن شديد", "severe abdominal"}):
        facts["severe_abdominal_pain"] = True
    if _has_any(text, {"ضعف في الرجل", "مش قادر اتحكم في البول", "مش قادر اتحكم في البراز", "bladder", "bowel", "leg weakness"}):
        facts["bladder_bowel_or_leg_weakness"] = True
    duration_match = re.search(r"(\d+)\s*(?:ايام|أيام|يوم|ساعات|اسابيع|أسبوع|days?|hours?|weeks?)", text)
    if duration_match:
        facts["duration"] = duration_match.group(0)
    elif _has_any(text, {"من امبارح", "من إمبارح", "امبارح", "yesterday", "since yesterday"}):
        facts["duration"] = "since yesterday" if _has_any(text, {"yesterday", "since yesterday"}) else "من امبارح"
    elif _has_any(text, {"يومين", "يومين تقريبا", "من يومين", "بقاله يومين", "بقالها يومين"}):
        facts["duration"] = "يومين"
    elif _has_any(text, {"يوم", "من يوم", "بقاله يوم", "بقالها يوم"}):
        facts["duration"] = "يوم"
    elif _has_any(text, {"اسبوع", "أسبوع", "اسبوعين", "أسبوعين", "week", "two weeks"}):
        facts["duration"] = "أسبوعين" if _has_any(text, {"اسبوعين", "أسبوعين", "two weeks"}) else "أسبوع"
    facts = {key: value for key, value in facts.items() if value not in (None, "", [])}
    merged_symptoms = _dedupe(symptoms + classifier_symptoms)
    inactive_or_denied = _expanded_fact_keys(set(explicit_denials) | set(inactive_mentions))
    facts = _facts_without_denials(facts, inactive_or_denied)
    filtered_symptoms = _symptoms_without_denied_facts(_without_explicitly_negated_red_flags(message, merged_symptoms), inactive_or_denied)
    if "chest_pain" in merged_symptoms and "chest_pain" not in filtered_symptoms:
        facts.pop("chest_pain", None)
        if domain_hint == "chest_pain":
            domain_hint = None
        if body_part == "chest":
            body_part = None
    if body_part == "back" and "back_pain" in explicit_denials:
        body_part = None
    if body_part == "side" and "flank_pain" in explicit_denials:
        body_part = None
    return facts, filtered_symptoms, body_part, domain_hint, explicit_denials


def _negated_scope_spans(normalized_text: str) -> list[tuple[int, int]]:
    text = normalized_text
    spans: list[tuple[int, int]] = []
    normalized_markers = sorted({normalize_text(marker) for marker in NEGATION_SCOPE_MARKERS}, key=len, reverse=True)
    for marker in normalized_markers:
        if not marker:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(marker)}(?!\w)")
        for match in pattern.finditer(text):
            start = match.start()
            end = len(text)
            for contrast in CONTRAST_SCOPE_MARKERS:
                contrast_index = text.find(normalize_text(contrast), match.end())
                if contrast_index >= 0:
                    end = min(end, contrast_index)
            spans.append((start, end))
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _negated_scope_texts(message: str) -> list[str]:
    text = f" {normalize_text(_repair_mojibake(message or ''))} "
    return [text[start:end].strip() for start, end in _negated_scope_spans(text)]


def _message_without_negated_scopes(message: str) -> str:
    text = f" {normalize_text(_repair_mojibake(message or ''))} "
    spans = _negated_scope_spans(text)
    if not spans:
        return text.strip()
    kept: list[str] = []
    cursor = 0
    for start, end in spans:
        kept.append(text[cursor:start])
        cursor = max(cursor, end)
    kept.append(text[cursor:])
    return normalize_text(" ".join(kept))


def _mentioned_fact_keys(message: str) -> set[str]:
    text = normalize_text(_repair_mojibake(message or ""))
    mentioned: set[str] = set()
    for concept in FACT_CONCEPTS.values():
        patterns = {normalize_text(pattern) for pattern in concept.get("patterns", ())}
        if _has_any(text, patterns):
            mentioned.update(concept.get("facts", ()))
    return _expanded_fact_keys(mentioned)


def _is_general_or_hypothetical_fact_question(message: str) -> bool:
    text = normalize_text(_repair_mojibake(message or ""))
    if not text or not _mentioned_fact_keys(message):
        return False
    question_like = "؟" in message or "?" in message or text.startswith(("هل ", "is ", "can ", "could "))
    if _has_any(text, GENERAL_FACT_QUESTION_MARKERS):
        if _has_any(text, {"خايف", "خايفة", "ممكن يكون", "could it be", "worried it is", "doctor asked", "asked me if"}):
            return True
        return question_like and not _has_any(text, PERSONAL_ASSERTION_MARKERS)
    return False


def _resolved_fact_keys(message: str) -> set[str]:
    text = normalize_text(_repair_mojibake(message or ""))
    if not text or not _has_any(text, RESOLVED_FACT_MARKERS):
        return set()
    return _mentioned_fact_keys(message)


def _inactive_fact_mentions(message: str) -> set[str]:
    inactive: set[str] = set()
    if _is_general_or_hypothetical_fact_question(message):
        inactive.update(_mentioned_fact_keys(message))
    inactive.update(_resolved_fact_keys(message))
    return _expanded_fact_keys(inactive)


def _explicitly_denied_fact_keys(message: str) -> set[str]:
    denied: set[str] = set()
    for scope in _negated_scope_texts(message):
        for concept in FACT_CONCEPTS.values():
            patterns = {normalize_text(pattern) for pattern in concept.get("patterns", ())}
            if _has_any(scope, patterns):
                denied.update(concept.get("facts", ()))
    return _expanded_fact_keys(denied)


def _expanded_fact_keys(fact_keys: Iterable[str]) -> set[str]:
    expanded = set(fact_keys or [])
    changed = True
    while changed:
        changed = False
        for concept in FACT_CONCEPTS.values():
            related = set(concept.get("facts", ()))
            if expanded.intersection(related) and not related.issubset(expanded):
                expanded.update(related)
                changed = True
    return expanded


def _facts_without_denials(facts: dict[str, Any], denied_facts: Iterable[str]) -> dict[str, Any]:
    denied = _expanded_fact_keys(denied_facts)
    return {key: value for key, value in facts.items() if key not in denied}


def _symptoms_without_denied_facts(symptoms: list[str], denied_facts: Iterable[str]) -> list[str]:
    denied = _expanded_fact_keys(denied_facts)
    drop: set[str] = set()
    for concept in FACT_CONCEPTS.values():
        if denied.intersection(set(concept.get("facts", ()))):
            drop.update(concept.get("symptoms", ()))
    return [symptom for symptom in symptoms if symptom not in drop]


def _is_v3_closing_turn(message: str, symptoms: list[str]) -> bool:
    if symptoms:
        return False
    text = normalize_text(_repair_mojibake(message))
    if _is_closing(message):
        return True
    closing_terms = {
        "شكرا",
        "شكرا سلام",
        "شكرا مع السلامة",
        "تمام شكرا",
        "تسلم",
        "متشكر",
        "سلام",
        "مع السلامة",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
    }
    medical_terms = {
        "صداع",
        "ألم",
        "وجع",
        "كحة",
        "حرارة",
        "ضيق تنفس",
        "pain",
        "fever",
        "cough",
        "dizzy",
    }
    return _has_any(text, closing_terms) and not _has_any(text, medical_terms) and len(text.split()) <= 6


def _is_v3_meta_smalltalk_turn(message: str, symptoms: list[str]) -> bool:
    if symptoms:
        return False
    text = normalize_text(_repair_mojibake(message))
    return _has_any(
        text,
        {
            "اسمك ايه",
            "اسمك إيه",
            "انت اسمك ايه",
            "مين انت",
            "انت مين",
            "بتشتغل ايه",
            "على فكرة اسمك",
            "what is your name",
            "who are you",
            "this makes no sense",
            "makes no sense",
        },
    )


def _is_family_or_personal_chat(message: str) -> bool:
    return _has_any(
        message,
        {
            "ماما عامله اي",
            "ماما عاملة اي",
            "ماما عاملة إيه",
            "امك عامله اي",
            "امك عاملة اي",
            "اخوك عامل اي",
            "اختك عاملة اي",
            "اختك عامله اي",
            "ابوك عامل اي",
            "باباك عامل اي",
            "خالتك عاملة اي",
            "خالتك عامله اي",
            "خالك عامل اي",
            "عمتك عاملة اي",
            "عمتك عامله اي",
            "how is your mom",
            "how is your mother",
            "your mom",
            "your family",
        },
    )


def _is_identity_or_name_chat(message: str) -> bool:
    text = normalize_text(_repair_mojibake(message or "")).lower()
    return _has_any(
        text,
        {
            "اسمك",
            "انت مين",
            "مين انت",
            "who are you",
            "what is your name",
            "what's your name",
            "whats your name",
        },
    )


def _v3_zero_evidence_non_medical_intent(message: str) -> str | None:
    text = normalize_text(_repair_mojibake(message or "")).lower()
    tokens = [token for token in re.split(r"\s+", text) if token]
    if _is_family_or_personal_chat(message):
        return "off_topic"
    if _has_any(
        text,
        {
            "كسم",
            "كس ام",
            "غبي",
            "عبيط",
            "بتفهم",
            "stupid",
            "dumb",
            "this makes no sense",
            "makes no sense",
            "fuck",
            "idiot",
        },
    ):
        return "abuse"
    if _has_any(
        text,
        {
            "ازيك",
            "عامل اي",
            "عاملة اي",
            "اي الدنيا",
            "الجو حر",
            "هاي",
            "اهلا",
            "هلا",
            "how are you",
            "what's up",
            "whats up",
            "are you okay",
            "you okay",
            "اسمك",
            "مين انت",
            "انت مين",
            "who are you",
            "what can you do",
            "what do you do",
            "bro what",
        },
    ):
        return "casual"
    if len(tokens) <= 5 and _has_any(text, {"برو", "bro", "يبر", "؟؟", "??", "what"}):
        return "nonsense"
    return None


def _v3_should_own_message(
    message: str,
    meaning: MessageMeaning,
    case_state: ClinicalCaseState,
    case_action: str,
) -> bool:
    domain = case_state.active_domain or meaning.domain
    if domain not in V3_OWNED_DOMAINS:
        return False
    text = normalize_text(_repair_mojibake(message))
    protected_specialty_context = _has_any(
        text,
        {
            "ألم عين",
            "الم عين",
            "عيني",
            "ضغط",
            "ضغطي",
            "سكر",
            "عطش",
            "رقبة",
            "رقبتي",
            "حامل",
            "طفل",
            "سناني",
            "ضرس",
            "بول",
            "حرقان بول",
            "eye pain",
            "blood pressure",
            "diabetes",
            "sugar",
            "neck",
            "pregnant",
            "urine",
        },
    )
    explicit_headache = _has_any(
        text,
        {"صداع", "وجع راس", "وجع دماغ", "راسي", "دماغي", "headache", "head pain"},
    )
    continuing_headache_case = case_action == "continue" and case_state.active_domain in V3_OWNED_DOMAINS
    if protected_specialty_context and not continuing_headache_case:
        return False
    return explicit_headache or continuing_headache_case or meaning.domain in V3_OWNED_DOMAINS


def _new_fact_names(meaning: MessageMeaning) -> list[str]:
    return sorted(str(key) for key in meaning.facts.keys())


def _summary_for_case(case_state: ClinicalCaseState, language: str) -> str:
    pieces: list[str] = []
    domain = case_state.active_domain or "unknown"
    if language == "en":
        pieces.append(f"active domain: {domain}")
        if case_state.symptoms:
            pieces.append("symptoms: " + ", ".join(case_state.symptoms[:8]))
        if case_state.facts:
            pieces.append("facts: " + ", ".join(f"{k}={v}" for k, v in list(case_state.facts.items())[:8]))
        return "; ".join(pieces)
    domain_ar = {
        "headache": "صداع",
        "neurology_vague": "أعراض عصبية",
        "chest_pain": "ألم صدر",
        "back_pain": "ألم ظهر",
        "gynecology": "أعراض نساء",
    }.get(domain, domain)
    pieces.append(f"الحالة النشطة: {domain_ar}")
    if case_state.symptoms:
        pieces.append("الأعراض: " + "، ".join(case_state.symptoms[:8]))
    if case_state.facts:
        pieces.append("المعلومات: " + "، ".join(f"{k}={v}" for k, v in list(case_state.facts.items())[:8]))
    return "؛ ".join(pieces)


def _avoid_repeated_question(question: str | None, case_state: ClinicalCaseState) -> str | None:
    if not question:
        return None
    if _question_mentions_denied_fact(question, case_state):
        return None
    slot = _question_slot(question)
    if slot and slot in case_state.answered_questions:
        return None
    if _question_is_repeated(question, case_state):
        return None
    return question


def _safe_follow_up_questions(questions: Iterable[str], case_state: ClinicalCaseState) -> list[str]:
    safe: list[str] = []
    for question in questions or []:
        cleaned = _avoid_repeated_question(str(question or "").strip(), case_state)
        if cleaned and cleaned not in safe:
            safe.append(cleaned)
    return safe


def _final_guard_state(case_state: ClinicalCaseState, current_message: str) -> ClinicalCaseState:
    if not current_message:
        return case_state
    try:
        current_denials = set(_explicitly_denied_fact_keys(current_message))
        current_denials.update(_inactive_fact_mentions(current_message))
        current_denials.update(_resolved_fact_keys(current_message))
    except Exception:
        return case_state
    current_denials = _expanded_fact_keys(current_denials)
    if not current_denials:
        return case_state
    merged_denials = set(case_state.denials).union(current_denials)
    return replace(
        case_state,
        denials=merged_denials,
        facts=_facts_without_denials(case_state.facts, merged_denials),
        symptoms=_symptoms_without_denied_facts(case_state.symptoms, merged_denials),
    )


def _question_mentions_denied_fact(question: str, case_state: ClinicalCaseState) -> bool:
    denied = _expanded_fact_keys(case_state.denials)
    if not denied:
        return False
    return bool(_mentioned_fact_keys(question).intersection(denied))


def _answer_conflicts_with_denied_facts(answer: str, case_state: ClinicalCaseState) -> bool:
    denied = _expanded_fact_keys(case_state.denials)
    if not answer or not denied:
        return False
    text = normalize_text(_repair_mojibake(answer)).lower()
    chunks = [chunk.strip() for chunk in re.split(r"[\n\.؟?،؛!]+", text) if chunk.strip()]
    for fact in denied:
        terms = _terms_for_fact(fact)
        if not terms:
            continue
        for chunk in chunks:
            if not _has_any(chunk, terms):
                continue
            if _is_future_warning_chunk(chunk):
                continue
            if _has_any(chunk, {"نفي", "منفي", "مفيش", "مافيش", "لا يوجد", "بدون", "denied", "no ", "without"}):
                continue
            return True
    return False


def _is_future_warning_chunk(chunk: str) -> bool:
    text = f" {normalize_text(_repair_mojibake(chunk)).lower()} "
    future_markers = {
        " لو ",
        " اذا ",
        " إذا ",
        " في حالة ",
        " ظهر ",
        " ظهرت ",
        " تظهر ",
        " if ",
        " warning",
        " appears",
        " develops",
    }
    return any(marker in text for marker in future_markers)


def _terms_for_fact(fact: str) -> set[str]:
    terms: set[str] = set()
    for concept in FACT_CONCEPTS.values():
        if fact in set(concept.get("facts", ())):
            terms.update(normalize_text(_repair_mojibake(pattern)).lower() for pattern in concept.get("patterns", ()))
    aliases = {
        "chest_pain": {"ألم الصدر", "الم الصدر", "وجع الصدر", "صدر", "chest pain", "chest pressure"},
        "breathlessness": {"ضيق نفس", "صعوبة تنفس", "نهجان", "shortness of breath", "breathless"},
        "fever": {"حرارة", "سخونية", "حمى", "fever", "temperature"},
        "temperature": {"حرارة عالية", "سخونية عالية", "high fever", "temperature"},
        "weakness": {"ضعف", "weakness", "weak"},
        "numbness": {"تنميل", "خدر", "numbness", "numb"},
        "speech_change": {"لخبطة كلام", "لخبطة في الكلام", "slurred speech", "speech trouble"},
    }
    terms.update(normalize_text(_repair_mojibake(term)).lower() for term in aliases.get(fact, set()))
    return {term for term in terms if term}


def _denial_safe_fallback(plan: ClinicalPlan, case_state: ClinicalCaseState) -> str:
    language = case_state.language or "ar"
    domain = case_state.active_domain or plan.domain
    symptoms = set(case_state.symptoms)
    respiratory_like = domain in {"respiratory", "infectious", "throat_ent"} or bool(
        {"cough", "throat_irritation", "mild_fever", "continuous_sneezing"}.intersection(symptoms)
    )
    if language == "en":
        if respiratory_like:
            return (
                "Based on the confirmed symptoms, this sounds more like a mild upper-respiratory infection or common cold pattern. "
                "Rest, fluids, and monitoring are reasonable for now. Seek medical review if symptoms persist, worsen, or if new warning signs develop such as shortness of breath, chest pain, or high persistent fever."
            )
        return (
            "I will rely on the symptoms you confirmed and not the denied ones. The information is not enough for a definite diagnosis, "
            "so monitor the symptoms and seek medical care if they persist, worsen, or new warning signs appear."
        )
    if respiratory_like:
        return (
            "بناءً على الأعراض المؤكدة، النمط أقرب لعدوى بسيطة في الجهاز التنفسي العلوي أو دور برد خفيف، مع كحة ورشح وتهيج بسيط في الحلق. "
            "طالما لا يوجد ضيق نفس أو ألم صدر أو حرارة عالية حسب كلامك، فالتعامل المبدئي يكون بالراحة، شرب سوائل، ومتابعة الأعراض. "
            "راجع طبيبًا لو الأعراض استمرت أو زادت، واطلب مساعدة عاجلة فقط لو ظهرت لاحقًا علامات جديدة مثل ضيق نفس، ألم صدر، أو حرارة عالية مستمرة."
        )
    return (
        "هعتمد على الأعراض التي أكدت وجودها فقط، وليس الأعراض التي نفيتها. لا يمكن تأكيد تشخيص نهائي من المحادثة وحدها، "
        "فراقب الأعراض وراجع طبيبًا إذا استمرت أو زادت أو ظهرت علامات خطورة جديدة."
    )


def _doctor_route_conflicts_with_denials(doctor: str | None, case_state: ClinicalCaseState) -> bool:
    if not doctor:
        return False
    lowered = doctor.lower()
    denied = _expanded_fact_keys(case_state.denials)
    if "chest_pain" in denied and any(term in lowered for term in {"cardio", "cardiologist"}):
        return True
    if {"chest_pain", "breathlessness"}.issubset(denied) and "emergency" in lowered:
        return True
    return False


def _is_migraine_like_headache(case_state: ClinicalCaseState) -> bool:
    if case_state.active_domain != "headache":
        return False
    facts = case_state.facts
    if facts.get("sudden_onset") or facts.get("weakness") or facts.get("numbness") or facts.get("speech_change"):
        return False
    migraine_features = {
        "unilateral_headache",
        "pulsating_headache",
        "photophobia",
        "phonophobia",
        "nausea_present",
    }
    score = sum(1 for feature in migraine_features if facts.get(feature))
    return score >= 3 and {"weakness", "numbness", "speech_change"}.issubset(_expanded_fact_keys(case_state.denials))


def _question_is_repeated(question: str, case_state: ClinicalCaseState) -> bool:
    normalized_question = normalize_text(question)
    if any(normalized_question and normalized_question in normalize_text(old) for old in case_state.pending_questions):
        return True
    slot = _question_slot(question)
    return bool(slot and slot in case_state.asked_question_slots and slot in case_state.answered_questions)


def _question_slot(question: str) -> str | None:
    text = normalize_text(question or "")
    ids = _question_ids_for_text(text)
    if ids:
        return ids[0]
    if _has_any(text, {"فجأة", "تدريجي", "sudden", "gradual"}):
        return "sudden_onset"
    if _has_any(text, {"زغللة", "النظر", "vision"}):
        return "vision_change"
    if _has_any(text, {"شدة", "شديد", "severe", "worst"}):
        return "severity"
    if _has_any(text, {"من إمتى", "بدأ", "when did", "how long"}):
        return "duration"
    if _has_any(text, {"تنميل", "numb"}):
        return "numbness"
    if _has_any(text, {"ضعف", "weak"}):
        return "weakness"
    if _has_any(text, {"ضيق نفس", "عرق بارد", "ذراع", "shortness", "cold sweat", "arm"}):
        return "chest_red_flags"
    if _has_any(text, {"أسفل", "أعلى", "lower", "upper"}):
        return "location"
    if _has_any(text, {"لون", "حكة", "ريحة", "color", "itch"}):
        return "discharge_details"
    if _has_any(text, {"حرارة", "سخونية", "fever", "temperature"}):
        return "fever"
    if _has_any(text, {"كحة", "صدر", "صفير", "cough", "wheezing"}):
        return "respiratory_red_flags"
    if _has_any(text, {"بطن", "قيء", "ترجيع", "إسهال", "دم", "abdomen", "vomiting", "diarrhea", "blood"}):
        return "digestive_red_flags"
    if _has_any(text, {"بول", "حرقان", "urine", "urinary"}):
        return "urinary_red_flags"
    if _has_any(text, {"طفح", "تورم", "شفاه", "وجه", "rash", "swelling"}):
        return "skin_allergy_red_flags"
    if _has_any(text, {"عين", "النظر", "vision", "eye"}):
        return "eye_red_flags"
    if _has_any(text, {"سني", "سنى", "سنان", "ضرس", "لثة", "tooth", "gum"}):
        return "dental_red_flags"
    if _has_any(text, {"سكر", "عطش", "جوع", "رعشة", "sugar", "glucose", "thirst", "hunger", "shaking"}):
        return "endocrine_context"
    if _has_any(text, {"إيذاء", "انتحار", "harm", "suicide", "safe"}):
        return "mental_health_safety"
    return None


def _extract_questions(answer: str) -> list[str]:
    parts = re.split(r"(?<=[؟?])", answer or "")
    questions: list[str] = []
    for part in parts:
        if "?" not in part and "؟" not in part:
            continue
        candidate = part.strip()
        if "\n" in candidate:
            candidate = candidate.splitlines()[-1].strip()
        sentence_bits = re.split(r"(?<=[.!。])\s+", candidate)
        candidate = sentence_bits[-1].strip() if sentence_bits else candidate
        if candidate:
            questions.append(candidate)
    return questions


def _clean_question(value: Any) -> str | None:
    if value is None:
        return None
    question = str(value).strip()
    if not question:
        return None
    if any(pattern in question.lower() for pattern in INTERNAL_LEAK_PATTERNS):
        return None
    if _question_count(question) > 1:
        question = re.split(r"[؟?]", question)[0].strip() + ("؟" if _contains_arabic(question) else "?")
    return question


def _question_count(text: str) -> int:
    value = text or ""
    question_marks = {"?", "\u061f", "\ufe56", "\uff1f"}
    return sum(1 for char in value if char in question_marks) + value.count("\u00d8\u0178")


def _limit_to_one_question(text: str) -> str:
    if _question_count(text) <= MAX_V3_QUESTIONS:
        return text
    seen = 0
    kept: list[str] = []
    for char in text:
        kept.append(char)
        if char in {"?", "؟"}:
            seen += 1
            if seen >= MAX_V3_QUESTIONS:
                break
    return "".join(kept).strip()


def _strings(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _candidate_condition_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value[:5]:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("condition") or "").strip()
            reason = str(item.get("reason") or item.get("supporting_reason") or "").strip()
            if name:
                cleaned.append({"name": name, "reason": reason})
        elif str(item).strip():
            cleaned.append({"name": str(item).strip(), "reason": ""})
    return cleaned


QUESTION_ID_PATTERNS: dict[str, tuple[str, ...]] = {
    "abdominal_location": (
        "الألم فين",
        "مكان الألم",
        "فين بالظبط",
        "where exactly",
        "where is the pain",
        "location",
    ),
    "nausea_present": (
        "غثيان",
        "عايز ترجع",
        "عايزة ترجعي",
        "حاسس إنك هترجع",
        "nausea",
        "nauseated",
        "feel like vomiting",
    ),
    "vomiting_present": (
        "ترجيع",
        "قيء",
        "بتستفرغ",
        "vomiting",
        "throwing up",
    ),
    "diarrhea_present": (
        "إسهال",
        "اسهال",
        "diarrhea",
        "diarrhoea",
    ),
    "previous_gastric_history": (
        "تاريخ سابق",
        "حصل قبل كده",
        "قرحة",
        "قولون",
        "previous history",
        "happened before",
        "ulcer",
    ),
    "duration": (
        "من إمتى",
        "من امتى",
        "بقاله",
        "بقالها",
        "قد إيه",
        "كم عدد أيام",
        "أيام أو أسابيع",
        "how long",
        "when did",
    ),
    "fever_present": (
        "حرارة",
        "سخونية",
        "حمى",
        "fever",
        "temperature",
    ),
    "blood_present": (
        "دم",
        "blood",
    ),
    "dehydration_present": (
        "جفاف",
        "مش بتبول",
        "دوخة شديدة",
        "dehydration",
    ),
    "chest_red_flags": (
        "ضيق نفس",
        "عرق بارد",
        "إغماء",
        "يمتد للذراع",
        "shortness of breath",
        "cold sweat",
        "fainting",
        "arm or jaw",
    ),
    "neurological_red_flags": (
        "تنميل",
        "ضعف",
        "صعوبة كلام",
        "زغللة",
        "numbness",
        "weakness",
        "slurred speech",
        "vision",
    ),
    "sudden_onset": (
        "فجأة",
        "فجاءه",
        "تدريجي",
        "بالتدريج",
        "sudden",
        "gradual",
    ),
    "skin_allergy_red_flags": (
        "تورم الشفاه",
        "تورم الوجه",
        "ضيق تنفس",
        "lip swelling",
        "face swelling",
        "breathing trouble",
    ),
}


QUESTION_TO_FACT: dict[str, str] = {
    "abdominal_location": "abdominal_location",
    "nausea_present": "nausea_present",
    "vomiting_present": "vomiting_present",
    "diarrhea_present": "diarrhea_present",
    "previous_gastric_history": "previous_gastric_history",
    "duration": "duration",
    "fever_present": "fever_present",
    "blood_present": "blood_vomit_or_stool",
    "dehydration_present": "dehydration",
    "chest_red_flags": "chest_red_flags",
    "neurological_red_flags": "neurological_red_flags",
    "skin_allergy_red_flags": "skin_allergy_red_flags",
    "sudden_onset": "sudden_onset",
}


QUESTION_BANK: dict[str, list[tuple[str, str, str]]] = {
    "digestive": [
        (
            "abdominal_location",
            "الألم فين بالظبط: فوق، تحت، يمين، شمال، ولا في نص البطن؟",
            "Where exactly is the pain: upper, lower, right, left, or middle abdomen?",
        ),
        (
            "previous_gastric_history",
            "هل حصلت لك نوبات شبه كده قبل كده أو عندك تاريخ قرحة/قولون؟",
            "Has this happened before, or do you have a history of ulcer or irritable bowel symptoms?",
        ),
        (
            "duration",
            "الألم والإحساس بالترجيع بقالهم قد إيه؟",
            "How long have the pain and nausea been going on?",
        ),
        (
            "blood_present",
            "هل في دم في القيء أو البراز، أو ألم شديد بيزيد بسرعة؟",
            "Is there blood in vomit or stool, or severe pain that is getting worse quickly?",
        ),
        (
            "fever_present",
            "هل في حرارة أو إسهال مع ألم البطن؟",
            "Is there fever or diarrhea with the abdominal pain?",
        ),
    ],
    "headache": [
        ("sudden_onset", "الصداع بدأ فجأة ولا تدريجي؟", "Did the headache start suddenly or gradually?"),
        ("neurological_red_flags", "هل معاه تنميل/ضعف، زغللة، أو صعوبة كلام؟", "Any numbness, weakness, vision change, or speech difficulty?"),
        ("duration", "بقاله قد إيه وبيزيد ولا ثابت؟", "How long has it been going on, and is it worsening or stable?"),
    ],
    "chest_pain": [
        ("chest_red_flags", "هل مع ألم الصدر في ضيق نفس، عرق بارد، إغماء، أو ألم ممتد للذراع أو الفك؟", "Any shortness of breath, cold sweat, fainting, or pain spreading to the arm or jaw?"),
        ("duration", "بدأ من إمتى، وهل ظهر فجأة أو مع مجهود؟", "When did it start, and did it start suddenly or with exertion?"),
    ],
}


def _question_ids_for_text(text: str, domain: str | None = None) -> list[str]:
    normalized = normalize_text(text or "")
    ids: list[str] = []
    for question_id, patterns in QUESTION_ID_PATTERNS.items():
        if _has_any(normalized, set(patterns)):
            ids.append(question_id)
    if domain == "digestive" or _has_any(normalized, {"بطن", "معدة", "abdomen", "stomach"}):
        if _has_any(normalized, {"فين", "مكان", "نص البطن", "نصف البطن", "وسط البطن", "يمين", "شمال", "فوق", "تحت", "upper", "lower", "right", "left", "middle"}):
            ids.append("abdominal_location")
    return _dedupe(ids)


def _question_ids_answered_by_message(message: str, meaning: MessageMeaning, state: ClinicalCaseState) -> list[str]:
    text = normalize_text(message or "")
    ids: list[str] = []
    if (
        meaning.duration
        or meaning.facts.get("duration")
        or re.search(r"\b\d+\s*(?:يوم|ايام|أيام|ساعة|ساعات|days?|hours?)", text)
        or _has_any(text, {"يومين", "من يومين", "بقاله يومين", "بقالها يومين", "اسبوع", "أسبوع", "اسبوعين", "أسبوعين"})
    ):
        ids.append("duration")
    if meaning.facts.get("abdominal_location") or _has_any(text, {"نص البطن", "نصف البطن", "وسط البطن", "يمين", "شمال", "فوق", "تحت", "middle abdomen", "right side", "left side"}):
        ids.append("abdominal_location")
    if "sudden_onset" in meaning.facts or _has_any(text, {"فجأة", "فجاءه", "تدريجي", "بالتدريج", "sudden", "gradual"}):
        ids.append("sudden_onset")
    if meaning.facts.get("nausea_present") or _has_any(text, {"غثيان", "عايز ارجع", "عايزة ارجع", "احساس بالترجيع", "إحساس بالترجيع", "nausea", "nauseated"}):
        ids.append("nausea_present")
    if meaning.facts.get("vomiting_present") or _has_any(text, {"ترجيع", "قيء", "استفراغ", "vomiting", "throwing up"}):
        ids.append("vomiting_present")
    if meaning.facts.get("diarrhea_present") or _has_any(text, {"إسهال", "اسهال", "diarrhea", "diarrhoea"}):
        ids.append("diarrhea_present")
    if "previous_gastric_history" in meaning.facts or _has_any(text, {"تاريخ سابق", "تاريخ", "حصل قبل كده", "قبل كده", "قرحة", "قولون", "previous history", "history", "happened before", "ulcer"}):
        ids.append("previous_gastric_history")
    if meaning.facts.get("fever") or _has_any(text, {"حرارة", "سخونية", "حمى", "fever"}):
        ids.append("fever_present")
    if meaning.facts.get("blood_vomit_or_stool") or _has_any(text, {"دم", "blood"}):
        ids.append("blood_present")
    if meaning.facts.get("dehydration"):
        ids.append("dehydration_present")
    for pending_id in state.pending_question_ids:
        fact_key = _fact_key_for_question_id(pending_id)
        if fact_key and fact_key in meaning.facts:
            ids.append(pending_id)
    return _dedupe(ids)


def _fact_key_for_question_id(question_id: str) -> str | None:
    return QUESTION_TO_FACT.get(question_id)


def _is_negative_short_answer(message: str) -> bool:
    text = normalize_text(message or "")
    tokens = set(text.split())
    negative_tokens = {"لا", "لأ", "مفيش", "مافيش", "مش", "no", "not", "none"}
    negative_phrases = {"مش موجود", "مش عندي", "معنديش", "ما عنديش", "no history", "not present", "doesn't", "doesnt", "does not"}
    return len(tokens) <= 8 and (bool(tokens.intersection(negative_tokens)) or _has_any(text, negative_phrases))


def _is_positive_short_answer(message: str) -> bool:
    text = normalize_text(message or "")
    tokens = set(text.split())
    return len(tokens) <= 6 and bool(tokens.intersection({"اه", "أه", "ايوه", "أيوه", "نعم", "yes", "yeah"}))


def _select_next_question(domain: str, language: str, case_state: ClinicalCaseState) -> tuple[str | None, str | None]:
    bank = QUESTION_BANK.get(domain, [])
    facts = case_state.facts
    answered = set(case_state.answered_question_ids) | set(case_state.answered_questions)
    if domain == "digestive":
        ordered_ids = ["abdominal_location"]
        if facts.get("nausea_present") or facts.get("vomiting_present"):
            ordered_ids.append("previous_gastric_history")
        ordered_ids.extend(["duration", "blood_present", "fever_present"])
        bank_by_id = {qid: (ar, en) for qid, ar, en in bank}
        for question_id in ordered_ids:
            fact_key = _fact_key_for_question_id(question_id)
            if question_id in answered or question_id in case_state.pending_question_ids:
                continue
            if fact_key and fact_key in facts and facts.get(fact_key) not in (None, ""):
                continue
            if question_id in bank_by_id:
                ar, en = bank_by_id[question_id]
                return question_id, en if language == "en" else ar
        return None, None
    for question_id, ar_question, en_question in bank:
        fact_key = _fact_key_for_question_id(question_id)
        if question_id in answered or question_id in case_state.pending_question_ids:
            continue
        if fact_key and fact_key in facts and facts.get(fact_key) not in (None, ""):
            continue
        return question_id, en_question if language == "en" else ar_question
    profile = DOMAIN_PROFILES.get(domain)
    if profile:
        question = profile["en_question"] if language == "en" else profile["ar_question"]
        question_id = _question_slot(question)
        if question_id and question_id not in answered and question_id not in case_state.pending_question_ids:
            return question_id, question
    return None, None


def _digestive_possibilities(case_state: ClinicalCaseState) -> list[str]:
    facts = case_state.facts
    symptoms = set(case_state.symptoms)
    possibilities: list[str] = []
    if facts.get("blood_vomit_or_stool") or facts.get("dehydration") or facts.get("severe_abdominal_pain"):
        possibilities.append("urgent digestive condition")
    if facts.get("vomiting_present") or "vomiting" in symptoms or facts.get("diarrhea_present") or "diarrhoea" in symptoms:
        possibilities.append("Gastroenteritis")
    if facts.get("nausea_present") or facts.get("post_meal_worse") or facts.get("abdominal_location") == "middle_abdomen":
        possibilities.append("gastritis or indigestion")
    possibilities.append("abdominal pain needing clinical context")
    return _dedupe(possibilities)[:3]


def _digestive_analysis_for_patient(case_state: ClinicalCaseState, language: str) -> str:
    facts = case_state.facts
    has_nausea = facts.get("nausea_present") or "nausea" in case_state.symptoms
    location = facts.get("abdominal_location")
    post_meal = facts.get("post_meal_worse")
    if language == "en":
        if post_meal and location == "middle_abdomen":
            return "Middle abdominal pain that worsens after eating can fit stomach irritation, gastritis, or indigestion. This is still an initial direction, not a final diagnosis."
        if post_meal:
            return "Pain that worsens after eating keeps the focus on a digestive or stomach-related cause, while warning signs and the exact pattern still matter."
        if location == "middle_abdomen" and has_nausea:
            return "Middle abdominal pain with nausea can fit stomach irritation, indigestion, or an early digestive infection, but it needs a few details before naming a diagnosis."
        if has_nausea:
            return "Abdominal pain with nausea points toward a digestive cause, but the exact location, duration, and warning signs matter."
        return "Abdominal pain can come from several digestive causes, so I’ll keep the reasoning focused on the abdomen and ask one useful detail at a time."
    if post_meal and location == "middle_abdomen":
        return "ألم نص البطن اللي بيزيد بعد الأكل ممكن يتماشى مع تهيج المعدة، التهاب المعدة، أو عسر الهضم. ده اتجاه مبدئي وليس تشخيص نهائي، ومع الترجيع أو زيادة الألم الأفضل تقييم طبي."
    if post_meal:
        return "كون الوجع بيزيد بعد الأكل يخلينا نركز أكتر على سبب هضمي أو متعلق بالمعدة، مع الانتباه لأي علامات خطورة أو زيادة سريعة في الألم."
    if location == "middle_abdomen" and has_nausea:
        return "ألم نص البطن مع إحساس بالترجيع ممكن يحصل مع تهيج المعدة، عسر هضم، أو بداية عدوى هضمية، لكن محتاجين نكمل بسؤال واحد مهم قبل ما نثبت اتجاه."
    if has_nausea:
        return "ألم البطن مع إحساس بالترجيع يخلينا نفكر في سبب هضمي، لكن مكان الألم ومدته ووجود علامات خطورة هم اللي يحددوا الخطوة الأنسب."
    return "ألم البطن له أسباب كتير، فخلينا نخلي التفكير مركز على البطن والجهاز الهضمي ونسأل سؤال واحد مفيد في كل مرة."


def _question_frequency(answer: str, question: str) -> int:
    normalized_answer = normalize_text(re.sub(r"\s+", " ", answer or ""))
    normalized_question = normalize_text(re.sub(r"\s+", " ", question or "").strip(" ؟?"))
    if not normalized_answer or not normalized_question:
        return 0
    return normalized_answer.count(normalized_question)


def _has_answered_question_contamination(answer: str, case_state: ClinicalCaseState) -> bool:
    answered = set(case_state.answered_question_ids) | set(case_state.answered_questions)
    if not answered:
        return False
    for question in _extract_questions(answer):
        if answered.intersection(_question_ids_for_text(question, case_state.active_domain)):
            return True
    return False


def _has_unrelated_body_part(answer: str, domain: str, case_state: ClinicalCaseState) -> bool:
    text = normalize_text(answer or "")
    blocked_by_domain = {
        "digestive": {"صداع", "زغللة", "فقرات الرقبة", "ألم الصدر", "chest pain", "headache", "neck"},
        "headache": {"البطن", "معدة", "إسهال", "abdominal", "stomach", "diarrhea"},
        "chest_pain": {"إسهال", "طفح", "abdominal", "rash"},
        "back_pain": {"معدة", "إسهال", "malaria", "aids"},
        "dental": {"بول", "حرقان البول", "مسالك", "urinary", "urine", "urination"},
        "urinary": {"سن", "ضرس", "أسنان", "tooth", "dental", "gum"},
    }
    blocked = blocked_by_domain.get(domain, set())
    if not blocked:
        return False
    allowed = " ".join([case_state.active_body_part or "", " ".join(case_state.symptoms)])
    return any(term in text and term not in allowed for term in blocked)


def _safe_risk_level(value: Any, fallback: str) -> str:
    risk = str(value or "").strip().lower()
    return risk if risk in {"low", "moderate", "urgent", "emergency"} else fallback


def _urgency_from_risk(risk: str) -> str:
    if risk == "emergency":
        return "High"
    if risk in {"urgent", "moderate"}:
        return "Medium"
    return "Low"


def _doctor_for_domain(domain: str | None) -> str:
    return {
        "headache": "Neurologist",
        "neurology_vague": "Neurologist",
        "chest_pain": "Cardiologist",
        "back_pain": "Orthopedic doctor",
        "neck_pain": "Orthopedic doctor",
        "orthopedics": "Orthopedic doctor",
        "gynecology": "Gynecologist",
        "throat_ent": "ENT Specialist",
        "digestive": "Gastroenterologist",
        "urinary": "Urologist",
        "respiratory": "Pulmonologist",
        "skin": "Dermatologist",
        "eye": "Ophthalmologist",
        "dental": "Dentist",
        "mental_health": "Psychiatrist",
        "endocrine": "Endocrinologist",
        "cardiology": "Cardiologist",
        "pediatrics": "Pediatrician",
        "vestibular_ent": "ENT specialist",
        "body_ache": "General Practitioner",
        "infectious": "General Practitioner",
        "closing": "Not needed",
    }.get(domain or "", "General Practitioner")


def _is_high_risk_by_existing_safety(message: str, symptoms: list[str]) -> bool:
    checked_symptoms = _without_explicitly_negated_red_flags(message, symptoms)
    if _is_negated_self_harm_statement(message) and set(checked_symptoms).issubset({"anxiety", "fatigue"}):
        return False
    if "chest_pain" in checked_symptoms and not _has_chest_emergency_companion(message, checked_symptoms):
        return False
    return has_red_flags(message, checked_symptoms) or choose_urgency(message, checked_symptoms, 0) == "High"


def _without_explicitly_negated_red_flags(message: str, symptoms: list[str]) -> list[str]:
    text = normalize_text(_repair_mojibake(message or ""))
    denied = _explicitly_denied_fact_keys(text)
    return _symptoms_without_denied_facts(symptoms, denied)


def _denies_fever(message: str) -> bool:
    text = normalize_text(_repair_mojibake(message or "")).lower()
    fever_terms = {"حرارة", "سخونية", "سخونيه", "حمى", "fever", "temperature"}
    denial_terms = {
        "مقولتش",
        "ما قولتش",
        "مش عندي",
        "معنديش",
        "ما عنديش",
        "مفيش",
        "مافيش",
        "ولا",
        "no ",
        "not ",
        "without",
    }
    return _has_any(text, fever_terms) and _has_any(text, denial_terms)


def _is_negated_self_harm_statement(message: str) -> bool:
    text = normalize_text(_repair_mojibake(message or ""))
    return _has_any(
        text,
        {
            "مش هاذي نفسي",
            "مش هأذي نفسي",
            "مش هاذي",
            "مش هأذي",
            "مش هؤذي نفسي",
            "مش هؤذي",
            "مش هاذي روحي",
            "مش هأذي روحي",
            "not going to hurt myself",
            "i will not hurt myself",
            "no self harm",
            "not suicidal",
        },
    )


def _has_chest_emergency_companion(message: str, symptoms: list[str]) -> bool:
    symptom_set = set(symptoms)
    if {"breathlessness", "sweating", "cold_sweat", "coma", "dizziness"}.intersection(symptom_set):
        return True
    text = normalize_text(_repair_mojibake(message or ""))
    return _has_any(
        text,
        {
            "ضيق نفس",
            "صعوبة تنفس",
            "عرق بارد",
            "تعرق بارد",
            "اغماء",
            "إغماء",
            "امتداد",
            "دراعي الشمال",
            "الذراع",
            "الفك",
            "shortness of breath",
            "cold sweat",
            "fainting",
            "radiating",
            "left arm",
            "jaw",
        },
    )


def _generic_red_flag_signal(domain: str | None, language: str | None = None) -> EmergencySignal:
    doctor = {
        "headache": "Emergency Department / Neurologist",
        "neurology_vague": "Emergency Department / Neurologist",
        "chest_pain": "Emergency Department / Cardiologist",
        "back_pain": "Emergency Department / Orthopedic doctor",
        "neck_pain": "Emergency Department / Orthopedic doctor",
        "orthopedics": "Emergency Department / Orthopedic doctor",
        "digestive": "Emergency Department / Gastroenterologist",
        "urinary": "Emergency Department / Urologist",
        "skin": "Emergency Department / Allergist",
        "eye": "Emergency Department / Ophthalmologist",
        "dental": "Emergency Department / Dentist",
        "mental_health": "Emergency Department / Psychiatrist",
        "endocrine": "Emergency Department / Endocrinologist",
        "cardiology": "Emergency Department / Cardiologist",
        "pediatrics": "Emergency Department / Pediatrician",
        "vestibular_ent": "Emergency Department / ENT specialist",
        "gynecology": "Emergency Department / Gynecologist",
        "respiratory": "Emergency Department / Pulmonologist",
    }.get(domain or "", "Emergency Department")
    display_doctor = {
        "headache": "الطوارئ فورًا / طبيب مخ وأعصاب",
        "neurology_vague": "الطوارئ فورًا / طبيب مخ وأعصاب",
        "chest_pain": "الطوارئ فورًا / طبيب قلب",
        "back_pain": "الطوارئ فورًا / طبيب عظام",
        "neck_pain": "الطوارئ فورًا / طبيب عظام",
        "orthopedics": "الطوارئ فورًا / طبيب عظام",
        "digestive": "الطوارئ فورًا / طبيب جهاز هضمي",
        "urinary": "الطوارئ فورًا / طبيب مسالك",
        "skin": "الطوارئ فورًا / طبيب حساسية",
        "eye": "الطوارئ فورًا / طبيب عيون",
        "dental": "الطوارئ فورًا / طبيب أسنان",
        "mental_health": "الطوارئ فورًا / طبيب نفسي",
        "endocrine": "الطوارئ فورًا / طبيب غدد",
        "cardiology": "الطوارئ فورًا / طبيب قلب",
        "pediatrics": "الطوارئ فورًا / طبيب أطفال",
        "vestibular_ent": "الطوارئ فورًا / طبيب أنف وأذن وحنجرة",
        "gynecology": "الطوارئ فورًا / طبيب نساء",
        "respiratory": "الطوارئ فورًا / طبيب صدر",
    }.get(domain or "", "الطوارئ فورًا")
    return EmergencySignal(
        domain or "general",
        "Emergency red-flag concern",
        "اشتباه حالة طارئة",
        doctor,
        display_doctor,
        "وجود علامة خطورة في الأعراض يحتاج تقييمًا عاجلًا في الطوارئ.",
    )


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text or "")


def _patient_answer_erases_medical_meaning(answer: str) -> bool:
    text = normalize_text(_repair_mojibake(answer or "")).lower()
    return _has_any(
        text,
        {
            "مش قادر افهم الرسالة دي كشكوى صحية",
            "مش قادر أفهم الرسالة دي كشكوى صحية",
            "اكتب العرض بشكل اوضح",
            "اكتب العرض بشكل أوضح",
            "could not understand that as a health concern",
            "cannot understand that as a health concern",
            "write the symptom clearly",
        },
    )


def _is_low_value_medical_answer(answer: str, deterministic_plan: ClinicalPlan) -> bool:
    if deterministic_plan.domain in {"unknown", "non_medical", "casual", "off_topic", "abuse", "nonsense", "closing"}:
        return False
    text = normalize_text(_repair_mojibake(answer or "")).lower()
    if _question_count(answer) > 0:
        return False
    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) <= 8:
        return True
    return _has_any(
        text,
        {
            "لا اعرف ما السبب",
            "لا أعرف ما السبب",
            "i do not know the cause",
            "i don't know the cause",
        },
    )


def _raw_plan_has_emergency_reason(raw: dict[str, Any]) -> bool:
    text = " ".join(_strings(raw.get("risk_reasons")) + _strings(raw.get("clinical_summary"))).lower()
    return any(
        term in text
        for term in {
            "emergency",
            "red flag",
            "stroke",
            "heart attack",
            "anaphylaxis",
            "severe allergic",
            "blood",
            "fainting",
            "vision loss",
            "bladder",
            "bowel",
        }
    )


def _raw_text_has_emergency_signal(message: str, case_state: ClinicalCaseState) -> bool:
    text = normalize_text(f"{message} {_summary_for_case(case_state, case_state.language)}")
    return _has_any(
        text,
        {
            "ضيق تنفس",
            "ألم صدر شديد",
            "عرق بارد",
            "إغماء",
            "فقدان الوعي",
            "تنميل ناحية",
            "صعوبة كلام",
            "صداع فجأة",
            "فقدان النظر",
            "دم في البول",
            "دم في القيء",
            "دم في البراز",
            "تورم الشفاه",
            "shortness of breath",
            "cold sweat",
            "fainting",
            "slurred speech",
            "sudden headache",
            "vision loss",
            "blood in urine",
            "vomiting blood",
            "blood in stool",
            "lip swelling",
        },
    )


def _has_generic_viral_cluster(message: str, symptoms: list[str]) -> bool:
    if _denies_fever(message):
        return False
    symptom_set = set(symptoms)
    has_fever = bool({"mild_fever", "high_fever"}.intersection(symptom_set)) or _has_any(
        message,
        {"سخونية", "سخونيه", "حرارة", "حمى", "fever", "temperature"},
    )
    has_aches_or_respiratory = bool({"muscle_pain", "fatigue", "cough", "throat_irritation"}.intersection(symptom_set)) or _has_any(
        message,
        {"تكسير", "جسمي واجعني", "كحة", "زوري", "body aches", "cough", "sore throat"},
    )
    return has_fever and has_aches_or_respiratory and not _has_malaria_context_denial_or_support(message)[0]


def _has_malaria_support(message: str, case_state: ClinicalCaseState) -> bool:
    support, _denial = _has_malaria_context_denial_or_support(message)
    text = normalize_text(_repair_mojibake(message))
    fever_chills = _has_any(text, {"حرارة", "حمى", "سخونية", "رعشة", "تعرق", "fever", "chills", "sweating"})
    return support and fever_chills


def _has_malaria_context_denial_or_support(message: str) -> tuple[bool, bool]:
    message = _repair_mojibake(message)
    support = _has_any(
        message,
        {
            "سفر",
            "مسافر",
            "ناموس",
            "بعوض",
            "قرصة ناموس",
            "mosquito",
            "travel",
        },
    )
    denial = _has_any(
        message,
        {
            "معنديش سفر",
            "مفيش سفر",
            "ولا سفر",
            "مفيش ناموس",
            "ولا ناموس",
            "no travel",
            "no mosquito",
        },
    )
    return support and not denial, denial


def _looks_like_malaria_explanation_followup(message: str, active_case: ClinicalCaseState | None = None) -> bool:
    text = normalize_text(_repair_mojibake(message or "")).lower()
    mentions_malaria = _has_any(text, {"malaria", "\u0645\u0644\u0627\u0631\u064a\u0627", "\u0627\u0644\u0645\u0644\u0627\u0631\u064a\u0627"})
    if not mentions_malaria:
        return False
    asks_explanation = _has_any(
        text,
        {
            "why",
            "what is",
            "explain",
            "\u064a\u0639\u0646\u064a \u0627\u064a\u0647",
            "\u064a\u0639\u0646\u064a \u0625\u064a\u0647",
            "\u0644\u064a\u0647",
            "\u0644\u0645\u0627\u0630\u0627",
            "\u0627\u0634\u0631\u062d",
            "\u0627\u064a",
            "\u0625\u064a\u0647",
        },
    )
    if asks_explanation:
        return True
    return bool(active_case and active_case.active_domain == "infectious")


def _is_previous_malaria_question(request: ChatRequest) -> bool:
    if _looks_like_malaria_explanation_followup(request.message):
        return True
    text = normalize_text(_repair_mojibake(request.message))
    if not _has_any(text, {"ملاريا", "malaria"}):
        return False
    if _has_any(text, {"اي", "إيه", "ما هي", "اشرح", "ليه", "لماذا", "why", "what is"}):
        return True
    history_text = _repair_mojibake(" ".join(item.content or "" for item in request.history[-6:]))
    return _has_any(history_text, {"ملاريا", "malaria"})
