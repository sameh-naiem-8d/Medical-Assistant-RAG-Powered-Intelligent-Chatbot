from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .clarification_service import (
    CLARIFICATION_MODE,
    CLOSING_MODE,
    DIAGNOSIS_MODE,
    EMERGENCY_MODE,
    clarification_answer,
    closing_answer,
    determine_response_mode,
    filter_repeated_questions,
    is_closing_message,
    is_vague_or_body_area_only,
    smart_follow_up_questions,
)
from .chat_engine_v2 import ChatEngineV2
from .chat_engine_v3 import ChatEngineV3
from .classifier_service import ClassifierService
from .config import get_settings
from .conversation_orchestrator import (
    ConversationIntent,
    ConversationState,
    build_conversation_state,
    has_any_normalized as conversation_has_any_normalized,
    is_weak_body_ache_only,
    is_weak_nonspecific_case,
    rare_diagnosis_unsupported,
)
from .display_labels import display_diagnosis_ar, display_doctor_ar
from .knowledge_service import KnowledgeService
from .llm_service import HIGH_URGENCY_PREFIX, LLMService
from .rag_service import RAGService
from .safety import EmergencySignal, detect_context_flags, detect_emergency_signal, normalize_text
from .schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse, RetrievedCase

settings = get_settings()

classifier_service = ClassifierService(settings)
knowledge_service = KnowledgeService(settings)
rag_service = RAGService(settings)
llm_service = LLMService(settings)
chat_engine_v2 = ChatEngineV2(
    classifier_service=classifier_service,
    knowledge_service=knowledge_service,
    rag_service=rag_service,
    llm_service=llm_service,
)
chat_engine_v3 = ChatEngineV3(
    classifier_service=classifier_service,
    knowledge_service=knowledge_service,
    rag_service=rag_service,
    llm_service=llm_service,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)

allow_credentials = "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


SPECIALTY_CONTEXT_FLAGS = {
    "dental_context",
    "eye_context",
    "pregnancy",
    "pregnancy_red_flag",
    "gynecology_context",
    "pediatric",
    "pediatric_red_flag",
    "self_harm",
    "mental_health",
    "trauma_context",
    "severe_trauma",
    "urinary_context",
    "urinary_retention",
    "endocrine_context",
    "ent_context",
    "infectious_context",
}

MAX_CLARIFICATION_QUESTIONS = 3
MAX_DIAGNOSIS_QUESTIONS = 2
BROAD_VIRAL_DIAGNOSIS = "Viral or flu-like illness"
GENERAL_CLARIFICATION_DIAGNOSIS = "General symptoms needing clarification"
LOCAL_DEMO_SESSION_TTL_SECONDS = 4 * 60 * 60
LOCAL_DEMO_MAX_SESSIONS = 200
LOCAL_DEMO_MAX_HISTORY_MESSAGES = 40

_local_demo_sessions: dict[str, dict[str, Any]] = {}


def _request_conversation_id(request: ChatRequest) -> str | None:
    return request.conversation_id or request.session_id


def _dedupe_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    cleaned: list[ChatMessage] = []
    previous_key: tuple[str, str] | None = None
    for message in messages:
        role = message.role if message.role in {"user", "assistant"} else "user"
        content = (message.content or "").strip()
        if not content:
            continue
        key = (role, content)
        if key == previous_key:
            continue
        cleaned.append(ChatMessage(role=role, content=content))
        previous_key = key
    return cleaned


def _copy_request_with_history(request: ChatRequest, history: list[ChatMessage]) -> ChatRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"history": history})
    return request.copy(update={"history": history})


def _prune_local_demo_sessions(now: float | None = None) -> None:
    current = now or time.time()
    expired = [
        key
        for key, record in _local_demo_sessions.items()
        if current - float(record.get("updated_at", 0.0)) > LOCAL_DEMO_SESSION_TTL_SECONDS
    ]
    for key in expired:
        _local_demo_sessions.pop(key, None)

    overflow = len(_local_demo_sessions) - LOCAL_DEMO_MAX_SESSIONS
    if overflow <= 0:
        return
    oldest = sorted(
        _local_demo_sessions,
        key=lambda key: float(_local_demo_sessions[key].get("updated_at", 0.0)),
    )
    for key in oldest[:overflow]:
        _local_demo_sessions.pop(key, None)


def _augment_local_demo_request(request: ChatRequest) -> tuple[ChatRequest, dict[str, Any] | None]:
    conversation_id = _request_conversation_id(request)
    if request.source != "local_demo" or not conversation_id:
        return request, None

    _prune_local_demo_sessions()
    record = _local_demo_sessions.get(conversation_id)
    if record and record.get("user_id") and request.user_id and record.get("user_id") != request.user_id:
        record = None
        _local_demo_sessions.pop(conversation_id, None)
    if not record:
        record = {"history": [], "user_id": request.user_id, "updated_at": time.time()}
        _local_demo_sessions[conversation_id] = record

    stored_history = list(record.get("history") or [])
    merged_history = _dedupe_messages(stored_history + list(request.history or []))[-LOCAL_DEMO_MAX_HISTORY_MESSAGES:]
    return _copy_request_with_history(request, merged_history), record


def _remember_local_demo_turn(
    *,
    original_request: ChatRequest,
    effective_request: ChatRequest,
    response: ChatResponse,
    record: dict[str, Any] | None,
) -> None:
    conversation_id = _request_conversation_id(original_request)
    if original_request.source != "local_demo" or not conversation_id or record is None:
        return

    updated_history = _dedupe_messages(
        list(effective_request.history or [])
        + [
            ChatMessage(role="user", content=original_request.message),
            ChatMessage(role="assistant", content=response.answer),
        ]
    )[-LOCAL_DEMO_MAX_HISTORY_MESSAGES:]
    record["history"] = updated_history
    record["user_id"] = original_request.user_id
    record["updated_at"] = time.time()
    record["case_state_update"] = response.case_state_update
    _local_demo_sessions[conversation_id] = record
    response.case_state_update.setdefault("local_memory", {})
    response.case_state_update["local_memory"].update(
        {
            "enabled": True,
            "stored_history_messages": len(updated_history),
            "ttl_seconds": LOCAL_DEMO_SESSION_TTL_SECONDS,
        }
    )


SAFE_BROAD_PRECAUTIONS = [
    "الراحة وتقليل المجهود مؤقتا",
    "شرب سوائل كافية على فترات صغيرة",
    "متابعة درجة الحرارة وطلب رعاية طبية إذا زادت الأعراض أو استمرت",
]


DIAGNOSIS_ALIASES = {
    "Malaria": {"ملاريا", "الملاريا", "malaria"},
    BROAD_VIRAL_DIAGNOSIS: {
        "عدوى فيروسية",
        "دور برد",
        "نزلة برد",
        "انفلونزا",
        "إنفلونزا",
        "viral",
        "flu",
    },
    "Common Cold": {"نزلة برد", "دور برد", "common cold"},
    "Pneumonia": {"التهاب رئوي", "pneumonia"},
    "Hypertension": {"ضغط", "ارتفاع ضغط", "hypertension"},
}


MALARIA_RISK_TERMS = {
    "سفر",
    "مسافر",
    "بعد سفر",
    "راجع من سفر",
    "منطقة فيها ملاريا",
    "منطقة موبوءة",
    "ناموس",
    "بعوض",
    "قرصة ناموس",
    "لدغة ناموس",
    "لدغة بعوض",
    "mosquito",
    "travel",
}


MALARIA_CYCLIC_TERMS = {
    "نوبات حرارة",
    "نوبات حمى",
    "بتروح وترجع",
    "تروح وترجع",
    "كل يومين",
    "كل تلات ايام",
    "كل ثلاثة أيام",
    "رعشة شديدة متكررة",
    "رعشة وتعرق متكرر",
}


MALARIA_CONTEXT_DENIAL_TERMS = {
    "معنديش سفر",
    "ماعنديش سفر",
    "مفيش سفر",
    "مافيش سفر",
    "مسافرتش",
    "ولا سفر",
    "معنديش ناموس",
    "مفيش ناموس",
    "مافيش ناموس",
    "ولا ناموس",
    "معنديش بعوض",
    "مفيش بعوض",
    "no travel",
    "no mosquito",
}


CHALLENGE_TERMS = {
    "ليه",
    "لماذا",
    "ازاي",
    "إزاي",
    "مش منطقي",
    "مش صح",
    "غلط",
    "معنديش",
    "ماعنديش",
    "مفيش",
    "مافيش",
    "انا معنديش",
    "not right",
    "why",
}


DIAGNOSIS_QUESTION_TERMS = {
    "اي",
    "إيه",
    "ايه",
    "ما هو",
    "يعني ايه",
    "يعني إيه",
    "اصلا",
    "أصلا",
    "اشرح",
    "التشخيص ده",
    "المرض ده",
    "خطير",
    "what is",
}


ROUTINE_EMERGENCY_BLOCKED_TERMS = {
    "الخطورة منخفضة",
    "انتظر وشوف",
    "انتظر",
    "راقب فقط",
    "استرخاء",
    "استرخ",
    "تأمل",
    "حمام ملح",
    "نام",
    "عيادة الصحة العامة",
    "relax",
    "meditation",
    "salt bath",
    "sleep it off",
}


ARABIC_DIGIT_TRANSLATION = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "٫": ".",
        "٬": ".",
    }
)

FEVER_CONTEXT_TERMS = {
    "حرارة",
    "حراره",
    "درجة الحرارة",
    "درجه الحراره",
    "سخونية",
    "سخونيه",
    "سخن",
    "حمى",
    "حمي",
    "fever",
    "temperature",
    "temp",
}

TEMPERATURE_QUESTION_TERMS = {
    "الحرارة وصلت كام",
    "الحراره وصلت كام",
    "درجة الحرارة وصلت كام",
    "درجه الحراره وصلت كام",
    "قست الحرارة",
    "قست الحراره",
    "وصلت كام تقريبا",
    "وصلت كام تقريب",
    "كام درجة",
    "كام درجه",
}

DURATION_TERMS = {
    "بقالها",
    "بقالي",
    "بقاله",
    "من امتى",
    "من امتي",
    "منذ متى",
    "منذ متي",
    "يوم",
    "يومين",
    "ايام",
    "أيام",
    "اسبوع",
    "أسبوع",
    "ساعه",
    "ساعة",
    "ساعات",
}

CASUAL_OR_OFFTOPIC_TERMS = {
    "عامل اي",
    "عامل ايه",
    "انت عامل ايه",
    "انتي عامله ايه",
    "اي الاخبار",
    "ايه الاخبار",
    "السلام عليكم",
    "هاي",
    "هلا",
    "ازيك",
    "ازيك يا دكتور",
    "صباح الخير",
    "مساء الخير",
}

STRONG_FRUSTRATION_TERMS = {
    "انت كويس",
    "انتي كويسه",
    "انت عبيط",
    "انتي عبيطه",
    "عبيط",
    "غبي",
    "غبية",
    "بتهزر",
    "تهزر",
    "مش فاهم",
    "مش فاهمة",
    "مش واضح",
    "مش منطقي",
    "ده غبي",
    "رد غريب",
    "كلامك غلط",
}

SOFT_FRUSTRATION_TERMS = {
    "يبني",
    "يا ابني",
    "يا بني",
}

ANSWERED_TEMPERATURE_QUESTION_TERMS = {
    "الحرارة وصلت كام",
    "الحراره وصلت كام",
    "درجة الحرارة وصلت كام",
    "درجه الحراره وصلت كام",
    "كام درجة",
    "كام درجه",
}

HIGH_FEVER_RED_FLAG_QUESTION = (
    "هل في ضيق تنفس، تيبس رقبة، طفح، قيء متكرر، ألم صدر، تشوش، أو علامات جفاف؟"
)


@dataclass
class ConversationCaseSummary:
    user_history_text: str
    assistant_history_text: str
    current_symptoms: list[str]
    known_symptoms: list[str]
    temperature_c: float | None = None
    has_high_temperature: bool = False
    duration_known: bool = False
    previous_diagnosis: str | None = None
    asked_temperature: bool = False
    asked_duration: bool = False
    asked_red_flags: bool = False


def _has_any_normalized(text: str, terms: set[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in terms)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _translate_arabic_digits(text: str) -> str:
    translated = (text or "").translate(ARABIC_DIGIT_TRANSLATION)
    return re.sub(r"(?<=\d),(?=\d)", ".", translated)


def _asked_temperature(text: str) -> bool:
    return _has_any_normalized(text, TEMPERATURE_QUESTION_TERMS)


def _asked_duration(text: str) -> bool:
    return _has_any_normalized(
        text,
        {
            "بدأت من إمتى",
            "بدات من امتى",
            "من إمتى بدأت",
            "من امتى بدأت",
            "بقالها قد إيه",
            "بقالها قد ايه",
            "منذ متى بدأت",
        },
    )


def _asked_red_flags(text: str) -> bool:
    return _has_any_normalized(
        text,
        {
            "ضيق تنفس",
            "ألم صدر",
            "الم صدر",
            "تيبس رقبة",
            "طفح",
            "قيء متكرر",
            "ترجيع متكرر",
            "تشوش",
            "جفاف",
            "تدهور سريع",
        },
    )


def _has_duration_answer(text: str) -> bool:
    return _has_any_normalized(text, DURATION_TERMS)


def _extract_temperature_c(text: str, assistant_context: str = "") -> float | None:
    translated = _translate_arabic_digits(text)
    normalized = normalize_text(translated)
    if not normalized:
        return None

    has_fever_context = _has_any_normalized(translated, FEVER_CONTEXT_TERMS) or _asked_temperature(
        assistant_context
    )
    candidates: list[float] = []
    for match in re.finditer(r"(?<!\d)(3[5-9](?:\.\d)?|4[0-2](?:\.\d)?)(?!\d)", normalized):
        value = float(match.group(1))
        local_window = normalized[max(0, match.start() - 45) : match.end() + 45]
        local_fever_context = any(normalize_text(term) in local_window for term in FEVER_CONTEXT_TERMS)
        if has_fever_context or local_fever_context:
            candidates.append(value)
    return max(candidates) if candidates else None


def _is_casual_or_offtopic(message: str, current_symptoms: list[str]) -> bool:
    if current_symptoms or detect_emergency_signal(message, current_symptoms):
        return False
    normalized = normalize_text(message)
    if not normalized:
        return False
    word_count = len(normalized.split())
    return word_count <= 6 and _has_any_normalized(message, CASUAL_OR_OFFTOPIC_TERMS)


def _is_insult_or_frustration(message: str, current_symptoms: list[str]) -> bool:
    if detect_emergency_signal(message, current_symptoms):
        return False
    normalized = normalize_text(message)
    if not normalized:
        return False
    if _has_any_normalized(message, STRONG_FRUSTRATION_TERMS):
        return True
    if current_symptoms:
        return False
    return _has_any_normalized(message, SOFT_FRUSTRATION_TERMS) and (
        "انت" in normalized or "الرد" in normalized or "كلام" in normalized
    )


def _filter_questions_without_fallback(
    questions: list[str],
    previous_assistant_messages: list[str],
    limit: int,
) -> list[str]:
    previous_text = normalize_text(" ".join(previous_assistant_messages))
    filtered = [
        question
        for question in questions
        if normalize_text(question) and normalize_text(question) not in previous_text
    ]
    return filtered[:limit]


def _remove_duplicate_question_lines(answer: str, follow_up_questions: list[str]) -> str:
    if not answer or not follow_up_questions:
        return answer
    normalized_questions = {normalize_text(question).strip(" -:؟?") for question in follow_up_questions}
    kept_lines: list[str] = []
    removed_any = False
    for line in answer.splitlines():
        stripped = line.strip()
        normalized_line = normalize_text(stripped.lstrip("-•*0123456789. ")).strip(" -:؟?")
        if normalized_line and normalized_line in normalized_questions:
            removed_any = True
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    for question in follow_up_questions:
        if question:
            cleaned = cleaned.replace(question, "")
    if removed_any:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([؟?.,،])", r"\1", cleaned).strip()
    return cleaned or answer


def _question_mentions_temperature(question: str) -> bool:
    return _has_any_normalized(question, ANSWERED_TEMPERATURE_QUESTION_TERMS)


def _answer_has_urgent_action(answer: str) -> bool:
    return _has_any_normalized(
        answer,
        {
            "الطوارئ",
            "الإسعاف",
            "الاسعاف",
            "مساعدة طبية عاجلة",
            "رعاية عاجلة",
            "توجه فورًا",
            "توجه فورا",
            "اتصل بالإسعاف",
            "اتصل بالاسعاف",
        },
    )


def _deterministic_emergency_answer(
    display_diagnosis: str,
    display_doctor: str,
    reason: str,
) -> str:
    return (
        f"{HIGH_URGENCY_PREFIX}\n\n"
        f"القلق الطبي الأقرب: {display_diagnosis}.\n\n"
        f"السبب: {reason}\n\n"
        "من فضلك اطلب مساعدة عاجلة الآن، وخلي شخص قريب يفضل معاك لحد ما توصل للرعاية المناسبة. "
        "لو الأعراض شديدة أو بدأت فجأة، لا تقود السيارة بنفسك.\n\n"
        f"الطبيب المناسب: {display_doctor}.\n\n"
        "تنبيه: هذا ليس تشخيصًا نهائيًا، لكنه موقف لا ينفع الانتظار فيه."
    )


def _deterministic_emergency_answer_en(
    display_diagnosis: str,
    display_doctor: str,
    reason: str,
) -> str:
    return (
        "These symptoms may be urgent. Please go to the emergency department now or call emergency services.\n\n"
        f"Main medical concern: {display_diagnosis}.\n\n"
        f"Why: {reason}\n\n"
        "Please do not wait, monitor only, or drive yourself if symptoms are severe or started suddenly. "
        "Stay with someone nearby until you reach appropriate care.\n\n"
        f"Suggested care: {display_doctor}.\n\n"
        "Note: this is not a final diagnosis, but it is not a situation to manage with routine home advice."
    )


EMERGENCY_REASON_EN: dict[str, str] = {
    "وجود أفكار أو نية لإيذاء النفس يحتاج مساعدة فورية ولا يجب التعامل معه وحدك.": "Thoughts or intent to self-harm need immediate support and should not be handled alone.",
    "وجود صعوبة تنفس أو خمول شديد أو تشنجات أو رفض رضاعة عند طفل/رضيع يحتاج طوارئ أطفال.": "Breathing difficulty, severe lethargy, seizures, or refusal to feed in a child or baby needs pediatric emergency care.",
    "صعوبة التنفس الشديدة أو زرقة الشفاه أو عدم القدرة على الكلام بسبب النفس قد تكون خطيرة.": "Severe breathing difficulty, blue lips, or being unable to speak because of breathlessness can be dangerous.",
    "القيء أو الإسهال الشديد مع علامات جفاف/لخبطة أو الحرارة العالية مع تدهور سريع يحتاج تقييمًا عاجلًا.": "Severe vomiting or diarrhea with dehydration/confusion, or high fever with rapid worsening, needs urgent assessment.",
    "اضطراب السكر مع إغماء أو لخبطة أو جفاف شديد قد يحتاج تدخلًا عاجلًا.": "A diabetes-related sugar problem with fainting, confusion, or severe dehydration may need urgent care.",
    "وجود ابتلاع منظف/كلور/سم أو جرعة كبيرة يحتاج تقييمًا عاجلًا.": "Swallowing a cleaner, bleach, poison, or a large medication dose needs urgent assessment.",
    "تورم الوجه أو الشفاه أو اللسان مع صعوبة التنفس قد يكون طارئًا.": "Swelling of the face, lips, or tongue with breathing difficulty can be an emergency.",
    "تنميل أو ضعف في ناحية من الجسم مع صعوبة كلام/ميلان الوجه/دوخة شديدة قد يشير لحالة عصبية طارئة.": "Numbness or weakness on one side of the body with speech difficulty, facial droop, or severe dizziness may suggest a neurological emergency.",
    "ألم أو ضغط الصدر مع ضيق تنفس أو عرق بارد أو امتداد الألم يحتاج طوارئ.": "Chest pain or pressure with shortness of breath, cold sweating, or pain spreading to the arm/jaw/back needs emergency care.",
    "نزيف أو قلة حركة الجنين أو ألم شديد أثناء الحمل يحتاج تقييمًا عاجلًا.": "Bleeding, reduced fetal movement, or severe pain during pregnancy needs urgent assessment.",
    "ألم البطن الشديد مع تيبس/تحجر أو قيء دم أو براز أسود أو ألم يمين أسفل البطن مع حرارة قد يحتاج طوارئ.": "Severe abdominal pain with rigidity, vomiting blood, black stool, or right-lower abdominal pain with fever may need emergency care.",
    "احتباس البول أو دم في البول مع ألم شديد أو ألم جنب مع حرارة يحتاج تقييمًا عاجلًا.": "Urinary retention, blood in urine with severe pain, or flank pain with fever needs urgent assessment.",
    "إصابة الرأس مع قيء أو فقدان وعي تحتاج تقييمًا عاجلًا.": "Head injury with vomiting or loss of consciousness needs urgent assessment.",
    "فقدان نظر مفاجئ أو ألم/إصابة شديدة في العين يحتاج تقييمًا عاجلًا.": "Sudden vision loss or severe eye pain/injury needs urgent assessment.",
    "وجود علامات خطورة في الرسالة يستدعي التقييم العاجل.": "The message includes red-flag symptoms that need urgent assessment.",
}


def _emergency_reason_for_language(reason: str, language: str) -> str:
    if language != "en":
        return reason
    if reason in EMERGENCY_REASON_EN:
        return EMERGENCY_REASON_EN[reason]
    for arabic_reason, english_reason in EMERGENCY_REASON_EN.items():
        if arabic_reason in reason:
            return english_reason
    if any("\u0600" <= char <= "\u06FF" for char in reason):
        return "The message includes red-flag symptoms that need urgent assessment."
    return reason


def _sanitize_patient_answer(answer: str) -> str:
    cleaned = answer or ""
    cleaned = re.sub(
        r"(الملاريا|ملاريا)[^\n.،]{0,80}(بكتيريا|بكتيرية)",
        "الملاريا عدوى سببها طفيل وينتقل غالبًا عن طريق لدغة نوع معين من البعوض",
        cleaned,
    )
    cleaned = re.sub(
        r"(malaria)[^\n.]{0,80}(bacteria|bacterial)",
        "malaria is caused by a parasite and is usually spread by certain mosquito bites",
        cleaned,
        flags=re.IGNORECASE,
    )
    blocked_replacements = {
        "حمام ملح": "راحة ومتابعة الأعراض",
        "تأمل": "راحة ومتابعة الأعراض",
        "avoid oily food": "stay hydrated and monitor symptoms",
        "avoid non-vegetarian food": "stay hydrated and monitor symptoms",
        "non-vegetarian": "specific irritating foods",
        "salt bath": "safe rest and symptom monitoring",
        "meditation": "rest and symptom monitoring",
        "classifier": "التقييم المبدئي",
        "RAG": "المعرفة الطبية",
        "retrieved cases": "المراجع الطبية",
        "confidence": "درجة الترجيح",
    }
    for old, new in blocked_replacements.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    return cleaned


def _final_response_guardrail(
    response: ChatResponse,
    emergency_signal: EmergencySignal | None = None,
    language: str = "ar",
) -> ChatResponse:
    if response.mode == CLARIFICATION_MODE:
        response.possible_diagnosis = None
        response.display_diagnosis_ar = None
        response.confidence = 0.0
        response.follow_up_questions = response.follow_up_questions[:MAX_CLARIFICATION_QUESTIONS]
        response.needs_follow_up = bool(response.follow_up_questions)
        response.answer = _remove_duplicate_question_lines(response.answer, response.follow_up_questions)
        if response.answer.strip().startswith("محتاج أعرف كام حاجة بسيطة"):
            response.answer = clarification_answer(response.follow_up_questions)
        elif not response.answer.strip():
            response.answer = clarification_answer(response.follow_up_questions)
        response.answer = _remove_duplicate_question_lines(response.answer, response.follow_up_questions)

    if response.mode == CLOSING_MODE:
        response.needs_follow_up = False
        response.follow_up_questions = []

    is_emergency = bool(emergency_signal) or response.mode == EMERGENCY_MODE or response.urgency_level == "High"
    if not is_emergency:
        if response.mode == DIAGNOSIS_MODE:
            response.follow_up_questions = response.follow_up_questions[:MAX_DIAGNOSIS_QUESTIONS]
            response.needs_follow_up = bool(response.follow_up_questions)
            response.answer = _remove_duplicate_question_lines(response.answer, response.follow_up_questions)
        if language != "en" and response.possible_diagnosis and response.display_diagnosis_ar:
            response.answer = response.answer.replace(response.possible_diagnosis, response.display_diagnosis_ar)
        if language != "en" and response.suggested_doctor and response.display_doctor_ar:
            response.answer = response.answer.replace(response.suggested_doctor, response.display_doctor_ar)
        response.answer = _sanitize_patient_answer(response.answer)
        return response

    response.mode = EMERGENCY_MODE
    response.urgency_level = "High"
    response.needs_follow_up = False
    response.follow_up_questions = []

    if emergency_signal:
        response.possible_diagnosis = emergency_signal.diagnosis
        response.display_diagnosis_ar = emergency_signal.display_diagnosis_ar
        response.suggested_doctor = emergency_signal.doctor
        response.display_doctor_ar = emergency_signal.display_doctor_ar
        response.confidence = max(response.confidence, 0.99)
        response.retrieved_cases = []
        reason = emergency_signal.reason
    else:
        response.display_diagnosis_ar = response.display_diagnosis_ar or display_diagnosis_ar(
            response.possible_diagnosis
        ) or "حالة قد تحتاج طوارئ"
        response.display_doctor_ar = response.display_doctor_ar or display_doctor_ar(response.suggested_doctor)
        reason = "وجود علامات خطورة في الرسالة يستدعي التقييم العاجل."

    if not response.display_doctor_ar or (
        "الطوارئ" not in response.display_doctor_ar and "طوارئ" not in response.display_doctor_ar
    ):
        response.suggested_doctor = "Emergency care"
        response.display_doctor_ar = display_doctor_ar(response.suggested_doctor)

    if response.possible_diagnosis and response.display_diagnosis_ar:
        response.answer = response.answer.replace(response.possible_diagnosis, response.display_diagnosis_ar)
    if response.suggested_doctor and response.display_doctor_ar:
        response.answer = response.answer.replace(response.suggested_doctor, response.display_doctor_ar)

    if language == "en":
        response.answer = _deterministic_emergency_answer_en(
            response.possible_diagnosis or response.display_diagnosis_ar or "possible emergency condition",
            response.suggested_doctor or response.display_doctor_ar or "Emergency care",
            _emergency_reason_for_language(reason, language),
        )
    else:
        response.answer = _deterministic_emergency_answer(
            response.display_diagnosis_ar or "حالة طارئة محتملة",
            response.display_doctor_ar or "الطوارئ فورًا",
            reason,
        )
    response.answer = _sanitize_patient_answer(response.answer)

    return response


def _consistency_normalize(text: str) -> str:
    text = normalize_text(text)
    replacements = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _consistency_has_any(text: str, terms: set[str]) -> bool:
    normalized = _consistency_normalize(text)
    return any(_consistency_normalize(term) in normalized for term in terms)


def _response_conflicts_with_medical_meaning(response: ChatResponse, state: ConversationState | None) -> bool:
    if not state or response.mode in {EMERGENCY_MODE, CLOSING_MODE} or response.urgency_level == "High":
        return False
    meaning = state.medical_meaning
    domain = meaning.domain
    if not domain:
        return False

    combined_text = " ".join(
        str(part or "")
        for part in (
            response.answer,
            response.possible_diagnosis,
            response.display_diagnosis_ar,
            response.suggested_doctor,
            response.display_doctor_ar,
        )
    )

    neck_terms = {
        "neck",
        "cervical",
        "Cervical spondylosis",
        "\u0631\u0642\u0628",
        "\u0627\u0644\u0631\u0642\u0628\u0629",
        "\u0627\u0644\u0631\u0642\u0628\u0647",
        "\u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629",
        "\u062e\u0634\u0648\u0646\u0629 \u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629",
    }
    back_terms = {"back", "back pain", "\u0638\u0647\u0631", "\u0636\u0647\u0631", "\u0627\u0644\u0638\u0647\u0631", "\u0627\u0644\u0636\u0647\u0631"}
    abdomen_terms = {"stomach", "abdomen", "abdominal", "\u0628\u0637\u0646", "\u0645\u0639\u062f\u0629", "\u0645\u0639\u062f\u0647"}
    urinary_terms = {"urine", "urinary", "\u0628\u0648\u0644", "\u062a\u0628\u0648\u0644", "\u0645\u0633\u0627\u0644\u0643"}
    throat_terms = {"throat", "sore throat", "\u0632\u0648\u0631", "\u062d\u0644\u0642", "\u062d\u0646\u062c\u0631\u0629"}

    if domain == "musculoskeletal_back_pain":
        if response.possible_diagnosis == "Cervical spondylosis":
            return True
        if response.suggested_doctor in {"Neurologist", "Endocrinologist"} and not meaning.red_flags:
            return True
        if _consistency_has_any(combined_text, neck_terms) and "neck" not in meaning.body_parts:
            return True
        if "numbness" in meaning.denied and not _consistency_has_any(combined_text, {"numbness", "\u062a\u0646\u0645\u064a\u0644"}):
            return True
        return not _consistency_has_any(combined_text, back_terms)

    if domain == "musculoskeletal_neck_pain":
        if _consistency_has_any(combined_text, abdomen_terms):
            return True
        return not _consistency_has_any(combined_text, neck_terms)

    if domain == "throat_ent":
        if _has_throat_cough_fever_cluster(state.known_symptoms):
            return False
        if response.possible_diagnosis == "Cervical spondylosis":
            return True
        if _consistency_has_any(combined_text, abdomen_terms | back_terms):
            return True
        return False

    if domain == "digestive_abdominal":
        if _consistency_has_any(combined_text, neck_terms):
            return True
        return False

    if domain == "urinary":
        if response.suggested_doctor in {"Neurologist", "Dermatologist", "Gastroenterologist"}:
            return True
        if response.mode == DIAGNOSIS_MODE and not _consistency_has_any(combined_text, urinary_terms):
            return True
        return False

    return False


def _final_meaning_consistency_guardrail(
    response: ChatResponse,
    state: ConversationState | None,
    assistant_history_texts: list[str],
) -> ChatResponse:
    if not _response_conflicts_with_medical_meaning(response, state):
        return response
    assert state is not None
    replacement = _meaning_domain_clarification_response(state, assistant_history_texts)
    if replacement:
        return replacement

    questions = filter_repeated_questions(
        smart_follow_up_questions(state.current_message, state.known_symptoms),
        assistant_history_texts,
    )
    return _non_diagnostic_response(
        answer=clarification_answer(questions),
        state=state,
        suggested_doctor=_clarification_doctor(state.current_message, response.urgency_level),
        follow_up_questions=questions,
    )


def _clarification_doctor(message: str, urgency_level: str) -> str:
    flags = detect_context_flags(message)
    if urgency_level == "High" or flags.intersection(SPECIALTY_CONTEXT_FLAGS):
        return knowledge_service.suggest_doctor(None, [], urgency_level, message=message)
    if _has_any_normalized(
        message,
        {
            "\u0631\u0634\u062d",
            "\u0627\u062d\u062a\u0642\u0627\u0646",
            "\u0648\u062c\u0639 \u062d\u0644\u0642",
            "\u0623\u0644\u0645 \u062d\u0644\u0642",
            "\u0627\u0644\u0645 \u062d\u0644\u0642",
            "\u0648\u062c\u0639 \u0632\u0648\u0631",
            "runny nose",
            "congestion",
            "sore throat",
        },
    ):
        return "General Practitioner"
    return "Needs more information"


def _diagnosis_mentioned_in_text(*texts: str) -> str | None:
    normalized = normalize_text(" ".join(text for text in texts if text))
    if not normalized:
        return None
    for diagnosis, aliases in DIAGNOSIS_ALIASES.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            return diagnosis
    return None


def _is_diagnosis_question(message: str, diagnosis: str | None) -> bool:
    if not diagnosis:
        return False
    normalized = normalize_text(message)
    question_like = any(normalize_text(term) in normalized for term in DIAGNOSIS_QUESTION_TERMS)
    alias_mentioned = any(
        normalize_text(alias) in normalized for alias in DIAGNOSIS_ALIASES.get(diagnosis, set())
    )
    generic_reference = any(
        normalize_text(term) in normalized
        for term in {
            "التشخيص ده",
            "المرض ده",
            "الحالة دي",
            "الحاله دي",
            "ده خطير",
            "دي خطيرة",
            "اشرحلي ده",
            "اشرحلي المرض",
        }
    )
    return question_like and (alias_mentioned or generic_reference)


def _is_challenge_or_correction(message: str, diagnosis: str | None) -> bool:
    if not diagnosis:
        return False
    normalized = normalize_text(message)
    return any(normalize_text(term) in normalized for term in CHALLENGE_TERMS)


def _has_malaria_support_context(message: str) -> bool:
    normalized = normalize_text(message)
    has_risk_context = any(normalize_text(term) in normalized for term in MALARIA_RISK_TERMS)
    has_cyclic_context = any(normalize_text(term) in normalized for term in MALARIA_CYCLIC_TERMS)
    return has_risk_context or has_cyclic_context


def _denies_malaria_support_context(message: str) -> bool:
    normalized = normalize_text(message)
    return any(normalize_text(term) in normalized for term in MALARIA_CONTEXT_DENIAL_TERMS)


def _is_generic_fever_body_aches(symptoms: list[str], message: str) -> bool:
    symptom_set = set(symptoms)
    fever_present = bool({"high_fever", "mild_fever"}.intersection(symptom_set))
    constitutional_symptoms = {
        "muscle_pain",
        "fatigue",
        "joint_pain",
        "malaise",
        "lethargy",
        "headache",
        "chills",
        "shivering",
        "sweating",
    }
    organ_specific_or_red_flag = {
        "cough",
        "phlegm",
        "breathlessness",
        "chest_pain",
        "diarrhoea",
        "vomiting",
        "abdominal_pain",
        "skin_rash",
        "itching",
        "red_spots_over_body",
        "pain_behind_the_eyes",
        "burning_micturition",
        "continuous_feel_of_urine",
        "spotting_ urination",
        "yellowing_of_eyes",
        "yellowish_skin",
        "dark_urine",
        "stiff_neck",
        "weakness_of_one_body_side",
        "slurred_speech",
    }
    if not fever_present or not constitutional_symptoms.intersection(symptom_set):
        return False
    if organ_specific_or_red_flag.intersection(symptom_set):
        return False
    return not _has_malaria_support_context(message)


def _has_throat_cough_fever_cluster(symptoms: list[str]) -> bool:
    symptom_set = set(symptoms)
    return (
        "throat_irritation" in symptom_set
        and "cough" in symptom_set
        and bool({"mild_fever", "high_fever"}.intersection(symptom_set))
    )


def _is_malaria_supported_fever(symptoms: list[str], message: str) -> bool:
    symptom_set = set(symptoms)
    fever_present = bool({"high_fever", "mild_fever"}.intersection(symptom_set))
    chill_or_sweat = bool({"chills", "shivering", "sweating"}.intersection(symptom_set))
    return (
        fever_present
        and chill_or_sweat
        and _has_malaria_support_context(message)
        and not _denies_malaria_support_context(message)
    )


def _limited_followups(questions: list[str], assistant_history_texts: list[str], limit: int = MAX_DIAGNOSIS_QUESTIONS) -> list[str]:
    return filter_repeated_questions(questions, assistant_history_texts)[:limit]


def _build_case_summary(
    *,
    current_message: str,
    current_symptoms: list[str],
    recent_history: list,
    assistant_history_texts: list[str],
) -> ConversationCaseSummary:
    user_history_text = " ".join(
        item.content for item in recent_history if item.role == "user" and item.content
    )
    assistant_history_text = " ".join(assistant_history_texts)
    history_symptoms = (
        classifier_service.extract_symptoms("", history_text=user_history_text)
        if user_history_text
        else []
    )
    known_symptoms = _dedupe(history_symptoms + current_symptoms)
    combined_user_text = f"{user_history_text} {current_message}".strip()
    temperature_c = _extract_temperature_c(combined_user_text, assistant_history_text)
    if temperature_c is not None and temperature_c >= 39.0 and "high_fever" not in known_symptoms:
        known_symptoms.append("high_fever")
    previous_diagnosis = _diagnosis_mentioned_in_text(*assistant_history_texts[-4:])
    return ConversationCaseSummary(
        user_history_text=user_history_text,
        assistant_history_text=assistant_history_text,
        current_symptoms=current_symptoms,
        known_symptoms=known_symptoms,
        temperature_c=temperature_c,
        has_high_temperature=temperature_c is not None and temperature_c >= 39.5,
        duration_known=_has_duration_answer(combined_user_text),
        previous_diagnosis=previous_diagnosis,
        asked_temperature=_asked_temperature(assistant_history_text),
        asked_duration=_asked_duration(assistant_history_text),
        asked_red_flags=_asked_red_flags(assistant_history_text),
    )


def _is_answer_to_followup(message: str, summary: ConversationCaseSummary) -> bool:
    if summary.asked_temperature and _extract_temperature_c(message, summary.assistant_history_text) is not None:
        return True
    if summary.asked_duration and _has_duration_answer(message):
        return True
    return False


def _casual_response() -> ChatResponse:
    suggested_doctor = "Not needed"
    return ChatResponse(
        mode=CLARIFICATION_MODE,
        answer="أنا تمام، موجود عشان أساعدك طبيًا. لو عندك عرض أو سؤال صحي اكتبهولي.",
        extracted_symptoms=[],
        possible_diagnosis=None,
        display_diagnosis_ar=None,
        confidence=0.0,
        urgency_level="Low",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor_ar(suggested_doctor),
        precautions=[],
        needs_follow_up=False,
        follow_up_questions=[],
        retrieved_cases=[],
    )


def _is_english(language: str) -> bool:
    return language == "en"


def _non_diagnostic_response(
    *,
    answer: str,
    state: ConversationState,
    urgency_level: str = "Low",
    suggested_doctor: str = "Not needed",
    follow_up_questions: list[str] | None = None,
) -> ChatResponse:
    questions = (follow_up_questions or [])[:1 if state.intent in {
        ConversationIntent.GREETING_OR_CASUAL,
        ConversationIntent.NON_MEDICAL_CHAT,
        ConversationIntent.FAMILY_OR_OFFTOPIC,
        ConversationIntent.NONSENSE_OR_LOW_INFORMATION,
        ConversationIntent.PROFANITY_OR_ABUSE,
        ConversationIntent.INSULT_OR_FRUSTRATION,
    } else MAX_CLARIFICATION_QUESTIONS]
    return ChatResponse(
        mode=CLARIFICATION_MODE,
        answer=answer,
        extracted_symptoms=state.known_symptoms,
        possible_diagnosis=None,
        display_diagnosis_ar=None,
        confidence=0.0,
        urgency_level=urgency_level,
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor_ar(suggested_doctor),
        precautions=[],
        needs_follow_up=bool(questions),
        follow_up_questions=questions,
        retrieved_cases=[],
    )


def _route_for_conversation_intent(state: ConversationState) -> str:
    if state.intent == ConversationIntent.PROFANITY_OR_ABUSE:
        return "abuse"
    if state.intent == ConversationIntent.INSULT_OR_FRUSTRATION:
        return "abuse"
    if state.intent == ConversationIntent.FAMILY_OR_OFFTOPIC:
        return "family"
    if state.intent == ConversationIntent.NON_MEDICAL_CHAT:
        normalized = normalize_text(state.current_message)
        if any(term in normalized for term in {"who are you", "what are you", "مين انت", "انت مين"}):
            return "who_are_you"
        if any(term in normalized for term in {"what can you do", "تقدر تعمل ايه", "بتعمل ايه"}):
            return "capabilities"
        return "off_topic"
    if state.intent == ConversationIntent.NONSENSE_OR_LOW_INFORMATION:
        return "nonsense"
    if state.intent == ConversationIntent.GREETING_OR_CASUAL:
        return "casual"
    return "medical_clarification"


def _naturalize_chat_response(
    response: ChatResponse,
    *,
    state: ConversationState | None,
    route: str,
    message: str,
    history: list,
    language: str | None = None,
    medical_domain: str | None = None,
    diagnosis_allowed: bool = False,
) -> ChatResponse:
    stable_body_area_domains = {
        "musculoskeletal_back_pain",
        "musculoskeletal_neck_pain",
        "throat_ent",
        "digestive_abdominal",
        "urinary",
    }
    if (
        state
        and response.mode == CLARIFICATION_MODE
        and (
            state.medical_meaning.domain in stable_body_area_domains
            or state.active_case.get("active_domain") in stable_body_area_domains
        )
    ):
        return response
    response.answer = llm_service.naturalize_response(
        route=route,
        message=message,
        language=language or (state.language if state else "ar"),
        history=history,
        draft_answer=response.answer,
        follow_up_questions=response.follow_up_questions,
        suggested_doctor=response.suggested_doctor,
        medical_domain=medical_domain or (",".join(sorted(state.medical_domains)) if state else None),
        diagnosis_allowed=diagnosis_allowed,
    )
    return response


LLM_ROUTER_INTENT_MAP = {
    "casual": ConversationIntent.GREETING_OR_CASUAL,
    "off_topic": ConversationIntent.NON_MEDICAL_CHAT,
    "family": ConversationIntent.FAMILY_OR_OFFTOPIC,
    "abuse": ConversationIntent.PROFANITY_OR_ABUSE,
    "nonsense": ConversationIntent.NONSENSE_OR_LOW_INFORMATION,
    "correction": ConversationIntent.CORRECTION_OR_NEGATION,
    "challenge": ConversationIntent.CHALLENGE_PREVIOUS_DIAGNOSIS,
    "followup_answer": ConversationIntent.ANSWER_FOLLOWUP_QUESTION,
}

LLM_ROUTER_DOMAIN_MAP = {
    "body_ache": "general",
    "fever": "infection",
    "respiratory": "respiratory",
    "throat_ent": "throat_ent",
    "digestive": "digestive",
    "urinary": "urinary",
    "reproductive_gynecology": "reproductive",
    "pregnancy": "reproductive",
    "skin": "skin",
    "eye": "eye",
    "dental": "dental",
    "neurology": "neurological",
    "heart_chest": "cardiac",
    "mental_health": "mental_health",
    "allergy": "allergy",
    "injury": "trauma",
    "medication": "medication_poisoning",
    "poisoning": "medication_poisoning",
    "child_elderly": "pediatric",
    "chronic": "general",
}


def _apply_llm_router_for_uncertain_state(
    *,
    state: ConversationState,
    message: str,
    recent_history: list,
) -> None:
    if state.intent != ConversationIntent.NONSENSE_OR_LOW_INFORMATION or state.current_symptoms:
        return
    routed = llm_service.route_intent(
        message=message,
        language=state.language,
        history=recent_history,
    )
    if not routed:
        return
    intent = str(routed.get("intent") or "").strip().lower()
    needs_medical_path = bool(routed.get("needs_medical_path"))
    domain = str(routed.get("medical_domain") or "none").strip().lower()
    mapped_domain = LLM_ROUTER_DOMAIN_MAP.get(domain)
    if mapped_domain:
        state.medical_domains.add(mapped_domain)
    if needs_medical_path and intent in {"medical_complaint", "medical_question", "followup_answer"}:
        state.intent = ConversationIntent.VAGUE_UNCLEAR
        return
    mapped_intent = LLM_ROUTER_INTENT_MAP.get(intent)
    if mapped_intent:
        state.intent = mapped_intent


def _casual_or_nonmedical_response(state: ConversationState) -> ChatResponse:
    if _is_english(state.language):
        if state.intent == ConversationIntent.FAMILY_OR_OFFTOPIC:
            answer = "I’m here as a medical assistant, so let’s keep it to your health. Tell me your symptoms or medical question and I’ll guide you safely."
        elif state.intent == ConversationIntent.NON_MEDICAL_CHAT:
            answer = "I’m MedBridge AI, a medical assistant. I can help you think through symptoms safely, but I won’t diagnose from casual chat."
        elif state.intent == ConversationIntent.NONSENSE_OR_LOW_INFORMATION:
            answer = "I’m here to help with medical questions. Tell me your symptoms or health concern and I’ll guide you safely."
        else:
            answer = "I’m doing fine. I’m here to help with medical symptoms or health questions whenever you’re ready."
    else:
        if state.intent == ConversationIntent.FAMILY_OR_OFFTOPIC:
            answer = "أنا هنا كمساعد طبي، فخلّينا في صحتك أو أي سؤال طبي تحب تسأله."
        elif state.intent == ConversationIntent.NON_MEDICAL_CHAT:
            answer = "أنا MedBridge AI، مساعد طبي. أقدر أساعدك في فهم الأعراض والأسئلة الصحية بأمان، لكن مش هطلع تشخيص من دردشة عامة."
        elif state.intent == ConversationIntent.NONSENSE_OR_LOW_INFORMATION:
            answer = "أنا معاك. لو عندك عرض صحي أو سؤال طبي اكتبهولي وهساعدك خطوة بخطوة."
        else:
            answer = "أنا تمام، موجود عشان أساعدك طبيًا. لو عندك عرض أو سؤال صحي اكتبهولي."
    return _non_diagnostic_response(answer=answer, state=state)


def _frustration_state_response(state: ConversationState, assistant_history_texts: list[str]) -> ChatResponse:
    if _is_english(state.language):
        if state.intent == ConversationIntent.PROFANITY_OR_ABUSE:
            answer = (
                "Let’s keep it respectful so I can help. If you have symptoms or a medical question, tell me clearly and I’ll guide you safely."
            )
        else:
            answer = (
                "I hear the frustration. I won’t diagnose from that message. If there’s a health concern, describe the symptom clearly and I’ll help safely."
            )
    else:
        if state.intent == ConversationIntent.PROFANITY_OR_ABUSE:
            answer = (
                "خلينا نتكلم باحترام عشان أقدر أساعدك. لو عندك عرض صحي أو سؤال طبي اكتبهولي بوضوح."
            )
        else:
            answer = (
                "حاسس إنك متضايق، وحقك تطلب رد أوضح. مش هطلع تشخيص من الرسالة دي. لو في مشكلة صحية قولها بوضوح وهساعدك."
            )

    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor="Not needed",
        follow_up_questions=[],
    )


def _denied_concepts_text(state: ConversationState) -> str:
    labels_ar = {
        "fever": "الحرارة",
        "cough": "الكحة",
        "travel": "السفر",
        "mosquito": "التعرض للناموس",
        "breathlessness": "ضيق التنفس",
        "chest_pain": "ألم الصدر",
    }
    labels_en = {
        "fever": "fever",
        "cough": "cough",
        "travel": "travel",
        "mosquito": "mosquito exposure",
        "breathlessness": "breathlessness",
        "chest_pain": "chest pain",
    }
    labels = labels_en if _is_english(state.language) else labels_ar
    values = [labels.get(concept, concept) for concept in sorted(state.denied_concepts)]
    if not values:
        return "the previous assumption" if _is_english(state.language) else "الافتراض السابق"
    return ", ".join(values)


def _correction_response(state: ConversationState, assistant_history_texts: list[str]) -> ChatResponse:
    denied_text = _denied_concepts_text(state)
    if _is_english(state.language):
        answer = (
            f"Thanks for correcting me. We should remove {denied_text} from the picture and not keep using it as evidence.\n\n"
            "With the remaining information, this is still too broad for a specific diagnosis. A few focused details will help."
        )
        questions = [
            "What symptom is bothering you the most right now?",
            "When did it start, and was it after effort, injury, poor sleep, or an infection exposure?",
        ]
    else:
        answer = (
            f"تمام، شكرًا للتوضيح. كده نستبعد {denied_text} من الصورة ومش هنفضل نبني عليه.\n\n"
            "باقي الكلام لسه عام ومحتاج تفاصيل بسيطة بدل ما نثبت تشخيص غير مدعوم."
        )
        questions = [
            "أكتر عرض مضايقك دلوقتي إيه بالضبط؟",
            "بدأ من إمتى؟ وهل كان بعد مجهود/إصابة أو مع أعراض زي التهاب حلق، كحة، إسهال، أو حرقان بول؟",
        ]
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=2)
    suggested_doctor = _clarification_doctor(state.current_message, "Low")
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor=suggested_doctor,
        follow_up_questions=questions,
    )


def _body_ache_clarification_response(state: ConversationState, assistant_history_texts: list[str]) -> ChatResponse:
    if _is_english(state.language):
        answer = (
            "Body aches by themselves are nonspecific. They can happen with muscle strain, poor sleep, early viral infection, dehydration, or other causes. "
            "I should not jump to a specific disease from that alone."
        )
        questions = [
            "When did the body aches start?",
            "Do you also have fever, sore throat, cough, stomach symptoms, urinary burning, or a recent heavy effort/injury?",
            "Is the pain all over or focused in one area?",
        ]
    else:
        answer = (
            "وجع الجسم لوحده عرض عام جدًا. ممكن يحصل مع إجهاد عضلي، قلة نوم، بداية عدوى فيروسية، جفاف، أو أسباب تانية. "
            "مش منطقي نطلع منه تشخيص محدد أو مرض مقلق من غير تفاصيل."
        )
        questions = [
            "وجع الجسم بدأ من إمتى؟",
            "هل فيه أعراض تانية زي التهاب حلق، كحة، إسهال، حرقان بول، أو كان بعد مجهود/إصابة؟",
            "الوجع في الجسم كله ولا في مكان معين؟",
        ]
        if "fever" not in state.denied_concepts:
            questions.insert(1, "هل فيه حرارة مقاسة أو إحساس بسخونية؟")
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=3)
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor="General Practitioner",
        follow_up_questions=questions,
    )


BACK_LOWER_CONTEXT_TERMS = {
    "lower back",
    "\u0627\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
    "\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
    "\u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
    "\u0623\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
}


def _active_case_has_lower_back(state: ConversationState) -> bool:
    active_body_part = str(state.active_case.get("active_body_part") or "")
    combined_user_text = f"{state.user_history_text} {state.current_message}"
    return active_body_part == "lower_back" or "lower_back" in state.medical_meaning.body_parts or _has_any_normalized(
        combined_user_text,
        BACK_LOWER_CONTEXT_TERMS,
    )


def _active_case_denies(state: ConversationState, *concepts: str) -> bool:
    denied = set(state.denied_concepts) | set(state.medical_meaning.denied) | set(
        state.active_case.get("denied_facts") or []
    )
    return any(concept in denied for concept in concepts)


def _back_follow_up_questions_for_state(state: ConversationState) -> list[str]:
    lower = _active_case_has_lower_back(state)
    denied_numbness = _active_case_denies(state, "numbness")
    questions: list[str] = []
    if _is_english(state.language):
        if not lower:
            questions.append("Is the pain in the lower back or upper back?")
        if not state.duration_known:
            if lower:
                questions.append("When did it start?")
                questions.append("Did it start after heavy lifting, injury, a sudden movement, or sitting for a long time?")
            else:
                questions.append("When did it start, and did it follow injury, lifting, sudden movement, or sitting for a long time?")
        if not denied_numbness and not state.asked_red_flags:
            questions.append("Does it go down the leg, or come with numbness, weakness, or bladder/bowel control problems?")
        elif not state.asked_red_flags:
            questions.append("Does it go down the leg, or come with leg weakness or bladder/bowel control problems?")
        return questions[:3]

    if not lower:
        questions.append("\u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631 \u0648\u0644\u0627 \u0623\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631\u061f")
    if not state.duration_known:
        if lower:
            questions.append("\u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u061f")
            questions.append("\u0647\u0644 \u0628\u062f\u0623 \u0628\u0639\u062f \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629\u060c \u062d\u0631\u0643\u0629 \u0645\u0641\u0627\u062c\u0626\u0629\u060c \u0625\u0635\u0627\u0628\u0629\u060c \u0623\u0648 \u0642\u0639\u062f\u0629 \u0637\u0648\u064a\u0644\u0629\u061f")
        else:
            questions.append("\u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u060c \u0648\u0647\u0644 \u0643\u0627\u0646 \u0628\u0639\u062f \u0625\u0635\u0627\u0628\u0629 \u0623\u0648 \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629\u061f")
    if not denied_numbness and not state.asked_red_flags:
        questions.append("\u0647\u0644 \u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0631\u062c\u0644 \u0623\u0648 \u0645\u0639\u0627\u0647 \u062a\u0646\u0645\u064a\u0644/\u0636\u0639\u0641 \u0623\u0648 \u0645\u0634\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644 \u0623\u0648 \u0627\u0644\u0628\u0631\u0627\u0632\u061f")
    elif not state.asked_red_flags:
        questions.append("\u0647\u0644 \u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0631\u062c\u0644 \u0623\u0648 \u0645\u0639\u0627\u0647 \u0636\u0639\u0641 \u0623\u0648 \u0645\u0634\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644 \u0623\u0648 \u0627\u0644\u0628\u0631\u0627\u0632\u061f")
    return questions[:3]


def _back_pain_answer_for_state(state: ConversationState) -> str:
    lower = _active_case_has_lower_back(state)
    denied_numbness = _active_case_denies(state, "numbness")
    followup = state.intent == ConversationIntent.ANSWER_FOLLOWUP_QUESTION
    if _is_english(state.language):
        if lower and denied_numbness:
            return (
                "Good, no numbness is somewhat reassuring. Since the pain is in the lower back, common broad causes include muscle strain, posture, heavy lifting, or disc/spine irritation. "
                "Avoid heavy lifting for now, keep gentle movement instead of staying in bed all day, use a warm compress if comfortable, and rest from movements that clearly worsen the pain. "
                "Seek urgent care if leg weakness, loss of bladder or bowel control, severe trauma, fever with severe back pain, or rapidly worsening pain appears."
            )
        if followup and lower:
            return (
                "Thanks, so we are keeping this focused on lower back pain. It can come from muscle strain, posture, heavy lifting, or irritation around the spine/discs. "
                "Avoid heavy lifting, keep gentle movement, and watch for urgent warning signs like leg weakness, bladder/bowel control problems, fever with severe pain, trauma, or fast worsening."
            )
        return (
            "Back pain is usually related to muscle strain, posture, heavy lifting, or a spine/joint issue depending on the location and pattern. "
            "I should narrow it down before suggesting any diagnosis."
        )

    if lower and denied_numbness:
        return (
            "\u062a\u0645\u0627\u0645\u060c \u0639\u062f\u0645 \u0648\u062c\u0648\u062f \u062a\u0646\u0645\u064a\u0644 \u0645\u0637\u0645\u0646 \u0646\u0633\u0628\u064a\u064b\u0627. "
            "\u0637\u0627\u0644\u0645\u0627 \u0627\u0644\u0623\u0644\u0645 \u0641\u064a \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631\u060c \u0641\u0627\u0644\u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0634\u0627\u0626\u0639\u0629 \u0645\u0645\u0643\u0646 \u062a\u0643\u0648\u0646 \u0634\u062f \u0639\u0636\u0644\u064a\u060c \u0648\u0636\u0639\u064a\u0629 \u063a\u0644\u0637\u060c \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629\u060c \u0623\u0648 \u062a\u0647\u064a\u062c \u0641\u064a \u0627\u0644\u0641\u0642\u0631\u0627\u062a/\u0627\u0644\u063a\u0636\u0627\u0631\u064a\u0641. "
            "\u062d\u0627\u0648\u0644 \u062a\u062a\u062c\u0646\u0628 \u0627\u0644\u062d\u0645\u0644 \u0627\u0644\u062a\u0642\u064a\u0644\u060c \u0627\u062a\u062d\u0631\u0643 \u062d\u0631\u0643\u0629 \u062e\u0641\u064a\u0641\u0629 \u0645\u0646 \u063a\u064a\u0631 \u0639\u0646\u0641 \u0648\u0645\u0627 \u062a\u0641\u0636\u0644\u0634 \u0641\u064a \u0627\u0644\u0633\u0631\u064a\u0631 \u0637\u0648\u0644 \u0627\u0644\u064a\u0648\u0645\u060c \u0648\u0645\u0645\u0643\u0646 \u0643\u0645\u0627\u062f\u0627\u062a \u062f\u0627\u0641\u0626\u0629 \u0644\u0648 \u0645\u0631\u064a\u062d\u0629. "
            "\u0627\u0637\u0644\u0628 \u0631\u0639\u0627\u064a\u0629 \u0639\u0627\u062c\u0644\u0629 \u0644\u0648 \u0638\u0647\u0631 \u0636\u0639\u0641 \u0641\u064a \u0627\u0644\u0631\u062c\u0644\u060c \u0641\u0642\u062f\u0627\u0646 \u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644/\u0627\u0644\u0628\u0631\u0627\u0632\u060c \u0623\u0644\u0645 \u0634\u062f\u064a\u062f \u0628\u0639\u062f \u0625\u0635\u0627\u0628\u0629\u060c \u062d\u0631\u0627\u0631\u0629 \u0645\u0639 \u0623\u0644\u0645 \u0634\u062f\u064a\u062f\u060c \u0623\u0648 \u062a\u062f\u0647\u0648\u0631 \u0633\u0631\u064a\u0639."
        )
    if followup and lower:
        return (
            "\u062a\u0645\u0627\u0645\u060c \u0643\u062f\u0647 \u0647\u0646\u062e\u0644\u064a \u0627\u0644\u062a\u0642\u064a\u064a\u0645 \u0645\u0631\u0643\u0632 \u0639\u0644\u0649 \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631. "
            "\u0623\u0644\u0645 \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631 \u0645\u0645\u0643\u0646 \u064a\u062d\u0635\u0644 \u0645\u0646 \u0634\u062f \u0639\u0636\u0644\u064a\u060c \u0642\u0639\u062f\u0629 \u0623\u0648 \u0648\u0636\u0639\u064a\u0629 \u063a\u0644\u0637\u060c \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629\u060c \u0623\u0648 \u062a\u0647\u064a\u062c \u0641\u064a \u0627\u0644\u0641\u0642\u0631\u0627\u062a. "
            "\u0627\u062a\u062c\u0646\u0628 \u0627\u0644\u062d\u0645\u0644 \u0627\u0644\u062a\u0642\u064a\u0644 \u0648\u062e\u0644\u064a \u0627\u0644\u062d\u0631\u0643\u0629 \u062e\u0641\u064a\u0641\u0629 \u0648\u0622\u0645\u0646\u0629\u060c \u0648\u0627\u0637\u0644\u0628 \u0631\u0639\u0627\u064a\u0629 \u0639\u0627\u062c\u0644\u0629 \u0644\u0648 \u0638\u0647\u0631 \u0636\u0639\u0641 \u0641\u064a \u0627\u0644\u0631\u062c\u0644 \u0623\u0648 \u0641\u0642\u062f\u0627\u0646 \u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644/\u0627\u0644\u0628\u0631\u0627\u0632 \u0623\u0648 \u0623\u0644\u0645 \u0634\u062f\u064a\u062f \u0628\u0639\u062f \u0625\u0635\u0627\u0628\u0629."
        )
    lower_label = "\u0623\u0644\u0645 \u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631" if lower else "\u0623\u0644\u0645 \u0627\u0644\u0638\u0647\u0631"
    return (
        lower_label
        + " \u063a\u0627\u0644\u0628\u0627 \u0628\u064a\u0643\u0648\u0646 \u0645\u0646 \u0634\u062f \u0639\u0636\u0644\u064a\u060c \u0648\u0636\u0639\u064a\u0629 \u063a\u0644\u0637\u060c \u062d\u0645\u0644 \u062d\u0627\u062c\u0629 \u062a\u0642\u064a\u0644\u0629\u060c \u0623\u0648 \u0645\u0634\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u0641\u0642\u0631\u0627\u062a \u062d\u0633\u0628 \u0645\u0643\u0627\u0646 \u0627\u0644\u0623\u0644\u0645 \u0648\u0637\u0628\u064a\u0639\u062a\u0647. "
        "\u0647\u0633\u0623\u0644\u0643 \u0643\u0627\u0645 \u0633\u0624\u0627\u0644 \u062a\u062d\u062a \u0639\u0634\u0627\u0646 \u0623\u0648\u0636\u062d \u0627\u0644\u0635\u0648\u0631\u0629."
    )


def _meaning_domain_clarification_response(
    state: ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse | None:
    meaning = state.medical_meaning
    domain = meaning.domain
    if not domain:
        return None

    if _is_english(state.language):
        if domain == "musculoskeletal_back_pain":
            answer = _back_pain_answer_for_state(state)
            questions = _back_follow_up_questions_for_state(state)
            doctor = "Orthopedic doctor"
        elif domain == "musculoskeletal_neck_pain":
            if set(state.known_symptoms) - {"neck_pain", "excessive_hunger"}:
                return None
            answer = (
                "Neck pain can come from muscle strain, posture, sleep position, or cervical spine irritation. "
                "I should keep this neck-focused and not guess a diagnosis from one symptom."
            )
            questions = [
                "When did the neck pain start?",
                "Does it spread to the shoulder or arm, or come with numbness or weakness?",
                "Was there injury, heavy lifting, fever, severe headache, or neck stiffness?",
            ]
            doctor = "Orthopedic doctor"
        elif domain == "throat_ent":
            if {"cough", "phlegm", "mild_fever", "high_fever"}.intersection(state.current_symptoms):
                return None
            answer = (
                "This sounds like throat pain, so I should keep the assessment focused on your throat and swallowing symptoms "
                "before guessing a diagnosis."
            )
            questions = [
                "Do you also have fever or cough?",
                "Does the pain get worse when you swallow?",
                "Do you have breathing trouble, choking sensation, voice change, or trouble opening your mouth?",
            ]
            doctor = _clarification_doctor(state.current_message, "Low")
        elif domain == "digestive_abdominal":
            if set(state.current_symptoms) - {"abdominal_pain", "stomach_pain"}:
                return None
            answer = (
                "This is abdominal pain, so I should keep the next questions focused on the stomach/abdomen and digestive symptoms before guessing a diagnosis."
            )
            questions = [
                "Where exactly is the abdominal pain: upper, lower, right, left, or around the belly button?",
                "Do you have vomiting or nausea?",
                "Do you have diarrhea or constipation?",
            ]
            doctor = "Needs more information"
        elif domain == "urinary":
            if set(state.current_symptoms) - {"burning_micturition"} or _has_any_normalized(
                state.current_message,
                {
                    "\u0642\u0644\u0629 \u0628\u0648\u0644",
                    "\u0642\u0644\u0647 \u0628\u0648\u0644",
                    "\u062f\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644",
                    "\u0627\u062d\u062a\u0628\u0627\u0633 \u0628\u0648\u0644",
                    "low urine",
                    "blood in urine",
                    "urinary retention",
                },
            ):
                return None
            answer = (
                "This sounds urinary, especially burning with urination, so I should keep the assessment focused on urine symptoms and warning signs."
            )
            questions = [
                "Do you feel burning while urinating?",
                "Are you urinating more often or feeling urgent need to urinate?",
                "Have you noticed blood in the urine?",
            ]
            doctor = "Urologist"
        else:
            return None
    else:
        if domain == "musculoskeletal_back_pain":
            answer = _back_pain_answer_for_state(state)
            questions = _back_follow_up_questions_for_state(state)
            doctor = "Orthopedic doctor"
        elif domain == "musculoskeletal_neck_pain":
            if set(state.known_symptoms) - {"neck_pain", "excessive_hunger"}:
                return None
            answer = (
                "\u0623\u0644\u0645 \u0627\u0644\u0631\u0642\u0628\u0629 \u0645\u0645\u0643\u0646 \u064a\u062d\u0635\u0644 \u0645\u0646 \u0634\u062f \u0639\u0636\u0644\u064a\u060c \u0648\u0636\u0639\u064a\u0629 \u0646\u0648\u0645 \u0623\u0648 \u062c\u0644\u0648\u0633 \u063a\u0644\u0637\u060c \u0623\u0648 \u062a\u0647\u064a\u062c \u0641\u064a \u0641\u0642\u0631\u0627\u062a \u0627\u0644\u0631\u0642\u0628\u0629. "
                "\u0647\u062e\u0644\u064a \u0627\u0644\u0631\u062f \u0645\u0631\u0643\u0632 \u0639\u0644\u0649 \u0627\u0644\u0631\u0642\u0628\u0629 \u0648\u0647\u0633\u0623\u0644 \u0623\u0633\u0626\u0644\u0629 \u0645\u062d\u062f\u062f\u0629 \u0642\u0628\u0644 \u0623\u064a \u0627\u0633\u062a\u0646\u062a\u0627\u062c."
            )
            questions = [
                "\u0623\u0644\u0645 \u0627\u0644\u0631\u0642\u0628\u0629 \u0628\u062f\u0623 \u0645\u0646 \u0625\u0645\u062a\u0649\u061f",
                "\u0647\u0644 \u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0643\u062a\u0641 \u0623\u0648 \u0627\u0644\u0630\u0631\u0627\u0639\u060c \u0623\u0648 \u0645\u0639\u0627\u0647 \u062a\u0646\u0645\u064a\u0644/\u0636\u0639\u0641\u061f",
                "\u0647\u0644 \u0641\u064a \u0625\u0635\u0627\u0628\u0629\u060c \u062d\u0631\u0627\u0631\u0629\u060c \u0635\u062f\u0627\u0639 \u0634\u062f\u064a\u062f\u060c \u0623\u0648 \u062a\u064a\u0628\u0633 \u0634\u062f\u064a\u062f \u0641\u064a \u0627\u0644\u0631\u0642\u0628\u0629\u061f",
            ]
            doctor = "Orthopedic doctor"
        elif domain == "throat_ent":
            if {"cough", "phlegm", "mild_fever", "high_fever"}.intersection(state.current_symptoms):
                return None
            answer = (
                "\u0623\u0644\u0645 \u0627\u0644\u0632\u0648\u0631 \u0623\u0648 \u0627\u0644\u062d\u0644\u0642 \u0645\u062d\u062a\u0627\u062c \u0623\u0639\u0631\u0641 \u0643\u0627\u0645 \u062d\u0627\u062c\u0629 \u0628\u0633\u064a\u0637\u0629 \u0642\u0628\u0644 \u0623\u064a \u0627\u0633\u062a\u0646\u062a\u0627\u062c. "
                "\u0647\u062e\u0644\u064a \u0627\u0644\u0631\u062f \u0639\u0646 \u0627\u0644\u062d\u0644\u0642 \u0648\u0627\u0644\u0628\u0644\u0639\u060c \u0645\u0646 \u063a\u064a\u0631 \u0645\u0627 \u0623\u0631\u0628\u0637\u0647 \u0628\u0645\u0643\u0627\u0646 \u062a\u0627\u0646\u064a."
            )
            questions = [
                "\u0647\u0644 \u0639\u0646\u062f\u0643 \u062d\u0631\u0627\u0631\u0629 \u0623\u0648 \u0643\u062d\u0629\u061f",
                "\u0647\u0644 \u0627\u0644\u0623\u0644\u0645 \u0628\u064a\u0632\u064a\u062f \u0645\u0639 \u0627\u0644\u0628\u0644\u0639\u061f",
                "\u0647\u0644 \u0641\u064a \u0635\u0639\u0648\u0628\u0629 \u062a\u0646\u0641\u0633\u060c \u0625\u062d\u0633\u0627\u0633 \u0628\u0627\u062e\u062a\u0646\u0627\u0642\u060c \u062a\u063a\u064a\u0631 \u0641\u064a \u0627\u0644\u0635\u0648\u062a\u060c \u0623\u0648 \u0635\u0639\u0648\u0628\u0629 \u0641\u062a\u062d \u0627\u0644\u0641\u0645\u061f",
            ]
            doctor = _clarification_doctor(state.current_message, "Low")
        elif domain == "digestive_abdominal":
            if set(state.current_symptoms) - {"abdominal_pain", "stomach_pain"}:
                return None
            answer = (
                "\u0623\u0644\u0645 \u0627\u0644\u0628\u0637\u0646 \u0645\u062d\u062a\u0627\u062c \u0623\u0633\u0626\u0644\u0629 \u0645\u0631\u0643\u0632\u0629 \u0639\u0646 \u0645\u0643\u0627\u0646 \u0627\u0644\u0623\u0644\u0645 \u0648\u0623\u0639\u0631\u0627\u0636 \u0627\u0644\u0645\u0639\u062f\u0629 \u0648\u0627\u0644\u0647\u0636\u0645 \u0642\u0628\u0644 \u0623\u064a \u0627\u0633\u062a\u0646\u062a\u0627\u062c. "
                "\u0647\u0633\u064a\u0628 \u0627\u0644\u0623\u0633\u0626\u0644\u0629 \u0645\u0646\u0638\u0645\u0629 \u062a\u062d\u062a."
            )
            questions = [
                "\u0627\u0644\u0623\u0644\u0645 \u0641\u064a\u0646 \u0628\u0627\u0644\u0636\u0628\u0637 \u0641\u064a \u0627\u0644\u0628\u0637\u0646: \u0641\u0648\u0642\u060c \u062a\u062d\u062a\u060c \u064a\u0645\u064a\u0646\u060c \u0634\u0645\u0627\u0644\u060c \u0648\u0644\u0627 \u062d\u0648\u0627\u0644\u064a\u0646 \u0627\u0644\u0633\u0631\u0629\u061f",
                "\u0647\u0644 \u0641\u064a \u062a\u0631\u062c\u064a\u0639 \u0623\u0648 \u063a\u062b\u064a\u0627\u0646\u061f",
                "\u0647\u0644 \u0641\u064a \u0625\u0633\u0647\u0627\u0644 \u0623\u0648 \u0625\u0645\u0633\u0627\u0643\u061f",
            ]
            doctor = "Needs more information"
        elif domain == "urinary":
            if set(state.current_symptoms) - {"burning_micturition"} or _has_any_normalized(
                state.current_message,
                {
                    "\u0642\u0644\u0629 \u0628\u0648\u0644",
                    "\u0642\u0644\u0647 \u0628\u0648\u0644",
                    "\u062f\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644",
                    "\u0627\u062d\u062a\u0628\u0627\u0633 \u0628\u0648\u0644",
                    "low urine",
                    "blood in urine",
                    "urinary retention",
                },
            ):
                return None
            answer = (
                "\u062d\u0631\u0642\u0627\u0646 \u0627\u0644\u0628\u0648\u0644 \u064a\u062e\u0644\u064a\u0646\u0627 \u0646\u0631\u0643\u0632 \u0639\u0644\u0649 \u0623\u0639\u0631\u0627\u0636 \u0627\u0644\u062a\u0628\u0648\u0644 \u0648\u0627\u0644\u0645\u0633\u0627\u0644\u0643 \u0627\u0644\u0628\u0648\u0644\u064a\u0629. "
                "\u0647\u0633\u0623\u0644\u0643 \u0643\u0627\u0645 \u0633\u0624\u0627\u0644 \u0645\u062d\u062f\u062f \u0645\u0646 \u063a\u064a\u0631 \u0645\u0627 \u0623\u063a\u064a\u0631 \u0627\u062a\u062c\u0627\u0647 \u0627\u0644\u062d\u0627\u0644\u0629."
            )
            questions = [
                "\u0647\u0644 \u0641\u064a \u062d\u0631\u0642\u0627\u0646 \u0623\u062b\u0646\u0627\u0621 \u0627\u0644\u062a\u0628\u0648\u0644\u061f",
                "\u0647\u0644 \u0628\u062a\u062f\u062e\u0644 \u0627\u0644\u062d\u0645\u0627\u0645 \u0643\u062a\u064a\u0631 \u0623\u0648 \u062d\u0627\u0633\u0633 \u0625\u0646\u0643 \u0645\u062d\u062a\u0627\u062c \u062a\u062a\u0628\u0648\u0644 \u0628\u0627\u0633\u062a\u0645\u0631\u0627\u0631\u061f",
                "\u0647\u0644 \u0644\u0627\u062d\u0638\u062a \u062f\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644\u061f",
            ]
            doctor = "Urologist"
        else:
            return None

    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=3)
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor=doctor,
        follow_up_questions=questions,
    )


def _reproductive_clarification_response(
    state: ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse:
    if _is_english(state.language):
        answer = (
            "This sounds related to menstrual or gynecological symptoms, so I should ask focused questions rather than guess a diagnosis."
        )
        questions = [
            "One important question so I can guide you safely: how old are you roughly, and is pregnancy possible?",
            "Is the pain or bleeding around the expected period time or outside it, and how severe is it?",
            "Any heavy bleeding, fever, foul-smelling discharge, severe lower abdominal pain, dizziness, or fainting?",
        ]
    else:
        answer = (
            "ده يبدو متعلقًا بالدورة أو الصحة النسائية، فالأفضل أسأل أسئلة محددة بدل ما أطلع تشخيص عشوائي."
        )
        questions = [
            "سؤال مهم عشان أوجهك صح: السن كام تقريبًا، وهل في احتمال حمل؟",
            "الألم أو النزيف مرتبط بميعاد الدورة ولا خارج ميعادها؟ وشدته قد إيه؟",
            "هل في نزيف غزير، حرارة، إفرازات برائحة غير طبيعية، ألم شديد أسفل البطن، دوخة شديدة، أو إغماء؟",
        ]
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=3)
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor="Gynecologist",
        follow_up_questions=questions,
    )


def _english_domain_clarification_response(
    state: ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse | None:
    if not _is_english(state.language) or state.current_symptoms:
        return None

    domains = state.medical_domains
    if "throat_ent" in domains:
        answer = "Let’s narrow down the throat symptoms before guessing a diagnosis."
        questions = [
            "How long has your throat been hurting?",
            "Do you have fever, cough, runny nose, or swollen tonsils?",
            "Is swallowing difficult, or do you have breathing trouble or a changed voice?",
        ]
        doctor = "ENT specialist"
    elif "digestive" in domains:
        answer = "Let’s narrow down the stomach or abdominal symptoms safely."
        questions = [
            "Where is the pain exactly: upper abdomen, lower abdomen, right, left, or around the belly button?",
            "Do you have vomiting, nausea, diarrhea, constipation, fever, or blood in stool/vomit?",
            "When did it start, and is it getting worse?",
        ]
        doctor = "Gastroenterologist"
    elif "urinary" in domains:
        answer = "This sounds urinary or kidney-related, so a few focused details matter."
        questions = [
            "Do you have burning while urinating or needing to urinate often?",
            "Any blood in urine, flank/back pain, fever, or difficulty passing urine?",
            "When did this start?",
        ]
        doctor = "Urologist"
    elif "skin" in domains:
        answer = "Let’s clarify the skin symptoms before naming a cause."
        questions = [
            "Where is the rash, itching, or skin change?",
            "Is it spreading quickly, painful, or associated with swelling of the lips/face or breathing trouble?",
            "Did it start after new food, medicine, skincare product, or exposure?",
        ]
        doctor = "Dermatologist"
    elif "dental" in domains:
        answer = "Dental symptoms need a few focused details."
        questions = [
            "Is the pain in a tooth, gum, or jaw, and how long has it been there?",
            "Is there facial/gum swelling, fever, pus, bleeding, or trouble swallowing?",
        ]
        doctor = "Dentist"
    elif "eye" in domains:
        answer = "Eye symptoms can range from mild irritation to urgent problems, so let’s clarify."
        questions = [
            "Is there severe eye pain or sudden vision change?",
            "Is it one eye or both, and is there redness, discharge, trauma, or light sensitivity?",
        ]
        doctor = "Ophthalmologist"
    elif "infection" in domains:
        answer = "Fever or infection-like symptoms need a little context before any diagnosis."
        questions = [
            "What temperature did you measure, and how long has it been going on?",
            "Do you also have cough, sore throat, rash, vomiting, diarrhea, stiff neck, or breathing trouble?",
        ]
        doctor = "General Practitioner"
    else:
        return None

    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=3)
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor=doctor,
        follow_up_questions=questions,
    )


def _unsupported_prediction_response(
    *,
    state: ConversationState,
    diagnosis: str | None,
    assistant_history_texts: list[str],
) -> ChatResponse:
    if _is_english(state.language):
        answer = (
            "The symptoms are too nonspecific for a confident diagnosis. I’m going to keep this broad instead of forcing a rare or unsupported condition."
        )
        questions = [
            "What is the main symptom now?",
            "When did it start, and are there any red flags like chest pain, breathing trouble, fainting, severe headache, or high fever?",
        ]
    else:
        answer = (
            "الأعراض لسه عامة ومش كفاية لتشخيص محدد بثقة. الأفضل نخليها كأعراض تحتاج توضيح بدل ما نطلع مرض نادر أو غير مدعوم."
        )
        questions = [
            "أكتر عرض مضايقك دلوقتي إيه؟",
            "بدأ من إمتى؟ وهل في ألم صدر، ضيق تنفس، إغماء، صداع شديد، أو حرارة عالية؟",
        ]
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=2)
    suggested_doctor = _clarification_doctor(state.current_message, "Low")
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor=suggested_doctor,
        follow_up_questions=questions,
    )


def _frustration_response(
    *,
    summary: ConversationCaseSummary,
    assistant_history_texts: list[str],
) -> ChatResponse:
    suggested_doctor = "General Practitioner" if summary.has_high_temperature or summary.known_symptoms else "Not needed"
    display_doctor = display_doctor_ar(suggested_doctor)
    follow_up_questions: list[str] = []
    if summary.has_high_temperature and not summary.duration_known:
        follow_up_questions.append("الحرارة بقالها قد إيه؟")
    follow_up_questions = _filter_questions_without_fallback(
        follow_up_questions,
        assistant_history_texts,
        limit=1,
    )

    if summary.has_high_temperature:
        answer = (
            "حقك تستغرب لو الرد مش واضح. خلينا نراجع الحالة بهدوء: عندك سخونية وصلت "
            f"{summary.temperature_c:g} وتكسير في الجسم. حرارة قريبة من 40 تعتبر عالية وتحتاج تقييم طبي قريب/اليوم، "
            "خصوصًا لو مستمرة أو مش بتنزل أو معها ضيق تنفس، تيبس رقبة، طفح، قيء متكرر، ألم صدر، تشوش، أو جفاف.\n\n"
            f"الطبيب المناسب: {display_doctor} أو رعاية عاجلة حسب شدة الحالة."
        )
    elif summary.known_symptoms:
        answer = (
            "حقك تستغرب لو الرد مش واضح. خلينا نرتب الأعراض من جديد بدل ما نثبت تشخيص غير مؤكد. "
            "اكتبلي أهم عرض عندك حاليًا، بدأ من إمتى، وهل فيه حرارة أو ضيق تنفس أو ألم شديد؟"
        )
    else:
        answer = (
            "حقك تستغرب لو الرد مش واضح. أنا هنا للمساعدة الطبية فقط، ومش هطلع تشخيص من رسالة مش فيها أعراض. "
            "اكتبلي العرض اللي مضايقك وبدأ من إمتى."
        )

    return ChatResponse(
        mode=CLARIFICATION_MODE,
        answer=answer,
        extracted_symptoms=summary.known_symptoms if summary.has_high_temperature else [],
        possible_diagnosis=None,
        display_diagnosis_ar=None,
        confidence=0.0,
        urgency_level="Medium" if summary.has_high_temperature else "Low",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor,
        precautions=[],
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _high_fever_guidance_response(
    *,
    summary: ConversationCaseSummary | ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse:
    diagnosis = BROAD_VIRAL_DIAGNOSIS
    suggested_doctor = "General Practitioner"
    display_diagnosis = display_diagnosis_ar(diagnosis)
    display_doctor = display_doctor_ar(suggested_doctor)
    language = getattr(summary, "language", "ar")
    questions: list[str] = []
    if not summary.duration_known:
        questions.append("How long has the fever been going on?" if language == "en" else "الحرارة بقالها قد إيه؟")
    if not summary.asked_red_flags:
        questions.append(
            "Any shortness of breath, stiff neck, rash, repeated vomiting, chest pain, confusion, or dehydration?"
            if language == "en"
            else HIGH_FEVER_RED_FLAG_QUESTION
        )
    follow_up_questions = _filter_questions_without_fallback(
        questions,
        assistant_history_texts,
        limit=MAX_DIAGNOSIS_QUESTIONS,
    )
    follow_up_questions = [
        question for question in follow_up_questions if not _question_mentions_temperature(question)
    ][:MAX_DIAGNOSIS_QUESTIONS]

    temperature_text = f"{summary.temperature_c:g}" if summary.temperature_c is not None else "40"
    if language == "en":
        answer = (
            f"A temperature around {temperature_text} is important. Fever close to 40°C is high and needs same-day medical assessment, "
            "especially if it persists, does not come down, or comes with breathing trouble, stiff neck, rash, repeated vomiting, chest pain, confusion, or dehydration.\n\n"
            "With fever and body aches, a broad viral or flu-like illness can be one possibility, but the high temperature matters more than forcing a diagnosis in chat.\n\n"
            "Suggested care: general practitioner / internal medicine, or urgent care if symptoms are severe.\n\n"
            "For now: rest, fluids in small frequent amounts, and monitor temperature. Seek medical help quickly if any warning sign appears.\n\n"
            "Note: this is initial guidance, not a final diagnosis."
        )
    else:
        answer = (
            f"تمام، حرارة {temperature_text} معلومة مهمة. حرارة قريبة من 40 تعتبر عالية وتحتاج تقييم طبي قريب/اليوم، "
            "خصوصًا لو مستمرة أو مش بتنزل أو معها تدهور، تيبس رقبة، طفح، قيء متكرر، ضيق تنفس، ألم صدر، تشوش، أو جفاف.\n\n"
            f"مع السخونية وتكسير الجسم، الاحتمال المبدئي الأقرب يكون {display_diagnosis}، لكن الرقم العالي يخلي المتابعة الطبية أهم من تثبيت تشخيص من الشات.\n\n"
            f"الطبيب المناسب: {display_doctor} أو رعاية عاجلة لو الأعراض شديدة.\n\n"
            "حاليًا اهتم بالراحة وشرب سوائل على فترات صغيرة ومتابعة الحرارة. اطلب مساعدة طبية بسرعة لو ظهرت أي علامة من علامات الخطر اللي فوق.\n\n"
            "تنبيه: ده توجيه مبدئي وليس تشخيصًا نهائيًا."
        )
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=summary.known_symptoms,
        possible_diagnosis=diagnosis,
        display_diagnosis_ar=display_diagnosis,
        confidence=0.42,
        urgency_level="Medium",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor,
        precautions=SAFE_BROAD_PRECAUTIONS,
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _broad_fever_response(
    *,
    extracted_symptoms: list[str],
    urgency_level: str,
    assistant_history_texts: list[str],
    language: str = "ar",
) -> ChatResponse:
    diagnosis = BROAD_VIRAL_DIAGNOSIS
    suggested_doctor = "General Practitioner"
    if language == "en":
        follow_up_questions = _limited_followups(
            [
                "How high is the temperature?",
                "When did the symptoms start, and is there cough, sore throat, exposure to infection, or severe chills?",
            ],
            assistant_history_texts,
        )
        display_diagnosis = "viral or flu-like illness"
        display_doctor = "General practitioner / internal medicine"
        urgency_display = "medium" if urgency_level == "Medium" else "low"
        answer = (
            "Fever with body aches often fits a broad viral or flu-like illness, especially early on. "
            "That does not confirm a specific or serious infection without stronger supporting signs.\n\n"
            f"Initial direction: {display_diagnosis}. Urgency: {urgency_display}. Suggested care: {display_doctor}.\n\n"
            "For now: rest, drink enough fluids, and monitor temperature. Seek medical care quickly if fever is very high or persistent, "
            "or if breathing trouble, chest pain, fainting, stiff neck, severe rash, or fast worsening appears.\n\n"
            "Note: this is initial guidance, not a final diagnosis."
        )
        return ChatResponse(
            mode=DIAGNOSIS_MODE,
            answer=answer,
            extracted_symptoms=extracted_symptoms,
            possible_diagnosis=diagnosis,
            display_diagnosis_ar=display_diagnosis_ar(diagnosis),
            confidence=0.45,
            urgency_level=urgency_level if urgency_level in {"Low", "Medium"} else "Medium",
            suggested_doctor=suggested_doctor,
            display_doctor_ar=display_doctor_ar(suggested_doctor),
            precautions=SAFE_BROAD_PRECAUTIONS,
            needs_follow_up=bool(follow_up_questions),
            follow_up_questions=follow_up_questions,
            retrieved_cases=[],
        )
    follow_up_questions = _limited_followups(
        [
            "الحرارة وصلت كام تقريبًا؟",
            "الأعراض بدأت من إمتى؟ وهل في كحة، التهاب حلق، مخالطة لحد عنده عدوى، أو رعشة شديدة؟",
        ],
        assistant_history_texts,
    )
    display_diagnosis = display_diagnosis_ar(diagnosis)
    display_doctor = display_doctor_ar(suggested_doctor)
    urgency_display = "متوسط" if urgency_level == "Medium" else "منخفض"
    answer = (
        f"واضح إن عندك سخونية مع تكسير في الجسم، وده غالبًا بيظهر مع عدوى فيروسية أو بداية دور برد/إنفلونزا. "
        f"الاحتمال الأقرب مبدئيًا هو: {display_diagnosis}.\n\n"
        "الأعراض دي لوحدها ما تكفيش لتثبيت مرض محدد أو عدوى خطيرة بثقة، خصوصًا من غير علامات داعمة زي "
        "كحة شديدة، ضيق تنفس، طفح، تعرض واضح لعدوى، أو تدهور سريع.\n\n"
        f"مستوى الخطورة: {urgency_display}. "
        f"الطبيب المناسب: {display_doctor}.\n\n"
        "حاليًا الأفضل الراحة، شرب سوائل كفاية، ومتابعة درجة الحرارة. اطلب رعاية طبية بسرعة لو الحرارة عالية جدًا "
        "أو مستمرة، أو ظهر ضيق تنفس، ألم صدر، إغماء، تيبس رقبة، طفح شديد، أو تدهور سريع.\n\n"
        "تنبيه: ده توجيه مبدئي وليس تشخيصًا نهائيًا."
    )
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=extracted_symptoms,
        possible_diagnosis=diagnosis,
        display_diagnosis_ar=display_diagnosis,
        confidence=0.45,
        urgency_level=urgency_level if urgency_level in {"Low", "Medium"} else "Medium",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor,
        precautions=SAFE_BROAD_PRECAUTIONS,
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _cautious_malaria_response(
    *,
    extracted_symptoms: list[str],
    assistant_history_texts: list[str],
) -> ChatResponse:
    diagnosis = "Malaria"
    suggested_doctor = "Infectious disease specialist"
    follow_up_questions = _limited_followups(
        [
            "السفر كان فين وإمتى أو التعرض للناموس كان واضح قد إيه؟",
            "هل الحرارة بتيجي في نوبات مع رعشة وتعرق شديد وبتتحسن ثم ترجع؟",
        ],
        assistant_history_texts,
    )
    display_diagnosis = display_diagnosis_ar(diagnosis)
    display_doctor = display_doctor_ar(suggested_doctor)
    answer = (
        f"وجود حرارة مع رعشة أو تعرق ومعاهم سفر قريب أو تعرض واضح للناموس يخلي {display_diagnosis} احتمال محتاج تقييم، "
        "لكن لا يتأكد من الأعراض فقط.\n\n"
        "الملاريا عدوى بسبب طفيل بينتقل غالبًا عن طريق لدغة نوع معين من البعوض، وليست بكتيريا. "
        "عشان كده التقييم الطبي مهم لو الأعراض متكررة أو قوية.\n\n"
        f"مستوى الخطورة: متوسط. الطبيب المناسب: {display_doctor}.\n\n"
        "حاليًا اهتم بالسوائل والراحة ومتابعة الحرارة. اطلب رعاية عاجلة لو حصل تدهور سريع، إغماء، لخبطة، ضيق تنفس، "
        "قيء مستمر، أو حرارة شديدة لا تتحسن.\n\n"
        "تنبيه: ده احتمال مبدئي وليس تشخيصًا نهائيًا."
    )
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=extracted_symptoms,
        possible_diagnosis=diagnosis,
        display_diagnosis_ar=display_diagnosis,
        confidence=0.58,
        urgency_level="Medium",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor,
        precautions=SAFE_BROAD_PRECAUTIONS,
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _malaria_explanation_response(assistant_history_texts: list[str]) -> ChatResponse:
    suggested_doctor = "General Practitioner"
    follow_up_questions = _limited_followups(
        [
            "هل كان في سفر قريب أو تعرض واضح لناموس كتير؟",
            "هل الحرارة بتيجي في نوبات مع رعشة وتعرق شديد؟",
        ],
        assistant_history_texts,
    )
    answer = (
        "الملاريا عدوى سببها طفيل، وبيتنتقل غالبًا عن طريق لدغة نوع معين من البعوض. "
        "ممكن تعمل حرارة ورعشة وتكسير وتعرق، لكنها لا تتأكد من سخونية وتكسير فقط.\n\n"
        "لو مفيش سفر قريب، أو تعرض واضح لناموس في منطقة خطورة، أو نوبات حرارة متكررة مع رعشة وتعرق، "
        "فالاحتمال بيبقى أضعف، وساعتها بنفكر أكثر في عدوى فيروسية أو دور برد/إنفلونزا حسب باقي الأعراض.\n\n"
        "جاوبني على سؤالين بس عشان أوضح الاتجاه."
    )
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=[],
        possible_diagnosis=None,
        display_diagnosis_ar=None,
        confidence=0.0,
        urgency_level="Low",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor_ar(suggested_doctor),
        precautions=[],
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _malaria_challenge_response(
    *,
    extracted_symptoms: list[str],
    assistant_history_texts: list[str],
) -> ChatResponse:
    diagnosis = BROAD_VIRAL_DIAGNOSIS
    suggested_doctor = "General Practitioner"
    follow_up_questions = _limited_followups(
        [
            "الحرارة وصلت كام تقريبًا؟",
            "هل في كحة، التهاب حلق، رشح، أو مخالطة لشخص عنده عدوى؟",
        ],
        assistant_history_texts,
    )
    display_diagnosis = display_diagnosis_ar(diagnosis)
    display_doctor = display_doctor_ar(suggested_doctor)
    answer = (
        "معاك حق تسأل. من غير سفر قريب أو تعرض واضح لناموس، احتمال الملاريا بيبقى أضعف، "
        "وسخونية وتكسير الجسم لوحدهم ما يكفوش لتأكيدها.\n\n"
        f"الاتجاه الأقرب مبدئيًا يكون {display_diagnosis} أو عدوى عامة بسيطة، حسب مدة الحرارة وباقي الأعراض. "
        "خلينا نركز على التفاصيل المهمة بدل ما نثبت تشخيص غير مدعوم.\n\n"
        f"مستوى الخطورة غالبًا منخفض إلى متوسط حسب درجة الحرارة. الطبيب المناسب: {display_doctor}.\n\n"
        "اطلب رعاية بسرعة لو الحرارة عالية جدًا أو مستمرة، أو ظهر ضيق تنفس، ألم صدر، إغماء، تيبس رقبة، طفح شديد، أو تدهور سريع."
    )
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=extracted_symptoms,
        possible_diagnosis=diagnosis,
        display_diagnosis_ar=display_diagnosis,
        confidence=0.42,
        urgency_level="Low",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor,
        precautions=SAFE_BROAD_PRECAUTIONS,
        needs_follow_up=bool(follow_up_questions),
        follow_up_questions=follow_up_questions,
        retrieved_cases=[],
    )


def _previous_diagnosis_explanation_response(
    *,
    diagnosis: str | None,
    state: ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse:
    diagnosis = diagnosis or "General medical evaluation"
    suggested_doctor = "General Practitioner"
    display_name = display_diagnosis_ar(diagnosis) or diagnosis
    if diagnosis == "Malaria":
        if _is_english(state.language):
            answer = (
                "Malaria is an infection caused by a parasite, usually spread by certain mosquito bites. "
                "It can cause fever, chills, sweating, and body aches, but it is not confirmed from body aches or vague symptoms alone.\n\n"
                "If there was no travel to a malaria-risk area, no clear mosquito exposure, and no repeated fever/chills pattern, malaria becomes less likely."
            )
            questions = [
                "Was there recent travel to a malaria-risk area or heavy mosquito exposure?",
                "Do symptoms come in repeated fever/chills/sweating attacks?",
            ]
        else:
            return _malaria_explanation_response(assistant_history_texts)
    elif diagnosis == "Hypertension":
        if _is_english(state.language):
            answer = (
                "Hypertension means high blood pressure. Headache or dizziness alone does not prove it; it needs an actual blood pressure reading or known history."
            )
            questions = ["Have you measured your blood pressure recently?"]
        else:
            answer = (
                "ارتفاع ضغط الدم يعني إن قياس الضغط أعلى من الطبيعي. الصداع أو الدوخة لوحدهم ما يثبتوش الضغط؛ لازم قياس ضغط فعلي أو تاريخ مرضي معروف."
            )
            questions = ["هل قست الضغط مؤخرًا؟ ولو قسته كان كام؟"]
    else:
        if _is_english(state.language):
            answer = (
                f"{display_name} was only a possible direction, not a confirmed diagnosis. "
                "A final diagnosis needs enough symptom details and usually medical examination."
            )
            questions = ["Which symptom is most important right now?", "When did it start?"]
        else:
            answer = (
                f"{display_name} كان احتمالًا مبدئيًا فقط، وليس تشخيصًا مؤكدًا. "
                "تأكيد التشخيص يحتاج تفاصيل كفاية وغالبًا كشف طبي."
            )
            questions = ["أهم عرض عندك دلوقتي إيه؟", "بدأ من إمتى؟"]
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=2)
    return ChatResponse(
        mode=DIAGNOSIS_MODE,
        answer=answer,
        extracted_symptoms=state.known_symptoms,
        possible_diagnosis=None,
        display_diagnosis_ar=None,
        confidence=0.0,
        urgency_level="Low",
        suggested_doctor=suggested_doctor,
        display_doctor_ar=display_doctor_ar(suggested_doctor),
        precautions=[],
        needs_follow_up=bool(questions),
        follow_up_questions=questions,
        retrieved_cases=[],
    )


def _diagnosis_challenge_response(
    *,
    diagnosis: str | None,
    state: ConversationState,
    assistant_history_texts: list[str],
) -> ChatResponse:
    diagnosis = diagnosis or state.previous_diagnosis
    if diagnosis == "Malaria":
        if "fever" in state.denied_concepts:
            return _correction_response(state, assistant_history_texts)
        return _malaria_challenge_response(
            extracted_symptoms=state.known_symptoms,
            assistant_history_texts=assistant_history_texts,
        )
    if _is_english(state.language):
        answer = (
            "You’re right to challenge that. I should not defend a label if the evidence is weak. "
            "Let’s re-check the actual symptoms and keep the assessment broad until there is enough support."
        )
        questions = ["Which symptoms are definitely present now?", "Which symptoms or exposures should I remove from the picture?"]
    else:
        answer = (
            "معاك حق تسأل. مش المفروض أدافع عن تشخيص لو الدليل ضعيف. "
            "خلينا نراجع الأعراض المؤكدة ونشيل أي افتراض غلط من الصورة."
        )
        questions = ["إيه الأعراض الموجودة فعلًا دلوقتي؟", "وفيه أي عرض أو تعرض عايز تستبعده؟"]
    questions = _filter_questions_without_fallback(questions, assistant_history_texts, limit=2)
    return _non_diagnostic_response(
        answer=answer,
        state=state,
        suggested_doctor="Needs more information",
        follow_up_questions=questions,
    )


def _conversation_intent_response(
    *,
    message: str,
    state: ConversationState,
    assistant_history_texts: list[str],
    recent_history: list,
) -> ChatResponse | None:
    current_or_previous_diagnosis = state.previous_diagnosis or _diagnosis_mentioned_in_text(
        message,
        *assistant_history_texts[-4:],
    )

    if state.intent in {
        ConversationIntent.GREETING_OR_CASUAL,
        ConversationIntent.NON_MEDICAL_CHAT,
        ConversationIntent.FAMILY_OR_OFFTOPIC,
        ConversationIntent.NONSENSE_OR_LOW_INFORMATION,
    }:
        response = _casual_or_nonmedical_response(state)
        return _naturalize_chat_response(
            response,
            state=state,
            route=_route_for_conversation_intent(state),
            message=message,
            history=recent_history,
        )

    if state.intent in {
        ConversationIntent.PROFANITY_OR_ABUSE,
        ConversationIntent.INSULT_OR_FRUSTRATION,
    }:
        response = _frustration_state_response(
            state=state,
            assistant_history_texts=assistant_history_texts,
        )
        return _naturalize_chat_response(
            response,
            state=state,
            route="abuse",
            message=message,
            history=recent_history,
        )

    if state.intent == ConversationIntent.CORRECTION_OR_NEGATION:
        response = _correction_response(state, assistant_history_texts)
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    if state.intent == ConversationIntent.CHALLENGE_PREVIOUS_DIAGNOSIS:
        response = _diagnosis_challenge_response(
            diagnosis=current_or_previous_diagnosis,
            state=state,
            assistant_history_texts=assistant_history_texts,
        )
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    if state.intent == ConversationIntent.ASK_ABOUT_PREVIOUS_DIAGNOSIS:
        response = _previous_diagnosis_explanation_response(
            diagnosis=current_or_previous_diagnosis,
            state=state,
            assistant_history_texts=assistant_history_texts,
        )
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    if state.has_high_temperature and (
        state.intent == ConversationIntent.ANSWER_FOLLOWUP_QUESTION
        or _has_any_normalized(message, FEVER_CONTEXT_TERMS)
    ):
        response = _high_fever_guidance_response(
            summary=state,
            assistant_history_texts=assistant_history_texts,
        )
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
            diagnosis_allowed=True,
        )

    if state.intent == ConversationIntent.REPRODUCTIVE_OR_GENDER_SENSITIVE_COMPLAINT:
        response = _reproductive_clarification_response(state, assistant_history_texts)
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    meaning_domain_response = _meaning_domain_clarification_response(state, assistant_history_texts)
    if meaning_domain_response:
        return _naturalize_chat_response(
            meaning_domain_response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    english_domain_response = _english_domain_clarification_response(state, assistant_history_texts)
    if english_domain_response:
        return _naturalize_chat_response(
            english_domain_response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    if state.intent == ConversationIntent.VAGUE_UNCLEAR and is_weak_body_ache_only(
        state.current_symptoms,
        message,
    ):
        response = _body_ache_clarification_response(state, assistant_history_texts)
        return _naturalize_chat_response(
            response,
            state=state,
            route="medical_clarification",
            message=message,
            history=recent_history,
        )

    return None


def _emergency_chat_response(
    *,
    message: str,
    history: list,
    extracted_symptoms: list[str],
    emergency_signal: EmergencySignal,
    language: str = "ar",
) -> ChatResponse:
    answer = llm_service.generate_answer(
        message=message,
        history=history,
        extracted_symptoms=extracted_symptoms,
        diagnosis=emergency_signal.diagnosis,
        confidence=0.99,
        urgency_level="High",
        suggested_doctor=emergency_signal.doctor,
        precautions=[],
        diagnosis_description=emergency_signal.reason,
        follow_up_questions=[],
        retrieved_cases=[],
    )
    return _final_response_guardrail(
        ChatResponse(
            mode=EMERGENCY_MODE,
            answer=answer,
            extracted_symptoms=extracted_symptoms,
            possible_diagnosis=emergency_signal.diagnosis,
            display_diagnosis_ar=emergency_signal.display_diagnosis_ar,
            confidence=0.99,
            urgency_level="High",
            suggested_doctor=emergency_signal.doctor,
            display_doctor_ar=emergency_signal.display_doctor_ar,
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        ),
        emergency_signal,
        language=language,
    )


def _response_language(request: ChatRequest, state: ConversationState | None, fallback: str = "ar") -> str:
    if request.language and request.language.strip():
        hint = request.language.strip().lower()
        if hint.startswith("en"):
            return "en"
        if hint.startswith("ar"):
            return "ar"
    return state.language if state else fallback


def _case_state_update(
    *,
    response: ChatResponse,
    request: ChatRequest,
    state: ConversationState | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "mode": response.mode,
        "urgency_level": response.urgency_level,
        "possible_diagnosis": response.possible_diagnosis,
        "display_diagnosis_ar": response.display_diagnosis_ar,
        "suggested_doctor": response.suggested_doctor,
        "display_doctor_ar": response.display_doctor_ar,
        "confidence": response.confidence,
        "needs_follow_up": response.needs_follow_up,
        "follow_up_questions": response.follow_up_questions,
        "known_symptoms": response.extracted_symptoms,
        "language": _response_language(request, state),
        "source": request.source,
        "safety_flags": ["emergency"] if response.mode == EMERGENCY_MODE or response.urgency_level == "High" else [],
    }
    if state:
        update.update(
            {
                "intent": state.intent,
                "medical_domains": sorted(state.medical_domains),
                "medical_meaning": {
                    "language": state.medical_meaning.language,
                    "domain": state.medical_meaning.domain,
                    "body_parts": state.medical_meaning.body_parts,
                    "symptoms": state.medical_meaning.symptoms,
                    "red_flags": state.medical_meaning.red_flags,
                    "denied": state.medical_meaning.denied,
                },
                "active_case": state.active_case,
                "denied_concepts": sorted(state.denied_concepts),
                "denied_symptoms": sorted(state.denied_symptoms),
                "temperature_c": state.temperature_c,
                "has_high_temperature": state.has_high_temperature,
                "duration_known": state.duration_known,
                "previous_diagnosis": state.previous_diagnosis,
                "asked_temperature": state.asked_temperature,
                "asked_duration": state.asked_duration,
                "asked_red_flags": state.asked_red_flags,
            }
        )
    return update


def _attach_integration_metadata(
    response: ChatResponse,
    request: ChatRequest,
    state: ConversationState | None = None,
) -> ChatResponse:
    response.conversation_id = request.conversation_id or request.session_id
    response.case_state_update = _case_state_update(response=response, request=request, state=state)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = {}
    artifacts.update(rag_service.artifact_status())
    artifacts.update(classifier_service.artifact_status())
    artifacts.update(knowledge_service.artifact_status())
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        app_version=settings.app_version,
        build_id=settings.build_id,
        frontend_build_id=settings.frontend_build_id,
        llm_configured=llm_service.configured,
        llm_key_count=llm_service.key_count,
        llm_model=settings.groq_model,
        llm_fallback_model=settings.groq_fallback_model,
        artifacts=artifacts,
    )


def _legacy_chat_impl(request: ChatRequest) -> ChatResponse:
    recent_history = request.history[-10:]
    user_history_text = " ".join(
        item.content for item in recent_history if item.role == "user" and item.content
    )
    assistant_history_texts = [
        item.content for item in recent_history if item.role == "assistant" and item.content
    ]
    combined_user_message = f"{user_history_text} {request.message}".strip()

    def finalize(
        response: ChatResponse,
        state: ConversationState | None = None,
        emergency_signal: EmergencySignal | None = None,
        language: str | None = None,
    ) -> ChatResponse:
        guarded = _final_response_guardrail(
            response,
            emergency_signal=emergency_signal,
            language=language or _response_language(request, state),
        )
        guarded = _final_meaning_consistency_guardrail(guarded, state, assistant_history_texts)
        guarded = _final_response_guardrail(
            guarded,
            emergency_signal=emergency_signal,
            language=language or _response_language(request, state),
        )
        return _attach_integration_metadata(guarded, request, state)

    if is_closing_message(request.message):
        suggested_doctor = "Not needed"
        closing_response = ChatResponse(
            mode=CLOSING_MODE,
            answer=closing_answer(),
            extracted_symptoms=[],
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level="Low",
            suggested_doctor=suggested_doctor,
            display_doctor_ar=display_doctor_ar(suggested_doctor),
            precautions=[],
            needs_follow_up=False,
            follow_up_questions=[],
            retrieved_cases=[],
        )
        closing_response = _naturalize_chat_response(
            closing_response,
            state=None,
            route="closing",
            message=request.message,
            history=recent_history,
            language=_response_language(request, None),
        )
        return finalize(closing_response)

    current_symptoms = classifier_service.extract_symptoms(request.message, history_text="")
    history_symptoms = (
        classifier_service.extract_symptoms("", history_text=user_history_text)
        if user_history_text
        else []
    )
    conversation_state = build_conversation_state(
        current_message=request.message,
        recent_history=recent_history,
        current_symptoms=current_symptoms,
        history_symptoms=history_symptoms,
        diagnosis_aliases=DIAGNOSIS_ALIASES,
    )
    _apply_llm_router_for_uncertain_state(
        state=conversation_state,
        message=request.message,
        recent_history=recent_history,
    )

    current_emergency_signal = detect_emergency_signal(request.message, current_symptoms)
    if current_emergency_signal:
        return _attach_integration_metadata(
            _emergency_chat_response(
                message=request.message,
                history=request.history,
                extracted_symptoms=current_symptoms,
                emergency_signal=current_emergency_signal,
                language=conversation_state.language,
            ),
            request,
            conversation_state,
        )

    conversation_response = _conversation_intent_response(
        message=request.message,
        state=conversation_state,
        assistant_history_texts=assistant_history_texts,
        recent_history=recent_history,
    )
    if conversation_response:
        return finalize(conversation_response, conversation_state)

    extracted_symptoms = conversation_state.known_symptoms
    emergency_signal = detect_emergency_signal(combined_user_message, extracted_symptoms)
    if emergency_signal:
        return _attach_integration_metadata(
            _emergency_chat_response(
                message=combined_user_message,
                history=request.history,
                extracted_symptoms=extracted_symptoms,
                emergency_signal=emergency_signal,
                language=conversation_state.language,
            ),
            request,
            conversation_state,
        )

    severity = knowledge_service.score_symptoms(extracted_symptoms, combined_user_message)
    urgency_level = str(severity["urgency"])

    if urgency_level != "High" and _is_malaria_supported_fever(extracted_symptoms, combined_user_message):
        return finalize(
            _cautious_malaria_response(
                extracted_symptoms=extracted_symptoms,
                assistant_history_texts=assistant_history_texts,
            ),
            conversation_state,
        )

    if urgency_level != "High" and _is_generic_fever_body_aches(extracted_symptoms, combined_user_message):
        broad_urgency = "Medium" if "high_fever" in set(extracted_symptoms) else urgency_level
        return finalize(
            _broad_fever_response(
                extracted_symptoms=extracted_symptoms,
                urgency_level=broad_urgency,
                assistant_history_texts=assistant_history_texts,
                language=conversation_state.language,
            ),
            conversation_state,
        )

    if urgency_level != "High" and _has_throat_cough_fever_cluster(extracted_symptoms):
        broad_urgency = "Medium" if "high_fever" in set(extracted_symptoms) else urgency_level
        return finalize(
            _broad_fever_response(
                extracted_symptoms=extracted_symptoms,
                urgency_level=broad_urgency,
                assistant_history_texts=assistant_history_texts,
                language=conversation_state.language,
            ),
            conversation_state,
        )

    if urgency_level != "High" and is_weak_body_ache_only(
        extracted_symptoms,
        combined_user_message,
    ):
        return finalize(
            _body_ache_clarification_response(conversation_state, assistant_history_texts),
            conversation_state,
        )

    obvious_clarification = urgency_level != "High" and (
        not extracted_symptoms or is_vague_or_body_area_only(combined_user_message, extracted_symptoms)
    )
    if obvious_clarification:
        follow_up_questions = filter_repeated_questions(
            smart_follow_up_questions(combined_user_message, extracted_symptoms),
            assistant_history_texts,
        )
        suggested_doctor = _clarification_doctor(combined_user_message, urgency_level)
        clarification_response = ChatResponse(
            mode=CLARIFICATION_MODE,
            answer=clarification_answer(follow_up_questions),
            extracted_symptoms=extracted_symptoms,
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level=urgency_level,
            suggested_doctor=suggested_doctor,
            display_doctor_ar=display_doctor_ar(suggested_doctor),
            precautions=[],
            needs_follow_up=True,
            follow_up_questions=follow_up_questions,
            retrieved_cases=[],
        )
        clarification_response = _naturalize_chat_response(
            clarification_response,
            state=conversation_state,
            route="medical_clarification",
            message=request.message,
            history=recent_history,
        )
        return finalize(clarification_response, conversation_state)

    prediction = classifier_service.predict(extracted_symptoms)

    retrieved_cases = rag_service.filter_evidence(
        rag_service.retrieve(combined_user_message),
        urgency_level=urgency_level,
    )
    prediction = classifier_service.fuse_prediction(
        prediction=prediction,
        symptoms=extracted_symptoms,
        message=combined_user_message,
        severity_score=int(severity["score"]),
        retrieved_cases=retrieved_cases,
        descriptions=knowledge_service.descriptions,
        precautions=knowledge_service.precautions,
    )

    if urgency_level != "High" and rare_diagnosis_unsupported(
        prediction.diagnosis,
        extracted_symptoms,
        combined_user_message,
        conversation_state.denied_concepts,
    ):
        return finalize(
            _unsupported_prediction_response(
                state=conversation_state,
                diagnosis=prediction.diagnosis,
                assistant_history_texts=assistant_history_texts,
            ),
            conversation_state,
        )

    mode = determine_response_mode(
        message=combined_user_message,
        symptoms=extracted_symptoms,
        confidence=prediction.confidence,
        urgency_level=urgency_level,
    )
    if mode == CLARIFICATION_MODE and _has_throat_cough_fever_cluster(extracted_symptoms):
        mode = DIAGNOSIS_MODE
    if mode == CLARIFICATION_MODE:
        follow_up_questions = filter_repeated_questions(
            smart_follow_up_questions(combined_user_message, extracted_symptoms),
            assistant_history_texts,
        )
        suggested_doctor = _clarification_doctor(combined_user_message, urgency_level)
        clarification_response = ChatResponse(
            mode=CLARIFICATION_MODE,
            answer=clarification_answer(follow_up_questions),
            extracted_symptoms=extracted_symptoms,
            possible_diagnosis=None,
            display_diagnosis_ar=None,
            confidence=0.0,
            urgency_level=urgency_level,
            suggested_doctor=suggested_doctor,
            display_doctor_ar=display_doctor_ar(suggested_doctor),
            precautions=[],
            needs_follow_up=True,
            follow_up_questions=follow_up_questions,
            retrieved_cases=[],
        )
        clarification_response = _naturalize_chat_response(
            clarification_response,
            state=conversation_state,
            route="medical_clarification",
            message=request.message,
            history=recent_history,
        )
        return finalize(clarification_response, conversation_state)

    if mode != EMERGENCY_MODE:
        mode = DIAGNOSIS_MODE

    precautions = knowledge_service.get_precautions(prediction.diagnosis)
    suggested_doctor = knowledge_service.suggest_doctor(
        prediction.diagnosis,
        extracted_symptoms,
        urgency_level,
        message=combined_user_message,
    )
    follow_up_questions = knowledge_service.follow_up_questions(
        extracted_symptoms,
        prediction.confidence,
        urgency_level,
        message=combined_user_message,
    )
    follow_up_questions = filter_repeated_questions(follow_up_questions, assistant_history_texts)
    if urgency_level == "High":
        follow_up_questions = []
    needs_follow_up = urgency_level != "High" and bool(follow_up_questions)

    answer = llm_service.generate_answer(
        message=combined_user_message,
        history=request.history,
        extracted_symptoms=extracted_symptoms,
        diagnosis=prediction.diagnosis,
        confidence=prediction.confidence,
        urgency_level=urgency_level,
        suggested_doctor=suggested_doctor,
        precautions=precautions,
        diagnosis_description=knowledge_service.get_description(prediction.diagnosis),
        follow_up_questions=follow_up_questions,
        retrieved_cases=retrieved_cases,
    )

    return finalize(
        ChatResponse(
            mode=mode,
            answer=answer,
            extracted_symptoms=extracted_symptoms,
            possible_diagnosis=prediction.diagnosis,
            display_diagnosis_ar=display_diagnosis_ar(prediction.diagnosis),
            confidence=prediction.confidence,
            urgency_level=urgency_level,
            suggested_doctor=suggested_doctor,
            display_doctor_ar=display_doctor_ar(suggested_doctor),
            precautions=precautions,
            needs_follow_up=needs_follow_up,
            follow_up_questions=follow_up_questions,
            retrieved_cases=[RetrievedCase(**case) for case in retrieved_cases],
        ),
        conversation_state,
        language=conversation_state.language,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    effective_request, local_record = _augment_local_demo_request(request)
    response = chat_engine_v3.handle_chat(
        effective_request,
        fallback_handler=lambda forwarded_request: chat_engine_v2.handle_chat(
            forwarded_request,
            legacy_handler=_legacy_chat_impl,
        ),
    )
    _remember_local_demo_turn(
        original_request=request,
        effective_request=effective_request,
        response=response,
        record=local_record,
    )
    return response
