from __future__ import annotations

from .safety import detect_context_flags, has_red_flags, normalize_text


DIAGNOSIS_MODE = "diagnosis"
CLARIFICATION_MODE = "clarification"
EMERGENCY_MODE = "emergency"
CLOSING_MODE = "closing"

MIN_CONFIDENCE_FOR_DIAGNOSIS = 0.35
MIN_CONFIDENCE_WITH_MEANINGFUL_SYMPTOMS = 0.30
MEANINGFUL_SYMPTOM_COUNT = 3


CLOSING_PHRASES = [
    "شكرا",
    "شكرًا",
    "تمام",
    "خلاص",
    "كده كفاية",
    "كدة كفاية",
    "مش محتاج حاجة تاني",
    "مش محتاج حاجه تاني",
    "تسلم",
    "تسلم يا دكتور",
    "thank you",
    "thanks",
    "thx",
]

NON_REPEATED_FALLBACK_QUESTIONS = [
    "هل ظهر عرض جديد أو زادت الأعراض عن آخر رسالة؟",
    "هل يوجد ألم صدر، ضيق تنفس، إغماء، نزيف، أو تدهور سريع؟",
    "منذ متى بدأت المشكلة، وهل تتحسن أم تزيد؟",
]

NEGATION_TERMS = {
    "مفيش",
    "مافيش",
    "من غير",
    "ولا",
    "بدون",
    "لا يوجد",
}

NEGATED_QUESTION_TERMS: dict[str, list[str]] = {
    "ضيق نفس": ["ضيق تنفس", "صعوبة تنفس", "صعوبة في التنفس"],
    "ضيق تنفس": ["ضيق تنفس", "صعوبة تنفس", "صعوبة في التنفس"],
    "الم صدر": ["ألم صدر", "الم صدر", "ألم في الصدر"],
    "ألم صدر": ["ألم صدر", "الم صدر", "ألم في الصدر"],
    "اغماء": ["إغماء", "اغماء"],
}


BODY_AREA_TERMS: dict[str, list[str]] = {
    "throat": [
        "زوري",
        "زورى",
        "حلقي",
        "حلقى",
        "حلق",
        "بلع",
        "حنجرة",
        "حاجة واقفة في حلقي",
        "حاجة واقفة في زوري",
    ],
    "abdomen": ["بطني", "بطن", "معدتي", "معده", "كرشي", "وجع بطن", "الم بطن"],
    "chest": ["صدري", "صدر", "الصدر", "وجع صدر", "الم صدر"],
    "head": ["دماغي", "راسي", "رأسي", "وجع راس", "وجع رأس"],
    "urinary": ["بول", "تبول", "بتبول", "حمام", "حرقان بول", "حرقان"],
    "skin": ["جلدي", "جلد", "حبوب", "طفح", "هرش", "حكة", "حكه"],
}

BODY_AREA_ONLY_TERMS: dict[str, list[str]] = {
    "throat": [
        "زوري",
        "زورى",
        "حلقي",
        "حلقى",
        "حلق",
        "بلع",
        "حنجرة",
        "حاجة واقفة في حلقي",
        "حاجة واقفة في زوري",
    ],
    "abdomen": ["بطني", "بطن", "معدتي", "معده", "كرشي"],
    "chest": ["صدري", "صدر", "الصدر"],
    "head": ["دماغي", "راسي", "رأسي"],
    "urinary": ["بول", "تبول", "بتبول", "حمام"],
    "skin": ["جلدي", "جلد"],
}

SPECIFIC_SYMPTOM_CONTEXT_TERMS = [
    "كحة",
    "سعال",
    "سخونية",
    "حرارة",
    "حمى",
    "بلغم",
    "صفير",
    "مزمنة",
    "رشح",
    "احتقان",
    "احمرار",
    "حكة",
    "حكه",
    "هرش",
    "طفح",
    "حبوب",
    "بقع",
    "يتقشر",
    "تقشر",
    "إسهال",
    "اسهال",
    "ترجيع",
    "قيء",
    "غثيان",
    "إمساك",
    "امساك",
    "فقدان شهية",
    "دوخة",
    "صداع",
    "زغللة",
    "تنميل",
    "حرقان",
    "دم",
    "تورم",
    "ضيق تنفس",
]

SPECIFIC_SYMPTOM_CONTEXT_TERMS.extend(
    [
        "نزيف",
        "حامل",
        "حمل",
        "طفل",
        "ابني",
        "بنتي",
        "رضيعي",
        "مش بيرضع",
        "مش عايز اعيش",
        "أفكار انتحارية",
        "اؤذي نفسي",
        "قلق",
        "اكتئاب",
        "ضرس",
        "ألم عين",
        "إصابة",
        "جرح",
        "وقعت",
        "احتباس بول",
        "مش عارف اتبول",
        "طنين",
    ]
)

VAGUE_TERMS = [
    "تعبان",
    "تعبانة",
    "مش مرتاح",
    "مش مرتاحة",
    "حاسس بحاجة غلط",
    "حاسه بحاجة غلط",
    "مش قادر احدد",
    "مش قادر أحدد",
    "مش عارف احدد",
    "مش عارفة احدد",
    "مش طبيعي",
    "مش مظبوط",
    "جسمي مش مظبوط",
    "من غير تفاصيل",
    "بسيطة",
    "بسيطه",
]

TOPIC_QUESTIONS: dict[str, list[str]] = {
    "throat": [
        "هل عندك حرارة أو كحة؟",
        "هل الألم بيزيد مع البلع؟",
        "هل في صعوبة تنفس أو إحساس باختناق أو تغير في الصوت؟",
        "هل في صعوبة في الكلام أو إحساس إن حاجة واقفة في الحلق؟",
        "منذ متى بدأ ألم الزور أو الإحساس ده؟",
    ],
    "abdomen": [
        "الألم فين بالضبط في البطن: فوق، تحت، يمين، شمال، ولا حوالين السرة؟",
        "هل في ترجيع أو غثيان؟",
        "هل في إسهال أو إمساك؟",
        "هل في حرارة؟",
        "هل لاحظت دم في البراز أو القيء؟",
        "منذ متى بدأت المشكلة؟",
    ],
    "chest": [
        "هل ألم الصدر شديد أو ضاغط؟",
        "هل معاه ضيق تنفس أو تعرق شديد؟",
        "هل الألم بيمتد للذراع أو الفك أو الظهر؟",
        "الألم بدأ من إمتى وبيستمر قد إيه؟",
        "هل في دوخة شديدة أو إغماء؟",
    ],
    "head": [
        "هل الصداع شديد جدًا أو بدأ فجأة؟",
        "هل في دوخة أو قيء؟",
        "هل في زغللة أو تغير في النظر؟",
        "هل في ضعف أو تنميل في ناحية من الجسم؟",
        "هل في تيبس في الرقبة أو حرارة؟",
        "منذ متى بدأت الأعراض؟",
    ],
    "urinary": [
        "هل في حرقان أثناء التبول؟",
        "هل بتدخل الحمام كتير أو حاسس إنك محتاج تتبول باستمرار؟",
        "هل لاحظت دم في البول؟",
        "هل في ألم في الجنب أو أسفل الظهر؟",
        "هل في حرارة؟",
        "هل في صعوبة أو عدم قدرة على التبول؟",
    ],
    "skin": [
        "الطفح أو الحبوب موجودة فين بالضبط؟",
        "هل في حكة أو ألم؟",
        "هل الطفح بينتشر بسرعة؟",
        "هل في تورم في الشفاه أو الوجه؟",
        "هل في صعوبة تنفس؟",
        "هل ظهر بعد أكل أو منتج جديد على الجلد؟",
    ],
    "infection": [
        "الحرارة بقالها قد إيه؟",
        "هل قست درجة الحرارة؟ وصلت كام تقريبًا؟",
        "هل في كحة، بلغم، ألم حلق، طفح، ألم في الجسم، أو مخالطة لشخص عنده عدوى؟",
        "هل في سفر قريب أو مخالطة لشخص عنده عدوى؟",
        "هل في ضيق تنفس، تيبس رقبة، إغماء، أو تدهور سريع؟",
    ],
    "endocrine": [
        "هل عندك سكر أو قست السكر مؤخرًا؟",
        "هل في عطش شديد أو تبول كتير؟",
        "هل في تعرق ورعشة وجوع شديد؟",
        "هل في زغللة، دوخة شديدة، إغماء، أو لخبطة؟",
    ],
    "neuro_ent": [
        "هل الدوخة إحساس دوران ولا عدم اتزان؟",
        "هل في طنين، ألم، انسداد، أو إفرازات من الأذن؟",
        "هل في قيء، زغللة، صداع شديد، أو تنميل في ناحية من الجسم؟",
        "هل الأعراض بتزيد مع حركة الرأس؟",
    ],
    "pediatric": [
        "عمر الطفل كام؟",
        "هل الطفل بيرضع أو بياكل ويشرب طبيعي؟",
        "هل في خمول شديد، تشنجات، صعوبة تنفس، أو قلة تبول؟",
        "درجة الحرارة وصلت كام وبدأت من إمتى؟",
    ],
    "pregnancy": [
        "هل يوجد حمل حاليًا؟ وفي أي شهر تقريبًا؟",
        "هل يوجد نزيف مهبلي أو ألم شديد أسفل البطن؟",
        "هل يوجد صداع شديد، زغللة، تورم، أو ضغط مرتفع؟",
        "هل الأعراض بدأت فجأة أم تدريجيًا؟",
    ],
    "mental_health": [
        "هل القلق أو الحزن مأثر على النوم أو الأكل أو الدراسة/الشغل؟",
        "منذ متى بدأت الأعراض؟",
        "هل توجد نوبات هلع أو خوف شديد؟",
        "هل عندك أفكار لإيذاء نفسك أو إنك مش عايز تعيش؟",
    ],
    "dental": [
        "الألم في سن ولا ضرس؟ ومنذ متى بدأ؟",
        "هل في ورم في اللثة أو الوجه أو صعوبة في البلع؟",
        "هل توجد حرارة أو صديد أو نزيف؟",
    ],
    "eye": [
        "هل يوجد ألم شديد في العين أو تغير مفاجئ في النظر؟",
        "هل في احمرار، إفرازات، أو حساسية من الضوء؟",
        "هل الزغللة في عين واحدة أم العينين؟",
    ],
    "trauma": [
        "الإصابة حصلت إزاي ومنذ متى؟",
        "هل يوجد نزيف مستمر، جرح عميق، تشوه، أو عدم قدرة على الحركة؟",
        "هل كانت الإصابة في الرأس أو معها قيء أو لخبطة؟",
    ],
    "general": [
        "هل عندك ألم في مكان معين، حرارة، كحة، دوخة، قيء، إسهال، طفح جلدي، أو ضيق تنفس؟",
        "منذ متى بدأت المشكلة؟",
        "هل الأعراض ثابتة، بتزيد، ولا بتتحسن؟",
        "هل عندك مرض مزمن أو بتاخد أدوية حاليًا؟",
        "هل يوجد عرض شديد مثل ألم في الصدر، صعوبة في التنفس، إغماء، نزيف، أو تدهور سريع؟",
    ],
}


def detect_clarification_topics(message: str) -> list[str]:
    normalized = normalize_text(message)
    topics: list[str] = []
    for topic, terms in BODY_AREA_TERMS.items():
        if any(normalize_text(term) in normalized for term in terms):
            topics.append(topic)
    flags = detect_context_flags(message)
    flag_topic_map = {
        "infectious_context": "infection",
        "endocrine_context": "endocrine",
        "urinary_context": "urinary",
        "urinary_retention": "urinary",
        "ent_context": "neuro_ent",
        "pediatric": "pediatric",
        "pediatric_red_flag": "pediatric",
        "pregnancy": "pregnancy",
        "pregnancy_red_flag": "pregnancy",
        "gynecology_context": "pregnancy",
        "mental_health": "mental_health",
        "self_harm": "mental_health",
        "dental_context": "dental",
        "eye_context": "eye",
        "trauma_context": "trauma",
        "severe_trauma": "trauma",
    }
    for flag, topic in flag_topic_map.items():
        if flag in flags and topic not in topics:
            topics.append(topic)
    if ("high_fever" in message or any(term in normalized for term in {normalize_text("سخونية"), normalize_text("حرارة")})):
        if "infection" not in topics:
            topics.append("infection")
    if not topics and any(normalize_text(term) in normalized for term in VAGUE_TERMS):
        topics.append("general")
    return topics


def is_closing_message(message: str) -> bool:
    normalized = normalize_text(message)
    if not normalized:
        return False
    if len(normalized.split()) > 6:
        return False
    if has_red_flags(message):
        return False
    if _has_specific_symptom_context(message) or _has_body_area_only_topic(message):
        return False
    return any(normalize_text(phrase) in normalized for phrase in CLOSING_PHRASES)


def filter_repeated_questions(questions: list[str], previous_assistant_messages: list[str]) -> list[str]:
    if not questions:
        return []

    previous_text = normalize_text(" ".join(previous_assistant_messages))
    filtered = [
        question
        for question in questions
        if normalize_text(question) and normalize_text(question) not in previous_text
    ]
    if filtered:
        return filtered[:6]

    fallback = [
        question
        for question in NON_REPEATED_FALLBACK_QUESTIONS
        if normalize_text(question) not in previous_text
    ]
    return fallback[:3]


def _message_negates_term(message: str, term: str) -> bool:
    normalized = normalize_text(message)
    normalized_term = normalize_text(term)
    start = 0
    while True:
        index = normalized.find(normalized_term, start)
        if index < 0:
            return False
        prefix = normalized[:index].split()
        window = " ".join(prefix[-6:])
        if any(normalize_text(negation) in window for negation in NEGATION_TERMS):
            return True
        start = index + len(normalized_term)


def filter_negated_follow_up_questions(questions: list[str], message: str) -> list[str]:
    blocked_terms: list[str] = []
    for denied_term, question_terms in NEGATED_QUESTION_TERMS.items():
        if _message_negates_term(message, denied_term):
            blocked_terms.extend(question_terms)
    if not blocked_terms:
        return questions
    return [
        question
        for question in questions
        if not any(normalize_text(term) in normalize_text(question) for term in blocked_terms)
    ]


def closing_answer() -> str:
    return (
        "تمام، أتمنى لك السلامة. لو ظهرت أعراض جديدة أو زادت الأعراض، ابعتلي في أي وقت.\n\n"
        "لو في ألم صدر شديد، صعوبة تنفس، إغماء، نزيف، أو تدهور سريع، اطلب مساعدة طبية عاجلة فورًا."
    )


def _has_body_area_only_topic(message: str) -> bool:
    normalized = normalize_text(message)
    return any(
        normalize_text(term) in normalized
        for terms in BODY_AREA_ONLY_TERMS.values()
        for term in terms
    )


def _has_specific_symptom_context(message: str) -> bool:
    normalized = normalize_text(message)
    return any(normalize_text(term) in normalized for term in SPECIFIC_SYMPTOM_CONTEXT_TERMS)


def is_vague_or_body_area_only(message: str, symptoms: list[str]) -> bool:
    normalized = normalize_text(message)
    has_body_area_only_topic = _has_body_area_only_topic(message)
    has_specific_context = _has_specific_symptom_context(message)
    word_count = len(normalized.split())
    has_vague_term = any(normalize_text(term) in normalized for term in VAGUE_TERMS)
    if any(normalize_text(term) in normalized for term in {"من غير تفاصيل", "بسيطة", "بسيطه"}):
        return len(symptoms) <= 2
    return (
        has_body_area_only_topic and len(symptoms) <= 1 and not has_specific_context
    ) or (
        has_vague_term and len(symptoms) <= 2 and not has_specific_context
    ) or (
        word_count <= 2 and len(symptoms) <= 1
    )


def determine_response_mode(
    *,
    message: str,
    symptoms: list[str],
    confidence: float,
    urgency_level: str,
) -> str:
    if urgency_level == "High":
        return EMERGENCY_MODE
    context_flags = detect_context_flags(message)
    if "mental_health" in context_flags and "self_harm" not in context_flags:
        return CLARIFICATION_MODE
    if ("pregnancy" in context_flags or "gynecology_context" in context_flags) and "pregnancy_red_flag" not in context_flags:
        return CLARIFICATION_MODE
    if "eye_context" in context_flags:
        return CLARIFICATION_MODE
    if "pediatric" in context_flags and len(symptoms) <= 1:
        return CLARIFICATION_MODE
    if not symptoms:
        return CLARIFICATION_MODE
    if is_vague_or_body_area_only(message, symptoms):
        return CLARIFICATION_MODE
    if confidence < MIN_CONFIDENCE_FOR_DIAGNOSIS:
        if len(symptoms) >= MEANINGFUL_SYMPTOM_COUNT and confidence >= MIN_CONFIDENCE_WITH_MEANINGFUL_SYMPTOMS:
            return DIAGNOSIS_MODE
        return CLARIFICATION_MODE
    return DIAGNOSIS_MODE


def smart_follow_up_questions(message: str, symptoms: list[str]) -> list[str]:
    topics = detect_clarification_topics(message)
    if not topics:
        topics = ["general"]

    questions: list[str] = []
    for topic in topics:
        for question in TOPIC_QUESTIONS.get(topic, []):
            if question not in questions:
                questions.append(question)
    return filter_negated_follow_up_questions(questions, message)[:6]


def clarification_answer(questions: list[str]) -> str:
    questions = questions or NON_REPEATED_FALLBACK_QUESTIONS[:2]
    question_lines = "\n".join(f"- {question}" for question in questions)
    return (
        "محتاج أعرف كام حاجة بسيطة عشان أوجهك صح.\n\n"
        "ممكن تجاوبني على الأسئلة دي:\n"
        f"{question_lines}\n\n"
        "لو عندك ألم صدر شديد، صعوبة في التنفس، إغماء، نزيف، أو تدهور سريع، اطلب مساعدة طبية عاجلة فورًا."
    )
