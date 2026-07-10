from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = _DIACRITICS.sub("", text)
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[\u0640]+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class EmergencySignal:
    category: str
    diagnosis: str
    display_diagnosis_ar: str
    doctor: str
    display_doctor_ar: str
    reason: str


NEGATION_TERMS = {
    "مفيش",
    "مافيش",
    "مش عندي",
    "ماعنديش",
    "ما عنديش",
    "لا يوجد",
    "بدون",
    "من غير",
    "لا اعاني من",
    "لا أعاني من",
    "ولا",
    "no",
    "without",
}


RED_FLAG_TERMS = [
    "الم شديد في الصدر",
    "الم صدر شديد",
    "صعوبه تنفس",
    "ضيق تنفس شديد",
    "اختناق",
    "اغماء",
    "فقدان الوعي",
    "نزيف شديد",
    "قيء دم",
    "دم في البراز",
    "تشنج",
    "تشنجات",
    "شلل",
    "زرقه الشفاه",
    "تيبس الرقبه",
    "حراره عاليه جدا",
    "حمى شديده",
    "chest pain",
    "severe chest",
    "shortness of breath",
    "difficulty breathing",
    "fainting",
    "seizure",
    "blood in stool",
    "vomiting blood",
    "مش قادر اتنفس",
    "مش عارف اتنفس",
    "نفسي مقطوع",
    "حاسس باختناق",
    "بختنق",
    "مخنوق",
    "زرقة في الشفاه",
    "زرقه في الشفاه",
    "ألم صدر شديد",
    "الم صدر شديد",
    "ضغط على الصدر",
    "وجع صدر مع عرق",
    "وجع صدر مع ضيق تنفس",
    "صداع شديد مفاجئ",
    "اسوء صداع في حياتي",
    "أسوأ صداع في حياتي",
    "صداع مع زغللة شديدة",
    "تنميل في نص جسمي",
    "تنميل ناحية واحدة",
    "تنميل في ايدي",
    "ضعف ناحية واحدة",
    "تلعثم الكلام",
    "ثقل الكلام",
    "اعوجاج الفم",
    "دم في البراز",
    "براز اسود",
    "قيء دم",
    "ترجيع دم",
    "مش بتبول",
    "جفاف شديد",
    "تورم الشفاه",
    "تورم الوجه مع ضيق تنفس",
    "حساسية مع اختناق",
    "بستفرغ دم",
    "استفرغ دم",
    "بترجع دم",
    "دم في البول ووجع شديد",
    "بول بدم ووجع شديد",
    "مش بتبول وجفاف",
    "جفاف شديد ودوخة",
    "كلامي تقيل",
    "لساني تقيل",
    "وشي معوج",
    "اعوجاج الوجه",
    "صداع شديد مع تيبس رقبة",
    "حكة شديدة وتورم في الوجه",
    "طفح مع تورم الوجه",
    "تورم الوجه مع حكة",
    "حكة شديدة وتورم في وشي",
    "حكه شديده وتورم في وشي",
    "تورم في وشي",
    "وشي وارم",
    "وشي ورم",
    "فقدت الوعي",
    "فقد وعي",
    "اغمى عليا",
    "اغمي عليا",
    "صدري ومعاه عرق بارد",
    "صدري مع عرق بارد",
    "وجع في صدري ومعاه عرق بارد",
    "عرق بارد مع وجع صدر",
    "مش قادر اشرب",
    "مش قادر اشرب مياه",
    "قلة بول ودوخة",
    "قله بول ودوخه",
    "اسهال شديد وقلة بول",
    "اسهال شديد وقله بول",
    "جوع شديد مع لخبطة",
    "جوع شديد مع لخبطه",
    "تعرق وجوع شديد مع لخبطة",
    "تعرق وجوع شديد مع لخبطه",
    "مغص كلوي شديد",
    "حرارة عالية وتدهور سريع",
    "حراره عاليه وتدهور سريع",
    "تدهور سريع",
    "مش قادر اقف",
    "مش قادر أقف",
    "طفل وقع على دماغه وبيستفرغ",
    "طفل وقع على راسه وبيستفرغ",
    "طفل وقع على رأسه وبيستفرغ",
    "وقع على دماغه وبيستفرغ",
    "وقع على راسه وبيستفرغ",
    "وقع على رأسه وبيستفرغ",
    "pregnant ومعايا bleeding",
    "pregnant with bleeding",
    "pregnant bleeding",
    "بفكر اذي نفسي",
    "بفكر أذي نفسي",
    "بفكر أأذي نفسي",
    "افكر اذي نفسي",
    "أفكر أذي نفسي",
    "الم في العين بعد خبطة",
    "ألم في العين بعد خبطة",
    "eye pain after trauma",
]

PHASE4B_RED_FLAG_TERMS = [
    "مش عايز اعيش",
    "مش عايز أعيش",
    "افكار انتحارية",
    "أفكار انتحارية",
    "بفكر اذي نفسي",
    "بفكر أذي نفسي",
    "بفكر أأذي نفسي",
    "افكر اذي نفسي",
    "أفكر أذي نفسي",
    "اؤذي نفسي",
    "أؤذي نفسي",
    "هأذي نفسي",
    "هقتل نفسي",
    "انتحار",
    "حامل وعندي نزيف",
    "نزيف مهبلي مع حمل",
    "نزيف مع حمل",
    "ألم شديد مع حمل",
    "الم شديد مع حمل",
    "صداع وزغللة مع حمل",
    "ضغط عالي مع حمل",
    "الطفل مش بيرضع",
    "رضيعي مش بيرضع",
    "الطفل خامل",
    "طفل خامل",
    "الطفل مش بيفوق",
    "الطفل عنده تشنج",
    "تشنجات مع الحرارة",
    "الطفل مش قادر يتنفس",
    "مش عارف اتبول",
    "مش قادر اتبول",
    "احتباس بول",
    "صعوبة التبول مع ألم شديد",
    "صعوبة التبول مع الم شديد",
    "صداع شديد مع تيبس رقبة",
    "صداع شديد مع حرارة وتيبس رقبة",
    "صداع شديد مع لخبطة",
    "مش شايف كويس فجأة",
    "فقدان نظر",
    "وقعت على راسي",
    "وقعت على رأسي",
    "خبطة في الرأس",
    "خبطة في الراس",
    "جرح عميق",
    "نزيف مستمر",
    "حادث ونزيف",
    "كسر ونزيف",
    "تسمم",
    "بلعت دواء كتير",
    "جرعة زيادة",
    "اخدت حبوب كتير",
    "أخدت حبوب كتير",
    "اخدت جرعة كبيرة",
    "أخدت جرعة كبيرة",
    "جرعة كبيرة من الدوا",
    "جرعة كبيرة من الدواء",
    "شربت منظف",
    "بلعت منظف",
    "شربت كلور",
    "بلعت سم",
    "overdose",
    "poison",
    "حامل ومش حاسة بحركة الجنين",
    "مش حاسة بحركة الجنين",
    "حركة الجنين قلت",
    "حركة الجنين وقفت",
    "فقدت النظر فجأة",
    "فقدان النظر فجأة",
    "فقدان مفاجئ للنظر",
    "مش شايف فجأة",
    "خبطة في العين",
    "إصابة في العين",
    "اصابة في العين",
    "اتخبطت في دماغي وبعدها بقيت بستفرغ",
    "اتخبطت في راسي وبعدها بقيت بستفرغ",
    "خبطة في الرأس مع قيء",
    "خبطة في الراس مع قيء",
    "وقعت على دماغي وبستفرغ",
    "حرق كبير",
    "حرق عميق",
    "الجلد مفتوح",
    "خراج في الضرس ووشي وارم",
    "ورم في الوش مع ألم ضرس",
    "ورم في الوجه مع ألم ضرس",
    "اتكسر سني وفي نزيف",
    "نفسي ضاق",
    "نفسي ضايق",
    "تورم رجل واحدة مع ألم",
]

RED_FLAG_TERMS.extend(PHASE4B_RED_FLAG_TERMS)


SELF_HARM_TERMS = [
    "مش عايز اعيش",
    "مش عايز أعيش",
    "افكار انتحارية",
    "أفكار انتحارية",
    "اؤذي نفسي",
    "أؤذي نفسي",
    "هأذي نفسي",
    "هقتل نفسي",
    "انتحار",
]

MENTAL_HEALTH_TERMS = [
    "قلق",
    "توتر",
    "اكتئاب",
    "نوبات هلع",
    "خوف شديد",
    "مش بنام",
    "أرق",
    "ارق",
]

PREGNANCY_TERMS = ["حامل", "حمل", "أنا حامل", "انا حامل", "pregnant"]
PREGNANCY_RED_FLAG_TERMS = [
    "نزيف مهبلي",
    "نزيف مع حمل",
    "حامل وعندي نزيف",
    "ألم شديد مع حمل",
    "الم شديد مع حمل",
    "صداع وزغللة مع حمل",
    "ضغط عالي مع حمل",
    "حامل ومش حاسة بحركة الجنين",
    "مش حاسة بحركة الجنين",
    "حركة الجنين قلت",
    "حركة الجنين وقفت",
    "pregnant ومعايا bleeding",
    "pregnant with bleeding",
    "pregnant bleeding",
    "bleeding with pregnancy",
]

GYNECOLOGY_CONTEXT_TERMS = [
    "الدورة",
    "الدوره",
    "الدورة متأخرة",
    "الدوره متاخره",
    "تأخر الدورة",
    "تاخر الدوره",
    "افرازات مهبلية",
    "إفرازات مهبلية",
    "حكة مهبلية",
    "حكه مهبليه",
    "نزيف مهبلي",
    "بعد العلاقة",
    "بعد العلاقه",
]

PEDIATRIC_TERMS = [
    "طفل",
    "طفلي",
    "ابني",
    "بنتي",
    "رضيعي",
    "رضيع",
    "البيبي",
    "عنده سنتين",
    "عندها حرارة",
]
PEDIATRIC_RED_FLAG_TERMS = [
    "الطفل مش بيرضع",
    "رضيعي مش بيرضع",
    "مش بيرضع",
    "الطفل خامل",
    "طفل خامل",
    "مش بيفوق",
    "الطفل بيصرخ",
    "بيصرخ ومش بيهدى",
    "تشنجات مع الحرارة",
    "الطفل مش قادر يتنفس",
    "طفل وقع على دماغه وبيستفرغ",
    "طفل وقع على راسه وبيستفرغ",
    "طفل وقع على رأسه وبيستفرغ",
]

URINARY_RETENTION_TERMS = ["احتباس بول", "مش عارف اتبول", "مش قادر اتبول"]
URINARY_CONTEXT_TERMS = ["حرقان بول", "دم في البول", "بول كتير", "بول قليل", "ألم جنب", "الم جنب", "مغص كلوي", "صعوبة التبول"]
ENDOCRINE_CONTEXT_TERMS = [
    "هبوط سكر",
    "ارتفاع سكر",
    "سكر عالي",
    "سكر واطي",
    "السكر واطي",
    "السكر عالي",
    "عطش",
    "عطش شديد",
    "جوع شديد",
    "جوووع",
]
INFECTIOUS_CONTEXT_TERMS = [
    "التهاب",
    "عدوى",
    "عدوي",
    "سخونية",
    "حرارة",
    "حمى",
    "حرارة بقالها أيام",
    "حرارة بقالها ايام",
    "تعرق مع حرارة",
]
ENT_CONTEXT_TERMS = ["طنين", "ألم أذن", "الم اذن", "ودني", "الأذن", "الاذن", "الدنيا بتلف"]
DENTAL_CONTEXT_TERMS = ["ألم سن", "الم سن", "ضرس", "لثة", "لثه", "ورم في الفم", "سنان"]
EYE_CONTEXT_TERMS = [
    "ألم عين",
    "الم عين",
    "احمرار عين",
    "مش شايف",
    "فقدان نظر",
    "تغير في النظر",
    "eye pain",
    "blurred vision",
    "خبطة في العين",
    "إصابة في العين",
    "اصابة في العين",
]
TRAUMA_CONTEXT_TERMS = ["إصابة", "اصابة", "كدمة", "وقعت", "جرح", "حادث", "كسر", "اتخبطت", "خبطة"]
SEVERE_TRAUMA_TERMS = [
    "جرح عميق",
    "نزيف مستمر",
    "نزيف شديد",
    "وقعت على راسي",
    "وقعت على رأسي",
    "حادث ونزيف",
    "كسر ونزيف",
    "اتخبطت في دماغي وبعدها بقيت بستفرغ",
    "اتخبطت في راسي وبعدها بقيت بستفرغ",
    "خبطة في الرأس مع قيء",
    "خبطة في الراس مع قيء",
    "حرق كبير",
    "حرق عميق",
    "الجلد مفتوح",
    "طفل وقع على دماغه وبيستفرغ",
    "طفل وقع على راسه وبيستفرغ",
    "طفل وقع على رأسه وبيستفرغ",
    "الم في العين بعد خبطة",
    "ألم في العين بعد خبطة",
]


def _contains_any(normalized_message: str, terms: Iterable[str]) -> bool:
    return any(normalize_text(term) in normalized_message for term in terms)


def detect_context_flags(message: str) -> set[str]:
    normalized = normalize_text(message)
    flags: set[str] = set()
    checks = [
        ("self_harm", SELF_HARM_TERMS),
        ("mental_health", MENTAL_HEALTH_TERMS),
        ("pregnancy", PREGNANCY_TERMS),
        ("pregnancy_red_flag", PREGNANCY_RED_FLAG_TERMS),
        ("gynecology_context", GYNECOLOGY_CONTEXT_TERMS),
        ("pediatric", PEDIATRIC_TERMS),
        ("pediatric_red_flag", PEDIATRIC_RED_FLAG_TERMS),
        ("urinary_retention", URINARY_RETENTION_TERMS),
        ("urinary_context", URINARY_CONTEXT_TERMS),
        ("endocrine_context", ENDOCRINE_CONTEXT_TERMS),
        ("infectious_context", INFECTIOUS_CONTEXT_TERMS),
        ("ent_context", ENT_CONTEXT_TERMS),
        ("dental_context", DENTAL_CONTEXT_TERMS),
        ("eye_context", EYE_CONTEXT_TERMS),
        ("trauma_context", TRAUMA_CONTEXT_TERMS),
        ("severe_trauma", SEVERE_TRAUMA_TERMS),
    ]
    for flag, terms in checks:
        if _contains_any(normalized, terms):
            flags.add(flag)
    signal = detect_emergency_signal(message)
    if signal:
        flags.add(f"{signal.category}_emergency")
        if signal.category == "pregnancy":
            flags.add("pregnancy_red_flag")
        if signal.category in {"head_injury", "eye"}:
            flags.add("severe_trauma")
    return flags


HIGH_RISK_SYMPTOMS = {
    "chest_pain",
    "breathlessness",
    "coma",
    "altered_sensorium",
    "blood_in_sputum",
    "stomach_bleeding",
    "weakness_of_one_body_side",
    "slurred_speech",
    "coma",
    "acute_liver_failure",
}

LOW_RISK_URI_SYMPTOMS = {
    "cough",
    "runny_nose",
    "congestion",
    "throat_irritation",
    "continuous_sneezing",
    "fatigue",
    "loss_of_smell",
    "watering_from_eyes",
    "redness_of_eyes",
}

LOW_RISK_SKIN_SYMPTOMS = {
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "red_spots_over_body",
    "dischromic _patches",
    "internal_itching",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
}

LOW_RISK_DIGESTIVE_SYMPTOMS = {
    "stomach_pain",
    "acidity",
    "indigestion",
    "nausea",
    "vomiting",
    "constipation",
    "abdominal_pain",
    "belly_pain",
    "loss_of_appetite",
    "passage_of_gases",
    "distention_of_abdomen",
}

LOW_RISK_URINARY_SYMPTOMS = {
    "burning_micturition",
    "bladder_discomfort",
    "continuous_feel_of_urine",
    "polyuria",
    "back_pain",
    "foul_smell_of urine",
}

LOW_RISK_VASCULAR_SYMPTOMS = {
    "swollen_legs",
    "swollen_blood_vessels",
    "prominent_veins_on_calf",
    "painful_walking",
}


def _only_low_risk(symptoms: set[str], allowed: set[str]) -> bool:
    return bool(symptoms) and symptoms.issubset(allowed)


def _has_hypertension_context(normalized_message: str) -> bool:
    return any(term in normalized_message for term in {"ضغط عالي", "ارتفاع الضغط", "ضغط الدم عالي"})


def _is_negated_at(normalized_message: str, match_index: int) -> bool:
    prefix = normalized_message[:match_index].split()
    window = " ".join(prefix[-6:])
    return any(normalize_text(term) in window for term in NEGATION_TERMS)


def _is_soft_low_urine_context(normalized_message: str, normalized_term: str) -> bool:
    if normalized_term != normalize_text("مش بتبول"):
        return False
    return any(
        normalize_text(term) in normalized_message
        for term in {
            "مش بتبول كويس",
            "مش بتبول كتير",
            "مش بتبول طبيعي",
        }
    )


def _contains_non_negated(normalized_message: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    start = 0
    while True:
        match_index = normalized_message.find(normalized_term, start)
        if match_index < 0:
            return False
        if not _is_negated_at(normalized_message, match_index) and not _is_soft_low_urine_context(
            normalized_message,
            normalized_term,
        ):
            return True
        start = match_index + len(normalized_term)


def _has_negated_self_harm_context(normalized_message: str) -> bool:
    negated_self_harm_terms = {
        "مش هاذي نفسي",
        "مش هأذي نفسي",
        "مش هؤذي نفسي",
        "مش اذي نفسي",
        "مش أذي نفسي",
        "مش بفكر اذي نفسي",
        "مش بفكر أذي نفسي",
        "لا افكر اذي نفسي",
        "لا أفكر أذي نفسي",
        "not going to hurt myself",
        "not thinking of hurting myself",
        "do not want to hurt myself",
        "don't want to hurt myself",
    }
    return any(normalize_text(term) in normalized_message for term in negated_self_harm_terms)


NEURO_ONE_SIDE_DIRECT_TERMS = [
    "نص جسمي تنمل",
    "نص جسمي الشمال تنمل",
    "نصف جسمي تنمل",
    "تنميل في نص جسمي",
    "تنميل في نص الجسم",
    "تنميل في جهه واحده",
    "تنميل في جهة واحدة",
    "تنميل في ناحيه واحده",
    "تنميل في ناحية واحدة",
    "ناحيه من جسمي مش حاسس بيها",
    "ناحية من جسمي مش حاسس بيها",
    "مش حاسس بدراعي ورجلي",
    "numb on one side",
    "one side is numb",
    "left side is numb",
    "right side is numb",
    "weakness one side",
]

NEURO_SIDE_TERMS = [
    "نص جسمي",
    "نصف جسمي",
    "ناحيه واحده",
    "ناحية واحدة",
    "جهة واحدة",
    "جهه واحده",
    "الشمال",
    "اليسار",
    "اليمين",
    "يمين",
    "left side",
    "right side",
    "one side",
]

NEURO_NUMBNESS_TERMS = [
    "تنميل",
    "متنمل",
    "تنمل",
    "مش حاسس",
    "فقدان احساس",
    "خدر",
    "numb",
    "numbness",
]

NEURO_WEAKNESS_TERMS = [
    "ضعف",
    "ضعاف",
    "تقال",
    "تقيل",
    "ثقيل",
    "مش قادر احرك",
    "مش قادر احرك دراعي",
    "مش قادره احرك",
    "شلل",
    "weak",
    "weakness",
    "heavy",
]

NEURO_SPEECH_TERMS = [
    "الكلام تقيل",
    "كلامي تقيل",
    "بتكلم بصعوبه",
    "بتكلم بصعوبة",
    "مش عارف اتكلم",
    "مش عارفه اتكلم",
    "مش قادر اتكلم",
    "الكلام مش طالع",
    "لساني تقيل",
    "كلامي ملخبط",
    "لخبطه في الكلام",
    "لخبطة في الكلام",
    "تلعثم الكلام",
    "slurred speech",
    "speech is slurred",
]

NEURO_FACE_DROOP_TERMS = [
    "وشي مايل",
    "وشى مايل",
    "وجهي مايل",
    "بوقي معوج",
    "بؤي معوج",
    "فمي معوج",
    "نص وشي واقع",
    "نصف وشي واقع",
    "اعوجاج الفم",
    "اعوجاج الوجه",
    "face drooping",
]

NEURO_IMBALANCE_TERMS = [
    "عدم اتزان",
    "مش متزن",
    "مش متواز",
    "دوخه شديده",
    "دوخة شديدة",
    "دايخ جدا",
    "severe dizziness",
    "loss of balance",
]

SUDDEN_TERMS = ["فجاه", "فجاة", "فجأة", "مفاجئ", "مفاجئه", "من ساعه", "من ساعة", "sudden", "suddenly"]
SEVERE_HEADACHE_TERMS = ["صداع شديد", "صداع جامد", "اسوء صداع", "أسوأ صداع", "severe headache"]
VISION_NEURO_TERMS = [
    "زغلله مفاجئه",
    "زغللة مفاجئة",
    "فقدان نظر مفاجئ",
    "فقدت النظر فجاة",
    "مش شايف فجاة",
    "sudden vision loss",
    "vision loss",
]

CHEST_PAIN_TERMS = [
    "الم صدر",
    "ألم صدر",
    "الم في الصدر",
    "ألم في الصدر",
    "وجع صدر",
    "وجع في الصدر",
    "صدري",
    "chest pain",
]
CHEST_PRESSURE_TERMS = ["عصره في صدري", "عصرة في صدري", "ضغط في صدري", "ضغط على الصدر", "chest pressure"]
BREATHING_TERMS = ["ضيق تنفس", "صعوبه تنفس", "صعوبة تنفس", "مش قادر اتنفس", "shortness of breath", "difficulty breathing"]
COLD_SWEAT_TERMS = ["عرق بارد", "تعرق بارد", "cold sweat", "sweating"]
CHEST_RADIATION_TERMS = ["واصل لدراعي", "يمتد للذراع", "دراعي الشمال", "الفك", "الظهر", "jaw", "arm", "back"]
FAINT_OR_SEVERE_DIZZY_TERMS = ["اغماء", "فقدان الوعي", "دوخه شديده", "دوخة شديدة", "fainting"]

PREGNANCY_BLEEDING_TERMS = ["نزيف", "bleeding"]
PREGNANCY_REDUCED_MOVEMENT_TERMS = ["حركه الجنين قلت", "حركة الجنين قلت", "مش حاسه بحركه الجنين", "reduced fetal movement"]
PREGNANCY_SEVERE_PAIN_TERMS = ["الم بطن شديد", "ألم بطن شديد", "وجع شديد في بطني", "severe abdominal pain"]
PREGNANCY_NEURO_TERMS = ["صداع شديد", "زغلله", "زغللة", "تورم شديد", "ضغط عالي"]

POISONING_TERMS = [
    "شربت منظف",
    "بلعت منظف",
    "شربت كلور",
    "بلعت كلور",
    "شربت سم",
    "بلعت سم",
    "جرعه كبيره",
    "جرعة كبيرة",
    "جرعه زياده",
    "خدت جرعه كبيره",
    "اخدت جرعه كبيره",
    "overdose",
    "poisoning",
    "poison",
    "bleach",
    "cleaner ingestion",
]

ALLERGY_SWELLING_TERMS = [
    "تورم وش",
    "تورم الوجه",
    "وشي ورم",
    "وشي وارم",
    "وشي وشفايفي ورموا",
    "تورم الشفايف",
    "شفايفي ورمت",
    "تورم اللسان",
    "لساني ورم",
    "facial swelling",
    "tongue swelling",
]
ALLERGY_CONTEXT_TERMS = ["حساسيه", "حساسية", "طفح", "حكه", "حكة", "allergy", "allergic"]

HEAD_INJURY_TERMS = [
    "خبطه في الراس",
    "خبطة في الراس",
    "خبطه في الرأس",
    "خبطة في الرأس",
    "اصابه راس",
    "اصابة رأس",
    "وقعت على راسي",
    "وقعت على دماغي",
    "اتخبطت في دماغي",
    "head injury",
    "head trauma",
]
VOMITING_TERMS = ["قيء", "ترجيع", "بستفرغ", "بيستفرغ", "vomiting"]
LOSS_OF_CONSCIOUSNESS_TERMS = ["فقدان وعي", "فقدت الوعي", "اغمي عليا", "اغمى عليا", "loss of consciousness"]

EYE_SEVERE_TERMS = [
    "فقدان نظر مفاجئ",
    "فقدت النظر فجاة",
    "مش شايف فجاة",
    "الم عين شديد مفاجئ",
    "ألم عين شديد مفاجئ",
    "اصابه عين شديده",
    "اصابة عين شديدة",
    "sudden vision loss",
]
EYE_TRAUMA_TERMS = ["خبطه في العين", "خبطة في العين", "اصابه في العين", "اصابة في العين", "eye trauma"]

RESPIRATORY_DANGER_TERMS = [
    "ضيق تنفس شديد",
    "صعوبة تنفس شديدة",
    "مش قادر اتنفس",
    "مش عارف اتنفس",
    "مش قادر اتكلم من ضيق النفس",
    "مش قادر اتكلم من النفس",
    "نهجان شديد",
    "زرقة الشفاه",
    "زرقه الشفاه",
    "الشفايف زرقا",
    "صفير شديد",
    "wheezing severe",
    "blue lips",
    "can't speak because of breathlessness",
    "cannot speak because of breathlessness",
]

SEVERE_DEHYDRATION_TERMS = [
    "جفاف شديد",
    "بق ناشف جدا",
    "فمي ناشف جدا",
    "مش قادر اشرب",
    "مش قادر اشرب مياه",
    "قلة بول ودوخة",
    "قله بول ودوخه",
    "مش بتبول وجفاف",
    "تدهور سريع",
    "لخبطة",
    "مش مركز",
    "confusion",
    "severe dehydration",
]
DIARRHEA_VOMITING_TERMS = [
    "اسهال شديد",
    "إسهال شديد",
    "اسهال مستمر",
    "ترجيع مستمر",
    "قيء مستمر",
    "بستفرغ كتير",
    "vomiting and diarrhea",
]
SEPSIS_DANGER_TERMS = [
    "حرارة عالية وتدهور سريع",
    "حراره عاليه وتدهور سريع",
    "حرارة عالية مع لخبطة",
    "حمى شديدة مع لخبطة",
    "high fever with confusion",
    "severe fever and confusion",
]

DIABETES_CONTEXT_TERMS = [
    "عندي سكر",
    "مريض سكر",
    "السكر",
    "سكري",
    "diabetic",
    "diabetes",
    "هبوط سكر",
    "ارتفاع سكر",
    "سكر عالي",
    "سكر واطي",
]
DIABETES_DANGER_TERMS = [
    "لخبطة",
    "مش مركز",
    "اغماء",
    "فقدان الوعي",
    "جفاف شديد",
    "عطش شديد",
    "ترجيع مستمر",
    "confusion",
    "fainting",
    "severe dehydration",
]

ABDOMINAL_SEVERE_TERMS = [
    "الم بطن شديد",
    "ألم بطن شديد",
    "وجع بطن شديد",
    "مغص شديد",
    "بطني متحجرة",
    "بطني متحجره",
    "البطن متحجرة",
    "البطن ناشفة",
    "right lower abdominal pain",
    "severe abdominal pain",
]
RIGHT_LOWER_ABDOMEN_TERMS = [
    "يمين تحت البطن",
    "الناحية اليمين تحت",
    "الجانب اليمين تحت",
    "right lower abdomen",
]
GI_BLEEDING_TERMS = [
    "قيء دم",
    "ترجيع دم",
    "بستفرغ دم",
    "براز اسود",
    "براز أسود",
    "دم في البراز",
    "vomiting blood",
    "black stool",
]

URINARY_DANGER_TERMS = [
    "دم في البول ووجع شديد",
    "بول بدم ووجع شديد",
    "مغص كلوي شديد",
    "ألم جنب شديد",
    "الم جنب شديد",
    "flank pain with fever",
]

PEDIATRIC_DANGER_EXTRA_TERMS = [
    "طفل مش قادر يتنفس",
    "الطفل مش قادر يتنفس",
    "ابني مش قادر يتنفس",
    "بنتي مش قادرة تتنفس",
    "طفل عنده تشنج",
    "الطفل عنده تشنج",
    "طفل عنده حرارة وتيبس رقبة",
    "رضيع خامل",
    "baby not feeding",
    "lethargic baby",
    "child seizure",
]


def _has_concept(normalized_message: str, terms: Iterable[str]) -> bool:
    return any(_contains_non_negated(normalized_message, term) for term in terms)


def _has_one_sided_neuro(normalized_message: str) -> bool:
    if _has_concept(normalized_message, NEURO_ONE_SIDE_DIRECT_TERMS):
        return True
    has_side = _has_concept(normalized_message, NEURO_SIDE_TERMS)
    has_neuro = _has_concept(normalized_message, NEURO_NUMBNESS_TERMS + NEURO_WEAKNESS_TERMS)
    arm_leg_pair = _has_concept(normalized_message, ["دراعي ورجلي", "ايدي ورجلي", "ذراعي ورجلي", "arm and leg"])
    return has_side and (has_neuro or arm_leg_pair)


def detect_emergency_signal(message: str, symptoms: Iterable[str] = ()) -> EmergencySignal | None:
    normalized = normalize_text(message)
    symptom_set = set(symptoms)

    if _has_concept(normalized, SELF_HARM_TERMS) and not _has_negated_self_harm_context(normalized):
        return EmergencySignal(
            "self_harm",
            "Self-harm emergency concern",
            "خطر إيذاء النفس يحتاج مساعدة عاجلة",
            "Emergency Department / Psychiatrist",
            "الطوارئ فورًا / طبيب نفسي",
            "وجود أفكار أو نية لإيذاء النفس يحتاج مساعدة فورية ولا يجب التعامل معه وحدك.",
        )

    pediatric = _has_concept(normalized, PEDIATRIC_TERMS)
    if pediatric and (
        _has_concept(normalized, PEDIATRIC_RED_FLAG_TERMS + PEDIATRIC_DANGER_EXTRA_TERMS)
        or _has_concept(normalized, RESPIRATORY_DANGER_TERMS)
    ):
        return EmergencySignal(
            "pediatric",
            "Pediatric emergency concern",
            "أعراض خطيرة عند طفل تحتاج طوارئ",
            "Pediatric Emergency Department",
            "طوارئ أطفال فورًا",
            "وجود صعوبة تنفس أو خمول شديد أو تشنجات أو رفض رضاعة عند طفل/رضيع يحتاج طوارئ أطفال.",
        )

    if _has_concept(normalized, RESPIRATORY_DANGER_TERMS):
        return EmergencySignal(
            "respiratory",
            "Respiratory emergency concern",
            "أعراض تنفسية خطيرة محتملة",
            "Emergency Department / Pulmonologist",
            "الطوارئ فورًا / طبيب صدر",
            "صعوبة التنفس الشديدة أو زرقة الشفاه أو عدم القدرة على الكلام بسبب النفس قد تكون خطيرة.",
        )

    if (
        _has_concept(normalized, SEPSIS_DANGER_TERMS)
        or (
            _has_concept(normalized, DIARRHEA_VOMITING_TERMS)
            and _has_concept(normalized, SEVERE_DEHYDRATION_TERMS)
        )
    ):
        return EmergencySignal(
            "dehydration_or_infection",
            "Severe dehydration or infection concern",
            "اشتباه جفاف شديد أو عدوى خطيرة",
            "Emergency Department / Internal Medicine",
            "الطوارئ فورًا / باطنة",
            "القيء أو الإسهال الشديد مع علامات جفاف/لخبطة أو الحرارة العالية مع تدهور سريع يحتاج تقييمًا عاجلًا.",
        )

    if _has_concept(normalized, DIABETES_CONTEXT_TERMS) and _has_concept(normalized, DIABETES_DANGER_TERMS):
        return EmergencySignal(
            "endocrine",
            "Diabetes emergency concern",
            "اشتباه اضطراب سكر خطير",
            "Emergency Department / Internal Medicine or Endocrinologist",
            "الطوارئ فورًا / باطنة أو غدد صماء",
            "اضطراب السكر مع إغماء أو لخبطة أو جفاف شديد قد يحتاج تدخلًا عاجلًا.",
        )

    poisoning = _has_concept(normalized, POISONING_TERMS)
    if poisoning:
        return EmergencySignal(
            "poisoning",
            "Poisoning emergency concern",
            "اشتباه تسمم أو جرعة زائدة",
            "Poison Control / Emergency Department",
            "الطوارئ أو مركز السموم فورًا",
            "وجود ابتلاع منظف/كلور/سم أو جرعة كبيرة يحتاج تقييمًا عاجلًا.",
        )

    allergy_airway = (
        _has_concept(normalized, ALLERGY_SWELLING_TERMS) and _has_concept(normalized, BREATHING_TERMS)
    ) or (_has_concept(normalized, ALLERGY_CONTEXT_TERMS) and _has_concept(normalized, BREATHING_TERMS))
    if allergy_airway:
        return EmergencySignal(
            "allergy",
            "Severe allergic reaction concern",
            "اشتباه حساسية شديدة أو تورم تحسسي",
            "Emergency Department / Allergy or Pulmonology",
            "الطوارئ فورًا / حساسية ومناعة أو صدر حسب الحالة",
            "تورم الوجه أو الشفاه أو اللسان مع صعوبة التنفس قد يكون طارئًا.",
        )

    one_sided = _has_one_sided_neuro(normalized) or bool(
        {"weakness_of_one_body_side", "slurred_speech"}.intersection(symptom_set)
    )
    speech = _has_concept(normalized, NEURO_SPEECH_TERMS)
    face = _has_concept(normalized, NEURO_FACE_DROOP_TERMS)
    imbalance = _has_concept(normalized, NEURO_IMBALANCE_TERMS)
    severe_headache = _has_concept(normalized, SEVERE_HEADACHE_TERMS)
    sudden = _has_concept(normalized, SUDDEN_TERMS)
    vision_neuro = _has_concept(normalized, VISION_NEURO_TERMS)
    dizziness = _has_concept(normalized, ["دوخة", "دوخه", "دوار", "دايخ"])
    neuro_symptom = one_sided or speech or face or imbalance
    if (
        (one_sided and speech)
        or (one_sided and face)
        or (speech and face)
        or (severe_headache and neuro_symptom)
        or (vision_neuro and neuro_symptom)
        or (imbalance and speech)
        or (one_sided and imbalance and sudden)
        or (one_sided and sudden)
        or (one_sided and dizziness)
    ):
        return EmergencySignal(
            "neurological",
            "Neurological emergency concern",
            "أعراض عصبية خطيرة محتملة",
            "Emergency Department / Neurologist",
            "الطوارئ فورًا / طبيب مخ وأعصاب",
            "تنميل أو ضعف في ناحية من الجسم مع صعوبة كلام/ميلان الوجه/دوخة شديدة قد يشير لحالة عصبية طارئة.",
        )

    chest = _has_concept(normalized, CHEST_PAIN_TERMS + CHEST_PRESSURE_TERMS)
    if chest and (
        _has_concept(normalized, BREATHING_TERMS)
        or _has_concept(normalized, COLD_SWEAT_TERMS)
        or _has_concept(normalized, CHEST_RADIATION_TERMS)
        or _has_concept(normalized, FAINT_OR_SEVERE_DIZZY_TERMS)
    ):
        return EmergencySignal(
            "cardiac",
            "Cardiac emergency concern",
            "اشتباه مشكلة قلبية طارئة",
            "Emergency Department / Cardiologist",
            "الطوارئ فورًا / طبيب قلب",
            "ألم أو ضغط الصدر مع ضيق تنفس أو عرق بارد أو امتداد الألم يحتاج طوارئ.",
        )

    pregnancy = _has_concept(normalized, PREGNANCY_TERMS)
    if pregnancy and (
        _has_concept(normalized, PREGNANCY_BLEEDING_TERMS)
        or _has_concept(normalized, PREGNANCY_REDUCED_MOVEMENT_TERMS)
        or _has_concept(normalized, PREGNANCY_SEVERE_PAIN_TERMS)
        or (
            _has_concept(normalized, ["صداع شديد"])
            and _has_concept(normalized, ["زغلله", "زغللة", "تورم شديد", "ضغط عالي"])
        )
    ):
        return EmergencySignal(
            "pregnancy",
            "Pregnancy emergency concern",
            "أعراض حمل طارئة محتملة",
            "Emergency Department / Gynecologist",
            "طوارئ نساء وتوليد فورًا",
            "نزيف أو قلة حركة الجنين أو ألم شديد أثناء الحمل يحتاج تقييمًا عاجلًا.",
        )

    abdominal_danger = (
        _has_concept(normalized, GI_BLEEDING_TERMS)
        or (
            _has_concept(normalized, ABDOMINAL_SEVERE_TERMS)
            and (
                _has_concept(normalized, ["متحجر", "متحجرة", "ناشفة", "rigid"])
                or (
                    _has_concept(normalized, RIGHT_LOWER_ABDOMEN_TERMS)
                    and _has_concept(normalized, ["حرارة", "حمى", "ترجيع", "قيء", "vomiting", "fever"])
                )
            )
        )
    )
    if abdominal_danger:
        return EmergencySignal(
            "abdominal",
            "Abdominal surgical emergency concern",
            "ألم بطن خطير يحتاج طوارئ",
            "Emergency Department / Surgery or Gastroenterologist",
            "الطوارئ فورًا / جراحة أو جهاز هضمي",
            "ألم البطن الشديد مع تيبس/تحجر أو قيء دم أو براز أسود أو ألم يمين أسفل البطن مع حرارة قد يحتاج طوارئ.",
        )

    urinary_danger = (
        _has_concept(normalized, URINARY_RETENTION_TERMS + URINARY_DANGER_TERMS)
        or (
            _has_concept(normalized, ["الم جنب", "ألم جنب", "مغص كلوي", "flank pain"])
            and _has_concept(normalized, ["حرارة", "حمى", "سخونية", "fever"])
        )
        or (
            _has_concept(normalized, ["دم في البول", "بول بدم", "blood in urine"])
            and _has_concept(normalized, ["وجع شديد", "الم شديد", "ألم شديد", "severe pain"])
        )
    )
    if urinary_danger:
        return EmergencySignal(
            "urinary",
            "Urinary or kidney emergency concern",
            "اشتباه مشكلة كلى أو مسالك طارئة",
            "Emergency Department / Urologist",
            "الطوارئ فورًا / مسالك بولية",
            "احتباس البول أو دم في البول مع ألم شديد أو ألم جنب مع حرارة يحتاج تقييمًا عاجلًا.",
        )

    head_injury = _has_concept(normalized, HEAD_INJURY_TERMS)
    if head_injury and (_has_concept(normalized, VOMITING_TERMS) or _has_concept(normalized, LOSS_OF_CONSCIOUSNESS_TERMS)):
        return EmergencySignal(
            "head_injury",
            "Head injury emergency concern",
            "إصابة رأس تحتاج طوارئ",
            "Emergency Department / Neurology or Surgery",
            "الطوارئ فورًا / مخ وأعصاب أو جراحة",
            "إصابة الرأس مع قيء أو فقدان وعي تحتاج تقييمًا عاجلًا.",
        )

    if _has_concept(normalized, EYE_SEVERE_TERMS) or (
        _has_concept(normalized, EYE_TRAUMA_TERMS) and _has_concept(normalized, ["الم", "ألم", "وجع", "vision loss"])
    ):
        return EmergencySignal(
            "eye",
            "Eye emergency concern",
            "أعراض عين طارئة محتملة",
            "Emergency Department / Ophthalmologist",
            "الطوارئ فورًا / طبيب عيون",
            "فقدان نظر مفاجئ أو ألم/إصابة شديدة في العين يحتاج تقييمًا عاجلًا.",
        )

    return None


def has_red_flags(message: str, symptoms: Iterable[str] = ()) -> bool:
    if detect_emergency_signal(message, symptoms):
        return True
    normalized = normalize_text(message)
    if any(_contains_non_negated(normalized, term) for term in RED_FLAG_TERMS):
        return True
    return bool(HIGH_RISK_SYMPTOMS.intersection(set(symptoms)))


def choose_urgency(message: str, symptoms: list[str], severity_score: int) -> str:
    symptom_set = set(symptoms)
    normalized = normalize_text(message)
    context_flags = detect_context_flags(message)

    if has_red_flags(message, symptoms):
        return "High"

    if {
        "self_harm",
        "pregnancy_red_flag",
        "pediatric_red_flag",
        "urinary_retention",
        "severe_trauma",
    }.intersection(context_flags):
        return "High"

    if "dental_context" in context_flags and any(
        term in normalized
        for term in {
            normalize_text("وشي وارم"),
            normalize_text("وشي ورم"),
            normalize_text("تورم في الوش"),
            normalize_text("تورم في الوجه"),
        }
    ):
        return "High"

    if "eye_context" in context_flags and any(
        term in normalized
        for term in {
            normalize_text("بعد خبطة"),
            normalize_text("بعد خبطه"),
            normalize_text("فقدت النظر"),
            normalize_text("فقدان النظر"),
            normalize_text("مش شايف فجأة"),
            normalize_text("الم شديد"),
            normalize_text("ألم شديد"),
        }
    ):
        return "High"

    if "trauma_context" in context_flags and any(
        term in normalized
        for term in {
            normalize_text("مش قادر اقف"),
            normalize_text("مش قادر أقف"),
            normalize_text("قيء"),
            normalize_text("بيستفرغ"),
            normalize_text("بستفرغ"),
            normalize_text("لخبطة"),
            normalize_text("لخبطه"),
            normalize_text("نزيف"),
            normalize_text("جرح عميق"),
        }
    ):
        return "High"

    if "pregnancy" in context_flags and any(
        term in normalized
        for term in {
            normalize_text("نزيف"),
            normalize_text("ألم شديد"),
            normalize_text("الم شديد"),
            normalize_text("زغللة"),
            normalize_text("صداع شديد"),
            normalize_text("bleeding"),
        }
    ):
        return "High"

    if "pediatric" in context_flags and (
        {"breathlessness", "coma", "altered_sensorium"}.intersection(symptom_set)
        or any(
            term in normalized
            for term in {
                normalize_text("مش بيرضع"),
                normalize_text("خامل"),
                normalize_text("تشنج"),
                normalize_text("مش بيفوق"),
                normalize_text("صعوبة تنفس"),
                normalize_text("وقع على دماغه"),
                normalize_text("وقع على راسه"),
                normalize_text("وقع على رأسه"),
                normalize_text("بيستفرغ"),
            }
        )
    ):
        return "High"

    if "stomach_bleeding" in symptom_set:
        return "High"

    if "stiff_neck" in symptom_set and {"headache", "high_fever", "mild_fever"}.intersection(symptom_set):
        return "High"

    if "spotting_ urination" in symptom_set:
        severe_urinary_bleeding = any(
            term in normalized
            for term in {
                normalize_text("وجع شديد"),
                normalize_text("ألم شديد"),
                normalize_text("الم شديد"),
                normalize_text("مش بتبول"),
                normalize_text("جفاف"),
            }
        )
        return "High" if severe_urinary_bleeding else "Medium"

    if "dehydration" in symptom_set and (
        {"dizziness", "sunken_eyes"}.intersection(symptom_set)
        or any(
            term in normalized
            for term in {
                normalize_text("مش بتبول"),
                normalize_text("قلة بول"),
                normalize_text("قله بول"),
                normalize_text("مش قادر اشرب"),
                normalize_text("مش قادر اقف"),
            }
        )
    ):
        if _is_soft_low_urine_context(normalized, normalize_text("مش بتبول")) and not {
            "dizziness",
            "sunken_eyes",
        }.intersection(symptom_set):
            pass
        else:
            return "High"

    if {"diarrhoea", "dizziness"}.issubset(symptom_set) and any(
        term in normalized
        for term in {
            normalize_text("قلة بول"),
            normalize_text("قله بول"),
            normalize_text("بول قليل"),
            normalize_text("البول قليل"),
        }
    ):
        return "High"

    if (
        "excessive_hunger" in symptom_set
        and {"sweating", "shivering", "chills"}.intersection(symptom_set)
        and (
            {"coma", "dizziness", "fatigue", "altered_sensorium"}.intersection(symptom_set)
            or any(
                term in normalized
                for term in {
                    normalize_text("مش قادر اقف"),
                    normalize_text("لخبطة"),
                    normalize_text("لخبطه"),
                }
            )
        )
    ):
        return (
            "High"
            if (
                {"coma", "altered_sensorium"}.intersection(symptom_set)
                or any(
                    term in normalized
                    for term in {
                        normalize_text("مش قادر اقف"),
                        normalize_text("لخبطة"),
                        normalize_text("لخبطه"),
                    }
                )
            )
            else "Medium"
        )

    if "back_pain" in symptom_set and any(
        term in normalized
        for term in {
            normalize_text("مغص كلوي شديد"),
            normalize_text("ألم جنب شديد"),
            normalize_text("الم جنب شديد"),
        }
    ):
        return "High"

    if {"high_fever", "breathlessness"}.issubset(symptom_set):
        return "High"

    urinary_complication = {"burning_micturition", "bladder_discomfort", "continuous_feel_of_urine"}.intersection(
        symptom_set
    )
    flank_context = any(
        term in normalized
        for term in {
            normalize_text("الم في الجنب"),
            normalize_text("وجع في الجنب"),
            normalize_text("ألم الخاصرة"),
            normalize_text("وجع الخاصرة"),
        }
    )
    if urinary_complication and (
        {"high_fever", "vomiting", "dark_urine"}.intersection(symptom_set) or flank_context
    ):
        return "Medium"

    if _only_low_risk(symptom_set, LOW_RISK_URI_SYMPTOMS):
        return "Low"

    if _only_low_risk(symptom_set, LOW_RISK_SKIN_SYMPTOMS):
        return "Low"

    if _only_low_risk(symptom_set, LOW_RISK_VASCULAR_SYMPTOMS):
        return "Low"

    if _only_low_risk(symptom_set, LOW_RISK_URINARY_SYMPTOMS):
        return "Low"

    if _only_low_risk(symptom_set, LOW_RISK_DIGESTIVE_SYMPTOMS):
        if {"vomiting", "diarrhoea"}.issubset(symptom_set):
            return "Medium"
        return "Low"

    if _has_hypertension_context(normalized) and {"headache", "dizziness"}.intersection(symptom_set):
        return "Medium"

    if {"yellowish_skin", "dark_urine"}.issubset(symptom_set) or {"yellowing_of_eyes", "yellowish_skin"}.issubset(
        symptom_set
    ):
        return "Medium"

    if {"vomiting", "diarrhoea"}.issubset(symptom_set):
        return "Medium"

    if "high_fever" in symptom_set:
        return "Medium"

    if severity_score >= 22:
        return "High"

    if severity_score >= 8:
        return "Medium"

    return "Low"


def safety_suffix(urgency_level: str) -> str:
    if urgency_level == "High":
        return "إذا كانت الأعراض شديدة أو تتدهور بسرعة، اطلب رعاية طبية عاجلة الآن."
    return "هذه إجابة مساعدة وليست تشخيصا نهائيا. راجع طبيبا إذا استمرت الأعراض أو ساءت."
