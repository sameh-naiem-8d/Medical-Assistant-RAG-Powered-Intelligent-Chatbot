from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .safety import normalize_text
from .schemas import ChatMessage


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


class ConversationIntent:
    GREETING_OR_CASUAL = "greeting_or_casual"
    NON_MEDICAL_CHAT = "non_medical_chat"
    FAMILY_OR_OFFTOPIC = "family_or_offtopic"
    NONSENSE_OR_LOW_INFORMATION = "nonsense_or_low_information"
    PROFANITY_OR_ABUSE = "profanity_or_abuse"
    INSULT_OR_FRUSTRATION = "insult_or_frustration"
    CORRECTION_OR_NEGATION = "correction_or_negation"
    ASK_ABOUT_PREVIOUS_DIAGNOSIS = "ask_about_previous_diagnosis"
    CHALLENGE_PREVIOUS_DIAGNOSIS = "challenge_previous_diagnosis"
    ANSWER_FOLLOWUP_QUESTION = "answer_followup_question"
    REPRODUCTIVE_OR_GENDER_SENSITIVE_COMPLAINT = "reproductive_or_gender_sensitive_complaint"
    NEW_SYMPTOMS = "new_symptoms"
    VAGUE_UNCLEAR = "vague_unclear"


@dataclass
class ConversationState:
    intent: str
    language: str
    current_message: str
    user_history_text: str
    assistant_history_text: str
    current_symptoms: list[str]
    known_symptoms: list[str]
    medical_domains: set[str] = field(default_factory=set)
    denied_concepts: set[str] = field(default_factory=set)
    denied_symptoms: set[str] = field(default_factory=set)
    temperature_c: float | None = None
    has_high_temperature: bool = False
    duration_known: bool = False
    previous_diagnosis: str | None = None
    previous_doctor: str | None = None
    asked_temperature: bool = False
    asked_duration: bool = False
    asked_red_flags: bool = False
    medical_meaning: "MedicalMeaning" = field(default_factory=lambda: MedicalMeaning())
    active_case: dict[str, object] = field(default_factory=dict)


@dataclass
class MedicalMeaning:
    language: str = "ar"
    domain: str | None = None
    body_parts: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)


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
    "what is your temperature",
    "how high is your fever",
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
    "day",
    "days",
    "week",
    "weeks",
    "hour",
    "hours",
}

GREETING_OR_CASUAL_TERMS = {
    "اهلا",
    "اهلين",
    "السلام عليكم",
    "ازيك",
    "عامل اي",
    "عامل ايه",
    "انت عامل ايه",
    "انتي عامله ايه",
    "اي الدنيا",
    "ايه الدنيا",
    "اي الاخبار",
    "ايه الاخبار",
    "انا بسالك اي الدنيا",
    "انا بقولك ازيك",
    "هاي",
    "هلا",
    "صباح الخير",
    "مساء الخير",
    "hello",
    "hi",
    "hey",
    "how are you",
    "what's up",
    "whats up",
    "what's up bro",
    "whats up bro",
}

NON_MEDICAL_CHAT_TERMS = {
    "اسمك ايه",
    "اسمك اي",
    "مين انت",
    "مين انتي",
    "بتعمل ايه",
    "بتعمل اي",
    "انت مين",
    "انتي مين",
    "what is your name",
    "who are you",
    "what do you do",
    "are you okay",
    "are you ok",
}

FAMILY_OR_OFFTOPIC_TERMS = {
    "امك عاملة اي",
    "امك عامله اي",
    "امك اخبارها",
    "ابوك عامل اي",
    "ابوك اخباره",
    "خالتك عاملة اي",
    "خالتك عامله اي",
    "اختك عاملة اي",
    "اختك عامله اي",
    "اخوك عامل اي",
    "طنط عاملة اي",
    "ماما عاملة اي",
    "بابا عامل اي",
    "how is your mom",
    "how is your mother",
    "how is your dad",
    "how is your father",
    "how is your sister",
    "how is your brother",
    "your mom",
    "your mother",
    "your dad",
    "your father",
    "your sister",
    "your brother",
}

PROFANITY_ABUSE_TERMS = {
    "كسمك",
    "كس امك",
    "كس أمك",
    "كسم",
    "يا عرص",
    "عرص",
    "خول",
    "وسخ",
    "يا وسخ",
    "fuck you",
    "fuck off",
    "fucking",
    "shit",
    "asshole",
    "bitch",
}

FRUSTRATION_TERMS = {
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
    "بتفهم",
    "مش بتفهم",
    "يا غبي",
    "makes no sense",
    "this makes no sense",
    "wrong",
    "stupid",
    "are you stupid",
    "dumb",
    "this is dumb",
    "idiot",
    "you idiot",
}

SOFT_FRUSTRATION_TERMS = {"يبني", "يا ابني", "يا بني"}

NEGATION_TERMS = {
    "\u0644\u0627",
    "\u0644\u0623",
    "\u0645\u0641\u064a\u0634",
    "\u0645\u0627\u0641\u064a\u0634",
    "\u0645\u0639\u0647\u0648\u0634",
    "\u0645\u0639\u0647\u0627\u0634",
    "\u0645\u0641\u064a\u0647\u0648\u0634",
    "\u0645\u0627\u0641\u064a\u0647\u0648\u0634",
    "مفيش",
    "مافيش",
    "معنديش",
    "ماعنديش",
    "مقولتش",
    "ماقولتش",
    "انا مقولتش",
    "مش عندي",
    "مش عندى",
    "لا يوجد",
    "بدون",
    "من غير",
    "ولا",
    "no",
    "not",
    "without",
    "didn't say",
    "did not say",
    "i didn't say",
    "i did not say",
}

DENIED_CONCEPT_TERMS: dict[str, set[str]] = {
    "fever": {"حرارة", "حراره", "سخونية", "سخونيه", "سخن", "حمى", "حمي", "fever", "temperature"},
    "cough": {"كحة", "كحه", "سعال", "cough"},
    "travel": {"سفر", "مسافر", "سافرت", "travel", "traveled", "travelled"},
    "mosquito": {"ناموس", "بعوض", "mosquito", "mosquitoes"},
    "breathlessness": {"ضيق تنفس", "نفس", "اتنفس", "breathless", "shortness of breath"},
    "chest_pain": {"ألم صدر", "الم صدر", "وجع صدر", "chest pain"},
    "numbness": {"تنميل", "خدر", "numbness", "numb", "tingling"},
    "weakness": {"ضعف", "ضعف الرجل", "weakness", "leg weakness"},
    "bladder_bowel_loss": {
        "فقدان تحكم",
        "التحكم في البول",
        "التحكم في البراز",
        "loss of bladder",
        "loss of bowel",
        "bladder control",
        "bowel control",
    },
}

CONCEPT_TO_SYMPTOMS: dict[str, set[str]] = {
    "fever": {"high_fever", "mild_fever", "chills", "shivering"},
    "cough": {"cough", "phlegm"},
    "breathlessness": {"breathlessness"},
    "chest_pain": {"chest_pain"},
    "numbness": {"numbness"},
    "weakness": {"weakness"},
}

DIAGNOSIS_QUESTION_TERMS = {
    "يعني ايه",
    "يعني إيه",
    "اي هي",
    "ايه هي",
    "ايه",
    "اي",
    "ما هو",
    "اشرح",
    "اصلا",
    "أصلا",
    "what is",
    "explain",
    "is it dangerous",
}

CHALLENGE_TERMS = {
    "ليه",
    "لماذا",
    "ازاي",
    "إزاي",
    "اي علاقة",
    "ايه علاقة",
    "مش منطقي",
    "مش صح",
    "غلط",
    "why",
    "not right",
    "what does this have to do",
}

BODY_ACHE_TERMS = {
    "جسمي واجعني",
    "جسمى واجعنى",
    "وجع في جسمي",
    "وجع في الجسم",
    "تكسير في الجسم",
    "تكسير",
    "تكسير جسمي",
    "جسمي مكسر",
    "مكسر",
    "my body aches",
    "body aches",
    "body pain",
    "muscle aches",
}

VAGUE_GENERAL_TERMS = {
    "تعبان",
    "تعبانة",
    "مش مرتاح",
    "مش مرتاحة",
    "مش مظبوط",
    "جسمي مش مظبوط",
    "حاسس بحاجة غلط",
    "مش قادر احدد",
    "i feel unwell",
    "not feeling well",
    "tired",
}

LOW_INFORMATION_OFFTOPIC_TERMS = {
    "اي يبررو",
    "اي يبررووو",
    "يبررو",
    "برو",
    "bro",
    "bro what",
    "random",
    "blah",
    "??",
}

MEDICAL_DOMAIN_TERMS: dict[str, set[str]] = {
    "infection": {
        "حرارة",
        "حراره",
        "سخونية",
        "سخونيه",
        "حمى",
        "حمي",
        "عدوى",
        "عدوي",
        "التهاب",
        "fever",
        "infection",
    },
    "pain_fatigue": {
        "ألم",
        "الم",
        "وجع",
        "تعب",
        "تعبان",
        "مرهق",
        "إرهاق",
        "ارهاق",
        "مكسر",
        "تكسير",
        "pain",
        "hurt",
        "hurts",
        "sore",
        "ache",
        "aches",
        "fatigue",
        "tired",
    },
    "respiratory": {
        "كحة",
        "كحه",
        "سعال",
        "بلغم",
        "نفس",
        "تنفس",
        "مخنوق",
        "صفير",
        "cough",
        "phlegm",
        "breath",
        "breathing",
        "wheeze",
    },
    "throat_ent": {
        "زوري",
        "زورى",
        "حلقي",
        "حلقى",
        "حلق",
        "بلع",
        "ودني",
        "اذن",
        "أذن",
        "طنين",
        "throat",
        "sore throat",
        "throat hurts",
        "swallow",
        "ear",
    },
    "digestive": {
        "بطني",
        "بطن",
        "معدة",
        "معده",
        "إسهال",
        "اسهال",
        "ترجيع",
        "قيء",
        "غثيان",
        "حموضة",
        "حموضه",
        "مغص",
        "stomach",
        "belly",
        "tummy",
        "abdomen",
        "diarrhea",
        "vomit",
        "nausea",
        "cramps",
    },
    "urinary": {
        "بول",
        "تبول",
        "حرقان بول",
        "مسالك",
        "كلى",
        "كلي",
        "urine",
        "urinary",
        "pee",
        "kidney",
    },
    "skin": {
        "طفح",
        "حكة",
        "حكه",
        "هرش",
        "جلد",
        "حبوب",
        "rash",
        "itching",
        "skin",
        "acne",
    },
    "eye": {
        "عين",
        "عيني",
        "نظر",
        "زغللة",
        "زغلله",
        "eye",
        "vision",
        "blurred",
    },
    "dental": {
        "سن",
        "سنان",
        "ضرس",
        "لثة",
        "لثه",
        "tooth",
        "teeth",
        "gum",
        "dentist",
    },
    "neurological": {
        "صداع",
        "دوخة",
        "دوخه",
        "تنميل",
        "ضعف",
        "كلامي",
        "اتكلم",
        "headache",
        "dizzy",
        "dizziness",
        "numb",
        "speech",
    },
    "cardiac": {
        "صدر",
        "قلبي",
        "خفقان",
        "ضغط",
        "chest",
        "heart",
        "palpitations",
        "blood pressure",
    },
    "mental_health": {
        "قلق",
        "اكتئاب",
        "توتر",
        "نوبات هلع",
        "مش عايز اعيش",
        "anxiety",
        "depression",
        "panic",
        "self harm",
    },
    "reproductive": {
        "بريود",
        "البريود",
        "الدورة",
        "الدوره",
        "الحيض",
        "دورة شهرية",
        "دوره شهريه",
        "وجع الدورة",
        "وجع الدوره",
        "مغص الدورة",
        "مغص الدوره",
        "تقلصات الدورة",
        "تقلصات الدوره",
        "تأخر الدورة",
        "تاخر الدورة",
        "نزيف مهبلي",
        "افرازات مهبلية",
        "إفرازات مهبلية",
        "حكة مهبلية",
        "حكه مهبليه",
        "حمل",
        "حامل",
        "pregnant",
        "pregnancy",
        "period",
        "period pain",
        "menstrual",
        "menstruation",
        "vaginal bleeding",
        "vaginal discharge",
        "cramps",
    },
    "pediatric": {
        "طفل",
        "طفلي",
        "ابني",
        "بنتي",
        "رضيع",
        "رضيعي",
        "baby",
        "child",
        "kid",
    },
    "trauma": {
        "إصابة",
        "اصابة",
        "وقعت",
        "جرح",
        "كدمة",
        "كدمه",
        "حادث",
        "كسر",
        "مكسور",
        "fracture",
        "injury",
        "fell",
        "wound",
        "accident",
    },
    "medication_poisoning": {
        "دواء",
        "دوا",
        "جرعة",
        "جرعه",
        "سم",
        "تسمم",
        "كلور",
        "منظف",
        "medicine",
        "medication",
        "dose",
        "poison",
        "bleach",
    },
    "allergy": {
        "حساسية",
        "حساسيه",
        "تورم",
        "شفايف",
        "وشي ورم",
        "allergy",
        "allergic",
        "swelling",
        "lips",
    },
    "chronic": {
        "سكر",
        "سكري",
        "ضغط",
        "ربو",
        "غدة",
        "غده",
        "diabetes",
        "hypertension",
        "asthma",
        "thyroid",
    },
}

RARE_OR_OVER_SPECIFIC_DIAGNOSES = {
    "AIDS",
    "Malaria",
    "Dengue",
    "Typhoid",
    "Tuberculosis",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Hypertension",
    "Pneumonia",
}

AIDS_SUPPORT_TERMS = {
    "aids",
    "hiv",
    "hiv positive",
    "positive hiv",
    "immunodeficiency",
    "\u0625\u064a\u062f\u0632",
    "\u0627\u064a\u062f\u0632",
    "\u0646\u0642\u0635 \u0627\u0644\u0645\u0646\u0627\u0639\u0629",
    "\u0641\u064a\u0631\u0648\u0633 \u0646\u0642\u0635 \u0627\u0644\u0645\u0646\u0627\u0639\u0629",
    "\u062a\u062d\u0644\u064a\u0644 hiv",
    "\u062a\u062d\u0644\u064a\u0644 \u0625\u064a\u062f\u0632",
    "\u062a\u062d\u0644\u064a\u0644 \u0627\u064a\u062f\u0632",
    "\u0639\u0644\u0627\u0642\u0629 \u063a\u064a\u0631 \u0645\u062d\u0645\u064a\u0629",
    "\u062c\u0646\u0633 \u063a\u064a\u0631 \u0645\u062d\u0645\u064a",
    "\u0625\u0628\u0631\u0629 \u0645\u0644\u0648\u062b\u0629",
    "\u0627\u0628\u0631\u0629 \u0645\u0644\u0648\u062b\u0629",
}

MEANING_BODY_PART_TERMS: dict[str, set[str]] = {
    "back": {
        "back",
        "back pain",
        "back hurts",
        "my back hurts",
        "lower back",
        "\u0638\u0647\u0631",
        "\u0638\u0647\u0631\u064a",
        "\u0636\u0647\u0631",
        "\u0636\u0647\u0631\u064a",
        "\u0627\u0644\u0638\u0647\u0631",
        "\u0627\u0644\u0636\u0647\u0631",
        "\u0627\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
        "\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
        "\u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
        "\u0623\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
        "\u0627\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631",
        "\u0623\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631",
        "\u0627\u0639\u0644\u0649 \u0627\u0644\u0636\u0647\u0631",
        "\u0623\u0639\u0644\u0649 \u0627\u0644\u0636\u0647\u0631",
        "\u0648\u062c\u0639 \u0638\u0647\u0631",
        "\u0648\u062c\u0639 \u0636\u0647\u0631",
        "\u0627\u0644\u0645 \u0638\u0647\u0631",
        "\u0623\u0644\u0645 \u0638\u0647\u0631",
    },
    "neck": {
        "neck",
        "neck pain",
        "neck hurts",
        "my neck hurts",
        "\u0631\u0642\u0628\u0629",
        "\u0631\u0642\u0628\u0647",
        "\u0631\u0642\u0628\u062a\u064a",
        "\u0631\u0642\u0628\u062a\u0649",
        "\u0627\u0644\u0631\u0642\u0628\u0629",
        "\u0627\u0644\u0631\u0642\u0628\u0647",
        "\u0648\u062c\u0639 \u0631\u0642\u0628\u0629",
        "\u0627\u0644\u0645 \u0631\u0642\u0628\u0629",
        "\u0623\u0644\u0645 \u0631\u0642\u0628\u0629",
    },
    "throat": {
        "throat",
        "sore throat",
        "throat hurts",
        "my throat hurts",
        "\u0632\u0648\u0631",
        "\u0632\u0648\u0631\u064a",
        "\u062d\u0644\u0642",
        "\u062d\u0644\u0642\u064a",
        "\u0627\u0644\u062d\u0644\u0642",
        "\u0627\u0644\u0632\u0648\u0631",
        "\u0628\u0644\u0639",
    },
    "abdomen": {
        "abdomen",
        "abdominal",
        "belly",
        "stomach",
        "stomach pain",
        "\u0628\u0637\u0646",
        "\u0628\u0637\u0646\u064a",
        "\u0645\u0639\u062f\u0629",
        "\u0645\u0639\u062f\u0647",
        "\u0645\u0639\u062f\u062a\u064a",
        "\u0643\u0631\u0634",
        "\u0643\u0631\u0634\u064a",
    },
    "chest": {
        "chest",
        "chest pain",
        "\u0635\u062f\u0631",
        "\u0635\u062f\u0631\u064a",
        "\u0627\u0644\u0635\u062f\u0631",
    },
    "urinary": {
        "urine",
        "urinary",
        "burning urination",
        "pee",
        "\u0628\u0648\u0644",
        "\u0627\u0644\u0628\u0648\u0644",
        "\u062a\u0628\u0648\u0644",
        "\u062d\u0631\u0642\u0627\u0646 \u0628\u0648\u0644",
        "\u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0628\u0648\u0644",
    },
    "head": {
        "head",
        "headache",
        "\u0631\u0627\u0633",
        "\u0631\u0623\u0633",
        "\u062f\u0645\u0627\u063a",
        "\u0635\u062f\u0627\u0639",
    },
    "whole_body": {
        "body aches",
        "body pain",
        "\u062c\u0633\u0645\u064a",
        "\u0627\u0644\u062c\u0633\u0645",
        "\u062a\u0643\u0633\u064a\u0631",
    },
}

LOWER_BACK_TERMS = {
    "lower back",
    "\u0627\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
    "\u0623\u0633\u0641\u0644 \u0627\u0644\u0638\u0647\u0631",
    "\u0627\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
    "\u0623\u0633\u0641\u0644 \u0627\u0644\u0636\u0647\u0631",
}

UPPER_BACK_TERMS = {
    "upper back",
    "\u0627\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631",
    "\u0623\u0639\u0644\u0649 \u0627\u0644\u0638\u0647\u0631",
    "\u0627\u0639\u0644\u0649 \u0627\u0644\u0636\u0647\u0631",
    "\u0623\u0639\u0644\u0649 \u0627\u0644\u0636\u0647\u0631",
}

MEANING_PAIN_TERMS = {
    "pain",
    "ache",
    "aches",
    "hurt",
    "hurts",
    "\u0648\u062c\u0639",
    "\u0648\u0627\u062c\u0639",
    "\u0648\u0627\u062c\u0639\u0646\u064a",
    "\u0648\u0627\u062c\u0639\u0627\u0646\u064a",
    "\u0627\u0644\u0645",
    "\u0623\u0644\u0645",
}

MEANING_RED_FLAG_TERMS: dict[str, set[str]] = {
    "breathlessness": {"shortness of breath", "breathing trouble", "\u0636\u064a\u0642 \u0646\u0641\u0633", "\u0635\u0639\u0648\u0628\u0629 \u062a\u0646\u0641\u0633"},
    "trauma": {"injury", "fall", "fell", "accident", "\u0627\u0635\u0627\u0628\u0629", "\u0625\u0635\u0627\u0628\u0629", "\u0648\u0642\u0639\u062a", "\u062d\u0627\u062f\u062b\u0629"},
    "leg_neuro": {"leg weakness", "leg numbness", "\u062a\u0646\u0645\u064a\u0644 \u0627\u0644\u0631\u062c\u0644", "\u0636\u0639\u0641 \u0627\u0644\u0631\u062c\u0644", "\u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0631\u062c\u0644"},
    "bladder_bowel_loss": {"loss of bladder", "loss of bowel", "\u0641\u0642\u062f\u0627\u0646 \u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0648\u0644", "\u0641\u0642\u062f\u0627\u0646 \u062a\u062d\u0643\u0645 \u0641\u064a \u0627\u0644\u0628\u0631\u0627\u0632"},
    "fever": {"fever", "\u062d\u0631\u0627\u0631\u0629", "\u0633\u062e\u0648\u0646\u064a\u0629", "\u062d\u0645\u0649"},
}


def _meaning_normalize(text: str) -> str:
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


def _meaning_has_any(normalized_text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        normalized_term = _meaning_normalize(term)
        if not normalized_term:
            continue
        if re.search(r"[a-z]", normalized_term):
            if re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text):
                return True
            continue
        if normalized_term in normalized_text:
            return True
    return False


def extract_medical_meaning(
    message: str,
    symptoms: list[str],
    language: str,
    denied_concepts: set[str] | None = None,
) -> MedicalMeaning:
    normalized = _meaning_normalize(message)
    body_parts = [
        body_part
        for body_part, terms in MEANING_BODY_PART_TERMS.items()
        if _meaning_has_any(normalized, terms)
    ]
    if "back" in body_parts:
        if _meaning_has_any(normalized, LOWER_BACK_TERMS) and "lower_back" not in body_parts:
            body_parts.append("lower_back")
        if _meaning_has_any(normalized, UPPER_BACK_TERMS) and "upper_back" not in body_parts:
            body_parts.append("upper_back")
    symptom_set = set(symptoms)
    meaning_symptoms: list[str] = []
    if _meaning_has_any(normalized, MEANING_PAIN_TERMS) or any(
        symptom in symptom_set
        for symptom in {
            "back_pain",
            "neck_pain",
            "abdominal_pain",
            "stomach_pain",
            "chest_pain",
            "headache",
            "muscle_pain",
            "throat_irritation",
        }
    ):
        meaning_symptoms.append("pain")
    if "burning_micturition" in symptom_set or _meaning_has_any(
        normalized,
        {"burning urination", "\u062d\u0631\u0642\u0627\u0646 \u0628\u0648\u0644", "\u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0628\u0648\u0644"},
    ):
        meaning_symptoms.append("burning_urination")
        if "urinary" not in body_parts:
            body_parts.append("urinary")

    red_flags = [
        flag
        for flag, terms in MEANING_RED_FLAG_TERMS.items()
        if _meaning_has_any(normalized, terms)
    ]

    domain = None
    if "chest" in body_parts:
        domain = "chest_pain"
    elif "urinary" in body_parts:
        domain = "urinary"
    elif "throat" in body_parts:
        domain = "throat_ent"
    elif "abdomen" in body_parts:
        domain = "digestive_abdominal"
    elif "back" in body_parts:
        domain = "musculoskeletal_back_pain"
    elif "neck" in body_parts:
        domain = "musculoskeletal_neck_pain"
    elif "head" in body_parts:
        domain = "neurological_headache"
    elif "whole_body" in body_parts:
        domain = "general_body_aches"

    return MedicalMeaning(
        language=language,
        domain=domain,
        body_parts=body_parts,
        symptoms=list(dict.fromkeys(meaning_symptoms)),
        red_flags=red_flags,
        denied=sorted(denied_concepts or set()),
    )


def _latest_history_medical_meaning(
    recent_history: list[ChatMessage],
    symptoms: list[str],
    language: str,
    denied_concepts: set[str],
) -> MedicalMeaning:
    for item in reversed(recent_history):
        if item.role != "user" or not item.content:
            continue
        meaning = extract_medical_meaning(item.content, symptoms, language, denied_concepts)
        if meaning.domain:
            return meaning
    return MedicalMeaning(language=language, denied=sorted(denied_concepts))


def _active_medical_meaning(
    current_meaning: MedicalMeaning,
    history_meaning: MedicalMeaning,
    denied_concepts: set[str],
) -> MedicalMeaning:
    base = current_meaning if current_meaning.domain else history_meaning
    if not base.domain:
        base = current_meaning
    body_parts = list(dict.fromkeys(base.body_parts + current_meaning.body_parts))
    symptoms = list(dict.fromkeys(base.symptoms + current_meaning.symptoms))
    red_flags = list(dict.fromkeys(base.red_flags + current_meaning.red_flags))
    return MedicalMeaning(
        language=current_meaning.language or history_meaning.language,
        domain=base.domain,
        body_parts=body_parts,
        symptoms=symptoms,
        red_flags=red_flags,
        denied=sorted(denied_concepts),
    )


def _active_body_part(meaning: MedicalMeaning) -> str | None:
    for body_part in ("lower_back", "upper_back", "back", "neck", "throat", "abdomen", "urinary", "chest", "head"):
        if body_part in meaning.body_parts:
            return body_part
    return meaning.body_parts[0] if meaning.body_parts else None


def _doctor_route_for_active_domain(domain: str | None) -> str | None:
    return {
        "musculoskeletal_back_pain": "Orthopedic doctor",
        "musculoskeletal_neck_pain": "Orthopedic doctor",
        "throat_ent": "Needs more information",
        "digestive_abdominal": "Needs more information",
        "urinary": "Urologist",
        "chest_pain": "Emergency care",
        "neurological_headache": "General Practitioner",
        "general_body_aches": "General Practitioner",
    }.get(domain or "")


def _build_active_case(
    meaning: MedicalMeaning,
    known_symptoms: list[str],
    denied_concepts: set[str],
    assistant_history_text: str,
) -> dict[str, object]:
    if not meaning.domain:
        return {}
    previous_questions: list[str] = []
    if asked_duration(assistant_history_text):
        previous_questions.append("duration")
    if asked_red_flags(assistant_history_text):
        previous_questions.append("red_flags")
    if asked_temperature(assistant_history_text):
        previous_questions.append("temperature")
    return {
        "active_domain": meaning.domain,
        "active_body_part": _active_body_part(meaning),
        "active_symptom": meaning.symptoms[0] if meaning.symptoms else None,
        "known_facts": list(dict.fromkeys(known_symptoms)),
        "denied_facts": sorted(denied_concepts),
        "unanswered_questions": [],
        "doctor_route": _doctor_route_for_active_domain(meaning.domain),
        "red_flags_checked": sorted(
            concept for concept in denied_concepts if concept in {"numbness", "weakness", "bladder_bowel_loss"}
        ),
        "previous_questions_asked": previous_questions,
    }


def build_conversation_state(
    *,
    current_message: str,
    recent_history: list[ChatMessage],
    current_symptoms: list[str],
    history_symptoms: list[str],
    diagnosis_aliases: dict[str, set[str]],
) -> ConversationState:
    user_history_text = " ".join(
        item.content for item in recent_history if item.role == "user" and item.content
    )
    assistant_history_text = " ".join(
        item.content for item in recent_history if item.role == "assistant" and item.content
    )
    combined_user_text = f"{user_history_text} {current_message}".strip()
    denied_concepts = _detect_denied_concepts(combined_user_text)
    denied_symptoms = _symptoms_for_denied_concepts(denied_concepts)
    language = detect_language(current_message, user_history_text)
    medical_domains = detect_medical_domains(current_message)
    known_symptoms = [
        symptom
        for symptom in _dedupe(history_symptoms + current_symptoms)
        if symptom not in denied_symptoms
    ]
    current_medical_meaning = extract_medical_meaning(
        current_message,
        known_symptoms,
        language,
        denied_concepts,
    )
    history_medical_meaning = _latest_history_medical_meaning(
        recent_history,
        known_symptoms,
        language,
        denied_concepts,
    )
    medical_meaning = _active_medical_meaning(
        current_medical_meaning,
        history_medical_meaning,
        denied_concepts,
    )
    if medical_meaning.domain:
        medical_domains.add(medical_meaning.domain)
    if "back" in medical_meaning.body_parts and "pain" in medical_meaning.symptoms and "back_pain" not in known_symptoms:
        known_symptoms.append("back_pain")
    if "neck" in medical_meaning.body_parts and "pain" in medical_meaning.symptoms and "neck_pain" not in known_symptoms:
        known_symptoms.append("neck_pain")
    if "abdomen" in medical_meaning.body_parts and "pain" in medical_meaning.symptoms and "abdominal_pain" not in known_symptoms:
        known_symptoms.append("abdominal_pain")
    if "chest" in medical_meaning.body_parts and "pain" in medical_meaning.symptoms and "chest_pain" not in known_symptoms:
        known_symptoms.append("chest_pain")
    if "urinary" in medical_meaning.body_parts and "burning_urination" in medical_meaning.symptoms and "burning_micturition" not in known_symptoms:
        known_symptoms.append("burning_micturition")
    current_medical_meaning = extract_medical_meaning(
        current_message,
        known_symptoms,
        language,
        denied_concepts,
    )
    history_medical_meaning = _latest_history_medical_meaning(
        recent_history,
        known_symptoms,
        language,
        denied_concepts,
    )
    medical_meaning = _active_medical_meaning(
        current_medical_meaning,
        history_medical_meaning,
        denied_concepts,
    )
    if medical_meaning.domain:
        medical_domains.add(medical_meaning.domain)
    if "fever" not in denied_concepts and has_any_normalized(combined_user_text, FEVER_CONTEXT_TERMS):
        if not {"high_fever", "mild_fever"}.intersection(known_symptoms):
            known_symptoms.append("mild_fever")
    if has_any_normalized(combined_user_text, BODY_ACHE_TERMS) and "muscle_pain" not in known_symptoms:
        known_symptoms.append("muscle_pain")
    temperature_c = None if "fever" in denied_concepts else extract_temperature_c(
        combined_user_text,
        assistant_history_text,
    )
    if temperature_c is not None and temperature_c >= 39.0 and "high_fever" not in known_symptoms:
        known_symptoms.append("high_fever")
    previous_diagnosis = diagnosis_mentioned_in_text(
        assistant_history_text,
        diagnosis_aliases,
    )
    state = ConversationState(
        intent=ConversationIntent.NEW_SYMPTOMS,
        language=language,
        current_message=current_message,
        user_history_text=user_history_text,
        assistant_history_text=assistant_history_text,
        current_symptoms=current_symptoms,
        known_symptoms=known_symptoms,
        medical_domains=medical_domains,
        denied_concepts=denied_concepts,
        denied_symptoms=denied_symptoms,
        temperature_c=temperature_c,
        has_high_temperature=temperature_c is not None and temperature_c >= 39.5,
        duration_known=has_duration_answer(combined_user_text),
        previous_diagnosis=previous_diagnosis,
        asked_temperature=asked_temperature(assistant_history_text),
        asked_duration=asked_duration(assistant_history_text),
        asked_red_flags=asked_red_flags(assistant_history_text),
        medical_meaning=medical_meaning,
        active_case=_build_active_case(
            medical_meaning,
            known_symptoms,
            denied_concepts,
            assistant_history_text,
        ),
    )
    state.intent = detect_intent(state, diagnosis_aliases)
    return state


def detect_language(current_message: str, history_text: str = "") -> str:
    text = current_message or ""
    if not text.strip():
        text = history_text or ""
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if latin > 0 and arabic == 0:
        return "en"
    if arabic > 0 and latin > 0:
        return "mixed"
    return "ar"


def detect_intent(
    state: ConversationState,
    diagnosis_aliases: dict[str, set[str]],
) -> str:
    message = state.current_message
    current_domains = detect_medical_domains(message)
    if is_profanity_or_abuse(message):
        return ConversationIntent.PROFANITY_OR_ABUSE
    if is_family_or_offtopic(message, state.current_symptoms, current_domains):
        return ConversationIntent.FAMILY_OR_OFFTOPIC
    if is_greeting_or_casual(message, state.current_symptoms):
        return ConversationIntent.GREETING_OR_CASUAL
    if is_non_medical_chat(message, state.current_symptoms):
        return ConversationIntent.NON_MEDICAL_CHAT
    if is_insult_or_frustration(message, state.current_symptoms):
        return ConversationIntent.INSULT_OR_FRUSTRATION
    if is_challenge_previous_diagnosis(message, state.previous_diagnosis):
        return ConversationIntent.CHALLENGE_PREVIOUS_DIAGNOSIS
    if is_ask_about_previous_diagnosis(message, state.previous_diagnosis, diagnosis_aliases):
        return ConversationIntent.ASK_ABOUT_PREVIOUS_DIAGNOSIS
    if state.denied_concepts and is_correction_or_negation(message):
        active_red_flag_reply = bool(
            state.medical_meaning.domain
            and state.asked_red_flags
            and state.denied_concepts.intersection({"numbness", "weakness", "bladder_bowel_loss"})
        )
        if not active_red_flag_reply:
            return ConversationIntent.CORRECTION_OR_NEGATION
    if is_answer_to_followup(message, state):
        return ConversationIntent.ANSWER_FOLLOWUP_QUESTION
    if "reproductive" in state.medical_domains:
        return ConversationIntent.REPRODUCTIVE_OR_GENDER_SENSITIVE_COMPLAINT
    if is_weak_body_ache_only(state.current_symptoms, message):
        return ConversationIntent.VAGUE_UNCLEAR
    if not state.current_symptoms and is_vague_general_message(message):
        return ConversationIntent.VAGUE_UNCLEAR
    if is_nonsense_or_low_information(message, state.current_symptoms, current_domains):
        return ConversationIntent.NONSENSE_OR_LOW_INFORMATION
    return ConversationIntent.NEW_SYMPTOMS


def extract_temperature_c(text: str, assistant_context: str = "") -> float | None:
    translated = translate_arabic_digits(text)
    normalized = normalize_text(translated)
    if not normalized:
        return None
    has_fever_context = has_any_normalized(translated, FEVER_CONTEXT_TERMS) or asked_temperature(
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


def translate_arabic_digits(text: str) -> str:
    translated = (text or "").translate(ARABIC_DIGIT_TRANSLATION)
    return re.sub(r"(?<=\d),(?=\d)", ".", translated)


def has_any_normalized(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in terms)


def detect_medical_domains(message: str) -> set[str]:
    normalized = normalize_text(message)
    if not normalized:
        return set()
    domains: set[str] = set()
    for domain, terms in MEDICAL_DOMAIN_TERMS.items():
        if any(normalize_text(term) in normalized for term in terms):
            domains.add(domain)
    return domains


def is_nonsense_or_low_information(
    message: str,
    current_symptoms: list[str],
    medical_domains: set[str] | None = None,
) -> bool:
    if current_symptoms or medical_domains:
        return False
    normalized = normalize_text(message)
    if not normalized:
        return True
    if not re.search(r"[A-Za-z\u0600-\u06FF]", normalized):
        return True
    words = normalized.split()
    if has_any_normalized(message, LOW_INFORMATION_OFFTOPIC_TERMS):
        return True
    repeated_noise = bool(re.search(r"([A-Za-z\u0600-\u06FF])\1{3,}", normalized))
    if repeated_noise and len(words) <= 5:
        return True
    if len(words) <= 3:
        return True
    if len(normalized) <= 16 and not any(char.isdigit() for char in normalized):
        return True
    return False


def asked_temperature(text: str) -> bool:
    return has_any_normalized(text, TEMPERATURE_QUESTION_TERMS)


def asked_duration(text: str) -> bool:
    return has_any_normalized(
        text,
        {
            "بدأت من إمتى",
            "بدات من امتى",
            "من إمتى بدأت",
            "من امتى بدأت",
            "بقالها قد إيه",
            "بقالها قد ايه",
            "منذ متى بدأت",
            "when did it start",
            "how long",
        },
    )


def asked_red_flags(text: str) -> bool:
    return has_any_normalized(
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
            "shortness of breath",
            "chest pain",
            "confusion",
            "dehydration",
            "\u062a\u0646\u0645\u064a\u0644",
            "\u0636\u0639\u0641",
            "\u0627\u0644\u0628\u0648\u0644",
            "\u0627\u0644\u0628\u0631\u0627\u0632",
            "\u0628\u064a\u0646\u0632\u0644 \u0639\u0644\u0649 \u0627\u0644\u0631\u062c\u0644",
            "numbness",
            "weakness",
            "bladder",
            "bowel",
        },
    )


def has_duration_answer(text: str) -> bool:
    return has_any_normalized(text, DURATION_TERMS)


def is_greeting_or_casual(message: str, current_symptoms: list[str]) -> bool:
    if current_symptoms:
        return False
    normalized = normalize_text(message)
    word_count = len(normalized.split())
    return word_count <= 14 and _has_greeting_or_casual_term(normalized)


def _has_greeting_or_casual_term(normalized_message: str) -> bool:
    for term in GREETING_OR_CASUAL_TERMS:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if re.fullmatch(r"[a-z]+", normalized_term) and len(normalized_term) <= 3:
            if re.search(rf"(?<![a-z]){re.escape(normalized_term)}(?![a-z])", normalized_message):
                return True
            continue
        if normalized_term in normalized_message:
            return True
    return False


def is_non_medical_chat(message: str, current_symptoms: list[str]) -> bool:
    if current_symptoms:
        return False
    return has_any_normalized(message, NON_MEDICAL_CHAT_TERMS)


def is_family_or_offtopic(
    message: str,
    current_symptoms: list[str],
    medical_domains: set[str] | None = None,
) -> bool:
    if current_symptoms or medical_domains:
        return False
    normalized = normalize_text(message)
    word_count = len(normalized.split())
    return word_count <= 12 and has_any_normalized(message, FAMILY_OR_OFFTOPIC_TERMS)


def _collapse_repeated_letters(text: str) -> str:
    return re.sub(r"([A-Za-z\u0600-\u06FF])\1{1,}", r"\1", text)


def is_profanity_or_abuse(message: str) -> bool:
    normalized_variants = {
        normalize_text(message),
        _collapse_repeated_letters(normalize_text(message)),
    }
    for term in PROFANITY_ABUSE_TERMS:
        normalized_term = normalize_text(term)
        term_variants = {normalized_term, _collapse_repeated_letters(normalized_term)}
        for candidate_term in term_variants:
            if not candidate_term:
                continue
            for candidate_message in normalized_variants:
                if " " in candidate_term:
                    if candidate_term in candidate_message:
                        return True
                    continue
                if re.search(
                    rf"(?<![A-Za-z0-9_\u0600-\u06FF]){re.escape(candidate_term)}(?![A-Za-z0-9_\u0600-\u06FF])",
                    candidate_message,
                ):
                    return True
    return False


def is_insult_or_frustration(message: str, current_symptoms: list[str]) -> bool:
    if has_any_normalized(message, FRUSTRATION_TERMS):
        return True
    if current_symptoms:
        return False
    normalized = normalize_text(message)
    return has_any_normalized(message, SOFT_FRUSTRATION_TERMS) and (
        "انت" in normalized or "انتي" in normalized or "الرد" in normalized or "كلام" in normalized
    )


def is_correction_or_negation(message: str) -> bool:
    return has_any_normalized(message, NEGATION_TERMS) or has_any_normalized(
        message,
        {"لا ده مش صح", "لا مش صح", "ده مش صح", "not what i said", "that is not right"},
    )


def is_ask_about_previous_diagnosis(
    message: str,
    previous_diagnosis: str | None,
    diagnosis_aliases: dict[str, set[str]],
) -> bool:
    normalized = normalize_text(message)
    if not has_any_normalized(message, DIAGNOSIS_QUESTION_TERMS):
        return False
    if previous_diagnosis and _aliases_in_text(normalized, diagnosis_aliases.get(previous_diagnosis, set())):
        return True
    if previous_diagnosis and has_any_normalized(
        message,
        {"التشخيص ده", "المرض ده", "الحالة دي", "الحاله دي", "اشرحلي المرض", "is it dangerous"},
    ):
        return True
    return any(_aliases_in_text(normalized, aliases) for aliases in diagnosis_aliases.values())


def is_challenge_previous_diagnosis(message: str, previous_diagnosis: str | None) -> bool:
    mentions_malaria = has_any_normalized(message, {"ملاريا", "الملاريا", "malaria"})
    if not previous_diagnosis and not mentions_malaria:
        return False
    return has_any_normalized(message, CHALLENGE_TERMS) or (
        (previous_diagnosis == "Malaria" or mentions_malaria)
        and {"travel", "mosquito"}.intersection(_detect_denied_concepts(message))
    )


def is_answer_to_followup(message: str, state: ConversationState) -> bool:
    if state.asked_temperature and extract_temperature_c(message, state.assistant_history_text) is not None:
        return True
    if state.asked_duration and has_duration_answer(message):
        return True
    if state.asked_red_flags and state.denied_concepts:
        return True
    normalized = normalize_text(message)
    if len(normalized.split()) <= 5 and (
        state.denied_concepts
        or has_any_normalized(message, {"اه", "ايوه", "لا", "no", "yes"})
    ):
        return state.asked_temperature or state.asked_duration or state.asked_red_flags
    return False


def is_weak_body_ache_only(symptoms: list[str], message: str) -> bool:
    symptom_set = set(symptoms)
    has_body_ache_text = has_any_normalized(message, BODY_ACHE_TERMS)
    weak_symptoms = {"muscle_pain", "fatigue", "malaise", "lethargy", "joint_pain"}
    specific_symptoms = symptom_set - weak_symptoms
    return has_body_ache_text and not specific_symptoms


def is_vague_general_message(message: str) -> bool:
    return has_any_normalized(message, VAGUE_GENERAL_TERMS)


def is_weak_nonspecific_case(symptoms: list[str], message: str, denied_concepts: set[str] | None = None) -> bool:
    denied_concepts = denied_concepts or set()
    symptom_set = set(symptoms)
    if is_weak_body_ache_only(symptoms, message):
        return True
    if symptom_set <= {"fatigue", "malaise", "lethargy"}:
        return True
    if symptom_set <= {"headache", "dizziness"} and not has_any_normalized(
        message,
        {"ضغط", "قياس الضغط", "blood pressure", "hypertension", "ضعف", "تنميل", "كلام", "speech"},
    ):
        return True
    return False


def rare_diagnosis_unsupported(
    diagnosis: str | None,
    symptoms: list[str],
    message: str,
    denied_concepts: set[str],
) -> bool:
    if not diagnosis or diagnosis not in RARE_OR_OVER_SPECIFIC_DIAGNOSES:
        return False
    if diagnosis == "Malaria":
        return not (
            {"high_fever", "mild_fever"}.intersection(symptoms)
            and {"chills", "shivering", "sweating"}.intersection(symptoms)
            and has_any_normalized(message, {"سفر", "ناموس", "بعوض", "travel", "mosquito"})
            and not {"travel", "mosquito"}.intersection(denied_concepts)
        )
    if diagnosis == "AIDS":
        return not has_any_normalized(message, AIDS_SUPPORT_TERMS)
    if diagnosis == "Hypertension":
        return not has_any_normalized(message, {"ضغط", "قياس الضغط", "blood pressure", "hypertension"})
    if diagnosis == "Pneumonia":
        return not (
            {"cough", "high_fever"}.issubset(set(symptoms))
            and (
                "breathlessness" in symptoms
                or has_any_normalized(message, {"ضيق تنفس", "ألم صدر", "chest pain", "shortness of breath"})
            )
        )
    return is_weak_nonspecific_case(symptoms, message, denied_concepts)


def diagnosis_mentioned_in_text(
    text: str,
    diagnosis_aliases: dict[str, set[str]],
) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    for diagnosis, aliases in diagnosis_aliases.items():
        if _aliases_in_text(normalized, aliases):
            return diagnosis
    return None


def _detect_denied_concepts(text: str) -> set[str]:
    normalized = normalize_text(text)
    denied: set[str] = set()
    for concept, terms in DENIED_CONCEPT_TERMS.items():
        for term in terms:
            normalized_term = normalize_text(term)
            start = 0
            while True:
                index = normalized.find(normalized_term, start)
                if index < 0:
                    break
                prefix = " ".join(normalized[:index].split()[-7:])
                if any(normalize_text(negation) in prefix for negation in NEGATION_TERMS):
                    denied.add(concept)
                start = index + len(normalized_term)
    return denied


def _symptoms_for_denied_concepts(denied_concepts: set[str]) -> set[str]:
    symptoms: set[str] = set()
    for concept in denied_concepts:
        symptoms.update(CONCEPT_TO_SYMPTOMS.get(concept, set()))
    return symptoms


def _aliases_in_text(normalized_text: str, aliases: Iterable[str]) -> bool:
    return any(normalize_text(alias) in normalized_text for alias in aliases)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
