from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from .config import Settings
from .safety import choose_urgency, detect_context_flags, normalize_text

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.descriptions: dict[str, str] = {}
        self.precautions: dict[str, list[str]] = {}
        self.severity: dict[str, int] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        path = self.settings.artifacts_dir / "medical_knowledge.pkl"
        if not path.exists():
            return

        try:
            data: dict[str, Any] = joblib.load(path)
            self.descriptions = data.get("descriptions", {})
            self.precautions = data.get("precautions", {})
            self.severity = data.get("severity", {})
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.warning("Could not load medical knowledge artifact: %s", exc)

    @property
    def ready(self) -> bool:
        return bool(self.severity or self.precautions or self.descriptions)

    def artifact_status(self) -> dict[str, bool]:
        return {"medical_knowledge.pkl": (self.settings.artifacts_dir / "medical_knowledge.pkl").exists()}

    def _lookup_key(self, diagnosis: str | None, source: dict[str, Any]) -> str | None:
        if not diagnosis:
            return None
        if diagnosis in source:
            return diagnosis
        normalized = normalize_text(diagnosis)
        for key in source:
            if normalize_text(key) == normalized:
                return key
        return None

    def get_description(self, diagnosis: str | None) -> str | None:
        key = self._lookup_key(diagnosis, self.descriptions)
        return self.descriptions.get(key) if key else None

    def get_precautions(self, diagnosis: str | None) -> list[str]:
        key = self._lookup_key(diagnosis, self.precautions)
        if not key:
            return []
        return [item for item in self.precautions.get(key, []) if item]

    def score_symptoms(self, symptoms: list[str], message: str) -> dict[str, int | str]:
        score = sum(int(self.severity.get(symptom, 0)) for symptom in symptoms)
        urgency = choose_urgency(message, symptoms, score)
        return {"score": score, "urgency": urgency}

    def suggest_doctor(self, diagnosis: str | None, symptoms: list[str], urgency: str, message: str = "") -> str:
        if urgency == "High":
            return "Emergency care"

        symptom_set = set(symptoms)
        context_flags = detect_context_flags(message)
        normalized_message = normalize_text(message)

        if "self_harm" in context_flags:
            return "Emergency care"
        if "pregnancy" in context_flags or "pregnancy_red_flag" in context_flags or "gynecology_context" in context_flags:
            return "Gynecologist"
        if "pediatric" in context_flags or "pediatric_red_flag" in context_flags:
            return "Pediatrician"
        if self._has_respiratory_uri_context(symptom_set, normalized_message):
            return "General Practitioner"
        if self._has_hypertension_context(normalized_message):
            return "Cardiologist"
        if self._has_chronic_respiratory_context(symptom_set, normalized_message):
            return "Pulmonologist"
        if "dental_context" in context_flags:
            return "Dentist"
        if "eye_context" in context_flags:
            return "Ophthalmologist"
        if "mental_health" in context_flags:
            return "Psychiatrist"
        if "trauma_context" in context_flags or "severe_trauma" in context_flags:
            return "Orthopedic doctor"
        if "urinary_retention" in context_flags or "urinary_context" in context_flags:
            return "Urologist"
        if "endocrine_context" in context_flags:
            return "Endocrinologist"
        if "ent_context" in context_flags and not {"weakness_of_one_body_side", "slurred_speech", "stiff_neck"}.intersection(
            symptom_set
        ):
            return "ENT specialist"
        if "infectious_context" in context_flags and not diagnosis:
            return "General Practitioner"

        if normalize_text(diagnosis or "") == normalize_text("Bronchial Asthma") and not self._has_asthma_context(
            symptom_set,
            normalized_message,
        ):
            return "General Practitioner"

        if normalize_text(diagnosis or "") == normalize_text("General medical evaluation"):
            return "General Practitioner"
        if normalize_text(diagnosis or "") in {
            normalize_text("Viral or flu-like illness"),
            normalize_text("General symptoms needing clarification"),
        }:
            return "General Practitioner"

        if {
            "burning_micturition",
            "bladder_discomfort",
            "continuous_feel_of_urine",
            "spotting_ urination",
            "foul_smell_of urine",
        }.intersection(symptom_set):
            return "Urologist"
        if {"skin_rash", "itching", "nodal_skin_eruptions", "pus_filled_pimples", "blackheads"}.intersection(symptom_set):
            return "Dermatologist"
        if {"abdominal_pain", "stomach_pain", "vomiting", "diarrhoea", "constipation", "acidity", "indigestion"}.intersection(
            symptom_set
        ) and not {"dizziness", "weakness_of_one_body_side", "slurred_speech"}.intersection(symptom_set):
            return "Gastroenterologist"
        if {
            "polyuria",
            "excessive_hunger",
            "irregular_sugar_level",
            "weight_loss",
            "weight_gain",
            "enlarged_thyroid",
        }.intersection(symptom_set):
            return "Endocrinologist"
        if {
            "loss_of_balance",
            "spinning_movements",
            "weakness_of_one_body_side",
            "slurred_speech",
            "stiff_neck",
        }.intersection(symptom_set):
            return "Neurologist"
        if "throat_irritation" in symptom_set and not {"breathlessness", "chest_pain", "phlegm"}.intersection(symptom_set):
            return "ENT specialist"
        if {"breathlessness", "phlegm", "chest_pain"}.intersection(symptom_set) and "cough" in symptom_set:
            return "Pulmonologist"

        text = normalize_text(diagnosis or " ".join(symptoms))
        mapping = [
            (["general medical evaluation"], "General Practitioner"),
            (["viral", "flu-like", "flu"], "General Practitioner"),
            (["common cold"], "General Practitioner"),
            (["asthma", "pneumonia", "tuberculosis", "bronchial"], "Pulmonologist"),
            (["heart", "chest pain", "palpitations", "hypertension", "varicose"], "Cardiologist"),
            (["diabetes", "thyroid", "hypoglycemia"], "Endocrinologist"),
            (["migraine", "paralysis", "dizziness", "vertigo", "cervical spondylosis", "spondylosis"], "Neurologist"),
            (["hepatitis", "jaundice", "cholestasis", "gerd", "ulcer", "gastro"], "Gastroenterologist"),
            (["urinary", "urine", "kidney"], "Urologist"),
            (["chicken pox"], "Dermatologist"),
            (["fungal", "acne", "psoriasis", "impetigo", "rash", "skin"], "Dermatologist"),
            (["allergy"], "Allergist"),
            (["dengue", "malaria", "typhoid"], "Infectious disease specialist"),
        ]
        for keywords, doctor in mapping:
            if any(keyword in text for keyword in keywords):
                return doctor
        return "General Practitioner"

    def _has_respiratory_uri_context(self, symptom_set: set[str], normalized_message: str) -> bool:
        uri_symptoms = {
            "runny_nose",
            "congestion",
            "continuous_sneezing",
            "watering_from_eyes",
            "throat_irritation",
        }
        if uri_symptoms.intersection(symptom_set) and not {"breathlessness", "chest_pain"}.intersection(symptom_set):
            return True
        return any(
            normalize_text(term) in normalized_message
            for term in {
                "رشح",
                "احتقان",
                "عطس",
                "منخيري مسدودة",
                "انسداد أنف",
                "انسداد انف",
            }
        ) and not any(
            normalize_text(term) in normalized_message
            for term in {
                "ضيق تنفس",
                "ألم صدر",
                "الم صدر",
            }
        )

    def _has_hypertension_context(self, normalized_message: str) -> bool:
        return any(
            normalize_text(term) in normalized_message
            for term in {
                "ضغط عالي",
                "ضغطي عالي",
                "ارتفاع الضغط",
                "ضغط الدم عالي",
            }
        )

    def _has_chronic_respiratory_context(self, symptom_set: set[str], normalized_message: str) -> bool:
        return "cough" in symptom_set and (
            {"weight_loss", "sweating", "blood_in_sputum"}.intersection(symptom_set)
            or any(
                normalize_text(term) in normalized_message
                for term in {
                    "عرق بالليل",
                    "تعرق بالليل",
                    "كحة بقالها شهر",
                    "كحه بقالها شهر",
                    "كحة مزمنة",
                    "كحه مزمنه",
                }
            )
        )

    def _has_asthma_context(self, symptom_set: set[str], normalized_message: str) -> bool:
        if {"breathlessness", "chest_pain"}.intersection(symptom_set):
            return True
        return any(
            normalize_text(term) in normalized_message
            for term in {
                "صفير",
                "ازيز",
                "أزيز",
                "ربو",
                "مزمن",
                "بالليل",
            }
        )

    def follow_up_questions(self, symptoms: list[str], confidence: float, urgency: str, message: str = "") -> list[str]:
        symptom_set = set(symptoms)
        context_flags = detect_context_flags(message)

        if urgency == "High":
            if "self_harm" in context_flags:
                return [
                    "هل أنت في مكان آمن الآن ومعك شخص قريب؟",
                    "هل يمكنك الاتصال بالطوارئ أو بشخص موثوق فورًا؟",
                ]
            if "pregnancy" in context_flags or "pregnancy_red_flag" in context_flags:
                return [
                    "هل النزيف أو الألم شديد أو مستمر؟",
                    "هل يوجد صداع شديد، زغللة، إغماء، أو ضغط مرتفع؟",
                ]
            if "pediatric" in context_flags or "pediatric_red_flag" in context_flags:
                return [
                    "هل الطفل يتنفس بشكل طبيعي؟",
                    "هل يوجد خمول شديد، تشنجات، أو توقف عن الرضاعة أو الشرب؟",
                ]
            if "severe_trauma" in context_flags or "trauma_context" in context_flags:
                return [
                    "هل النزيف مستمر أو الجرح عميق؟",
                    "هل الإصابة في الرأس أو معها قيء، دوخة شديدة، أو لخبطة؟",
                ]
            return [
                "هل الألم شديد ومستمر أو يزيد مع الوقت؟",
                "هل يوجد إغماء، تعرق شديد، زرقة في الشفاه، أو صعوبة في الكلام؟",
            ]

        if "pediatric" in context_flags:
            return [
                "عمر الطفل كام؟",
                "هل الطفل بيرضع أو بياكل ويشرب طبيعي؟",
                "هل في خمول شديد، تشنجات، صعوبة تنفس، أو قلة تبول؟",
                "درجة الحرارة وصلت كام وبدأت من إمتى؟",
            ]

        if "pregnancy" in context_flags:
            return [
                "الحمل في أي شهر تقريبًا؟",
                "هل يوجد نزيف مهبلي أو ألم شديد أسفل البطن؟",
                "هل يوجد صداع شديد، زغللة، تورم، أو ضغط مرتفع؟",
            ]

        if "mental_health" in context_flags:
            return [
                "منذ متى بدأ القلق أو الحزن أو اضطراب النوم؟",
                "هل توجد نوبات هلع أو خوف شديد؟",
                "هل لديك أفكار لإيذاء نفسك أو إنك مش عايز تعيش؟",
            ]

        if "dental_context" in context_flags:
            return [
                "الألم في سن ولا ضرس؟ ومنذ متى بدأ؟",
                "هل يوجد ورم في اللثة أو الوجه أو صعوبة في البلع؟",
                "هل توجد حرارة أو صديد أو نزيف؟",
            ]

        if "eye_context" in context_flags:
            return [
                "هل يوجد ألم شديد أو تغير مفاجئ في النظر؟",
                "هل الاحمرار في عين واحدة أم العينين؟",
                "هل توجد إفرازات أو حساسية شديدة من الضوء؟",
            ]

        if "trauma_context" in context_flags:
            return [
                "الإصابة حصلت إزاي ومنذ متى؟",
                "هل يوجد نزيف مستمر، جرح عميق، تشوه، أو عدم قدرة على الحركة؟",
                "هل كانت الإصابة في الرأس أو معها قيء أو لخبطة؟",
            ]

        if "urinary_context" in context_flags or {
            "burning_micturition",
            "bladder_discomfort",
            "continuous_feel_of_urine",
            "spotting_ urination",
        }.intersection(symptom_set):
            return [
                "هل في حرقان أو صعوبة أثناء التبول؟",
                "هل لاحظت دم في البول أو قلة شديدة في كمية البول؟",
                "هل يوجد ألم في الجنب أو الخاصرة أو حرارة؟",
            ]

        if "endocrine_context" in context_flags:
            return [
                "هل عندك سكر أو قست السكر مؤخرًا؟",
                "هل في عطش شديد أو تبول كتير؟",
                "هل في تعرق ورعشة وجوع شديد أو إغماء؟",
            ]

        if "ent_context" in context_flags:
            return [
                "هل الدوخة إحساس دوران ولا عدم اتزان؟",
                "هل يوجد طنين أو ألم أو انسداد في الأذن؟",
                "هل يوجد قيء، زغللة، صداع شديد، أو تنميل في ناحية من الجسم؟",
            ]

        if "infectious_context" in context_flags or "high_fever" in symptom_set:
            return [
                "الحرارة بقالها قد إيه؟",
                "هل قست درجة الحرارة؟ وصلت كام تقريبًا؟",
                "هل يوجد كحة، بلغم، ألم حلق، طفح، أو ألم في الجسم؟",
                "هل يوجد سفر قريب أو مخالطة لشخص عنده عدوى؟",
            ]

        if "cough" in symptom_set and ("high_fever" in symptom_set or "mild_fever" in symptom_set):
            return [
                "منذ كم يوم بدأت الكحة والحرارة؟",
                "كم وصلت درجة الحرارة تقريبًا؟",
                "هل يوجد ألم في الصدر أو ضيق في التنفس؟",
                "هل الكحة مصحوبة ببلغم أو صفير في الصدر؟",
            ]

        if {"headache", "dizziness"}.issubset(symptom_set):
            return [
                "هل يوجد قيء متكرر أو عدم اتزان شديد؟",
                "هل يوجد ألم أو انسداد أو طنين في الأذن؟",
                "هل يوجد زغللة، ضعف في النظر، أو تنميل في طرف من الجسم؟",
                "هل قست ضغط الدم مؤخرًا؟",
            ]

        if "skin_rash" in symptom_set or "itching" in symptom_set:
            return [
                "أين مكان الطفح؟ وهل ينتشر في مناطق أخرى؟",
                "هل ظهر بعد دواء جديد أو أكل معين أو منتج على الجلد؟",
                "هل يوجد تورم في الوجه أو الشفاه أو صعوبة تنفس؟",
                "هل الطفح مؤلم أو فيه إفرازات؟",
            ]

        if {"diarrhoea", "vomiting"}.intersection(symptom_set) and (
            "abdominal_pain" in symptom_set or "nausea" in symptom_set
        ):
            return [
                "منذ متى بدأ الإسهال أو الترجيع؟",
                "هل يوجد دم في البراز أو القيء؟",
                "هل توجد علامات جفاف مثل قلة البول أو دوخة شديدة؟",
                "هل توجد حرارة مرتفعة أو ألم شديد في البطن؟",
            ]

        if confidence >= 0.65 and len(symptoms) >= 3:
            return []

        return [
            "منذ متى بدأت الأعراض؟",
            "هل توجد أعراض شديدة مثل ألم في الصدر، ضيق تنفس، إغماء، أو نزيف؟",
            "هل لديك أمراض مزمنة أو تتناول أدوية حاليا؟",
        ]
