from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .arabic_symptom_dictionary import flatten_symptom_synonyms
from .config import Settings
from .safety import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class CandidatePrediction:
    diagnosis: str
    confidence: float


@dataclass
class PredictionResult:
    diagnosis: str | None
    confidence: float
    top_candidates: list[CandidatePrediction] = field(default_factory=list)
    classifier_diagnosis: str | None = None
    classifier_confidence: float = 0.0
    fusion_notes: list[str] = field(default_factory=list)


SYMPTOM_SYNONYMS: dict[str, list[str]] = {
    "cough": ["كحه", "كحة", "سعال", "cough"],
    "high_fever": ["سخونيه", "سخونية", "حراره عاليه", "حمى", "fever", "high fever"],
    "mild_fever": ["حراره بسيطه", "حمى خفيفه", "mild fever"],
    "fatigue": ["تعب", "ارهاق", "اجهاد", "fatigue", "tired"],
    "headache": ["صداع", "وجع راس", "headache"],
    "nausea": ["غثيان", "nausea"],
    "vomiting": ["ترجيع", "قيء", "استفراغ", "vomiting"],
    "diarrhoea": ["اسهال", "diarrhea", "diarrhoea"],
    "constipation": ["امساك", "constipation"],
    "abdominal_pain": ["الم بطن", "وجع بطن", "abdominal pain"],
    "stomach_pain": ["الم معده", "وجع معده", "stomach pain"],
    "chest_pain": ["الم صدر", "وجع صدر", "chest pain"],
    "breathlessness": ["ضيق تنفس", "صعوبه تنفس", "shortness of breath"],
    "sore_throat": ["التهاب حلق", "وجع حلق", "sore throat"],
    "throat_irritation": ["حرقان حلق", "تهيج الحلق", "throat irritation"],
    "runny_nose": ["رشح", "سيلان الانف", "runny nose"],
    "congestion": ["احتقان", "انسداد الانف", "congestion"],
    "chills": ["قشعريره", "رعشه", "chills"],
    "shivering": ["رعشه", "ارتعاش", "shivering"],
    "skin_rash": ["طفح جلدي", "حساسيه جلد", "rash"],
    "itching": ["حكه", "هرش", "itching"],
    "joint_pain": ["الم مفاصل", "وجع مفاصل", "joint pain"],
    "muscle_pain": ["الم عضلات", "وجع عضلات", "muscle pain"],
    "dizziness": ["دوخه", "دوار", "dizziness"],
    "loss_of_appetite": ["فقدان شهيه", "loss of appetite"],
    "weight_loss": ["نقص وزن", "weight loss"],
    "weight_gain": ["زياده وزن", "weight gain"],
    "burning_micturition": ["حرقان بول", "حرقان اثناء التبول", "burning urination"],
    "bladder_discomfort": ["الم مثانه", "bladder discomfort"],
    "continuous_feel_of_urine": ["كثرة التبول", "احساس مستمر بالتبول"],
    "back_pain": ["الم ظهر", "وجع ظهر", "back pain"],
    "neck_pain": ["الم رقبه", "وجع رقبه", "neck pain"],
    "acidity": ["حموضه", "حرقان معده", "acidity"],
    "indigestion": ["عسر هضم", "indigestion"],
    "dehydration": ["جفاف", "dehydration"],
    "sweating": ["تعرق", "sweating"],
    "palpitations": ["خفقان", "palpitations"],
    "blurred_and_distorted_vision": ["زغلله", "تشوش الرؤيه", "blurred vision"],
    "watering_from_eyes": ["دموع", "دموع العين", "watering eyes"],
    "redness_of_eyes": ["احمرار العين", "red eyes"],
    "yellowish_skin": ["اصفرار الجلد", "yellow skin"],
    "yellowing_of_eyes": ["اصفرار العين", "yellow eyes"],
}


def _merge_symptom_synonyms(*sources: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source in sources:
        for symptom, phrases in source.items():
            merged.setdefault(symptom, [])
            merged[symptom].extend(phrases)
    return {symptom: list(dict.fromkeys(phrases)) for symptom, phrases in merged.items()}


PHASE3_TARGETED_SYNONYMS: dict[str, list[str]] = {
    "weakness_of_one_body_side": [
        "تنميل في نص الجسم",
        "تنميل في نصف الجسم",
        "نص الجسم متنمل",
        "نصف الجسم متنمل",
    ],
    "burning_micturition": [
        "حرقان في البول",
        "حرقان اثناء البول",
        "حرقان لما بتبول",
    ],
    "continuous_feel_of_urine": [
        "بروح الحمام كتير",
        "بدخل الحمام كتير",
        "عايز ادخل الحمام كتير",
        "كثرة دخول الحمام",
    ],
}


PHASE4B_TARGETED_SYNONYMS: dict[str, list[str]] = {
    "high_fever": [
        "سخونية شديدة",
        "حرارة شديدة",
        "حرارة بقالها أيام",
        "حرارة بقالها ايام",
        "حرارة مستمرة",
        "طفح مع حرارة",
        "ألم حلق مع حرارة",
        "الم حلق مع حرارة",
    ],
    "chills": [
        "رعشة",
        "رعشة مع حرارة",
        "برد ورعشة",
    ],
    "muscle_pain": [
        "تكسير في الجسم",
        "تكسير بالجسم",
        "جسمي متكسر",
        "جسمي مكسر",
    ],
    "sweating": [
        "تعرق مع حرارة",
        "عرق مع حرارة",
        "عرق ورعشة",
    ],
    "cough": [
        "كحة مع بلغم",
        "كحه مع بلغم",
    ],
    "phlegm": [
        "كحة مع بلغم",
        "كحه مع بلغم",
        "بلغم كتير",
    ],
    "throat_irritation": [
        "ألم حلق مع حرارة",
        "الم حلق مع حرارة",
        "وجع حلق مع حرارة",
    ],
    "skin_rash": [
        "طفح مع حرارة",
        "طفح جلدي مع حرارة",
    ],
    "polyuria": [
        "تبول كتير",
        "بول كتير",
        "بدخل الحمام كتير",
    ],
    "continuous_feel_of_urine": [
        "بول كتير",
        "تبول كتير",
        "بدخل الحمام كل شوية",
        "بروح الحمام كل شوية",
        "صعوبة التبول",
    ],
    "dehydration": [
        "بول قليل",
        "البول قليل",
        "مش بتبول كويس",
        "مش بتبول كتير",
    ],
    "excessive_hunger": [
        "جوع شديد",
        "جعان جدا",
        "جوع جامد",
    ],
    "weight_loss": [
        "نقص وزن",
        "وزني بينزل",
        "بنزل في الوزن",
    ],
    "blurred_and_distorted_vision": [
        "زغللة",
        "زغلله",
        "زغللة مع سكر",
        "دوخة مع سكر",
    ],
    "irregular_sugar_level": [
        "هبوط سكر",
        "انخفاض سكر",
        "ارتفاع سكر",
        "السكر واطي",
        "السكر عالي",
        "دوخة مع سكر",
        "ريحة نفس غريبة",
        "رائحة نفس غريبة",
    ],
    "burning_micturition": [
        "حرقان بول",
        "حرقان في البول",
        "حرقان وانا بتبول",
    ],
    "spotting_ urination": [
        "دم في البول",
        "بول بدم",
        "البول فيه دم",
    ],
    "back_pain": [
        "ألم جنب",
        "الم جنب",
        "وجع جنب",
        "ألم في الخاصرة",
        "الم في الخاصرة",
        "وجع في الخاصرة",
        "مغص كلوي",
    ],
    "bladder_discomfort": [
        "احتباس بول",
        "صعوبة التبول",
        "مش عارف اتبول",
        "مش قادر اتبول",
    ],
    "dizziness": [
        "دوخة",
        "الدنيا بتلف",
        "الدنيا بتلف بيا",
        "قيء مع دوخة",
        "دوخة مع قيء",
    ],
    "spinning_movements": [
        "الدنيا بتلف",
        "الدنيا بتلف بيا",
        "لفان",
    ],
    "loss_of_balance": [
        "عدم اتزان",
        "مش متزن",
        "مش متوازن",
    ],
    "unsteadiness": [
        "عدم اتزان",
        "مش متزن",
    ],
    "weakness_in_limbs": [
        "تنميل",
        "تنميل في الأطراف",
        "تنميل في الاطراف",
    ],
    "weakness_of_one_body_side": [
        "ضعف في ناحية",
        "ضعف ناحية واحدة",
        "تنميل ناحية واحدة",
        "تنميل في ناحية",
    ],
    "stiff_neck": [
        "تيبس رقبة",
        "تيبس في الرقبة",
        "رقبة ناشفة",
    ],
    "redness_of_eyes": [
        "احمرار عين",
        "احمرار العين",
        "عيني حمرا",
    ],
    "visual_disturbances": [
        "مش شايف كويس فجأة",
        "تغير في النظر",
        "فقدان نظر",
    ],
    "abnormal_menstruation": [
        "تأخر الدورة",
        "تاخر الدورة",
        "الدورة متأخرة",
        "الدورة متاخره",
        "نزيف مهبلي",
    ],
    "anxiety": [
        "قلق",
        "توتر",
        "نوبات هلع",
        "خوف شديد",
    ],
    "depression": [
        "اكتئاب",
        "حزن شديد",
    ],
    "bruising": [
        "كدمة",
        "كدمات",
        "إصابة",
        "اصابة",
    ],
}


P0_TARGETED_SYNONYMS: dict[str, list[str]] = {
    "high_fever": [
        "سخونه",
        "سخونة",
        "سخونه جامدة",
        "سخونه شديدة",
    ],
    "muscle_pain": [
        "تكسير فجسمي",
        "تكسير في جسمي",
        "تكسير فالجسم",
    ],
    "chest_pain": [
        "صدري واجعني",
        "وجع في صدري",
        "وجع صدري",
        "مش وجع عادي في صدري",
    ],
    "sweating": [
        "عرق بارد",
        "تعرق بارد",
    ],
    "excessive_hunger": [
        "جوووع",
        "جوووع شديد",
        "جوع شديد",
    ],
    "increased_appetite": [
        "جوووع",
        "جوع شديد",
        "جوع جامد",
    ],
    "irregular_sugar_level": [
        "سكر عالي",
        "سكر واطي",
        "سكر منخفض",
    ],
    "altered_sensorium": [
        "لخبطة",
        "لخبطه",
        "مش مركز",
        "مش قادر اركز",
        "تدهور سريع",
    ],
    "coma": [
        "فقدت الوعي",
        "فقد وعي",
        "اغمى عليا",
        "اغمي عليا",
    ],
    "vomiting": [
        "بيستفرغ",
        "بستفرغ",
        "استفراغ",
    ],
    "dehydration": [
        "قلة بول",
        "قله بول",
        "بول قليل",
        "مش قادر اشرب",
        "مش قادر اشرب مياه",
    ],
    "back_pain": [
        "مغص كلوي شديد",
    ],
    "visual_disturbances": [
        "فقدت النظر فجأة",
        "فقدان النظر فجأة",
    ],
    "abnormal_menstruation": [
        "bleeding",
        "pregnant ومعايا bleeding",
    ],
    "dark_urine": [
        "بولي غامق",
        "بول غامق",
        "بول داكن",
    ],
    "yellowish_skin": [
        "جسمي مصفر",
        "جسمي اصفر",
        "جلدي مصفر",
    ],
    "yellowing_of_eyes": [
        "عيني صفرا",
        "العين صفرا",
    ],
}


SYMPTOM_SYNONYMS = _merge_symptom_synonyms(
    SYMPTOM_SYNONYMS,
    flatten_symptom_synonyms(),
    PHASE3_TARGETED_SYNONYMS,
    PHASE4B_TARGETED_SYNONYMS,
    P0_TARGETED_SYNONYMS,
)


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
    "no",
    "without",
}


DISEASE_SYMPTOM_HINTS: dict[str, set[str]] = {
    "Common Cold": {
        "cough",
        "high_fever",
        "mild_fever",
        "fatigue",
        "headache",
        "runny_nose",
        "congestion",
        "throat_irritation",
        "phlegm",
        "chills",
    },
    "Bronchial Asthma": {
        "cough",
        "breathlessness",
        "chest_pain",
    },
    "Pneumonia": {
        "cough",
        "high_fever",
        "fatigue",
        "breathlessness",
        "phlegm",
        "chest_pain",
        "chills",
    },
    "Tuberculosis": {
        "cough",
        "high_fever",
        "mild_fever",
        "fatigue",
        "breathlessness",
        "phlegm",
        "chest_pain",
        "vomiting",
    },
    "Gastroenteritis": {
        "vomiting",
        "diarrhoea",
        "abdominal_pain",
        "nausea",
        "dehydration",
    },
    "Heart attack": {
        "chest_pain",
        "breathlessness",
        "sweating",
        "vomiting",
    },
    "Hypertension": {
        "headache",
        "dizziness",
        "chest_pain",
        "palpitations",
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "dizziness",
        "nausea",
        "headache",
        "loss_of_balance",
        "spinning_movements",
    },
    "Migraine": {
        "headache",
        "nausea",
        "vomiting",
        "visual_disturbances",
        "blurred_and_distorted_vision",
    },
    "Fungal infection": {
        "itching",
        "skin_rash",
        "nodal_skin_eruptions",
        "dischromic _patches",
    },
    "Allergy": {
        "itching",
        "skin_rash",
        "continuous_sneezing",
        "watering_from_eyes",
        "breathlessness",
    },
    "Hypoglycemia": {
        "sweating",
        "shivering",
        "chills",
        "excessive_hunger",
        "dizziness",
        "coma",
        "irregular_sugar_level",
    },
    "Diabetes": {
        "polyuria",
        "weight_loss",
        "fatigue",
        "increased_appetite",
        "irregular_sugar_level",
        "excessive_hunger",
    },
    "Hypothyroidism": {
        "weight_gain",
        "fatigue",
        "cold_hands_and_feets",
        "puffy_face_and_eyes",
        "brittle_nails",
        "enlarged_thyroid",
        "lethargy",
    },
    "Hyperthyroidism": {
        "weight_loss",
        "sweating",
        "fast_heart_rate",
        "palpitations",
        "increased_appetite",
        "irregular_sugar_level",
    },
    "Urinary tract infection": {
        "burning_micturition",
        "bladder_discomfort",
        "continuous_feel_of_urine",
        "foul_smell_of urine",
        "spotting_ urination",
        "back_pain",
        "polyuria",
        "dark_urine",
    },
    "Dengue": {
        "high_fever",
        "headache",
        "pain_behind_the_eyes",
        "red_spots_over_body",
        "skin_rash",
        "muscle_pain",
    },
    "Malaria": {
        "high_fever",
        "chills",
        "sweating",
        "headache",
        "muscle_pain",
    },
    "Typhoid": {
        "high_fever",
        "abdominal_pain",
        "diarrhoea",
        "loss_of_appetite",
        "headache",
        "fatigue",
    },
    "Jaundice": {
        "yellowish_skin",
        "yellowing_of_eyes",
        "dark_urine",
        "yellow_urine",
        "fatigue",
    },
    "General medical evaluation": {
        "fatigue",
        "weight_loss",
        "loss_of_appetite",
        "malaise",
        "lethargy",
        "anxiety",
        "restlessness",
        "depression",
        "lack_of_concentration",
    },
}


DISEASE_DOMAINS: dict[str, set[str]] = {
    "Common Cold": {"respiratory", "ent", "general"},
    "Bronchial Asthma": {"respiratory", "general"},
    "Pneumonia": {"respiratory", "general"},
    "Tuberculosis": {"respiratory", "general"},
    "Gastroenteritis": {"digestive", "general"},
    "Peptic ulcer diseae": {"digestive", "general"},
    "Heart attack": {"cardiac", "emergency", "general"},
    "Hypertension": {"cardiac", "general"},
    "Hypoglycemia": {"endocrine", "general"},
    "Diabetes": {"endocrine", "general"},
    "Hypothyroidism": {"endocrine", "general"},
    "Hyperthyroidism": {"endocrine", "general"},
    "Urinary tract infection": {"urinary", "general"},
    "Dengue": {"infectious", "general"},
    "Malaria": {"infectious", "general"},
    "Typhoid": {"infectious", "general"},
    "Jaundice": {"digestive", "general"},
    "General medical evaluation": {"general"},
    "(vertigo) Paroymsal  Positional Vertigo": {"neuro", "ent", "general"},
    "Migraine": {"neuro", "general"},
    "Fungal infection": {"dermatology", "general"},
    "Allergy": {"dermatology", "ent", "general"},
}


SYMPTOM_MEDICAL_TERMS: dict[str, set[str]] = {
    "cough": {"cough", "respiratory", "throat", "lungs", "mucus"},
    "high_fever": {"fever", "infection", "viral", "bacterial"},
    "mild_fever": {"fever", "infection", "viral", "bacterial"},
    "fatigue": {"fatigue", "tired", "weakness", "rest"},
    "headache": {"headache", "head", "migraine"},
    "dizziness": {"dizziness", "vertigo", "spinning"},
    "nausea": {"nausea", "vomiting"},
    "vomiting": {"vomiting", "nausea"},
    "diarrhoea": {"diarrhea", "diarrhoea", "intestinal", "digestive"},
    "abdominal_pain": {"abdominal", "stomach", "cramps", "digestive"},
    "skin_rash": {"skin", "rash", "area"},
    "itching": {"itching", "skin", "calamine"},
    "chest_pain": {"chest", "heart", "cardiac"},
    "breathlessness": {"breathe", "breathing", "breath", "lungs", "airway"},
}


ASTHMA_CONTEXT_TERMS = {
    "asthma",
    "wheeze",
    "wheezing",
    "ربو",
    "صفير",
    "ازيز",
    "أزيز",
    "مزمن",
}


HYPERTENSION_CONTEXT_TERMS = {
    "hypertension",
    "blood pressure",
    "high blood pressure",
    "ضغط",
    "ضغط الدم",
    "ارتفاع الضغط",
}


def _clean_symptom_name(symptom: str) -> str:
    return symptom.replace("_", " ").replace("  ", " ").strip()


class ClassifierService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.label_encoder = None
        self.symptom_columns: list[str] = []
        self._column_index: dict[str, int] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        artifacts_dir = self.settings.artifacts_dir
        try:
            classifier_path = artifacts_dir / "disease_classifier.pkl"
            if classifier_path.exists():
                self.model = joblib.load(classifier_path)

            encoder_path = artifacts_dir / "disease_label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)

            columns_path = artifacts_dir / "symptom_columns.pkl"
            if columns_path.exists():
                self.symptom_columns = list(joblib.load(columns_path))
                self._column_index = {name: idx for idx, name in enumerate(self.symptom_columns)}
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.warning("Could not load classifier artifacts: %s", exc)

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.symptom_columns)

    def artifact_status(self) -> dict[str, bool]:
        artifacts_dir = self.settings.artifacts_dir
        return {
            "disease_classifier.pkl": (artifacts_dir / "disease_classifier.pkl").exists(),
            "disease_label_encoder.pkl": (artifacts_dir / "disease_label_encoder.pkl").exists(),
            "symptom_columns.pkl": (artifacts_dir / "symptom_columns.pkl").exists(),
        }

    def extract_symptoms(self, message: str, history_text: str = "") -> list[str]:
        text = normalize_text(f"{history_text} {message}")
        candidates = self.symptom_columns or sorted(SYMPTOM_SYNONYMS)
        extracted: list[str] = []

        for symptom in candidates:
            phrases = [_clean_symptom_name(symptom), symptom]
            phrases.extend(SYMPTOM_SYNONYMS.get(symptom, []))

            normalized_phrases = {normalize_text(phrase) for phrase in phrases if phrase}
            if any(
                self._phrase_matches(text, phrase) and not self._is_negated_phrase(text, phrase)
                for phrase in normalized_phrases
            ):
                extracted.append(symptom)

        return list(dict.fromkeys(extracted))

    def _phrase_matches(self, normalized_text: str, normalized_phrase: str) -> bool:
        if not normalized_phrase:
            return False
        if " " not in normalized_phrase and len(normalized_phrase) <= 2:
            return normalized_phrase in normalized_text.split()
        return normalized_phrase in normalized_text

    def _is_negated_phrase(self, normalized_text: str, normalized_phrase: str) -> bool:
        match_index = normalized_text.find(normalized_phrase)
        if match_index < 0:
            return False

        prefix = normalized_text[:match_index].split()
        window = " ".join(prefix[-5:])
        return any(normalize_text(term) in window for term in NEGATION_TERMS)

    def _vectorize(self, symptoms: Iterable[str]) -> pd.DataFrame:
        vector = np.zeros((1, len(self.symptom_columns)), dtype=np.int8)
        for symptom in symptoms:
            index = self._column_index.get(symptom)
            if index is not None:
                vector[0, index] = 1
        return pd.DataFrame(vector, columns=self.symptom_columns)

    def predict(self, symptoms: list[str]) -> PredictionResult:
        if not self.ready or not symptoms:
            return PredictionResult(diagnosis=None, confidence=0.0)

        vector = self._vectorize(symptoms)
        if int(vector.to_numpy().sum()) == 0:
            return PredictionResult(diagnosis=None, confidence=0.0)

        if hasattr(self.model, "predict_proba"):
            top_candidates = self.predict_top_k(symptoms, k=5)
            if not top_candidates:
                return PredictionResult(diagnosis=None, confidence=0.0)
            diagnosis = top_candidates[0].diagnosis
            confidence = top_candidates[0].confidence
        else:
            raw_label = self.model.predict(vector)[0]
            diagnosis = self._decode_label(raw_label)
            confidence = 0.0
            top_candidates = [CandidatePrediction(diagnosis=diagnosis, confidence=confidence)]

        return PredictionResult(
            diagnosis=diagnosis,
            confidence=round(confidence, 4),
            top_candidates=top_candidates,
            classifier_diagnosis=diagnosis,
            classifier_confidence=round(confidence, 4),
        )

    def predict_top_k(self, symptoms: list[str], k: int = 5) -> list[CandidatePrediction]:
        if not self.ready or not symptoms or not hasattr(self.model, "predict_proba"):
            return []

        vector = self._vectorize(symptoms)
        if int(vector.to_numpy().sum()) == 0:
            return []

        probabilities = self.model.predict_proba(vector)[0]
        candidates: list[CandidatePrediction] = []
        for idx in probabilities.argsort()[::-1][:k]:
            raw_label = self.model.classes_[int(idx)]
            candidates.append(
                CandidatePrediction(
                    diagnosis=self._decode_label(raw_label),
                    confidence=round(float(probabilities[int(idx)]), 4),
                )
            )
        return candidates

    def fuse_prediction(
        self,
        prediction: PredictionResult,
        symptoms: list[str],
        message: str,
        severity_score: int,
        retrieved_cases: list[dict],
        descriptions: dict[str, str],
        precautions: dict[str, list[str]],
    ) -> PredictionResult:
        if not prediction.diagnosis:
            return prediction

        symptom_set = set(symptoms)
        normalized_message = normalize_text(message)
        top_candidates = prediction.top_candidates or [
            CandidatePrediction(prediction.diagnosis, prediction.confidence)
        ]
        candidate_scores = {candidate.diagnosis: candidate.confidence for candidate in top_candidates}
        notes: list[str] = []

        guardrail_candidates = self._guardrail_candidates(symptom_set, normalized_message)
        for diagnosis, boost, note in guardrail_candidates:
            candidate_scores[diagnosis] = max(candidate_scores.get(diagnosis, 0.0), boost)
            notes.append(note)

        rag_domains = self._rag_domain_scores(retrieved_cases)
        for diagnosis in list(candidate_scores):
            candidate_scores[diagnosis] += self._symptom_profile_score(diagnosis, symptom_set)
            candidate_scores[diagnosis] += self._knowledge_text_score(
                diagnosis,
                symptom_set,
                descriptions,
                precautions,
            )
            candidate_scores[diagnosis] += self._rag_support_score(diagnosis, rag_domains)
            candidate_scores[diagnosis] += self._severity_support_score(diagnosis, severity_score)
            candidate_scores[diagnosis] += self._sanity_adjustment(diagnosis, symptom_set, normalized_message)

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        final_diagnosis = ranked[0][0]
        final_score = ranked[0][1]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        raw_prob = next(
            (candidate.confidence for candidate in top_candidates if candidate.diagnosis == final_diagnosis),
            0.0,
        )
        confidence = self._fused_confidence(final_score, runner_up_score, raw_prob)

        return PredictionResult(
            diagnosis=final_diagnosis,
            confidence=confidence,
            top_candidates=top_candidates,
            classifier_diagnosis=prediction.classifier_diagnosis or prediction.diagnosis,
            classifier_confidence=prediction.classifier_confidence or prediction.confidence,
            fusion_notes=notes,
        )

    def _guardrail_candidates(
        self,
        symptom_set: set[str],
        normalized_message: str,
    ) -> list[tuple[str, float, str]]:
        candidates: list[tuple[str, float, str]] = []

        has_respiratory_basic = (
            "cough" in symptom_set
            and ("high_fever" in symptom_set or "mild_fever" in symptom_set)
            and (
                "fatigue" in symptom_set
                or "muscle_pain" in symptom_set
                or "throat_irritation" in symptom_set
                or "runny_nose" in symptom_set
                or "congestion" in symptom_set
            )
        )
        asthma_specific = self._has_asthma_specific_context(symptom_set, normalized_message)
        if has_respiratory_basic and not asthma_specific:
            candidates.append(
                (
                    "Common Cold",
                    0.68,
                    "respiratory fever/fatigue pattern without asthma-specific evidence",
                )
            )

        upper_airway_pattern = bool(
            {"runny_nose", "congestion", "continuous_sneezing", "watering_from_eyes", "throat_irritation"}.intersection(
                symptom_set
            )
        ) and not {"skin_rash", "itching", "breathlessness"}.intersection(symptom_set)
        if upper_airway_pattern:
            candidates.append(("Common Cold", 0.62, "upper-airway URI/allergic-rhinitis-like symptom guardrail"))

        if {"chest_pain", "breathlessness"}.issubset(symptom_set):
            candidates.append(("Heart attack", 0.82, "chest pain with breathlessness emergency guardrail"))

        allergic_airway_pattern = (
            "breathlessness" in symptom_set
            and {"skin_rash", "itching"}.intersection(symptom_set)
            and self._has_allergic_airway_context(normalized_message)
        )
        if allergic_airway_pattern:
            candidates.append(("Allergy", 0.86, "rash or itching with airway/lip swelling allergy emergency guardrail"))

        hypoglycemia_pattern = (
            "excessive_hunger" in symptom_set
            and (
                "sweating" in symptom_set
                or "irregular_sugar_level" in symptom_set
                or {"shivering", "chills", "fatigue", "dizziness", "coma"}.intersection(symptom_set)
            )
            and {"shivering", "chills", "fatigue", "anxiety", "dizziness", "coma", "irregular_sugar_level"}.intersection(
                symptom_set
            )
        )
        if hypoglycemia_pattern:
            candidates.append(("Hypoglycemia", 0.72, "sweating with tremor/chills and hunger endocrine guardrail"))

        diabetes_pattern = (
            "polyuria" in symptom_set
            and {"weight_loss", "increased_appetite", "irregular_sugar_level", "fatigue", "excessive_hunger"}.intersection(
                symptom_set
            )
        ) or ("irregular_sugar_level" in symptom_set and {"dizziness", "polyuria", "excessive_hunger"}.intersection(symptom_set))
        if diabetes_pattern:
            candidates.append(("Diabetes", 0.64, "polyuria or sugar-level symptoms endocrine guardrail"))
        if "irregular_sugar_level" in symptom_set and {"blurred_and_distorted_vision", "visual_disturbances"}.intersection(
            symptom_set
        ):
            candidates.append(("Diabetes", 0.66, "sugar-level symptoms with vision change endocrine guardrail"))

        hyperthyroid_pattern = {"weight_loss", "sweating", "fast_heart_rate"}.issubset(symptom_set) or (
            "weight_loss" in symptom_set and {"sweating", "palpitations", "fast_heart_rate"}.intersection(symptom_set)
        )
        if hyperthyroid_pattern:
            candidates.append(("Hyperthyroidism", 0.66, "weight loss with sweating and fast heart endocrine guardrail"))

        hypothyroid_pattern = (
            "weight_gain" in symptom_set
            and {"fatigue", "cold_hands_and_feets", "brittle_nails", "puffy_face_and_eyes"}.intersection(symptom_set)
        ) or (
            "enlarged_thyroid" in symptom_set
            or {"puffy_face_and_eyes", "fatigue"}.issubset(symptom_set)
        )
        if hypothyroid_pattern:
            candidates.append(("Hypothyroidism", 0.66, "weight gain/fatigue/thyroid swelling endocrine guardrail"))

        urinary_core = {
            "burning_micturition",
            "bladder_discomfort",
            "continuous_feel_of_urine",
            "foul_smell_of urine",
            "spotting_ urination",
        }
        urinary_pattern = bool(urinary_core.intersection(symptom_set)) and not self._has_liver_specific_evidence(
            symptom_set - {"dark_urine", "yellow_urine"},
            normalized_message,
        )
        if urinary_pattern or (
            {"burning_micturition", "dark_urine"}.issubset(symptom_set)
            and not {"yellowing_of_eyes", "yellowish_skin"}.intersection(symptom_set)
        ):
            candidates.append(("Urinary tract infection", 0.68, "urinary symptom cluster guardrail"))

        fever_present = "high_fever" in symptom_set or "mild_fever" in symptom_set
        if fever_present and {"headache", "pain_behind_the_eyes"}.issubset(symptom_set):
            candidates.append(("Dengue", 0.64, "fever with headache and pain behind eyes infectious guardrail"))
        if fever_present and {"red_spots_over_body", "skin_rash"}.intersection(symptom_set):
            candidates.append(("Dengue", 0.52, "fever with rash/red spots infectious guardrail"))
        travel_or_malaria_context = any(
            term in normalized_message
            for term in {
                normalize_text("بعد سفر"),
                normalize_text("سفر"),
                normalize_text("عرق بالليل"),
                normalize_text("تعرق بالليل"),
            }
        )
        respiratory_infection_context = {"cough", "phlegm", "breathlessness"}.intersection(symptom_set)
        if fever_present and {"chills", "sweating"}.intersection(symptom_set) and (
            travel_or_malaria_context or not respiratory_infection_context
        ):
            candidates.append(("Malaria", 0.60, "fever with chills/sweating infectious guardrail"))
        typhoid_context = (
            {"diarrhoea", "headache"}.intersection(symptom_set)
            or any(term in normalized_message for term in {normalize_text("مستمرة"), normalize_text("بعد سفر")})
        )
        if fever_present and {"abdominal_pain", "diarrhoea", "loss_of_appetite"}.intersection(symptom_set) and typhoid_context:
            candidates.append(("Typhoid", 0.58, "fever with abdominal/digestive infectious guardrail"))
        if "cough" in symptom_set and {"weight_loss", "sweating", "blood_in_sputum"}.intersection(symptom_set):
            candidates.append(("Tuberculosis", 0.80, "cough with weight loss/night sweat or blood sputum guardrail"))

        cardiovascular_palpitations_pattern = (
            {"palpitations", "dizziness"}.issubset(symptom_set)
            and "excessive_hunger" not in symptom_set
            and "sweating" not in symptom_set
        )
        if cardiovascular_palpitations_pattern:
            candidates.append(("Hypertension", 0.54, "palpitations with dizziness cardiovascular guardrail"))
        if self._has_hypertension_context(normalized_message) and {
            "headache",
            "dizziness",
            "blurred_and_distorted_vision",
            "visual_disturbances",
        }.intersection(symptom_set):
            candidates.append(("Hypertension", 0.66, "hypertension context with neurologic/vision symptoms guardrail"))

        vestibular_pattern = (
            "dizziness" in symptom_set
            and {"loss_of_balance", "spinning_movements"}.intersection(symptom_set)
            and {"vomiting", "nausea"}.intersection(symptom_set)
            and not (
                self._has_hypertension_context(normalized_message)
                or {"chest_pain", "palpitations"}.intersection(symptom_set)
            )
        )
        if vestibular_pattern:
            candidates.append(
                (
                    "(vertigo) Paroymsal  Positional Vertigo",
                    0.66,
                    "dizziness/imbalance/vomiting vestibular guardrail",
                )
            )
            candidates.append(("Migraine", 0.32, "vestibular symptoms neurologic alternative"))

        vague_constitutional_pattern = {"fatigue", "weight_loss", "loss_of_appetite"}.issubset(symptom_set)
        if vague_constitutional_pattern and not self._has_liver_specific_evidence(symptom_set, normalized_message):
            candidates.append(
                (
                    "General medical evaluation",
                    0.48,
                    "fatigue/appetite/weight loss pattern without liver-specific evidence",
                )
            )

        general_vague_symptoms = {
            "fatigue",
            "malaise",
            "lethargy",
            "anxiety",
            "restlessness",
            "depression",
            "lack_of_concentration",
            "bruising",
        }
        has_vague_general_pattern = bool(general_vague_symptoms.intersection(symptom_set)) and not bool(
            {
                "high_fever",
                "breathlessness",
                "chest_pain",
                "cough",
                "phlegm",
                "throat_irritation",
                "runny_nose",
                "congestion",
                "burning_micturition",
                "polyuria",
                "skin_rash",
                "itching",
                "yellowing_of_eyes",
                "yellowish_skin",
                "dark_urine",
            }.intersection(symptom_set)
        )
        if has_vague_general_pattern and not self._has_hypertension_context(normalized_message):
            candidates.append(("General medical evaluation", 0.52, "vague general symptoms without organ-specific evidence"))

        constitutional_two_of_three = sum(
            symptom in symptom_set for symptom in ("fatigue", "weight_loss", "loss_of_appetite")
        ) >= 2
        if constitutional_two_of_three and not self._has_liver_specific_evidence(symptom_set, normalized_message):
            candidates.append(("General medical evaluation", 0.54, "constitutional symptoms without liver-specific evidence"))

        non_respiratory_fever_aches = (
            {"joint_pain", "fatigue", "high_fever"}.issubset(symptom_set)
            and "cough" not in symptom_set
            and "breathlessness" not in symptom_set
        )
        if non_respiratory_fever_aches:
            candidates.append(("General medical evaluation", 0.46, "fever/fatigue/joint pain without respiratory evidence"))

        if {"vomiting", "diarrhoea"}.issubset(symptom_set) and (
            "abdominal_pain" in symptom_set or "nausea" in symptom_set
        ):
            candidates.append(("Gastroenteritis", 0.62, "digestive triad guardrail"))

        if {"itching", "skin_rash"}.issubset(symptom_set):
            candidates.append(("Fungal infection", 0.58, "rash with itching dermatology guardrail"))
            candidates.append(("Allergy", 0.38, "rash with itching allergy alternative"))

        neuro_triage = {"headache", "dizziness", "nausea"}.issubset(symptom_set)
        hypertension_specific = self._has_hypertension_context(normalized_message) or bool(
            {"chest_pain", "palpitations"}.intersection(symptom_set)
        )
        if neuro_triage and not hypertension_specific:
            candidates.append(
                (
                    "(vertigo) Paroymsal  Positional Vertigo",
                    0.44,
                    "headache/dizziness/nausea pattern without hypertension-specific evidence",
                )
            )
            candidates.append(("Migraine", 0.34, "headache with nausea neurologic alternative"))

        return candidates

    def _has_allergic_airway_context(self, normalized_message: str) -> bool:
        terms = {
            "تورم الشفاه",
            "تورم في الشفاه",
            "تورم الوجه",
            "تورم في الوجه",
            "حساسية",
            "بعد اكل",
            "بعد أكل",
            "بعد دواء",
            "بعد دوا",
            "اختناق",
            "صعوبة تنفس",
            "ضيق تنفس",
        }
        return any(normalize_text(term) in normalized_message for term in terms)

    def _has_liver_specific_evidence(self, symptom_set: set[str], normalized_message: str) -> bool:
        liver_symptoms = {
            "yellowish_skin",
            "yellowing_of_eyes",
            "dark_urine",
            "yellow_urine",
            "acute_liver_failure",
        }
        if liver_symptoms.intersection(symptom_set):
            return True
        liver_terms = {
            "اصفرار",
            "صفار",
            "بول غامق",
            "بول داكن",
            "عين صفرا",
            "العين صفرا",
            "جلد اصفر",
            "الكبد",
        }
        return any(normalize_text(term) in normalized_message for term in liver_terms)

    def _has_asthma_specific_context(self, symptom_set: set[str], normalized_message: str) -> bool:
        if "breathlessness" in symptom_set:
            return True
        return any(normalize_text(term) in normalized_message for term in ASTHMA_CONTEXT_TERMS)

    def _has_hypertension_context(self, normalized_message: str) -> bool:
        return any(normalize_text(term) in normalized_message for term in HYPERTENSION_CONTEXT_TERMS)

    def _symptom_profile_score(self, diagnosis: str, symptom_set: set[str]) -> float:
        hints = DISEASE_SYMPTOM_HINTS.get(diagnosis)
        if not hints:
            return 0.0
        matches = len(symptom_set.intersection(hints))
        if not matches:
            return 0.0
        coverage = matches / max(len(symptom_set), 1)
        specificity = matches / len(hints)
        return 0.22 * coverage + 0.10 * specificity

    def _knowledge_text_score(
        self,
        diagnosis: str,
        symptom_set: set[str],
        descriptions: dict[str, str],
        precautions: dict[str, list[str]],
    ) -> float:
        text = f"{descriptions.get(diagnosis, '')} {' '.join(precautions.get(diagnosis, []))}".lower()
        if not text:
            return 0.0

        matched = 0
        for symptom in symptom_set:
            terms = SYMPTOM_MEDICAL_TERMS.get(symptom, set())
            if any(term in text for term in terms):
                matched += 1
        return min(0.12, matched * 0.035)

    def _rag_domain_scores(self, retrieved_cases: list[dict]) -> dict[str, float]:
        domains: dict[str, float] = {}
        for rank, case in enumerate(retrieved_cases[:5]):
            weight = 1.0 / (rank + 1)
            text = normalize_text(f"{case.get('category', '')} {case.get('q_body', '')} {case.get('a_body', '')}")
            for domain, keywords in {
                "respiratory": ["تنفسي", "التنفسي", "صدر", "كحه", "كحة", "سعال", "ضيق تنفس", "بلغم"],
                "ent": ["انف", "اذن", "حنجره", "حلق", "رشح", "جيوب"],
                "digestive": ["هضمي", "باطنيه", "باطنية", "معده", "معدة", "بطن", "اسهال", "قيء", "غثيان"],
                "dermatology": ["جلديه", "جلدية", "جلد", "طفح", "حكه", "حكة"],
                "neuro": ["عصبيه", "عصبية", "صداع", "دوخه", "دوخة", "دوار"],
                "cardiac": ["قلب", "صدر", "خفقان"],
                "general": ["الطب العام"],
            }.items():
                if any(normalize_text(keyword) in text for keyword in keywords):
                    domains[domain] = domains.get(domain, 0.0) + weight

        if not domains:
            return domains
        max_score = max(domains.values())
        return {domain: score / max_score for domain, score in domains.items()}

    def _rag_support_score(self, diagnosis: str, rag_domains: dict[str, float]) -> float:
        domains = DISEASE_DOMAINS.get(diagnosis, set())
        if not domains or not rag_domains:
            return 0.0
        return min(0.12, sum(rag_domains.get(domain, 0.0) for domain in domains) * 0.04)

    def _severity_support_score(self, diagnosis: str, severity_score: int) -> float:
        if diagnosis == "Heart attack" and severity_score >= 10:
            return 0.08
        if diagnosis in {"Common Cold", "Gastroenteritis", "Fungal infection", "Allergy"} and severity_score <= 16:
            return 0.03
        return 0.0

    def _sanity_adjustment(self, diagnosis: str, symptom_set: set[str], normalized_message: str) -> float:
        if diagnosis == "Heart attack":
            allergic_airway_pattern = (
                "breathlessness" in symptom_set
                and {"skin_rash", "itching"}.intersection(symptom_set)
                and "chest_pain" not in symptom_set
                and not {"palpitations", "sweating"}.intersection(symptom_set)
                and self._has_allergic_airway_context(normalized_message)
            )
            if allergic_airway_pattern:
                return -0.40

            endocrine_cardiac_overlap = (
                {"weight_loss", "sweating"}.issubset(symptom_set)
                and {"fast_heart_rate", "palpitations"}.intersection(symptom_set)
                and "chest_pain" not in symptom_set
                and "breathlessness" not in symptom_set
            )
            if endocrine_cardiac_overlap:
                return -0.32

            urinary_or_digestive_without_cardiac = (
                {"burning_micturition", "spotting_ urination", "bladder_discomfort"}.intersection(symptom_set)
                or "stomach_bleeding" in symptom_set
            ) and "chest_pain" not in symptom_set and "breathlessness" not in symptom_set
            if urinary_or_digestive_without_cardiac:
                return -0.34

        if diagnosis in {"hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E"}:
            vague_constitutional_pattern = {"fatigue", "weight_loss", "loss_of_appetite"}.issubset(symptom_set)
            if vague_constitutional_pattern and not self._has_liver_specific_evidence(symptom_set, normalized_message):
                return -0.28

        if diagnosis == "Jaundice":
            urinary_dark_urine_pattern = (
                "dark_urine" in symptom_set
                and {"burning_micturition", "bladder_discomfort", "spotting_ urination"}.intersection(symptom_set)
                and not {"yellowing_of_eyes", "yellowish_skin"}.intersection(symptom_set)
            )
            if urinary_dark_urine_pattern:
                return -0.30
            if {"weight_loss", "fatigue"}.intersection(symptom_set) and not self._has_liver_specific_evidence(
                symptom_set,
                normalized_message,
            ):
                return -0.24

        if diagnosis == "Allergy":
            hypoglycemia_pattern = (
                "excessive_hunger" in symptom_set
                and "sweating" in symptom_set
                and {"shivering", "chills", "fatigue", "anxiety"}.intersection(symptom_set)
                and not {"skin_rash", "itching"}.intersection(symptom_set)
            )
            if hypoglycemia_pattern:
                return -0.24

            if not {"skin_rash", "itching", "continuous_sneezing", "breathlessness", "watering_from_eyes"}.intersection(
                symptom_set
            ):
                return -0.26

        if diagnosis == "Hypoglycemia":
            hypoglycemia_pattern = (
                "excessive_hunger" in symptom_set
                and (
                    "sweating" in symptom_set
                    or "irregular_sugar_level" in symptom_set
                    or {"shivering", "chills", "fatigue", "anxiety", "dizziness", "coma"}.intersection(symptom_set)
                )
            )
            if not hypoglycemia_pattern:
                return -0.30

        if diagnosis == "Bronchial Asthma":
            asthma_specific = self._has_asthma_specific_context(symptom_set, normalized_message)
            if (
                "cough" not in symptom_set
                and "breathlessness" not in symptom_set
                and not asthma_specific
            ):
                return -0.34

            has_respiratory_basic = (
                "cough" in symptom_set
                and ("high_fever" in symptom_set or "mild_fever" in symptom_set)
                and (
                    "fatigue" in symptom_set
                    or "muscle_pain" in symptom_set
                    or "throat_irritation" in symptom_set
                    or "runny_nose" in symptom_set
                    or "congestion" in symptom_set
                )
            )
            if has_respiratory_basic and not asthma_specific:
                return -0.42
            if symptom_set == {"cough"} and not asthma_specific:
                return -0.24

        if diagnosis == "Hypertension":
            neuro_triage = {"headache", "dizziness", "nausea"}.issubset(symptom_set)
            vestibular_pattern = (
                "dizziness" in symptom_set
                and {"loss_of_balance", "spinning_movements"}.intersection(symptom_set)
                and {"vomiting", "nausea"}.intersection(symptom_set)
            )
            hypertension_specific = self._has_hypertension_context(normalized_message) or bool(
                {"chest_pain", "palpitations"}.intersection(symptom_set)
            )
            if neuro_triage and not hypertension_specific:
                return -0.22
            if vestibular_pattern and not hypertension_specific:
                return -0.36

            endocrine_or_urinary_pattern = bool(
                {
                    "irregular_sugar_level",
                    "excessive_hunger",
                    "polyuria",
                    "burning_micturition",
                    "spotting_ urination",
                    "bladder_discomfort",
                    "foul_smell_of urine",
                }.intersection(symptom_set)
            )
            vague_without_pressure = bool({"fatigue", "malaise", "lethargy", "loss_of_appetite"}.intersection(symptom_set))
            if not hypertension_specific and (endocrine_or_urinary_pattern or vague_without_pressure):
                return -0.30

        if diagnosis == "Varicose veins":
            vascular_symptoms = {
                "swollen_legs",
                "swollen_blood_vessels",
                "prominent_veins_on_calf",
                "painful_walking",
            }
            endocrine_pattern = bool(
                {"weight_gain", "puffy_face_and_eyes", "cold_hands_and_feets", "enlarged_thyroid"}.intersection(symptom_set)
            )
            if not vascular_symptoms.intersection(symptom_set) or endocrine_pattern:
                return -0.32

        if diagnosis in {"Chicken pox", "Fungal infection", "Impetigo", "Drug Reaction"}:
            if not {"skin_rash", "itching", "red_spots_over_body", "yellow_crust_ooze", "bruising"}.intersection(
                symptom_set
            ):
                return -0.26

        if diagnosis in {"Migraine", "Paralysis (brain hemorrhage)"}:
            vague_general = {"fatigue", "malaise", "lethargy", "depression", "anxiety"}.intersection(symptom_set)
            neuro_specific = {
                "headache",
                "dizziness",
                "blurred_and_distorted_vision",
                "weakness_of_one_body_side",
                "slurred_speech",
                "stiff_neck",
            }.intersection(symptom_set)
            if vague_general and not neuro_specific:
                return -0.24

        return 0.0

    def _fused_confidence(self, best_score: float, runner_up_score: float, raw_probability: float) -> float:
        margin = max(0.0, best_score - runner_up_score)
        score_based = 0.28 + min(best_score, 1.0) * 0.30 + min(margin, 0.5) * 0.14
        confidence = max(raw_probability, score_based)
        return round(float(min(0.95, max(0.0, confidence))), 4)

    def _decode_label(self, raw_label: object) -> str:
        if self.label_encoder is not None and isinstance(raw_label, (int, np.integer)):
            return str(self.label_encoder.inverse_transform([int(raw_label)])[0]).strip()
        return str(raw_label).strip()
