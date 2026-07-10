from __future__ import annotations

import unittest

from app.classifier_service import ClassifierService, SYMPTOM_SYNONYMS
from app.config import get_settings
from app.knowledge_service import KnowledgeService
from app.safety import choose_urgency, has_red_flags


class SymptomExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ClassifierService(get_settings())
        cls.knowledge = KnowledgeService(get_settings())

    def assertExtracts(self, message: str, *expected: str) -> None:
        extracted = set(self.service.extract_symptoms(message))
        missing = set(expected) - extracted
        self.assertFalse(missing, f"Missing {missing} from {extracted} for message: {message}")

    def assertDoesNotExtract(self, message: str, *unexpected: str) -> None:
        extracted = set(self.service.extract_symptoms(message))
        found = set(unexpected).intersection(extracted)
        self.assertFalse(found, f"Unexpected {found} in {extracted} for message: {message}")

    def test_plan_examples_extract_expected_symptoms(self) -> None:
        self.assertExtracts("مش قادر اتنفس", "breathlessness")
        self.assertExtracts("بحس بدوخة", "dizziness")
        self.assertExtracts("بطني محروقة", "acidity")
        self.assertExtracts("مخنوق", "breathlessness")
        self.assertExtracts("جسمي مكسر", "fatigue")
        self.assertExtracts("نفسي مسدودة", "loss_of_appetite")

    def test_respiratory_colloquial_expressions(self) -> None:
        self.assertExtracts("عندي كحة وبلغم ووجع زور", "cough", "phlegm", "throat_irritation")
        self.assertExtracts("مناخيري مسدودة وبعطس كتير", "congestion", "continuous_sneezing")

    def test_neurological_red_flag_expressions(self) -> None:
        self.assertExtracts(
            "عندي صداع شديد مفاجئ وتنميل في ايدي",
            "headache",
            "weakness_of_one_body_side",
        )
        self.assertTrue(has_red_flags("عندي صداع شديد مفاجئ وتنميل في ايدي", ["headache"]))

    def test_diabetes_like_symptoms(self) -> None:
        self.assertExtracts(
            "عندي عطش شديد وكثرة تبول ونقص وزن",
            "polyuria",
            "weight_loss",
        )
        self.assertExtracts("عندي تعرق ورعشة وجوع شديد", "sweating", "shivering", "excessive_hunger")

    def test_jaundice_and_infectious_expressions(self) -> None:
        self.assertExtracts("عندي اصفرار جلد وبول غامق وتعب", "yellowish_skin", "dark_urine", "fatigue")
        self.assertExtracts("عندي حرارة عالية وصداع والم خلف العين", "high_fever", "headache", "pain_behind_the_eyes")

    def test_skin_and_varicose_expressions(self) -> None:
        self.assertExtracts(
            "عندي حبوب في الوجه وبثور ورؤوس سوداء",
            "nodal_skin_eruptions",
            "pus_filled_pimples",
            "blackheads",
        )
        self.assertExtracts(
            "عندي تورم في الرجلين ودوالي والم مع المشي",
            "swollen_legs",
            "prominent_veins_on_calf",
            "painful_walking",
        )

    def test_negated_high_risk_symptom_is_not_extracted(self) -> None:
        self.assertDoesNotExtract("عندي كحة ومفيش ضيق تنفس", "breathlessness")
        self.assertDoesNotExtract("صداع بس بدون زغللة", "blurred_and_distorted_vision")

    def test_dictionary_covers_most_symptom_columns(self) -> None:
        covered = [column for column in self.service.symptom_columns if column in SYMPTOM_SYNONYMS]
        coverage = len(covered) / len(self.service.symptom_columns)
        self.assertGreaterEqual(coverage, 0.90)

    def test_emergency_breathlessness_routes_high(self) -> None:
        symptoms = self.service.extract_symptoms("عندي زرقة في الشفاه ومش قادر اتنفس")
        self.assertIn("breathlessness", symptoms)
        self.assertEqual(choose_urgency("عندي زرقة في الشفاه ومش قادر اتنفس", symptoms, 0), "High")

    def test_low_risk_clusters_do_not_over_escalate(self) -> None:
        uri = self.service.extract_symptoms("عندي كحة ورشح واحتقان ووجع حلق")
        self.assertEqual(choose_urgency("عندي كحة ورشح واحتقان ووجع حلق", uri, 18), "Low")

        acne = self.service.extract_symptoms("عندي حبوب في الوجه وبثور ورؤوس سوداء")
        self.assertEqual(choose_urgency("عندي حبوب في الوجه وبثور ورؤوس سوداء", acne, 8), "Low")

        varicose = self.service.extract_symptoms("عندي تورم في الرجلين ودوالي والم مع المشي")
        self.assertEqual(choose_urgency("عندي تورم في الرجلين ودوالي والم مع المشي", varicose, 13), "Low")

    def test_hypertension_context_routes_medium(self) -> None:
        symptoms = self.service.extract_symptoms("عندي ضغط عالي وصداع ودوخة")
        self.assertEqual(choose_urgency("عندي ضغط عالي وصداع ودوخة", symptoms, 8), "Medium")

    def fused_diagnosis_for(self, message: str) -> tuple[str | None, str, list[str]]:
        symptoms = self.service.extract_symptoms(message)
        prediction = self.service.predict(symptoms)
        severity = self.knowledge.score_symptoms(symptoms, message)
        fused = self.service.fuse_prediction(
            prediction,
            symptoms,
            message,
            int(severity["score"]),
            [],
            self.knowledge.descriptions,
            self.knowledge.precautions,
        )
        return fused.diagnosis, str(severity["urgency"]), symptoms

    def test_allergic_airway_pattern_prefers_allergy_emergency(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي طفح جلدي مع تورم في الشفاه وصعوبة تنفس")
        self.assertIn("skin_rash", symptoms)
        self.assertIn("breathlessness", symptoms)
        self.assertEqual(diagnosis, "Allergy")
        self.assertEqual(urgency, "High")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Emergency care")

    def test_hypoglycemia_pattern_prefers_endocrine_candidate(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي تعرق ورعشة وجوع شديد")
        self.assertIn("sweating", symptoms)
        self.assertIn("excessive_hunger", symptoms)
        self.assertEqual(diagnosis, "Hypoglycemia")
        self.assertIn(urgency, {"Medium", "High"})
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Endocrinologist")

    def test_vague_constitutional_symptoms_do_not_default_to_hepatitis(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي تعب شديد وفقدان شهية ونقص وزن")
        self.assertIn("fatigue", symptoms)
        self.assertIn("loss_of_appetite", symptoms)
        self.assertIn("weight_loss", symptoms)
        self.assertEqual(diagnosis, "General medical evaluation")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "General Practitioner")

    def test_liver_specific_symptoms_can_still_support_jaundice(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي اصفرار جلد وبول غامق وتعب")
        self.assertIn("yellowish_skin", symptoms)
        self.assertIn("dark_urine", symptoms)
        self.assertEqual(diagnosis, "Jaundice")

    def test_requested_doctor_routing_fixes(self) -> None:
        self.assertEqual(self.knowledge.suggest_doctor("Hypertension", ["headache", "dizziness"], "Medium"), "Cardiologist")
        self.assertEqual(
            self.knowledge.suggest_doctor("Varicose veins", ["swollen_legs", "prominent_veins_on_calf"], "Low"),
            "Cardiologist",
        )
        self.assertEqual(
            self.knowledge.suggest_doctor("Cervical spondylosis", ["neck_pain", "headache"], "Medium"),
            "Neurologist",
        )

    def test_palpitations_pattern_does_not_become_hypoglycemia(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي خفقان ودوخة وتعب")
        self.assertIn("palpitations", symptoms)
        self.assertEqual(diagnosis, "Hypertension")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Cardiologist")

    def test_fever_joint_pain_fatigue_does_not_become_asthma(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("عندي الم مفاصل وتعب وسخونية")
        self.assertIn("joint_pain", symptoms)
        self.assertEqual(diagnosis, "General medical evaluation")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "General Practitioner")

    def test_phase2_urinary_bleeding_and_flank_patterns(self) -> None:
        symptoms = self.service.extract_symptoms("دم في البول ووجع شديد")
        self.assertIn("spotting_ urination", symptoms)
        self.assertEqual(choose_urgency("دم في البول ووجع شديد", symptoms, 0), "High")

        diagnosis, urgency, symptoms = self.fused_diagnosis_for("الم في الجنب وحرقان بول")
        self.assertIn("burning_micturition", symptoms)
        self.assertIn("back_pain", symptoms)
        self.assertEqual(diagnosis, "Urinary tract infection")
        self.assertEqual(urgency, "Medium")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Urologist")

    def test_phase2_endocrine_weak_cases(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("جعان جدا وبعرق ودايخ")
        self.assertIn("excessive_hunger", symptoms)
        self.assertIn("sweating", symptoms)
        self.assertEqual(diagnosis, "Hypoglycemia")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Endocrinologist")

        diagnosis, urgency, symptoms = self.fused_diagnosis_for("رقبتي فيها تورم من الغدة")
        self.assertIn("enlarged_thyroid", symptoms)
        self.assertEqual(diagnosis, "Hypothyroidism")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Endocrinologist")

    def test_phase2_infectious_fever_patterns(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("حرارة بعد سفر وقشعريرة")
        self.assertIn("high_fever", symptoms)
        self.assertIn("chills", symptoms)
        self.assertEqual(diagnosis, "Malaria")
        self.assertEqual(urgency, "Medium")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "Infectious disease specialist")

        diagnosis, urgency, symptoms = self.fused_diagnosis_for("حرارة وصداع والم خلف العين")
        self.assertIn("high_fever", symptoms)
        self.assertIn("pain_behind_the_eyes", symptoms)
        self.assertEqual(diagnosis, "Dengue")

    def test_phase2_emergency_red_flags(self) -> None:
        symptoms = self.service.extract_symptoms("بستفرغ دم ومعدتي واجعاني")
        self.assertIn("stomach_bleeding", symptoms)
        self.assertEqual(choose_urgency("بستفرغ دم ومعدتي واجعاني", symptoms, 0), "High")

        symptoms = self.service.extract_symptoms("كلامي تقيل ووشي معوج")
        self.assertIn("slurred_speech", symptoms)
        self.assertIn("weakness_of_one_body_side", symptoms)
        self.assertEqual(choose_urgency("كلامي تقيل ووشي معوج", symptoms, 0), "High")

        symptoms = self.service.extract_symptoms("صداع شديد مع تيبس رقبة وحرارة")
        self.assertIn("stiff_neck", symptoms)
        self.assertEqual(choose_urgency("صداع شديد مع تيبس رقبة وحرارة", symptoms, 0), "High")

        symptoms = self.service.extract_symptoms("حكة شديدة وتورم في الوجه")
        self.assertIn("itching", symptoms)
        self.assertEqual(choose_urgency("حكة شديدة وتورم في الوجه", symptoms, 0), "High")

    def test_phase2_vague_general_cases_stay_general(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("تعب شديد ودوخة ومفيش الم صدر")
        self.assertIn("fatigue", symptoms)
        self.assertIn("dizziness", symptoms)
        self.assertNotIn("chest_pain", symptoms)
        self.assertEqual(diagnosis, "General medical evaluation")
        self.assertEqual(self.knowledge.suggest_doctor(diagnosis, symptoms, urgency), "General Practitioner")

        diagnosis, urgency, symptoms = self.fused_diagnosis_for("فقدان شهية وتعب من يومين")
        self.assertIn("loss_of_appetite", symptoms)
        self.assertIn("fatigue", symptoms)
        self.assertEqual(diagnosis, "General medical evaluation")

    def test_phase3_3_throat_object_sensation_is_not_vomiting(self) -> None:
        symptoms = self.service.extract_symptoms("\u062d\u0627\u0633\u0633 \u0628\u062d\u0627\u062c\u0629 \u0648\u0627\u0642\u0641\u0629 \u0641\u064a \u062d\u0644\u0642\u064a")
        self.assertNotIn("vomiting", symptoms)

        symptoms = self.service.extract_symptoms("\u0639\u0646\u062f\u064a \u062a\u0631\u062c\u064a\u0639 \u0648\u0642\u064a\u0621")
        self.assertIn("vomiting", symptoms)

    def test_phase3_3_vestibular_pattern_prefers_neuro_not_cardiology(self) -> None:
        diagnosis, urgency, symptoms = self.fused_diagnosis_for("\u062f\u0648\u062e\u0629 \u0648\u0639\u062f\u0645 \u0627\u062a\u0632\u0627\u0646 \u0648\u0642\u064a\u0621")
        self.assertIn("dizziness", symptoms)
        self.assertIn("loss_of_balance", symptoms)
        self.assertIn("vomiting", symptoms)
        self.assertNotEqual(diagnosis, "Hypertension")
        self.assertIn(
            self.knowledge.suggest_doctor(diagnosis, symptoms, urgency),
            {"Neurologist", "ENT specialist"},
        )

    def test_phase3_3_stroke_style_numbness_routes_emergency(self) -> None:
        symptoms = self.service.extract_symptoms("\u062f\u0648\u062e\u0629 \u0648\u062a\u0646\u0645\u064a\u0644 \u0641\u064a \u0646\u0635 \u0627\u0644\u062c\u0633\u0645")
        self.assertIn("weakness_of_one_body_side", symptoms)
        self.assertEqual(choose_urgency("\u062f\u0648\u062e\u0629 \u0648\u062a\u0646\u0645\u064a\u0644 \u0641\u064a \u0646\u0635 \u0627\u0644\u062c\u0633\u0645", symptoms, 0), "High")

    def test_phase3_3_targeted_doctor_routing(self) -> None:
        rash = self.service.extract_symptoms("\u0639\u0646\u062f\u064a \u0637\u0641\u062d \u062c\u0644\u062f\u064a \u0648\u062d\u0643\u0629 \u0634\u062f\u064a\u062f\u0629")
        self.assertEqual(self.knowledge.suggest_doctor("Fungal infection", rash, "Low"), "Dermatologist")

        urinary = self.service.extract_symptoms(
            "\u0639\u0646\u062f\u064a \u062d\u0631\u0642\u0627\u0646 \u0641\u064a \u0627\u0644\u0628\u0648\u0644 \u0648\u0628\u0631\u0648\u062d \u0627\u0644\u062d\u0645\u0627\u0645 \u0643\u062a\u064a\u0631"
        )
        self.assertIn("burning_micturition", urinary)
        self.assertIn("continuous_feel_of_urine", urinary)
        self.assertEqual(self.knowledge.suggest_doctor("Urinary tract infection", urinary, "Low"), "Urologist")

        respiratory = self.service.extract_symptoms("\u0639\u0646\u062f\u064a \u0643\u062d\u0629 \u0648\u0633\u062e\u0648\u0646\u064a\u0629 \u0648\u062a\u0639\u0628")
        self.assertEqual(self.knowledge.suggest_doctor("Common Cold", respiratory, "Medium"), "General Practitioner")


if __name__ == "__main__":
    unittest.main()
