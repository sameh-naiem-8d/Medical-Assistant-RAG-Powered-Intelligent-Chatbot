from __future__ import annotations

import unittest

from app.safety import choose_urgency, detect_context_flags, has_red_flags


class Phase4FPreFinalSafetyTests(unittest.TestCase):
    def assert_high_urgency(self, message: str) -> None:
        self.assertTrue(has_red_flags(message, []), message)
        self.assertEqual(choose_urgency(message, [], 0), "High", message)

    def test_overdose_and_poisoning_phrases_are_high_urgency(self) -> None:
        self.assert_high_urgency("اخدت جرعة كبيرة من الدوا بالغلط")
        self.assert_high_urgency("شربت منظف بالغلط وبطني بتوجعني")
        self.assert_high_urgency("اخدت حبوب كتير ومش قادرة اركز")

    def test_pregnancy_reduced_fetal_movement_is_high_urgency(self) -> None:
        message = "حامل ومش حاسة بحركة الجنين"
        self.assert_high_urgency(message)
        self.assertIn("pregnancy_red_flag", detect_context_flags(message))

    def test_sudden_vision_loss_and_eye_trauma_are_high_urgency(self) -> None:
        self.assert_high_urgency("فقدت النظر فجأة في عين واحدة")
        self.assert_high_urgency("الم في العين بعد خبطة في العين")

    def test_head_injury_with_vomiting_is_high_urgency(self) -> None:
        message = "اتخبطت في دماغي وبعدها بقيت بستفرغ"
        self.assert_high_urgency(message)
        self.assertIn("severe_trauma", detect_context_flags(message))

    def test_large_burn_with_open_skin_is_high_urgency(self) -> None:
        message = "حرق كبير في ايدي والجلد مفتوح"
        self.assert_high_urgency(message)
        self.assertIn("severe_trauma", detect_context_flags(message))

    def test_facial_swelling_with_dental_abscess_is_high_urgency(self) -> None:
        self.assert_high_urgency("خراج في الضرس ووشي وارم")

    def test_insect_sting_with_face_swelling_and_breathing_phrase_is_high_urgency(self) -> None:
        self.assert_high_urgency("اتلسعت من حشرة وبعدها وشي ورم ونفسي ضاق")


if __name__ == "__main__":
    unittest.main()
