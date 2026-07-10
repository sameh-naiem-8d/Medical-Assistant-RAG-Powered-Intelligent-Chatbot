from __future__ import annotations

import re
import unittest
from pathlib import Path


class LocalDemoFrontendRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = Path("local_demo_frontend/index.html").read_text(encoding="utf-8")

    def test_frontend_does_not_render_internal_response_fields(self) -> None:
        forbidden_visible_labels = [
            "الوضع",
            "الخطورة",
            "confidence",
            "extracted_symptoms",
            "retrieved_cases",
            "follow_up_questions",
            "Clarifying Questions",
            "أسئلة توضيحية",
        ]
        body_text = re.sub(r"<script[\s\S]*?</script>", "", self.html, flags=re.IGNORECASE)
        for label in forbidden_visible_labels:
            self.assertNotIn(label, body_text)

    def test_assistant_bubble_renders_answer_text_only_with_optional_emergency_badge(self) -> None:
        self.assertIn('text.textContent = content;', self.html)
        self.assertIn('badge.textContent = "تنبيه عاجل";', self.html)
        self.assertNotRegex(self.html, r"textContent\s*=\s*data\.(mode|confidence|possible_diagnosis|suggested_doctor)")
        self.assertNotRegex(self.html, r"appendChild\([^)]*(follow_up_questions|retrieved_cases|extracted_symptoms)")

    def test_fetch_requests_have_abort_timeouts(self) -> None:
        self.assertIn("const HEALTH_TIMEOUT_MS", self.html)
        self.assertIn("const CHAT_TIMEOUT_MS", self.html)
        self.assertIn("new AbortController()", self.html)
        self.assertIn("fetchWithTimeout(HEALTH_URL", self.html)
        self.assertIn("fetchWithTimeout(CHAT_URL", self.html)


if __name__ == "__main__":
    unittest.main()
