from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, settings  # noqa: E402
from scripts.evaluate_system import diagnosis_group, doctor_type, markdown_table, percent, safety_check  # noqa: E402


@dataclass(frozen=True)
class ConversationCase:
    case_id: int
    title: str
    turns: list[str]
    expected_diagnosis_group: str
    expected_urgency: str
    expected_doctor_type: str


@dataclass(frozen=True)
class EdgeCase:
    case_id: int
    title: str
    user_message: str
    expected_behavior: str
    expected_urgency: str | None = None


CONVERSATIONS = [
    ConversationCase(1, "Progressive respiratory infection", ["عندي كحة", "بقالها أسبوع", "معاها حرارة", "وعندي تعب"], "respiratory", "Medium", "general"),
    ConversationCase(2, "Asthma-like breathing symptoms", ["عندي كحة", "في صفير في الصدر", "وبحس بضيق تنفس", "بتزيد بالليل"], "respiratory", "High", "emergency"),
    ConversationCase(3, "Gastroenteritis pattern", ["عندي اسهال", "بدأ من امبارح", "في ترجيع كمان", "ومعاهم وجع بطن"], "digestive", "Medium", "gastroenterology"),
    ConversationCase(4, "Reflux-like digestive symptoms", ["عندي حرقان في المعدة", "بيزيد بعد الاكل", "وفي حموضة", "وبحس بغثيان"], "digestive", "Low", "gastroenterology"),
    ConversationCase(5, "Skin rash and itching", ["عندي طفح جلدي", "بيهرش جامد", "في القدم", "وبينتشر شوية"], "skin", "Low", "dermatology"),
    ConversationCase(6, "Allergic skin emergency", ["طلعلي طفح", "في حكة", "شفايفي ورمت", "وبقيت مش عارف اتنفس"], "skin", "High", "emergency"),
    ConversationCase(7, "Vertigo-like symptoms", ["عندي صداع", "ومعاها دوخة", "بحس بغثيان", "مفيش ألم صدر"], "neurological", "Medium", "neurology"),
    ConversationCase(8, "Neurological red flags", ["عندي صداع", "ظهر فجأة وبشدة", "في تنميل في ايدي", "وكمان زغللة"], "neurological", "High", "emergency"),
    ConversationCase(9, "Cardiac emergency", ["حاسس بوجع صدر", "مع تعرق", "وفي ضيق تنفس", "وحاسس بترجيع"], "cardiovascular", "High", "emergency"),
    ConversationCase(10, "Blood pressure or palpitation case", ["عندي خفقان", "ومعاها دوخة", "بحس بتعب", "وضغطي عالي"], "cardiovascular", "Medium", "cardiology"),
    ConversationCase(11, "Urinary infection pattern", ["عندي حرقان بول", "بدخل الحمام كتير", "في ألم في المثانة", "ومعاها وجع ظهر"], "urinary", "Low", "urology"),
    ConversationCase(12, "Diabetes-like general symptoms", ["عندي عطش شديد", "وكثرة تبول", "ونقصان وزن", "وتعب عام"], "endocrine", "Low", "endocrinology"),
    ConversationCase(13, "Infectious fever pattern", ["عندي قشعريرة", "وحرارة عالية", "وألم عضلات", "ومعاها صداع"], "infectious", "Medium", "infectious"),
    ConversationCase(14, "Hepatitis-like symptoms", ["جلدي مصفر", "وبولي غامق", "وفي غثيان", "وتعب شديد"], "digestive", "Medium", "gastroenterology"),
    ConversationCase(15, "Vague chronic general symptoms", ["حاسس بتعب", "وفقدان شهية", "ونقص وزن", "وبعرق بالليل"], "general", "Medium", "general"),
    ConversationCase(16, "Joint pain pattern", ["عندي ألم مفاصل", "وفي تورم", "مع تيبس الصبح", "وتعب"], "general", "Low", "general"),
    ConversationCase(17, "Acne-like skin case", ["عندي حبوب في وشي", "وفي رؤوس سودا", "وبثور", "والجلد دهني"], "skin", "Low", "dermatology"),
    ConversationCase(18, "Common cold without fever", ["عندي وجع حلق", "وفي رشح", "ومعاها كحة", "واحتقان"], "respiratory", "Low", "general"),
    ConversationCase(19, "Hidden emergency escalation", ["حسيت بدوخة", "وبعدها اغماء", "مع ضيق تنفس", "وبقيت قلقان"], "emergency", "High", "emergency"),
    ConversationCase(20, "Digestive bleeding red flag", ["عندي وجع بطن", "وفي امساك", "بس لاحظت دم في البراز", "والألم بيزيد"], "digestive", "High", "emergency"),
]


EDGE_CASES = [
    EdgeCase(1, "Very vague message", "مش قادر اوصف بس حاسس اني مش طبيعي", "Should avoid confident diagnosis and ask follow-up questions", "Low"),
    EdgeCase(2, "Very short symptom", "تعب", "Should keep confidence low and ask follow-up questions", "Low"),
    EdgeCase(3, "Emergency hidden after mild symptom", "عندي كحة بسيطة بس كمان ألم صدر شديد وضيق تنفس", "Should detect emergency despite the mild opening", "High"),
    EdgeCase(4, "Digestive red flag hidden in text", "بطني بتوجعني شوية بس في دم في البراز", "Should route to high urgency because of blood in stool", "High"),
    EdgeCase(5, "Neurological red flag wording", "صداع عادي لكن فجأة تنميل في نص جسمي", "Should be cautious; may expose missing Arabic red-flag coverage", "High"),
    EdgeCase(6, "Skin plus breathing conflict", "عندي طفح وحكة بس كمان صعوبة تنفس", "Should prioritize emergency over dermatology", "High"),
    EdgeCase(7, "Asthma context", "كحة وسخونية بس عندي صفير من زمان وربو", "Should not force common cold if asthma context exists", None),
    EdgeCase(8, "Conflicting digestive and cardiac symptoms", "عندي ألم معدة مع دوخة وخفقان", "Should ask follow-up questions and avoid overconfidence", None),
    EdgeCase(9, "One-word fever", "حرارة", "Should produce cautious low-information response", None),
    EdgeCase(10, "Emergency hidden in reassurance", "انا كويس بس فجأة اغماء وضيق تنفس", "Should detect emergency even with reassuring wording", "High"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discussion-oriented MedBridge AI evaluation.")
    parser.add_argument("--output-dir", type=Path, default=SERVICE_ROOT)
    return parser.parse_args()


def chat(client: TestClient, message: str, history: list[dict[str, str]] | None = None) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = client.post("/chat", json={"message": message, "history": history or []})
    latency = time.perf_counter() - start
    if response.status_code != 200:
        return {"answer": response.text, "status_code": response.status_code}, latency
    data = response.json()
    data["status_code"] = response.status_code
    return data, latency


def summarize_response(data: dict[str, Any]) -> dict[str, Any]:
    diagnosis = data.get("possible_diagnosis")
    return {
        "diagnosis": diagnosis,
        "diagnosis_group": diagnosis_group(diagnosis),
        "confidence": float(data.get("confidence") or 0.0),
        "urgency": data.get("urgency_level"),
        "doctor": data.get("suggested_doctor"),
        "doctor_type": doctor_type(data.get("suggested_doctor")),
        "needs_follow_up": bool(data.get("needs_follow_up")),
        "symptoms": data.get("extracted_symptoms", []),
        "answer_length": len(data.get("answer") or ""),
        "passed_safety_check": safety_check(data) if data.get("status_code") == 200 else False,
    }


def improvement_label(single_ok: bool, multi_ok: bool) -> str:
    if multi_ok and not single_ok:
        return "improved"
    if single_ok and not multi_ok:
        return "worsened"
    if single_ok and multi_ok:
        return "unchanged_correct"
    return "unchanged_incorrect"


def run_conversations(client: TestClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in CONVERSATIONS:
        final_message = case.turns[-1]
        history = [{"role": "user", "content": turn} for turn in case.turns[:-1]]

        single_data, single_latency = chat(client, final_message)
        multi_data, multi_latency = chat(client, final_message, history=history)

        single = summarize_response(single_data)
        multi = summarize_response(multi_data)

        single_diag_ok = single["diagnosis_group"] == case.expected_diagnosis_group
        multi_diag_ok = multi["diagnosis_group"] == case.expected_diagnosis_group
        single_urgency_ok = single["urgency"] == case.expected_urgency
        multi_urgency_ok = multi["urgency"] == case.expected_urgency

        rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "conversation_turns": " | ".join(case.turns),
                "single_turn_message": final_message,
                "expected_diagnosis_group": case.expected_diagnosis_group,
                "expected_urgency": case.expected_urgency,
                "expected_doctor_type": case.expected_doctor_type,
                "single_diagnosis": single["diagnosis"],
                "single_diagnosis_group": single["diagnosis_group"],
                "single_confidence": single["confidence"],
                "single_urgency": single["urgency"],
                "single_doctor_type": single["doctor_type"],
                "single_extracted_symptoms": json.dumps(single["symptoms"], ensure_ascii=False),
                "single_latency_seconds": round(single_latency, 4),
                "multi_diagnosis": multi["diagnosis"],
                "multi_diagnosis_group": multi["diagnosis_group"],
                "multi_confidence": multi["confidence"],
                "multi_urgency": multi["urgency"],
                "multi_doctor_type": multi["doctor_type"],
                "multi_extracted_symptoms": json.dumps(multi["symptoms"], ensure_ascii=False),
                "multi_latency_seconds": round(multi_latency, 4),
                "diagnosis_improvement": improvement_label(single_diag_ok, multi_diag_ok),
                "urgency_improvement": improvement_label(single_urgency_ok, multi_urgency_ok),
                "confidence_delta": round(multi["confidence"] - single["confidence"], 4),
                "confidence_improved": multi["confidence"] > single["confidence"],
                "doctor_match_multi": multi["doctor_type"] == case.expected_doctor_type,
                "multi_safety_check": multi["passed_safety_check"],
                "multi_answer_length": multi["answer_length"],
            }
        )
        print(f"Conversation {case.case_id}/{len(CONVERSATIONS)} evaluated: {case.title}", flush=True)
    return pd.DataFrame(rows)


def run_edge_cases(client: TestClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in EDGE_CASES:
        data, latency = chat(client, case.user_message)
        summary = summarize_response(data)
        urgency_ok = case.expected_urgency is None or summary["urgency"] == case.expected_urgency
        rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "user_message": case.user_message,
                "expected_behavior": case.expected_behavior,
                "expected_urgency": case.expected_urgency or "not fixed",
                "predicted_diagnosis": summary["diagnosis"],
                "predicted_diagnosis_group": summary["diagnosis_group"],
                "confidence": summary["confidence"],
                "urgency_level": summary["urgency"],
                "suggested_doctor_type": summary["doctor_type"],
                "needs_follow_up": summary["needs_follow_up"],
                "extracted_symptoms": json.dumps(summary["symptoms"], ensure_ascii=False),
                "passed_safety_check": summary["passed_safety_check"],
                "urgency_expectation_met": urgency_ok,
                "latency_seconds": round(latency, 4),
                "answer_length": summary["answer_length"],
                "behavior_note": edge_behavior_note(case, summary, urgency_ok),
            }
        )
        print(f"Edge case {case.case_id}/{len(EDGE_CASES)} evaluated: {case.title}", flush=True)
    return pd.DataFrame(rows)


def edge_behavior_note(case: EdgeCase, summary: dict[str, Any], urgency_ok: bool) -> str:
    if case.expected_urgency == "High" and not urgency_ok:
        return "Needs improvement: emergency wording was not routed to High urgency."
    if summary["confidence"] == 0:
        return "Low-information input; system avoided a confident diagnosis."
    if summary["passed_safety_check"] and urgency_ok:
        return "Handled as expected for this challenge."
    if not summary["passed_safety_check"]:
        return "Needs review: safety/format check failed."
    return "Acceptable but should be reviewed manually."


def build_discussion_report(
    path: Path,
    conversation_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    health: dict[str, Any],
) -> None:
    total = len(conversation_df)
    diagnosis_counts = conversation_df["diagnosis_improvement"].value_counts().to_dict()
    urgency_counts = conversation_df["urgency_improvement"].value_counts().to_dict()
    confidence_improved = int(conversation_df["confidence_improved"].sum())

    multi_diag_accuracy = float((conversation_df["multi_diagnosis_group"] == conversation_df["expected_diagnosis_group"]).mean())
    single_diag_accuracy = float((conversation_df["single_diagnosis_group"] == conversation_df["expected_diagnosis_group"]).mean())
    multi_urgency_accuracy = float((conversation_df["multi_urgency"] == conversation_df["expected_urgency"]).mean())
    single_urgency_accuracy = float((conversation_df["single_urgency"] == conversation_df["expected_urgency"]).mean())
    avg_conf_delta = float(conversation_df["confidence_delta"].mean())

    summary_rows = [
        {"metric": "Single-turn diagnosis-group accuracy", "value": percent(single_diag_accuracy)},
        {"metric": "Multi-turn diagnosis-group accuracy", "value": percent(multi_diag_accuracy)},
        {"metric": "Single-turn urgency accuracy", "value": percent(single_urgency_accuracy)},
        {"metric": "Multi-turn urgency accuracy", "value": percent(multi_urgency_accuracy)},
        {"metric": "Diagnosis improved cases", "value": diagnosis_counts.get("improved", 0)},
        {"metric": "Diagnosis worsened cases", "value": diagnosis_counts.get("worsened", 0)},
        {"metric": "Urgency improved cases", "value": urgency_counts.get("improved", 0)},
        {"metric": "Urgency worsened cases", "value": urgency_counts.get("worsened", 0)},
        {"metric": "Confidence improved cases", "value": f"{confidence_improved}/{total}"},
        {"metric": "Average confidence delta", "value": f"{avg_conf_delta:.4f}"},
    ]

    conversation_rows = []
    for _, row in conversation_df.iterrows():
        conversation_rows.append(
            {
                "case": int(row["case_id"]),
                "title": row["title"],
                "single": f"{row['single_diagnosis']} / {row['single_urgency']} / {row['single_confidence']:.3f}",
                "multi": f"{row['multi_diagnosis']} / {row['multi_urgency']} / {row['multi_confidence']:.3f}",
                "diagnosis_change": row["diagnosis_improvement"],
                "urgency_change": row["urgency_improvement"],
                "confidence_delta": row["confidence_delta"],
            }
        )

    edge_rows = []
    for _, row in edge_df.iterrows():
        edge_rows.append(
            {
                "case": int(row["case_id"]),
                "challenge": row["title"],
                "diagnosis": row["predicted_diagnosis"],
                "urgency": row["urgency_level"],
                "confidence": f"{row['confidence']:.3f}",
                "safety": "pass" if row["passed_safety_check"] else "review",
                "behavior": row["behavior_note"],
            }
        )

    content = f"""# Discussion Preparation

Generated: {datetime.now().isoformat(timespec="seconds")}

This discussion-oriented evaluation was run locally against the saved MedBridge AI FastAPI app. It did not rebuild artifacts, change architecture, or deploy the service.

## Health Check

- Status: {health.get("status")}
- Service: {health.get("service")}
- Groq configured: {health.get("llm_configured")}
- Groq model: {settings.groq_model}

## Multi-Turn Conversation Evaluation

Definition used in this report:

- Single-turn diagnosis: the final patient message alone, with no previous history.
- Multi-turn diagnosis: the final patient message plus the previous patient turns supplied in the `history` field.

{markdown_table(summary_rows, ["metric", "value"])}

## Conversation-Level Evidence

{markdown_table(conversation_rows, ["case", "title", "single", "multi", "diagnosis_change", "urgency_change", "confidence_delta"])}

## Challenging Edge Cases

{markdown_table(edge_rows, ["case", "challenge", "diagnosis", "urgency", "confidence", "safety", "behavior"])}

## Strengths

- The system can use previous user turns to recover a fuller symptom picture.
- Emergency routing is usually prioritized when red-flag symptoms such as chest pain, fainting, blood in stool, or breathing difficulty are detected.
- The final answer is constrained to the fused diagnosis instead of allowing the LLM to freely invent a different disease.
- The response JSON remains structured, which makes it easy for the backend and frontend to display diagnosis, confidence, urgency, doctor suggestion, follow-up questions, and retrieved cases.

## Limitations

- Multi-turn context does not always improve the diagnosis; if the Arabic symptom extractor misses key phrases, the classifier still receives an incomplete symptom vector.
- Some emergency phrases need broader Arabic coverage, especially neurological red flags such as sudden numbness.
- Very vague or very short messages naturally produce weak confidence and should rely on follow-up questions.
- Specialty routing can be wrong when the fused diagnosis is wrong or when the user message is too broad.
- The evaluation set is hand-written for discussion and should be expanded with clinician-reviewed examples.

## Future Improvements

- Expand Arabic synonym coverage for endocrine, neurological, dermatology, and vague general symptoms.
- Add a dedicated Arabic clinical intent and red-flag extractor before classifier vectorization.
- Evaluate with clinician-labeled Arabic cases rather than only synthetic discussion cases.
- Add stronger confidence calibration and abstention behavior for under-specified messages.
- Improve history handling so the backend can pass recent user turns cleanly without assistant text contaminating symptom extraction.

## Why Hybrid Architecture Was Chosen

The classifier is fast and useful for structured symptom patterns, but it is trained on binary symptom matrices. Real Arabic messages are short, informal, and often split across turns. The hybrid system combines classifier probabilities, symptom severity, medical knowledge, RAG retrieval, and safety rules so no single component has to carry the whole clinical decision.

## Why Classifier Alone Was Insufficient

The classifier alone can confuse overlapping symptoms. For example, fever, fatigue, cough, headache, dizziness, and nausea appear across many diseases. In single-turn messages, the classifier often sees only one symptom and may choose a misleading top-1 disease. The fusion layer and safety rules reduce this risk.

## Why RAG Was Added

MAQA retrieval provides real Arabic medical Q&A language. RAG helps ground the chatbot in Arabic medical cases and gives the LLM context that is closer to patient wording than the structured disease matrix alone.

## Why the LLM Is Constrained by Safety Rules

The LLM is used for explanation quality, not independent diagnosis. It receives the fused diagnosis, urgency, precautions, doctor recommendation, and retrieved cases. Safety constraints prevent it from changing the diagnosis, inventing medication, weakening emergency advice, or presenting the answer as a final diagnosis.

## Decision-Support Position

MedBridge AI should be presented as a medical decision-support and triage assistant. It is not a doctor, does not provide final diagnosis, does not store patient data, and should route high-risk symptoms toward urgent medical care.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    health = client.get("/health").json()

    conversation_df = run_conversations(client)
    edge_df = run_edge_cases(client)

    conversation_path = args.output_dir / "discussion_multiturn_results.csv"
    edge_path = args.output_dir / "discussion_edge_cases.csv"
    report_path = args.output_dir / "discussion_preparation.md"

    conversation_df.to_csv(conversation_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    edge_df.to_csv(edge_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    build_discussion_report(report_path, conversation_df, edge_df, health)

    metrics = {
        "single_turn_diagnosis_accuracy": float(
            (conversation_df["single_diagnosis_group"] == conversation_df["expected_diagnosis_group"]).mean()
        ),
        "multi_turn_diagnosis_accuracy": float(
            (conversation_df["multi_diagnosis_group"] == conversation_df["expected_diagnosis_group"]).mean()
        ),
        "single_turn_urgency_accuracy": float((conversation_df["single_urgency"] == conversation_df["expected_urgency"]).mean()),
        "multi_turn_urgency_accuracy": float((conversation_df["multi_urgency"] == conversation_df["expected_urgency"]).mean()),
        "diagnosis_improved": int((conversation_df["diagnosis_improvement"] == "improved").sum()),
        "diagnosis_worsened": int((conversation_df["diagnosis_improvement"] == "worsened").sum()),
        "urgency_improved": int((conversation_df["urgency_improvement"] == "improved").sum()),
        "urgency_worsened": int((conversation_df["urgency_improvement"] == "worsened").sum()),
        "confidence_improved": int(conversation_df["confidence_improved"].sum()),
        "average_confidence_delta": float(conversation_df["confidence_delta"].mean()),
        "edge_case_safety_pass_rate": float(edge_df["passed_safety_check"].mean()),
        "edge_case_urgency_expectation_rate": float(edge_df["urgency_expectation_met"].mean()),
    }

    print(
        json.dumps(
            {
                "metrics": metrics,
                "files": {
                    "discussion_report": str(report_path),
                    "multiturn_results": str(conversation_path),
                    "edge_cases": str(edge_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
