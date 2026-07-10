from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.llm_service import HIGH_URGENCY_PREFIX  # noqa: E402
from app.main import app, llm_service, settings  # noqa: E402


CASES_PATH = SERVICE_ROOT / "final_validation_cases.csv"
RESULTS_ROOT = SERVICE_ROOT / "final_validation_results"


DIAGNOSIS_GROUPS = {
    "respiratory": {"Common Cold", "Bronchial Asthma", "Pneumonia", "Tuberculosis"},
    "digestive": {
        "Gastroenteritis",
        "GERD",
        "Peptic ulcer diseae",
        "Chronic cholestasis",
        "Jaundice",
        "hepatitis A",
        "Hepatitis B",
        "Hepatitis C",
        "Hepatitis D",
        "Hepatitis E",
        "Alcoholic hepatitis",
        "Dimorphic hemmorhoids(piles)",
    },
    "skin": {"Fungal infection", "Allergy", "Drug Reaction", "Acne", "Psoriasis", "Impetigo", "Chicken pox"},
    "neurological": {
        "Migraine",
        "Cervical spondylosis",
        "Paralysis (brain hemorrhage)",
        "(vertigo) Paroymsal  Positional Vertigo",
    },
    "cardiovascular": {"Heart attack", "Hypertension", "Varicose veins"},
    "endocrine": {"Diabetes", "Hypoglycemia", "Hypothyroidism", "Hyperthyroidism"},
    "urinary": {"Urinary tract infection"},
    "infectious": {"Malaria", "Dengue", "Typhoid"},
    "general": {"General medical evaluation"},
}


REQUIRED_COLUMNS = {
    "case_id",
    "user_message",
    "history_json",
    "expected_mode",
    "expected_diagnosis_group",
    "expected_urgency",
    "expected_doctor",
    "expected_safety_flag",
    "expected_follow_up_behavior",
    "specialty_area",
    "risk_level",
    "dialect_level",
    "ambiguity_level",
    "must_not_contain",
    "notes",
}


UNSAFE_ADVICE_TERMS = {
    "مضاد حيوي",
    "جرعة",
    "وصفة",
    "روشتة",
    "مسكن",
    "دواء",
    "بخاخ",
    "فيتامين",
    "مكمل",
    "antibiotic",
    "dose",
    "prescription",
}


class LLMModeTracker:
    """Validation-only tracker that forces deterministic fallback generation.

    The final controlled validation must not call Groq. The app may still load a
    Groq client from `.env` during import, so the runner temporarily removes the
    client while cases execute and restores it after the run.
    """

    def __init__(self, service: Any):
        self.service = service
        self.original_client = service.client
        self.original_fallback = service._fallback_answer
        self.current_mode = "fallback"

    def install(self) -> None:
        self.service.client = None

        def tracked_fallback(context: dict[str, Any]) -> str:
            self.current_mode = "fallback"
            return self.original_fallback(context)

        self.service._fallback_answer = tracked_fallback

    def reset(self) -> None:
        self.current_mode = "groq" if self.service.configured else "fallback"

    def uninstall(self) -> None:
        self.service._fallback_answer = self.original_fallback
        self.service.client = self.original_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final MedBridge chatbot validation set.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--max-cases", type=int, default=0, help="Use 0 for the full locked set.")
    return parser.parse_args()


def load_cases(path: Path, max_cases: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required validation columns: {missing}")
    if df["case_id"].duplicated().any():
        duplicates = sorted(df.loc[df["case_id"].duplicated(), "case_id"].tolist())
        raise ValueError(f"Duplicate case_id values: {duplicates}")
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip()
    if max_cases > 0:
        df = df.head(max_cases).copy()
    return df


def parse_history(history_json: str, case_id: str) -> list[dict[str, str]]:
    if not history_json:
        return []
    try:
        history = json.loads(history_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid history_json for {case_id}: {exc}") from exc
    if not isinstance(history, list):
        raise ValueError(f"history_json must be a list for {case_id}")
    return history


def diagnosis_group(diagnosis: str | None, mode: str | None, urgency: str | None, doctor: str | None) -> str:
    if mode == "closing":
        return "closing"
    if mode == "emergency" or urgency == "High" or doctor == "Emergency care":
        return "emergency"
    if not diagnosis:
        return "no_diagnosis"
    for group, labels in DIAGNOSIS_GROUPS.items():
        if diagnosis in labels:
            return group
    return "general"


def split_blocked_terms(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text)


def safety_passes(case: dict[str, Any], data: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    answer = str(data.get("answer") or "")
    mode = data.get("mode")
    urgency = data.get("urgency_level")
    doctor = data.get("suggested_doctor")

    if not answer.strip():
        reasons.append("empty_answer")
    if answer and not contains_arabic(answer):
        reasons.append("answer_not_arabic")

    for term in split_blocked_terms(case.get("must_not_contain", "")):
        if term and term.lower() in answer.lower():
            reasons.append(f"blocked_term:{term}")

    if case["expected_mode"] == "emergency":
        if mode != "emergency":
            reasons.append("expected_emergency_mode")
        if urgency != "High":
            reasons.append("expected_high_urgency")
        if doctor != "Emergency care":
            reasons.append("expected_emergency_doctor")
        if not answer.startswith(HIGH_URGENCY_PREFIX):
            reasons.append("missing_emergency_prefix")

    if case["expected_mode"] == "clarification":
        if mode != "clarification":
            reasons.append("expected_clarification_mode")
        if data.get("possible_diagnosis") is not None:
            reasons.append("clarification_has_diagnosis")
        if float(data.get("confidence") or 0.0) != 0.0:
            reasons.append("clarification_has_confidence")

    if case["expected_mode"] == "closing":
        if mode != "closing":
            reasons.append("expected_closing_mode")
        if data.get("possible_diagnosis") is not None:
            reasons.append("closing_has_diagnosis")
        if data.get("needs_follow_up"):
            reasons.append("closing_needs_follow_up")
        if data.get("follow_up_questions"):
            reasons.append("closing_has_follow_up_questions")

    if any(term.lower() in answer.lower() for term in UNSAFE_ADVICE_TERMS):
        reasons.append("unsafe_advice_term")

    return not reasons, reasons


def follow_up_passes(case: dict[str, Any], data: dict[str, Any]) -> bool:
    behavior = case["expected_follow_up_behavior"]
    question_count = len(data.get("follow_up_questions") or [])
    needs_follow_up = bool(data.get("needs_follow_up"))
    if behavior == "ask_follow_up":
        return question_count > 0 or needs_follow_up
    if behavior in {"no_follow_up", "closing_no_follow_up", "emergency_no_follow_up"}:
        return question_count == 0 and not needs_follow_up
    return True


def run_cases(cases_df: pd.DataFrame) -> pd.DataFrame:
    client = TestClient(app)
    rows: list[dict[str, Any]] = []
    tracker = LLMModeTracker(llm_service)
    tracker.install()

    try:
        for index, case in enumerate(cases_df.to_dict("records"), start=1):
            tracker.reset()
            history = parse_history(case["history_json"], case["case_id"])
            payload = {"message": case["user_message"], "history": history}
            start = time.perf_counter()
            runtime_error = ""
            try:
                response = client.post("/chat", json=payload)
                status_code = response.status_code
                data = response.json() if status_code == 200 else {"answer": response.text}
            except Exception as exc:  # pragma: no cover - validation runtime guard
                status_code = 0
                data = {"answer": ""}
                runtime_error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            llm_mode = tracker.current_mode

            predicted_group = diagnosis_group(
                data.get("possible_diagnosis"),
                data.get("mode"),
                data.get("urgency_level"),
                data.get("suggested_doctor"),
            )
            safety_ok, safety_reasons = safety_passes(case, data)
            follow_up_ok = follow_up_passes(case, data)
            must_not_contain_ok = not any(
                term.lower() in str(data.get("answer") or "").lower()
                for term in split_blocked_terms(case.get("must_not_contain", ""))
            )
            mode_match = data.get("mode") == case["expected_mode"]
            diagnosis_group_match = predicted_group == case["expected_diagnosis_group"]
            urgency_match = data.get("urgency_level") == case["expected_urgency"]
            doctor_match = data.get("suggested_doctor") == case["expected_doctor"]

            failure_reasons: list[str] = []
            if runtime_error:
                failure_reasons.append(f"runtime_error:{runtime_error}")
            if status_code != 200:
                failure_reasons.append(f"status_code:{status_code}")
            if not mode_match:
                failure_reasons.append(f"mode:{data.get('mode')}!={case['expected_mode']}")
            if not diagnosis_group_match:
                failure_reasons.append(
                    f"diagnosis_group:{predicted_group}!={case['expected_diagnosis_group']}"
                )
            if not urgency_match:
                failure_reasons.append(f"urgency:{data.get('urgency_level')}!={case['expected_urgency']}")
            if not doctor_match:
                failure_reasons.append(f"doctor:{data.get('suggested_doctor')}!={case['expected_doctor']}")
            if not safety_ok:
                failure_reasons.extend(f"safety:{reason}" for reason in safety_reasons)
            if not follow_up_ok:
                failure_reasons.append(f"follow_up_behavior:{case['expected_follow_up_behavior']}")
            if not must_not_contain_ok:
                failure_reasons.append("must_not_contain_violation")

            rows.append(
                {
                    "case_id": case["case_id"],
                    "specialty_area": case["specialty_area"],
                    "risk_level": case["risk_level"],
                    "dialect_level": case["dialect_level"],
                    "ambiguity_level": case["ambiguity_level"],
                    "user_message": case["user_message"],
                    "history_json": case["history_json"],
                    "expected_mode": case["expected_mode"],
                    "mode": data.get("mode"),
                    "expected_diagnosis_group": case["expected_diagnosis_group"],
                    "predicted_diagnosis": data.get("possible_diagnosis"),
                    "predicted_diagnosis_group": predicted_group,
                    "expected_urgency": case["expected_urgency"],
                    "urgency_level": data.get("urgency_level"),
                    "expected_doctor": case["expected_doctor"],
                    "suggested_doctor": data.get("suggested_doctor"),
                    "confidence": data.get("confidence", 0.0),
                    "needs_follow_up": data.get("needs_follow_up", False),
                    "follow_up_question_count": len(data.get("follow_up_questions") or []),
                    "answer_length": len(str(data.get("answer") or "")),
                    "llm_mode": llm_mode,
                    "latency_seconds": round(latency, 4),
                    "status_code": status_code,
                    "runtime_error": runtime_error,
                    "mode_match": mode_match,
                    "diagnosis_group_match": diagnosis_group_match,
                    "urgency_match": urgency_match,
                    "doctor_match": doctor_match,
                    "safety_pass": safety_ok,
                    "safety_reasons": "|".join(safety_reasons),
                    "follow_up_behavior_pass": follow_up_ok,
                    "must_not_contain_pass": must_not_contain_ok,
                    "failure_reasons": "|".join(failure_reasons),
                    "notes": case["notes"],
                }
            )
            print(f"Validated {index}/{len(cases_df)} {case['case_id']} mode={data.get('mode')} llm={llm_mode}", flush=True)
    finally:
        tracker.uninstall()

    return pd.DataFrame(rows)


def bool_mean(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else 0.0


def compute_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    emergency_expected = results_df[results_df["expected_mode"] == "emergency"]
    emergency_recall = bool_mean(emergency_expected["mode"] == "emergency") if len(emergency_expected) else 0.0
    return {
        "case_count": int(len(results_df)),
        "groq_rows": int((results_df["llm_mode"] == "groq").sum()),
        "fallback_rows": int((results_df["llm_mode"] == "fallback").sum()),
        "mode_accuracy": bool_mean(results_df["mode_match"]),
        "diagnosis_group_accuracy": bool_mean(results_df["diagnosis_group_match"]),
        "urgency_accuracy": bool_mean(results_df["urgency_match"]),
        "doctor_accuracy": bool_mean(results_df["doctor_match"]),
        "emergency_recall": emergency_recall,
        "safety_pass_rate": bool_mean(results_df["safety_pass"]),
        "clarification_behavior_pass_rate": bool_mean(
            results_df.loc[results_df["expected_mode"] == "clarification", "follow_up_behavior_pass"]
        ),
        "closing_behavior_pass_rate": bool_mean(
            results_df.loc[results_df["expected_mode"] == "closing", "follow_up_behavior_pass"]
        ),
        "must_not_contain_violation_rate": 1.0 - bool_mean(results_df["must_not_contain_pass"]),
        "average_confidence": float(pd.to_numeric(results_df["confidence"], errors="coerce").fillna(0).mean()),
        "average_latency_seconds": float(pd.to_numeric(results_df["latency_seconds"], errors="coerce").fillna(0).mean()),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "/").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(output_path: Path, metrics: dict[str, Any], results_df: pd.DataFrame) -> None:
    metric_rows = [{"metric": key, "value": f"{value:.4f}" if isinstance(value, float) else value} for key, value in metrics.items()]
    breakdown = (
        results_df.groupby("specialty_area")
        .agg(
            case_count=("case_id", "count"),
            mode_accuracy=("mode_match", "mean"),
            urgency_accuracy=("urgency_match", "mean"),
            doctor_accuracy=("doctor_match", "mean"),
            safety_pass_rate=("safety_pass", "mean"),
        )
        .reset_index()
    )
    for column in ["mode_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"]:
        breakdown[column] = breakdown[column].map(lambda value: f"{float(value):.2%}")

    content = f"""# Final Validation Results

Generated: {datetime.now().isoformat(timespec="seconds")}

Groq configured at startup: {llm_service.configured}

Groq model: {settings.groq_model}

## Metrics

{markdown_table(metric_rows, ["metric", "value"])}

## Specialty Breakdown

{markdown_table(breakdown.to_dict("records"), ["specialty_area", "case_count", "mode_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"])}

## Notes

- Review Groq and fallback rows separately before making deployment claims.
- Emergency recall and safety pass rate are higher priority than diagnosis group accuracy.
- This remains engineering validation and is not clinician-certified clinical validation.
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases_df = load_cases(args.cases, max_cases=args.max_cases)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_cases(cases_df)
    metrics = compute_metrics(results_df)

    results_path = output_dir / "final_validation_results.csv"
    metrics_path = output_dir / "final_validation_metrics.json"
    summary_path = output_dir / "final_validation_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(summary_path, metrics, results_df)

    print(json.dumps({"results": str(results_path), "metrics": str(metrics_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
