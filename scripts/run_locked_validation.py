from __future__ import annotations

import argparse
import csv
import json
import statistics
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

from app.main import app, llm_service, settings  # noqa: E402
from scripts.evaluate_system import diagnosis_group, doctor_type, percent, safety_check  # noqa: E402


LOCKED_SPLIT_NAME = "locked_validation_v1"
VALIDATION_CASES_PATH = SERVICE_ROOT / "validation_locked_cases.csv"
PHASE1_1_RESULTS_PATH = SERVICE_ROOT / "evaluation_results_phase1_1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked MedBridge AI validation set.")
    parser.add_argument("--cases", type=Path, default=VALIDATION_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=SERVICE_ROOT)
    parser.add_argument("--max-cases", type=int, default=0, help="Use 0 for the full locked set.")
    return parser.parse_args()


def clean_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    return cleaned


def load_locked_cases(path: Path, max_cases: int = 0) -> pd.DataFrame:
    df = clean_string_cells(pd.read_csv(path, encoding="utf-8-sig"))
    required_columns = {
        "case_id",
        "locked_split",
        "category",
        "user_message",
        "expected_diagnosis_group",
        "expected_urgency",
        "expected_doctor_type",
        "red_flags_present",
        "notes",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Locked validation set is missing required columns: {missing}")
    if df["case_id"].duplicated().any():
        duplicates = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate case_id values found: {duplicates}")
    invalid_split = df[df["locked_split"] != LOCKED_SPLIT_NAME]
    if len(invalid_split):
        raise ValueError(f"Unexpected locked_split values: {sorted(invalid_split['locked_split'].unique())}")
    if max_cases > 0:
        df = df.head(max_cases).copy()
    return df


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(bool_value)


class LLMModeTracker:
    """Validation-only instrumentation for Groq vs fallback answer generation."""

    def __init__(self, service: Any):
        self.service = service
        self.original_fallback = service._fallback_answer
        self.current_mode = "fallback"

    def install(self) -> None:
        def tracked_fallback(context: dict[str, Any]) -> str:
            self.current_mode = "fallback"
            return self.original_fallback(context)

        self.service._fallback_answer = tracked_fallback

    def reset(self) -> None:
        self.current_mode = "groq" if self.service.configured else "fallback"

    def uninstall(self) -> None:
        self.service._fallback_answer = self.original_fallback


def run_chat_validation(client: TestClient, cases_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    full_responses: list[dict[str, Any]] = []
    total = len(cases_df)
    mode_tracker = LLMModeTracker(llm_service)
    mode_tracker.install()

    try:
        for index, case in enumerate(cases_df.to_dict("records"), start=1):
            mode_tracker.reset()
            payload = {"message": case["user_message"], "history": []}
            start = time.perf_counter()
            response = client.post("/chat", json=payload)
            latency = time.perf_counter() - start
            status_code = response.status_code
            data = response.json() if status_code == 200 else {"answer": response.text}
            llm_mode = mode_tracker.current_mode

            predicted_diagnosis = data.get("possible_diagnosis")
            predicted_group = diagnosis_group(predicted_diagnosis)
            suggested_doctor = data.get("suggested_doctor")
            suggested_doctor_group = doctor_type(suggested_doctor)
            answer = data.get("answer") or ""
            extracted = data.get("extracted_symptoms") or []
            passed_safety = status_code == 200 and safety_check(data)

            row = {
                "case_id": case["case_id"],
                "locked_split": case["locked_split"],
                "category": case["category"],
                "user_message": case["user_message"],
                "expected_diagnosis_group": case["expected_diagnosis_group"],
                "expected_urgency": case["expected_urgency"],
                "expected_doctor_type": case["expected_doctor_type"],
                "red_flags_present": bool_value(case["red_flags_present"]),
                "notes": case["notes"],
                "extracted_symptoms": json.dumps(extracted, ensure_ascii=False),
                "extracted_symptom_count": len(extracted),
                "predicted_diagnosis": predicted_diagnosis,
                "predicted_diagnosis_group": predicted_group,
                "confidence": data.get("confidence", 0.0),
                "urgency_level": data.get("urgency_level"),
                "suggested_doctor": suggested_doctor,
                "suggested_doctor_type": suggested_doctor_group,
                "needs_follow_up": data.get("needs_follow_up", False),
                "follow_up_question_count": len(data.get("follow_up_questions") or []),
                "answer_length": len(answer),
                "latency_seconds": round(latency, 4),
                "llm_mode": llm_mode,
                "passed_safety_check": passed_safety,
                "diagnosis_group_match": predicted_group == case["expected_diagnosis_group"],
                "urgency_match": data.get("urgency_level") == case["expected_urgency"],
                "doctor_match": suggested_doctor_group == case["expected_doctor_type"],
                "status_code": status_code,
            }
            rows.append(row)
            full_responses.append(
                {
                    "case": case,
                    "response": data,
                    "latency_seconds": latency,
                    "llm_mode": llm_mode,
                    "passed_safety_check": passed_safety,
                }
            )
            print(
                f"Validated case {index}/{total}: case_id={case['case_id']} "
                f"category={case['category']} status={status_code} "
                f"llm_mode={llm_mode} latency={latency:.2f}s",
                flush=True,
            )
    finally:
        mode_tracker.uninstall()

    return pd.DataFrame(rows), full_responses


def metrics_from_results(results_df: pd.DataFrame) -> dict[str, Any]:
    expected_emergencies = results_df[results_df["expected_urgency"] == "High"]
    if len(expected_emergencies):
        emergency_recall = float(bool_series(expected_emergencies["urgency_level"] == "High").mean())
    else:
        emergency_recall = 0.0

    latencies = pd.to_numeric(results_df["latency_seconds"], errors="coerce").fillna(0)
    confidence = pd.to_numeric(results_df["confidence"], errors="coerce").fillna(0)
    llm_mode_counts = results_df["llm_mode"].value_counts().to_dict() if "llm_mode" in results_df else {}
    return {
        "case_count": int(len(results_df)),
        "groq_rows": int(llm_mode_counts.get("groq", 0)),
        "fallback_rows": int(llm_mode_counts.get("fallback", 0)),
        "diagnosis_group_accuracy": float(bool_series(results_df["diagnosis_group_match"]).mean()),
        "urgency_accuracy": float(bool_series(results_df["urgency_match"]).mean()),
        "doctor_recommendation_accuracy": float(bool_series(results_df["doctor_match"]).mean()),
        "emergency_detection_recall": emergency_recall,
        "safety_check_pass_rate": float(bool_series(results_df["passed_safety_check"]).mean()),
        "average_confidence": float(confidence.mean()),
        "average_latency_seconds": float(latencies.mean()),
        "median_latency_seconds": float(statistics.median(latencies.tolist())) if len(latencies) else 0.0,
        "p95_latency_seconds": float(latencies.quantile(0.95)) if len(latencies) else 0.0,
        "follow_up_question_rate": float(bool_series(results_df["needs_follow_up"]).mean()),
    }


def category_breakdown(results_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for category, group_df in results_df.groupby("category", sort=True):
        records.append(
            {
                "category": category,
                "case_count": int(len(group_df)),
                "groq_rows": int((group_df["llm_mode"] == "groq").sum()) if "llm_mode" in group_df else 0,
                "fallback_rows": int((group_df["llm_mode"] == "fallback").sum()) if "llm_mode" in group_df else 0,
                "diagnosis_accuracy": float(bool_series(group_df["diagnosis_group_match"]).mean()),
                "urgency_accuracy": float(bool_series(group_df["urgency_match"]).mean()),
                "doctor_accuracy": float(bool_series(group_df["doctor_match"]).mean()),
                "safety_pass_rate": float(bool_series(group_df["passed_safety_check"]).mean()),
                "average_confidence": float(pd.to_numeric(group_df["confidence"], errors="coerce").fillna(0).mean()),
            }
        )
    return pd.DataFrame(records)


def format_metrics_table(metrics: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"metric": "Case count", "value": str(metrics["case_count"])},
        {"metric": "Groq rows", "value": str(metrics["groq_rows"])},
        {"metric": "Fallback rows", "value": str(metrics["fallback_rows"])},
        {"metric": "Diagnosis accuracy", "value": percent(metrics["diagnosis_group_accuracy"])},
        {"metric": "Urgency accuracy", "value": percent(metrics["urgency_accuracy"])},
        {"metric": "Doctor recommendation accuracy", "value": percent(metrics["doctor_recommendation_accuracy"])},
        {"metric": "Emergency recall", "value": percent(metrics["emergency_detection_recall"])},
        {"metric": "Safety pass rate", "value": percent(metrics["safety_check_pass_rate"])},
        {"metric": "Average confidence", "value": f"{metrics['average_confidence']:.4f}"},
        {"metric": "Average latency seconds", "value": f"{metrics['average_latency_seconds']:.4f}"},
        {"metric": "Median latency seconds", "value": f"{metrics['median_latency_seconds']:.4f}"},
        {"metric": "P95 latency seconds", "value": f"{metrics['p95_latency_seconds']:.4f}"},
        {"metric": "Follow-up question rate", "value": percent(metrics["follow_up_question_rate"])},
    ]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "/").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pct_row(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    output = row.to_dict()
    for column in columns:
        output[column] = percent(float(output[column]))
    output["average_confidence"] = f"{float(output['average_confidence']):.4f}"
    return output


def verify_llm_modes(results_df: pd.DataFrame) -> None:
    if "llm_mode" not in results_df.columns:
        raise ValueError("validation_results is missing the llm_mode column")
    invalid = results_df[~results_df["llm_mode"].isin(["groq", "fallback"])]
    if len(invalid):
        case_ids = invalid["case_id"].astype(str).head(20).tolist()
        raise ValueError(f"Invalid or missing llm_mode values for case_ids: {case_ids}")


def deployment_recommendation(metrics: dict[str, Any]) -> str:
    if metrics["fallback_rows"] > 0:
        return (
            "Do not deploy from this validation result as a final Groq-only production validation. "
            "The run contains fallback rows, so the evidence is mixed-mode. Report the result honestly "
            "and wait for a consistent Groq-backed validation before production deployment."
        )
    if (
        metrics["diagnosis_group_accuracy"] >= 0.85
        and metrics["urgency_accuracy"] >= 0.90
        and metrics["doctor_recommendation_accuracy"] >= 0.90
        and metrics["emergency_detection_recall"] >= 0.95
        and metrics["safety_check_pass_rate"] >= 0.95
    ):
        return (
            "Ready for supervised deployment/demo review. The locked validation met the target quality "
            "thresholds with Groq configured and no fallback rows. Keep the medical disclaimer, safety "
            "guardrails, and stateless API contract in place."
        )
    return (
        "Not ready for production deployment yet. One or more locked validation metrics stayed below "
        "the review threshold, so the remaining weak categories should be reviewed before deployment."
    )


def load_phase1_1_metrics() -> dict[str, Any] | None:
    if not PHASE1_1_RESULTS_PATH.exists():
        return None
    old_df = pd.read_csv(PHASE1_1_RESULTS_PATH, encoding="utf-8-sig")
    return metrics_from_results(clean_string_cells(old_df))


def comparison_rows(old_metrics: dict[str, Any] | None, new_metrics: dict[str, Any]) -> list[dict[str, str]]:
    if not old_metrics:
        return [{"metric": "50-case comparison", "50_case_result": "not found", "locked_160_result": "available"}]
    percentage_metrics = [
        "diagnosis_group_accuracy",
        "urgency_accuracy",
        "doctor_recommendation_accuracy",
        "emergency_detection_recall",
        "safety_check_pass_rate",
        "follow_up_question_rate",
    ]
    numeric_metrics = [
        "average_confidence",
        "average_latency_seconds",
    ]
    labels = {
        "diagnosis_group_accuracy": "Diagnosis accuracy",
        "urgency_accuracy": "Urgency accuracy",
        "doctor_recommendation_accuracy": "Doctor recommendation accuracy",
        "emergency_detection_recall": "Emergency recall",
        "safety_check_pass_rate": "Safety pass rate",
        "follow_up_question_rate": "Follow-up question rate",
        "average_confidence": "Average confidence",
        "average_latency_seconds": "Average latency seconds",
    }
    rows: list[dict[str, str]] = []
    for key in percentage_metrics:
        delta = float(new_metrics[key]) - float(old_metrics[key])
        rows.append(
            {
                "metric": labels[key],
                "50_case_result": percent(float(old_metrics[key])),
                "locked_160_result": percent(float(new_metrics[key])),
                "delta": f"{delta:+.2%}",
            }
        )
    for key in numeric_metrics:
        delta = float(new_metrics[key]) - float(old_metrics[key])
        rows.append(
            {
                "metric": labels[key],
                "50_case_result": f"{float(old_metrics[key]):.4f}",
                "locked_160_result": f"{float(new_metrics[key]):.4f}",
                "delta": f"{delta:+.4f}",
            }
        )
    return rows


def mismatch_rows(results_df: pd.DataFrame, column: str, limit: int = 20) -> list[dict[str, Any]]:
    subset = results_df[~bool_series(results_df[column])].head(limit)
    fields = [
        "case_id",
        "category",
        "user_message",
        "expected_diagnosis_group",
        "predicted_diagnosis",
        "predicted_diagnosis_group",
        "expected_urgency",
        "urgency_level",
        "expected_doctor_type",
        "suggested_doctor",
        "extracted_symptom_count",
    ]
    return subset[fields].to_dict("records")


def symptom_gap_rows(results_df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    subset = results_df[
        (pd.to_numeric(results_df["extracted_symptom_count"], errors="coerce").fillna(0) <= 1)
        | (~bool_series(results_df["diagnosis_group_match"]))
    ].head(limit)
    fields = [
        "case_id",
        "category",
        "user_message",
        "extracted_symptoms",
        "predicted_diagnosis",
        "expected_diagnosis_group",
        "notes",
    ]
    return subset[fields].to_dict("records")


def safety_gap_rows(results_df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    subset = results_df[~bool_series(results_df["passed_safety_check"])].head(limit)
    fields = [
        "case_id",
        "category",
        "user_message",
        "predicted_diagnosis",
        "urgency_level",
        "suggested_doctor",
        "answer_length",
        "notes",
    ]
    return subset[fields].to_dict("records")


def write_report(
    path: Path,
    cases_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    old_metrics: dict[str, Any] | None,
    breakdown_df: pd.DataFrame,
    health: dict[str, Any],
    results_path: Path,
    metrics_path: Path,
) -> None:
    coverage_rows = cases_df.groupby("category").size().reset_index(name="case_count").to_dict("records")
    breakdown_rows = [
        pct_row(
            row,
            ["diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"],
        )
        for _, row in breakdown_df.sort_values("diagnosis_accuracy").iterrows()
    ]
    weak_categories = breakdown_df[
        (breakdown_df["diagnosis_accuracy"] < 0.85)
        | (breakdown_df["urgency_accuracy"] < 0.90)
        | (breakdown_df["doctor_accuracy"] < 0.90)
        | (breakdown_df["safety_pass_rate"] < 0.95)
    ].sort_values(["diagnosis_accuracy", "urgency_accuracy"])
    weak_rows = [
        pct_row(
            row,
            ["diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"],
        )
        for _, row in weak_categories.iterrows()
    ]

    if (
        metrics["diagnosis_group_accuracy"] >= 0.85
        and metrics["emergency_detection_recall"] >= 0.95
        and metrics["safety_check_pass_rate"] >= 0.95
    ):
        readiness = (
            "The locked validation result is strong enough to recommend deployment readiness "
            "for supervised graduation/demo deployment, while keeping the medical disclaimer and stateless design."
        )
    else:
        readiness = (
            "The locked validation result is not yet strong enough for broad production deployment. "
            "It can still support a controlled graduation demo if the weak categories and safety gaps are explained honestly."
        )

    content = f"""# Validation Report

Generated: {datetime.now().isoformat(timespec="seconds")}

This report validates the current MedBridge AI service on a locked Arabic evaluation set. The set is marked `{LOCKED_SPLIT_NAME}` and was not used for tuning, model changes, symptom extraction changes, prompt changes, or deployment changes in this validation phase.

No AI logic was modified during this run.

## Service Health

- Status: {health.get("status")}
- Service: {health.get("service")}
- Groq configured: {health.get("llm_configured")}
- Groq model: {settings.groq_model}

## Locked Set Coverage

Total locked cases: {len(cases_df)}

{markdown_table(coverage_rows, ["category", "case_count"])}

## Validation Metrics

{markdown_table(format_metrics_table(metrics), ["metric", "value"])}

## Comparison With Previous 50-Case Evaluation

{markdown_table(comparison_rows(old_metrics, metrics), ["metric", "50_case_result", "locked_160_result", "delta"])}

## Category Breakdown

{markdown_table(breakdown_rows, ["category", "case_count", "diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate", "average_confidence"])}

## Remaining Weak Categories

{markdown_table(weak_rows, ["category", "case_count", "diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate", "average_confidence"]) if weak_rows else "No weak category crossed the configured review thresholds."}

## Diagnosis Mismatches

{markdown_table(mismatch_rows(results_df, "diagnosis_group_match"), ["case_id", "category", "user_message", "expected_diagnosis_group", "predicted_diagnosis", "predicted_diagnosis_group", "expected_urgency", "urgency_level", "expected_doctor_type", "suggested_doctor", "extracted_symptom_count"])}

## Urgency Mismatches

{markdown_table(mismatch_rows(results_df, "urgency_match"), ["case_id", "category", "user_message", "expected_diagnosis_group", "predicted_diagnosis", "predicted_diagnosis_group", "expected_urgency", "urgency_level", "expected_doctor_type", "suggested_doctor", "extracted_symptom_count"])}

## Doctor Routing Mismatches

{markdown_table(mismatch_rows(results_df, "doctor_match"), ["case_id", "category", "user_message", "expected_diagnosis_group", "predicted_diagnosis", "predicted_diagnosis_group", "expected_urgency", "urgency_level", "expected_doctor_type", "suggested_doctor", "extracted_symptom_count"])}

## Remaining Symptom Gaps

These cases either had one or fewer extracted symptoms, or failed the diagnosis group check.

{markdown_table(symptom_gap_rows(results_df), ["case_id", "category", "user_message", "extracted_symptoms", "predicted_diagnosis", "expected_diagnosis_group", "notes"])}

## Remaining Safety Gaps

{markdown_table(safety_gap_rows(results_df), ["case_id", "category", "user_message", "predicted_diagnosis", "urgency_level", "suggested_doctor", "answer_length", "notes"])}

## Readiness Recommendation

{readiness}

## Files Created

- Locked validation results: `{results_path}`
- Locked validation metrics: `{metrics_path}`
- This report: `{path}`
"""
    path.write_text(content, encoding="utf-8")


def write_final_report(
    path: Path,
    cases_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    breakdown_df: pd.DataFrame,
    health: dict[str, Any],
    results_path: Path,
    metrics_path: Path,
) -> None:
    coverage_rows = cases_df.groupby("category").size().reset_index(name="case_count").to_dict("records")
    breakdown_rows = [
        pct_row(
            row,
            ["diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"],
        )
        for _, row in breakdown_df.sort_values("category").iterrows()
    ]
    weak_categories = breakdown_df[
        (breakdown_df["diagnosis_accuracy"] < 0.85)
        | (breakdown_df["urgency_accuracy"] < 0.90)
        | (breakdown_df["doctor_accuracy"] < 0.90)
        | (breakdown_df["safety_pass_rate"] < 0.95)
    ].sort_values(["diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy"])
    weak_rows = [
        pct_row(
            row,
            ["diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate"],
        )
        for _, row in weak_categories.iterrows()
    ]
    mode_counts = results_df["llm_mode"].value_counts().to_dict()
    all_modes_recorded = bool(results_df["llm_mode"].isin(["groq", "fallback"]).all())

    content = f"""# Final Validation Report

Generated: {datetime.now().isoformat(timespec="seconds")}

This report is the final locked validation run for the current MedBridge AI service before deployment review. The locked set is marked `{LOCKED_SPLIT_NAME}`. No AI logic, symptom extraction, classifier behavior, fusion logic, datasets, artifacts, or deployment settings were changed for this run.

## Run Conditions

- Validation cases: {len(cases_df)}
- Max cases argument: full locked set
- Groq model: {settings.groq_model}
- Health status: {health.get("status")}
- Health `llm_configured`: {health.get("llm_configured")}
- One locked validation run only: yes

## LLM Mode Verification

- `llm_mode` column present: {"yes" if "llm_mode" in results_df.columns else "no"}
- Every row records `groq` or `fallback`: {"yes" if all_modes_recorded else "no"}
- Groq rows: {int(mode_counts.get("groq", 0))}
- Fallback rows: {int(mode_counts.get("fallback", 0))}

## Locked Set Coverage

{markdown_table(coverage_rows, ["category", "case_count"])}

## Overall Metrics

{markdown_table(format_metrics_table(metrics), ["metric", "value"])}

## Category Breakdowns

{markdown_table(breakdown_rows, ["category", "case_count", "groq_rows", "fallback_rows", "diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate", "average_confidence"])}

## Remaining Weak Categories

{markdown_table(weak_rows, ["category", "case_count", "groq_rows", "fallback_rows", "diagnosis_accuracy", "urgency_accuracy", "doctor_accuracy", "safety_pass_rate", "average_confidence"]) if weak_rows else "No category crossed the configured weak-category thresholds."}

## Deployment Recommendation

{deployment_recommendation(metrics)}

## Files Created

- Validation results: `{results_path}`
- Validation metrics: `{metrics_path}`
- Final validation report: `{path}`
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_df = load_locked_cases(args.cases, args.max_cases)
    client = TestClient(app)
    health = client.get("/health").json()
    if health.get("llm_configured") is not True:
        raise RuntimeError(f"/health did not confirm Groq configuration: {health}")

    results_df, _ = run_chat_validation(client, cases_df)
    verify_llm_modes(results_df)
    metrics = metrics_from_results(results_df)
    old_metrics = load_phase1_1_metrics()
    breakdown_df = category_breakdown(results_df)

    results_path = args.output_dir / "validation_results.csv"
    metrics_path = args.output_dir / "validation_metrics.json"
    report_path = args.output_dir / "VALIDATION_REPORT.md"
    final_report_path = args.output_dir / "FINAL_VALIDATION_REPORT.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_path,
        cases_df,
        results_df,
        metrics,
        old_metrics,
        breakdown_df,
        health,
        results_path,
        metrics_path,
    )
    write_final_report(
        final_report_path,
        cases_df,
        results_df,
        metrics,
        breakdown_df,
        health,
        results_path,
        metrics_path,
    )

    print(
        json.dumps(
            {
                "health": health,
                "llm_mode_verification": {
                    "column_present": "llm_mode" in results_df.columns,
                    "all_rows_recorded": bool(results_df["llm_mode"].isin(["groq", "fallback"]).all()),
                    "groq_rows": metrics["groq_rows"],
                    "fallback_rows": metrics["fallback_rows"],
                },
                "metrics": metrics,
                "files": {
                    "validation_cases": str(args.cases),
                    "validation_results": str(results_path),
                    "validation_metrics": str(metrics_path),
                    "validation_report": str(report_path),
                    "final_validation_report": str(final_report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
