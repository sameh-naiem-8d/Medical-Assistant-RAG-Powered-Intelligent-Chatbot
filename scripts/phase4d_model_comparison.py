from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def clean_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_structured_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_root / "Training.csv")
    test = pd.read_csv(data_root / "Testing.csv")
    dataset = pd.read_csv(data_root / "dataset.csv")
    train["prognosis"] = clean_label(train["prognosis"])
    test["prognosis"] = clean_label(test["prognosis"])
    return train, test, dataset


def build_models() -> dict[str, Any]:
    return {
        "current_rf_300_balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "logistic_regression_lbfgs": LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "random_forest_100_balanced": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "extra_trees_300_balanced": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def model_size_bytes(model: Any) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size


def evaluate_model(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    labels: list[str],
    cv_splits: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_true: list[int] = []
    cv_pred: list[int] = []
    fit_times: list[float] = []
    predict_times: list[float] = []
    cv_confidences: list[float] = []

    for train_idx, valid_idx in splitter.split(X, y):
        fold_model = clone(model)
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y[train_idx]
        y_valid_fold = y[valid_idx]

        start = time.perf_counter()
        fold_model.fit(X_train_fold, y_train_fold)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        predictions = fold_model.predict(X_valid_fold)
        predict_times.append(time.perf_counter() - start)

        cv_true.extend(y_valid_fold.tolist())
        cv_pred.extend(predictions.tolist())
        if hasattr(fold_model, "predict_proba"):
            probabilities = fold_model.predict_proba(X_valid_fold)
            cv_confidences.extend(np.max(probabilities, axis=1).tolist())

    final_model = clone(model)
    start = time.perf_counter()
    final_model.fit(X, y)
    full_fit_time = time.perf_counter() - start

    start = time.perf_counter()
    official_predictions = final_model.predict(X_test)
    official_predict_time = time.perf_counter() - start

    official_confidences: list[float] = []
    if hasattr(final_model, "predict_proba"):
        probabilities = final_model.predict_proba(X_test)
        official_confidences = np.max(probabilities, axis=1).tolist()

    report = classification_report(
        cv_true,
        cv_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    weak_labels = sorted(
        (
            {
                "label": label,
                "f1": float(report[label]["f1-score"]),
                "recall": float(report[label]["recall"]),
                "precision": float(report[label]["precision"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        ),
        key=lambda item: (item["f1"], item["recall"], item["precision"], item["label"]),
    )[:8]

    cm = confusion_matrix(cv_true, cv_pred, labels=list(range(len(labels))))
    confusion_pairs: list[dict[str, Any]] = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confusion_pairs.append(
                    {
                        "true": labels[true_idx],
                        "predicted": labels[pred_idx],
                        "count": int(cm[true_idx, pred_idx]),
                    }
                )
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    return {
        "model": name,
        "supports_predict_proba": bool(hasattr(final_model, "predict_proba")),
        "needs_probability_calibration": not hasattr(final_model, "predict_proba") or name in {"linear_svm"},
        "cv": {
            "accuracy": float(accuracy_score(cv_true, cv_pred)),
            "macro_f1": float(f1_score(cv_true, cv_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(cv_true, cv_pred, average="weighted", zero_division=0)),
            "mean_fit_time_seconds_per_fold": float(np.mean(fit_times)),
            "mean_predict_latency_ms_per_sample": float(np.mean(predict_times) * 1000 / (len(y) / cv_splits)),
            "mean_confidence": float(np.mean(cv_confidences)) if cv_confidences else None,
            "top_confusion_pairs": confusion_pairs[:10],
            "weakest_labels_by_f1": weak_labels,
        },
        "official_testing_csv": {
            "accuracy": float(accuracy_score(y_test, official_predictions)),
            "macro_f1": float(f1_score(y_test, official_predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, official_predictions, average="weighted", zero_division=0)),
            "predict_latency_ms_per_sample": float(official_predict_time * 1000 / len(y_test)),
            "mean_confidence": float(np.mean(official_confidences)) if official_confidences else None,
        },
        "fit_time_seconds_full_training": float(full_fit_time),
        "serialized_model_size_bytes_estimate": int(model_size_bytes(final_model)),
    }


def parse_args() -> argparse.Namespace:
    service_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 4D structured symptom model comparison.")
    parser.add_argument("--data-root", type=Path, default=service_root.parent)
    parser.add_argument("--output", type=Path, default=service_root / "phase4d_model_comparison_results.json")
    parser.add_argument("--cv-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test, dataset = load_structured_data(args.data_root)
    symptom_columns = [column for column in train.columns if column != "prognosis"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["prognosis"])
    y_test = encoder.transform(test["prognosis"])
    labels = encoder.classes_.tolist()
    X = train[symptom_columns]
    X_test = test[symptom_columns]

    results = {
        "metadata": {
            "data_root": str(args.data_root),
            "training_shape": list(train.shape),
            "testing_shape": list(test.shape),
            "dataset_csv_shape": list(dataset.shape),
            "feature_count": len(symptom_columns),
            "label_count": len(labels),
            "train_class_distribution": dict(Counter(train["prognosis"])),
            "test_class_distribution": dict(Counter(test["prognosis"])),
            "testing_rows_per_class_min": int(test["prognosis"].value_counts().min()),
            "testing_rows_per_class_max": int(test["prognosis"].value_counts().max()),
            "cv_splits": args.cv_splits,
            "optional_packages": {
                "xgboost": importlib.util.find_spec("xgboost") is not None,
                "lightgbm": importlib.util.find_spec("lightgbm") is not None,
            },
            "labels": labels,
        },
        "models": [],
    }

    for name, model in build_models().items():
        results["models"].append(
            evaluate_model(
                name=name,
                model=model,
                X=X,
                y=y,
                X_test=X_test,
                y_test=y_test,
                labels=labels,
                cv_splits=args.cv_splits,
            )
        )

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
