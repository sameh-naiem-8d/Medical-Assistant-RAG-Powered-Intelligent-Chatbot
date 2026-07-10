# Phase 4D Future Training Notebook Plan

Date: 2026-06-17

This is a future experiment plan only. It does not approve replacing the current classifier.

## Goal

Create a controlled Colab/Kaggle notebook later to compare structured symptom classifiers and optionally prepare candidate artifacts without overwriting the accepted MedBridge AI production artifacts.

## Where To Run

For lightweight sklearn tabular models, local execution is acceptable.

Use Colab or Kaggle if:

- running larger repeated experiments,
- adding hyperparameter search,
- evaluating large candidate sets,
- testing transformer classifiers,
- running long validation batches.

No GPU is required for the structured sklearn models. GPU is only relevant for future transformer research.

## Input Datasets

Use copies of:

- `Training.csv`
- `Testing.csv`
- `dataset.csv`

Do not modify source datasets.

Required preprocessing:

- Strip whitespace from `prognosis` labels.
- Use the same 132 symptom columns as production.
- Confirm train/test labels match after normalization.
- Keep a fixed `random_state=42`.
- Save the exact feature order used by the candidate model.

## Models To Compare

Structured tabular candidates:

- Current production-style RandomForest, 300 trees, balanced.
- Logistic Regression.
- Linear SVM.
- RandomForest with fewer trees, such as 100.
- ExtraTrees.
- XGBoost only if already available in the environment or intentionally approved.
- LightGBM only if already available in the environment or intentionally approved.

Do not install or use heavy packages in the production service just for experimentation.

## Metrics To Report

Structured metrics:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted precision
- weighted recall
- weighted F1
- per-class precision/recall/F1
- top confusion pairs
- weakest labels by macro F1

Operational metrics:

- fit time
- prediction latency per sample
- serialized model size
- whether `predict_proba` is supported
- average confidence
- confidence distribution by correct/incorrect predictions
- calibration curve or expected calibration error if practical

Medical/chatbot metrics before replacement:

- diagnosis group accuracy
- urgency accuracy
- doctor recommendation accuracy
- emergency recall
- safety pass rate
- follow-up question behavior
- multi-turn improvement
- fallback/Groq mode split if LLM evaluation is included later

## Evaluation Protocol

Minimum structured experiment:

1. Load `Training.csv`.
2. Run 5-fold stratified cross-validation.
3. Train on all `Training.csv`.
4. Evaluate on official `Testing.csv`.
5. Compare metrics and confusion pairs.
6. Save results to a candidate-only report.

Do not rely only on `Testing.csv`, because it has one row per class.

Required chatbot-level validation before replacement:

1. Run existing unit tests.
2. Run the 50-case regression set.
3. Run the locked 160-case validation.
4. Separate Groq and fallback rows.
5. Review failures manually.
6. Confirm no regression in emergency recall or safety pass rate.

## Candidate Artifact Export

Candidate artifacts must be written to a separate folder, never directly to production `artifacts/`.

Recommended folder:

```text
candidate_artifacts/phase4d_<model_name>_<date>/
```

Candidate files:

- `disease_classifier.pkl`
- `disease_label_encoder.pkl`
- `symptom_columns.pkl`
- `candidate_metrics.json`
- `candidate_model_card.md`

The production files must remain untouched:

- `artifacts/disease_classifier.pkl`
- `artifacts/disease_label_encoder.pkl`
- `artifacts/symptom_columns.pkl`

## Candidate Model Card

Each candidate should document:

- model type and parameters
- training data paths
- feature count and feature order
- label count and label names
- cross-validation metrics
- official Testing.csv metrics
- realistic chatbot validation metrics
- known regressions
- confidence/probability behavior
- model size and latency
- whether calibration is needed
- replacement recommendation

## Safe Replacement Procedure

Only replace the production classifier after explicit approval.

Safe replacement steps:

1. Preserve the current baseline folder and production artifacts.
2. Train the candidate model in a separate folder.
3. Run structured metrics.
4. Run unit tests.
5. Run 50-case regression.
6. Run locked 160-case validation.
7. Run final Groq-backed validation if deployment is near.
8. Compare old vs candidate metrics.
9. Review medical safety failures manually.
10. If approved, copy candidate artifacts into `artifacts/`.
11. Restart local FastAPI and run smoke tests.
12. Update reports with exact artifact version and date.

Never overwrite `disease_classifier.pkl` during experimentation.

## Why Structured Accuracy Is Not Enough

The structured dataset already produces perfect scores for multiple lightweight models. That means:

- the one-hot symptom matrix is highly separable,
- Testing.csv is too small to be a real benchmark,
- structured accuracy does not measure Arabic symptom extraction,
- structured accuracy does not measure vague user messages,
- structured accuracy does not measure RAG behavior,
- structured accuracy does not measure answer safety,
- structured accuracy does not measure doctor-routing quality.

Any replacement must prove chatbot-level improvement, not just matrix accuracy.

## Future Transformer Research

Do not train transformers now.

Future transformer candidates only if enough high-quality labeled Arabic/Egyptian data exists:

- MARBERT
- CAMeLBERT
- XLM-R

Potential transformer task:

- input: Arabic/Egyptian natural-language symptom message,
- output: diagnosis group or disease label,
- optional output: urgency group.

Transformer requirements:

- labeled Arabic symptom-to-disease examples,
- train/validation/test split with no leakage,
- GPU runtime in Colab/Kaggle,
- careful safety review,
- comparison against the hybrid system, not just the structured classifier.

Transformer risks:

- hallucinated confidence if labels are noisy,
- overfitting to synthetic text,
- worse emergency safety if used without rules,
- difficult explainability,
- higher latency and deployment cost.

## Recommended Future Path

Short term:

- Keep the current production RandomForest.
- Consider Logistic Regression or RandomForest 100 only as future size/latency candidates.
- Do not replace based on Phase 4D alone.

Medium term:

- Expand realistic Arabic/Egyptian validation cases.
- Measure chatbot-level improvement, not only structured classification.
- Calibrate confidence if a new model changes probability behavior.

Long term:

- Explore transformer classifiers only after building or obtaining enough reliable Arabic natural-language labels.
