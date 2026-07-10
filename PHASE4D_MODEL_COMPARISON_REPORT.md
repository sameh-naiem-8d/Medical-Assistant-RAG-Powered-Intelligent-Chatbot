# Phase 4D: Lightweight Structured Model Comparison

Date: 2026-06-17

This phase is an isolated experiment/report only. No deployment, Groq call, production artifact rebuild, classifier replacement, dataset modification, 50-case regression, 160-case validation, or final validation was performed.

## Purpose

The goal was to check whether replacing the current structured symptom classifier is worth considering later. The comparison used only the structured symptom datasets and did not change the production AI pipeline.

## Files Created

- `scripts/phase4d_model_comparison.py`
- `phase4d_model_comparison_results.json`
- `PHASE4D_MODEL_COMPARISON_REPORT.md`
- `PHASE4D_TRAINING_NOTEBOOK_PLAN.md`

## Current Classifier Setup

Current production artifacts inspected:

- `artifacts/disease_classifier.pkl`
- `artifacts/disease_label_encoder.pkl`
- `artifacts/symptom_columns.pkl`

Current classifier loading path:

- `app/classifier_service.py`

Current artifact training path:

- `scripts/build_artifacts.py`

Detected current production model:

- Model type: `RandomForestClassifier`
- Training parameters from build script:
  - `n_estimators=300`
  - `class_weight="balanced"`
  - `random_state=42`
  - `n_jobs=-1`
- Feature count: 132 symptom columns
- Disease label count: 41
- Confidence support: yes, through `predict_proba`

Production classifier artifacts were not overwritten. Current observed artifact timestamps:

- `disease_classifier.pkl`: 2026-06-05 23:06:33
- `disease_label_encoder.pkl`: 2026-06-05 23:06:33
- `symptom_columns.pkl`: 2026-06-05 23:06:33

## Datasets Inspected

Structured classifier datasets:

- `D:/Project Graduation/Training.csv`
- `D:/Project Graduation/Testing.csv`
- `D:/Project Graduation/dataset.csv`

Dataset shapes:

| Dataset | Shape | Notes |
|---|---:|---|
| `Training.csv` | 4,920 x 133 | 132 symptom features + `prognosis` |
| `Testing.csv` | 41 x 133 | 132 symptom features + `prognosis` |
| `dataset.csv` | 4,920 x 18 | Disease plus symptom-list columns; inspected but not used for model metrics |

Class distribution:

- `Training.csv`: 41 classes, exactly 120 rows per class.
- `Testing.csv`: 41 classes, exactly 1 row per class after stripping label whitespace.
- Missing classes: none detected after normalizing labels.
- Imbalance: none in `Training.csv`; the dataset is perfectly balanced.

Important limitation:

`Testing.csv` is too small and too clean to prove real-world performance. It contains only one example per disease, and all examples are already structured one-hot symptom vectors. This does not measure Arabic symptom extraction, colloquial wording, history handling, RAG quality, safety rules, or user-facing answer quality.

## Disease Labels

The structured classifier covers these 41 internal labels:

`(vertigo) Paroymsal  Positional Vertigo`, `AIDS`, `Acne`, `Alcoholic hepatitis`, `Allergy`, `Arthritis`, `Bronchial Asthma`, `Cervical spondylosis`, `Chicken pox`, `Chronic cholestasis`, `Common Cold`, `Dengue`, `Diabetes`, `Dimorphic hemmorhoids(piles)`, `Drug Reaction`, `Fungal infection`, `GERD`, `Gastroenteritis`, `Heart attack`, `Hepatitis B`, `Hepatitis C`, `Hepatitis D`, `Hepatitis E`, `Hypertension`, `Hyperthyroidism`, `Hypoglycemia`, `Hypothyroidism`, `Impetigo`, `Jaundice`, `Malaria`, `Migraine`, `Osteoarthristis`, `Paralysis (brain hemorrhage)`, `Peptic ulcer diseae`, `Pneumonia`, `Psoriasis`, `Tuberculosis`, `Typhoid`, `Urinary tract infection`, `Varicose veins`, `hepatitis A`.

## Models Compared

Required lightweight tabular models:

- Current RandomForest-style baseline: `RandomForestClassifier`, 300 trees, balanced.
- Logistic Regression: `LogisticRegression`, lbfgs, balanced.
- Linear SVM: `LinearSVC`, balanced.
- RandomForest: 100 trees, balanced.
- ExtraTrees: 300 trees, balanced.

Optional packages:

- XGBoost: not installed, not used.
- LightGBM: not installed, not used.

No packages were installed.

## Evaluation Method

Evaluation used:

- 5-fold stratified cross-validation on `Training.csv`.
- Official `Testing.csv` check after fitting each model on all training rows.
- Metrics:
  - accuracy
  - macro F1
  - weighted F1
  - top confusion pairs
  - weakest labels by per-class F1
  - estimated serialized model size
  - prediction latency estimate
  - probability/confidence support

Timing notes:

- Latency was measured on this local laptop and should be treated as a relative estimate, not a production benchmark.
- Model size was estimated by serializing each trained model to a temporary joblib file outside `artifacts/`.

## Metrics Table

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 | Testing.csv Accuracy | Testing.csv Macro F1 | CV Latency ms/sample | Test Latency ms/sample | Size MB | Predict Proba | Mean CV Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| current_rf_300_balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0753 | 1.7274 | 20.789 | yes | 0.9984 |
| logistic_regression_lbfgs | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0041 | 0.0739 | 0.045 | yes | 0.9784 |
| linear_svm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0023 | 0.1951 | 0.045 | no | n/a |
| random_forest_100_balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0282 | 0.6515 | 6.927 | yes | 0.9985 |
| extra_trees_300_balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0606 | 1.1840 | 19.778 | yes | 1.0000 |

## Confusion and Weak Labels

Top confusion pairs:

- None found in 5-fold cross-validation for any compared model.

Weak labels:

- No weak labels appeared in the structured cross-validation output.
- Every label reported F1 = 1.0000 in this controlled matrix experiment.

Interpretation:

This is not proof that the chatbot will diagnose real Arabic messages perfectly. It means the one-hot structured symptom matrix is highly separable and probably partially synthetic or template-like.

## Practicality and Integration Notes

### Current RF 300 Balanced

- Structured performance: perfect.
- Probability support: yes.
- Fast enough for FastAPI: yes.
- Serialized size: about 20.8 MB.
- Integration risk: lowest, because this is the current artifact style.
- Replacement value: none, because it is already in production.

### Logistic Regression

- Structured performance: perfect.
- Probability support: yes.
- Fast enough for FastAPI: yes, fastest probability-supporting model in this run.
- Serialized size: about 0.045 MB.
- Calibration: may still need probability calibration checks on realistic chatbot cases.
- Likely real-chatbot benefit: uncertain. It may simplify deployment size/latency, but it does not solve Arabic symptom extraction or weak category issues.
- Replacement value: possible future candidate only after full regression and validation.

### Linear SVM

- Structured performance: perfect.
- Probability support: no direct `predict_proba`.
- Fast enough for FastAPI: yes.
- Serialized size: about 0.045 MB.
- Calibration: required if the current confidence/fusion behavior expects probabilities.
- Likely real-chatbot benefit: uncertain.
- Replacement value: not recommended now because the current pipeline depends on top probabilities.

### RandomForest 100 Balanced

- Structured performance: perfect.
- Probability support: yes.
- Fast enough for FastAPI: yes.
- Serialized size: about 6.9 MB, much smaller than the current 300-tree RF.
- Calibration: probably similar to current RF, but still needs realistic validation.
- Likely real-chatbot benefit: mostly size/latency, not accuracy.
- Replacement value: possible future deployment-size candidate, not justified now.

### ExtraTrees 300 Balanced

- Structured performance: perfect.
- Probability support: yes.
- Fast enough for FastAPI: yes.
- Serialized size: about 19.8 MB.
- Calibration: confidence looked extremely high, which may be overconfident on clean structured data.
- Likely real-chatbot benefit: uncertain.
- Replacement value: not compelling over current RF.

## Safety Decision

Changing the structured classifier could affect:

- top-5 disease candidates
- confidence values
- fusion behavior
- follow-up vs diagnosis mode thresholds
- doctor routing when diagnosis changes

Emergency safety is mostly protected by the separate safety layer, but classifier replacement could still affect non-emergency routing and diagnosis wording. Therefore replacement should not be approved based only on this matrix experiment.

## Is Any Model Better Than Current?

On structured metrics:

- No model is meaningfully better because all reached 100% CV and Testing.csv scores.

On deployment practicality:

- Logistic Regression is much smaller and faster while retaining probability output.
- RandomForest 100 is smaller than the current RF and still has probability output.

On medical chatbot quality:

- No model proved better. The real bottlenecks are still Arabic symptom extraction, coverage, fusion guardrails, and validation quality.

## Replacement Recommendation

Do not replace the production classifier now.

Reason:

- The current model is already perfect on this structured dataset.
- The official test file is only 41 rows, one per class.
- Cross-validation was also perfect for all models, suggesting the dataset is too separable to discriminate models.
- This experiment does not measure real Arabic patient messages.
- Full chatbot safety/regression/final validation is still postponed.

Future replacement may be worth testing only if:

- A candidate model keeps or improves structured CV performance.
- Confidence behavior is stable under realistic Arabic evaluation cases.
- Phase 3/4 regression cases do not regress.
- Final locked validation confirms diagnosis, urgency, doctor routing, emergency recall, and safety.
- The team explicitly approves replacing `disease_classifier.pkl`.

## Risks

- Over-interpreting perfect structured metrics.
- Replacing a stable model for smaller file size without proving chatbot-level improvement.
- Probability distribution changes causing fusion behavior shifts.
- Linear SVM lacking direct probability output.
- Tree models appearing overconfident on clean structured data.
- The structured dataset does not represent colloquial Arabic symptom extraction.

## Next Recommended Step

Do not train or replace now.

Recommended next step:

Build a candidate-training notebook or Colab experiment that exports candidate artifacts to a separate folder, then run:

1. structured metrics,
2. Phase 3/4 unit tests,
3. 50-case regression,
4. locked 160-case validation,
5. Groq/fallback-separated final validation,
6. manual medical safety review.

Only after those pass should the team consider replacing the current RandomForest artifact.
