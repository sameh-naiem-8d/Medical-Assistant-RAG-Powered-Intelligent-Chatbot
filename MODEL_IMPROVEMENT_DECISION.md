# Model Improvement Decision

Generated: 2026-06-17

No model training was performed in this handoff sprint.

## Decision

Keep the current classifier and hybrid fusion system for final team handoff.

Do not replace the classifier immediately before integration unless a new model:

- Clearly improves locked evaluation metrics.
- Improves weak categories, not only average accuracy.
- Preserves emergency recall and safety pass rate.
- Passes unit tests.
- Passes regression tests.
- Does not depend on unreviewed synthetic medical data.

## Why Immediate Training Is Not Worth It Today

Training a new model now would be risky because:

- The current service has an accepted Phase 3.3 behavior baseline.
- A rushed model could improve one metric while weakening emergency safety or doctor routing.
- The structured `Testing.csv` benchmark is too small and too clean to prove real-world performance.
- Real Arabic patient messages need a stronger evaluation set and preferably clinician review.
- Heavy transformer training should not run locally on the laptop.
- Any new model needs proper comparison, cross-validation, and locked validation before replacement.

The right action today is handoff readiness, not last-minute model replacement.

## Current Classifier Role

The current classifier is a structured symptom-matrix model trained from the project disease/symptom datasets.

It is useful because:

- It is fast.
- It is deterministic.
- It works with the 132 symptom columns.
- It provides top disease candidates for the fusion layer.
- It can run in the FastAPI service without GPU.

It is not enough alone because:

- Free-text Arabic symptoms are noisy.
- Many diseases share symptoms.
- Patient messages are often vague or multi-turn.
- The original structured benchmark is much easier than real conversation.
- It does not understand context unless extraction and fusion provide that context.

## Future Model Candidates

### RandomForest

Expected benefit:

- Strong baseline for structured symptom columns.
- Already compatible with current artifact format.

Risk:

- Can overfit structured symptom matrices.
- Does not understand natural language.

Training requirements:

- CPU is enough for current dataset size.
- Cross-validation needed.

### Logistic Regression

Expected benefit:

- Simple, interpretable linear baseline.
- Often strong on sparse binary symptom vectors.

Risk:

- May underperform on nonlinear symptom interactions.

Training requirements:

- CPU is enough.
- Needs class weighting and cross-validation.

### Linear SVM

Expected benefit:

- Strong baseline for high-dimensional sparse features.
- Good candidate for structured symptom vectors.

Risk:

- Probability calibration is needed if confidence scores are required.

Training requirements:

- CPU is enough for structured symptoms.
- Calibration step recommended.

### XGBoost

Expected benefit:

- Can capture nonlinear symptom interactions.
- Often strong for tabular classification.

Risk:

- More tuning complexity.
- Can overfit small structured datasets.

Training requirements:

- CPU possible for current size.
- Colab/Kaggle recommended for larger experiments.

### LightGBM

Expected benefit:

- Fast gradient boosting for tabular data.
- Good candidate for structured symptom classification.

Risk:

- Requires careful validation and tuning.
- May not improve if dataset is too small or too synthetic.

Training requirements:

- CPU is likely enough for structured data.
- Colab/Kaggle recommended for larger comparisons.

### MARBERT

Expected benefit:

- Strong Arabic language understanding.
- Better fit for Egyptian/Arabic symptom messages if enough labeled text exists.

Risk:

- Needs high-quality Arabic symptom-to-disease text data.
- Higher risk of hallucinated confidence if trained poorly.

Training requirements:

- GPU recommended.
- Colab/Kaggle required.

### CAMeLBERT

Expected benefit:

- Arabic-focused transformer family.
- Useful candidate for formal and dialect Arabic experiments.

Risk:

- Needs labeled natural-language medical data.
- Requires careful preprocessing and evaluation.

Training requirements:

- GPU recommended.
- Colab/Kaggle required.

### XLM-R

Expected benefit:

- Strong multilingual transformer.
- Can handle mixed Arabic/English if needed.

Risk:

- Larger models can be slow and expensive.
- Needs real labeled examples to beat the current hybrid system.

Training requirements:

- GPU required for fine-tuning.
- Colab/Kaggle required.

## Dataset Requirements Before Bigger Models

To justify a transformer or major model replacement, the project needs better data:

- Arabic medical QA datasets for stronger RAG.
- Arabic natural-language symptom-to-disease examples for classifier training.
- Disease-symptom structured datasets for tabular comparison.
- Medical dialogue datasets for multi-turn evaluation.
- Clinician-reviewed Arabic validation cases for reliable metrics.

Do not mix new datasets into production artifacts until license, quality, label mapping, and evaluation impact are reviewed.

## Why "All Diseases" Cannot Be Claimed

The current datasets cover a limited disease label set and a fixed symptom-column schema.

The chatbot can help with broad triage and routing, but it cannot honestly claim:

- Coverage of all diseases.
- Definitive diagnosis.
- Replacement of clinical examination.
- Medication or treatment authority.

Realistic claim:

```text
MedBridge AI is a broad Arabic symptom triage and doctor-routing assistant.
```

## Future Comparison Protocol

Recommended process before replacing the classifier:

1. Freeze a locked evaluation set.
2. Train each candidate model on the same training data.
3. Use cross-validation where possible.
4. Compare:
   - Diagnosis group accuracy.
   - Urgency accuracy.
   - Doctor routing accuracy.
   - Emergency recall.
   - Safety pass rate.
   - Average confidence.
   - Latency.
5. Inspect failed cases manually.
6. Run regression tests.
7. Replace the model only if improvement is clear and safety is preserved.

## Final Recommendation

For team handoff, keep the current RandomForest-based hybrid system with symptom extraction, fusion guardrails, RAG, knowledge layer, and constrained LLM response generation.

For the next research phase, prepare controlled Colab/Kaggle experiments comparing tabular baselines first, then transformer models only after better Arabic labeled data is available.

