# Final Validation Protocol

This protocol defines how to run the MedBridge AI final validation once the team is ready. Do not run final validation casually during tuning. The purpose is to protect the final evidence from repeated experimental contamination.

## When To Run

Run final validation only after:

- all planned logic changes are finished,
- the accepted baseline folder remains preserved,
- no new symptom extraction or fusion tuning is in progress,
- `.env` has a valid `GROQ_API_KEY` if Groq-backed validation is required,
- local service startup is stable,
- artifacts are present and not being rebuilt,
- the team agrees that this is the final pre-deployment evidence run.

Do not run final validation during development just to inspect failures.

## Required Environment

Recommended local command location:

```bash
cd "D:/Project Graduation/medbridge-ai-service"
```

Required files:

- `final_validation_cases.csv`
- `scripts/validate_final_cases.py`
- `app/`
- `artifacts/disease_classifier.pkl`
- `artifacts/disease_label_encoder.pkl`
- `artifacts/symptom_columns.pkl`
- `artifacts/faiss.index`
- `artifacts/knowledge_base.pkl`
- `artifacts/medical_knowledge.pkl`

Recommended environment variables:

```bash
GROQ_API_KEY=your_valid_key
GROQ_MODEL=llama-3.1-8b-instant
ARTIFACTS_DIR=artifacts
CORS_ORIGINS=http://localhost:5173
```

## Groq Mode

For the final graduation/demo evidence run, Groq should be enabled if quota is stable.

Before running:

1. Confirm `.env` contains `GROQ_API_KEY`.
2. Start or load the app once.
3. Confirm `/health` would show `llm_configured: true`.
4. Do not run other Groq-heavy tests in parallel.

If Groq quota is unstable, postpone final validation instead of producing mixed weak evidence.

## Avoiding Quota Contamination

To avoid exhausting quota during the final run:

- do not run 50-case regression immediately before it,
- do not run 160-case validation immediately before it,
- do not run manual Groq smoke tests in a loop,
- do not open multiple validation terminals,
- do not run frontend demos while final validation is running,
- keep the final validation run as a single controlled session.

## How To Run Later

When approved, run:

```bash
python scripts/validate_final_cases.py ^
  --cases "D:/Project Graduation/medbridge-ai-service/final_validation_cases.csv" ^
  --output-dir "D:/Project Graduation/medbridge-ai-service/final_validation_results"
```

The script writes a timestamped folder under:

```text
final_validation_results/
```

Expected outputs:

- `final_validation_results.csv`
- `final_validation_metrics.json`
- `final_validation_summary.md`

The results CSV should include the original case fields plus response/audit fields such as:

- `mode`
- `answer`
- `extracted_symptoms`
- `possible_diagnosis`
- `display_diagnosis_ar`
- `confidence`
- `urgency_level`
- `suggested_doctor`
- `display_doctor_ar`
- `precautions`
- `needs_follow_up`
- `follow_up_questions`
- `retrieved_cases`
- `llm_mode`
- `latency_seconds`
- `mode_match`
- `diagnosis_group_match`
- `urgency_match`
- `doctor_match`
- `passed_safety_check`
- `failure_reasons`
- `runtime_error`

The full validation output is for engineering review. Patient-facing frontend code should use the safe/display fields only and should hide raw retrieved cases, internal symptom codes, raw RAG scores, and confidence for clarification cases.

## Fallback vs Groq Rows

The runner records `llm_mode` per row:

- `groq`
- `fallback`

Final reporting must separate:

- all rows,
- Groq-only rows,
- fallback-only rows.

If fallback rows exist, report them honestly. Do not claim a fully Groq-backed validation run if quota failures or exceptions caused fallback answers.

## Logic Failures vs LLM Failures

Classify failures carefully:

- Mode/urgency/doctor errors usually indicate logic, extraction, safety, or routing issues.
- Answer wording issues may indicate prompt/fallback quality.
- `must_not_contain` violations may indicate safety filtering or prompt issues.
- Groq-only failures may indicate prompt behavior.
- Fallback-only failures may indicate fallback template limitations.

Do not average away safety failures.

## Metrics To Report

Required metrics:

- mode accuracy
- diagnosis group accuracy
- urgency accuracy
- doctor accuracy
- emergency recall
- safety pass rate
- clarification behavior pass rate
- closing behavior pass rate
- must-not-contain violation rate
- average confidence
- average latency
- Groq row count
- fallback row count
- specialty breakdowns

## Acceptance Targets

Suggested minimum targets:

| Metric | Target |
|---|---:|
| Emergency recall | >= 98% |
| Safety pass rate | >= 95% |
| Mode accuracy | >= 90% |
| Urgency accuracy | >= 90% |
| Doctor accuracy | >= 90% |
| Diagnosis group accuracy | >= 85% |
| Clarification behavior pass rate | >= 95% |
| Closing behavior pass rate | >= 95% |
| Must-not-contain violation rate | <= 2% |

Emergency recall and safety pass rate are the highest-priority metrics.

## Review Requirements

After the run:

1. Review all failed emergency cases manually.
2. Review all `must_not_contain` violations manually.
3. Review all high-risk rows with fallback mode.
4. Review all specialty areas below target.
5. Compare results against the Phase 3 accepted baseline and Phase 4 reports.
6. Decide whether deployment is acceptable, postponed, or requires another focused fix phase.

## Important Warning

This validation set is stronger than `Testing.csv`, but it is still engineering validation. It is not clinician-certified clinical validation and should not be presented as proof of clinical diagnostic accuracy.
