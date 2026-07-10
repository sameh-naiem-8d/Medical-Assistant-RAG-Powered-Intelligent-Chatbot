# Phase 4E Validation Set Preparation Report

Date: 2026-06-17

This phase prepared final validation files and documentation only. No validation run, Groq call, deployment, 50-case regression, 160-case validation, model training, classifier replacement, production artifact rebuild, original dataset modification, baseline-folder modification, or report/test removal was performed.

## Files Created

- `final_validation_cases.csv`
- `FINAL_VALIDATION_SCHEMA.md`
- `FINAL_VALIDATION_PROTOCOL.md`
- `scripts/validate_final_cases.py`
- `PHASE4E_VALIDATION_SET_REPORT.md`

## Existing Evaluation Assets Reviewed

Reviewed assets:

- `scripts/evaluate_system.py`
- `evaluation_results.csv`
- `phase3_regression/`
- `phase3_3_regression/`
- `PHASE3_3_POLISH_REPORT.md`
- `PHASE4A_WEAKNESS_MAP.md`
- `PHASE4A_DISEASE_COVERAGE_AUDIT.md`
- `PHASE4B_TARGETED_IMPROVEMENTS_REPORT.md`
- `PHASE4C_RAG_LABELS_POLISH_REPORT.md`
- `PHASE4D_MODEL_COMPARISON_REPORT.md`

## What Existing Evaluation Already Covers

The existing assets already cover:

- official `Testing.csv` classifier checks,
- a 50-case chatbot regression set,
- respiratory, digestive, skin, neurological, cardiovascular, emergency, general, urinary, endocrine, and infectious examples,
- Phase 3 to Phase 3.3 metric comparison,
- emergency recall and safety pass-rate tracking,
- doctor-routing checks,
- clarification mode behavior for vague/body-area-only messages,
- closing mode behavior,
- multi-turn behavior in focused unit tests,
- Phase 4B safety/routing contexts such as pregnancy, pediatrics, mental health, dental, eye, trauma, urinary, endocrine, and infectious improvements,
- Phase 4C RAG filtering and Arabic display-label polish,
- Phase 4D evidence that structured symptom-matrix models all score 100% and therefore do not justify classifier replacement alone.

Accepted Phase 3.3 50-case metrics from the existing reports:

- Diagnosis group accuracy: 96%
- Urgency accuracy: 94%
- Doctor accuracy: 98%
- Emergency recall: 100%
- Safety pass rate: 100%

## What Existing Evaluation Misses

The existing 50-case set is useful but too small for final graduation/deployment evidence.

Important gaps:

- only 50 chatbot cases,
- mostly single-turn,
- only 2 infectious cases,
- only 2 endocrine cases,
- only 2 urinary cases,
- limited pediatric coverage,
- limited pregnancy/gynecology coverage,
- limited dental and eye routing coverage,
- limited poisoning/overdose coverage,
- limited typo-heavy Arabic/Egyptian cases,
- limited mixed Arabic/English medical wording,
- no explicit `expected_mode` field,
- no explicit `expected_safety_flag` field,
- no explicit `expected_follow_up_behavior` field,
- no explicit `must_not_contain` checks,
- no dedicated final validation protocol,
- no larger final validation set designed to be run only once at the end.

## New Final Validation Set

Created:

`final_validation_cases.csv`

Case count:

- 200 cases.

This target meets the Phase 4E requirement of at least 200 cases. The set was kept at 200 to avoid rushing quality.

## Schema

The validation file uses the schema documented in:

`FINAL_VALIDATION_SCHEMA.md`

Columns:

- `case_id`
- `user_message`
- `history_json`
- `expected_mode`
- `expected_diagnosis_group`
- `expected_urgency`
- `expected_doctor`
- `expected_safety_flag`
- `expected_follow_up_behavior`
- `specialty_area`
- `risk_level`
- `dialect_level`
- `ambiguity_level`
- `must_not_contain`
- `notes`

## Case Distribution

### Mode Distribution

| Mode | Count |
|---|---:|
| diagnosis | 93 |
| emergency | 51 |
| clarification | 46 |
| closing | 10 |

### Urgency Distribution

| Urgency | Count |
|---|---:|
| Low | 80 |
| Medium | 69 |
| High | 51 |

### Specialty Distribution

| Specialty Area | Count |
|---|---:|
| infectious | 16 |
| endocrine | 16 |
| digestive | 15 |
| respiratory | 15 |
| neuro_ent | 15 |
| skin | 15 |
| general | 14 |
| urinary | 14 |
| multi_turn | 14 |
| cardiovascular | 12 |
| pediatric | 10 |
| pregnancy_gynecology | 10 |
| closing | 8 |
| mental_health | 8 |
| dental | 6 |
| eye | 6 |
| trauma | 3 |
| poisoning | 3 |

### Diagnosis Group Distribution

| Expected Diagnosis Group | Count |
|---|---:|
| emergency | 51 |
| no_diagnosis | 46 |
| respiratory | 17 |
| skin | 14 |
| endocrine | 14 |
| digestive | 12 |
| closing | 10 |
| neurological | 9 |
| infectious | 9 |
| urinary | 9 |
| cardiovascular | 6 |
| general | 3 |

### Risk Distribution

| Risk Level | Count |
|---|---:|
| low | 73 |
| medium | 76 |
| high | 51 |

### Dialect Distribution

| Dialect Level | Count |
|---|---:|
| egyptian | 174 |
| mixed | 13 |
| typo_heavy | 9 |
| formal_arabic | 4 |

### Ambiguity Distribution

| Ambiguity Level | Count |
|---|---:|
| low | 113 |
| medium | 54 |
| high | 33 |

## Required Coverage Checklist

Covered:

- clear diagnosis-like cases,
- vague/general symptoms,
- multi-turn cases with `history_json`,
- clarification cases,
- closing cases,
- emergency cases,
- infectious differentiation,
- endocrine/diabetes-like cases,
- urinary/kidney cases,
- neuro/ENT overlap,
- pediatrics,
- pregnancy/gynecology,
- mental health/self-harm,
- dental,
- eye,
- trauma,
- poisoning/overdose,
- skin,
- respiratory,
- digestive,
- typo-heavy Arabic/Egyptian messages,
- mixed Arabic/English medical words,
- cases where diagnosis should not be forced,
- cases where emergency must override everything.

## Key Counts

- Total cases: 200
- Emergency cases: 51
- Multi-turn cases: 14
- Typo-heavy or mixed Arabic/English cases: 22
- Clarification cases: 46
- Closing cases: 10
- Duplicate case IDs: 0
- Invalid `history_json` rows: 0

## Validation Runner

Created:

`scripts/validate_final_cases.py`

The script is designed to be run later. It was not executed in this phase.

Prepared capabilities:

- loads `final_validation_cases.csv`,
- uses local FastAPI `TestClient`,
- sends `message` and `history`,
- compares mode, diagnosis group, urgency, doctor routing, safety, follow-up behavior, and `must_not_contain`,
- records `llm_mode` as `groq` or `fallback`,
- writes timestamped results under `final_validation_results/`,
- computes:
  - mode accuracy,
  - diagnosis group accuracy,
  - urgency accuracy,
  - doctor accuracy,
  - emergency recall,
  - safety pass rate,
  - clarification behavior pass rate,
  - closing behavior pass rate,
  - must-not-contain violation rate,
  - average confidence,
  - average latency.

## Final Validation Protocol

Created:

`FINAL_VALIDATION_PROTOCOL.md`

It documents:

- when to run final validation,
- required environment,
- Groq-enabled vs fallback considerations,
- how to avoid quota contamination,
- how to separate Groq failures from logic failures,
- required metrics,
- acceptance targets,
- manual review requirements.

Suggested acceptance targets:

- emergency recall: >= 98%
- safety pass rate: >= 95%
- mode accuracy: >= 90%
- urgency accuracy: >= 90%
- doctor accuracy: >= 90%
- diagnosis group accuracy: >= 85%
- clarification behavior pass rate: >= 95%
- closing behavior pass rate: >= 95%
- must-not-contain violation rate: <= 2%

## Why This Is Stronger Than Testing.csv

`Testing.csv` has 41 structured one-hot symptom rows, one per disease label. It is useful for verifying the structured classifier artifact, but it does not test the real chatbot problem.

The new final validation set is stronger because it tests:

- natural Arabic/Egyptian user messages,
- spelling variation and colloquial wording,
- multi-turn history,
- clarification mode,
- closing mode,
- emergency override behavior,
- doctor routing for unsupported specialties,
- must-not-contain safety constraints,
- broad diagnosis grouping instead of only exact structured labels,
- user-facing behavior and safety expectations.

This directly targets the real MedBridge chatbot pipeline rather than only the clean symptom matrix.

## Limitations

- This is engineering validation, not clinician-certified clinical validation.
- Expected labels are practical QA expectations, not a medical gold standard.
- Some cases intentionally use broad groups because exact disease prediction is not clinically safe from a short message.
- The set should not be tuned against repeatedly. If it becomes a tuning set, it stops being a fair final validation set.
- Specialty contexts such as dental, eye, pregnancy, pediatrics, poisoning, and mental health are mostly routing/safety contexts, not classifier-supported disease labels.
- Groq behavior still needs a stable quota and a consistent run condition.

## Validation Status

Validation was not run.

No Groq call was made.

No 50-case regression was run.

No 160-case validation was run.

No final locked validation was run.

No production artifacts were rebuilt or replaced.

## Recommendation

Keep `final_validation_cases.csv` locked until the final evaluation session.

Before running final validation, confirm:

- the team approves this set,
- Groq quota is stable,
- no further logic tuning is planned,
- the accepted baseline remains preserved,
- the final validation will be reported honestly with Groq/fallback rows separated.
