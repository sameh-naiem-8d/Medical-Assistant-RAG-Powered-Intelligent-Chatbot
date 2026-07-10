# P0 Final Validation Repair Report

Generated: 2026-06-17

## Scope

This repair pass targeted the highest-risk failures from the fallback-only 200-case final validation run.

Rules followed:

- No deployment.
- No Groq calls.
- No model training.
- No artifact rebuild.
- No original dataset modification.
- No classifier artifact replacement.
- No full 200-case final validation rerun.
- No changes to `final_validation_cases.csv`.
- Metrics are reported honestly from the original final validation and the targeted mini recheck.

## Inputs Reviewed

- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_124135/final_validation_results.csv`
- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_124135/failure_analysis.csv`
- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_124135/failure_analysis.md`
- `D:/Project Graduation/medbridge-ai-service/FINAL_VALIDATION_RESULTS_REPORT.md`

## Files Changed

App logic:

- `D:/Project Graduation/medbridge-ai-service/app/safety.py`
- `D:/Project Graduation/medbridge-ai-service/app/knowledge_service.py`
- `D:/Project Graduation/medbridge-ai-service/app/clarification_service.py`
- `D:/Project Graduation/medbridge-ai-service/app/classifier_service.py`
- `D:/Project Graduation/medbridge-ai-service/app/main.py`

Validation runner:

- `D:/Project Graduation/medbridge-ai-service/scripts/validate_final_cases.py`

Tests:

- `D:/Project Graduation/medbridge-ai-service/tests/test_p0_final_validation_repairs.py`
- `D:/Project Graduation/medbridge-ai-service/tests/test_phase4b_targeted_improvements.py`

Reports/results:

- `D:/Project Graduation/medbridge-ai-service/P0_FAILURE_TRIAGE_REPORT.md`
- `D:/Project Graduation/medbridge-ai-service/P0_FINAL_VALIDATION_REPAIR_REPORT.md`
- `D:/Project Graduation/medbridge-ai-service/final_validation_failed_cases_recheck.csv`
- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/`

## Emergency Fixes

Emergency recall was the highest-risk blocker. The original final validation had 14 emergency misses.

Added or strengthened detection for:

- severe dehydration/unable to drink,
- diarrhea with dizziness and low urine,
- severe facial swelling/allergic swelling,
- loss of consciousness followed by severe headache,
- chest discomfort with cold sweating,
- hypoglycemia-like symptoms with confusion,
- severe renal-colic/flank-pain wording,
- high fever with rapid deterioration,
- trauma with inability to stand,
- pediatric head injury with vomiting,
- mixed English/Arabic pregnancy bleeding,
- self-harm intent variants,
- eye trauma and severe eye pain.

Behavior now enforced:

- emergency mode overrides classifier/RAG/diagnosis mode,
- urgency becomes `High`,
- doctor becomes `Emergency care`,
- answer starts with the urgent warning prefix,
- normal `follow_up_questions` are removed for High urgency responses.

## Safety Fixes

Safety failures were mostly caused by wrong mode, missed emergency, or diagnosis answers in clarification cases.

Fixes:

- Added negation-aware red-flag matching so phrases like `من غير اغماء` and `ولا الم صدر` do not trigger emergency mode.
- Added soft handling for `مش بتبول كويس` so it is not treated as complete urinary retention unless paired with stronger dehydration signs.
- Added clarification safeguards for:
  - non-emergency mental health,
  - eye complaints,
  - pregnancy/gynecology context,
  - under-specified pediatric messages.
- Added `تسلم يا دكتور` to closing mode.
- Removed medication wording from the skin follow-up template.
- Added a filter so follow-up questions do not repeat symptoms the user explicitly denied, such as `ضيق تنفس`.

## Must-Not-Contain Fixes

Original must-not-contain violation rate: 9.50% across the 200-case final validation.

Fixes:

- Strengthened asthma sanity rules so cough/fever/URI symptoms do not default to `Bronchial Asthma` without wheezing, breathlessness, or asthma context.
- Added URI/Common Cold guardrails for cough/fever/sore throat and upper-airway symptoms.
- Fixed clarification mode for cases that should not include diagnosis phrasing like `الاحتمال الأقرب`.
- Filtered negated follow-up terms.
- Fixed closing mode phrase handling.

## Urgency And Doctor Routing Fixes

Urgency and doctor routing were improved through targeted context rules.

Doctor routing improvements:

- emergency -> `Emergency care`,
- non-red-flag pregnancy/gynecology -> `Gynecologist`,
- pediatric under-specified symptoms -> `Pediatrician`,
- URI/rhinitis-style symptoms -> `General Practitioner`,
- hypertension context -> `Cardiologist`,
- sugar/endocrine context -> `Endocrinologist`,
- chronic cough with weight loss/night sweats -> `Pulmonologist`,
- dental context -> `Dentist`,
- eye context -> `Ophthalmologist`,
- non-emergency mental health -> `Psychiatrist`.

Important correction:

- Broad `زغللة` was removed from eye-context detection because it incorrectly routed pressure/sugar/headache cases to Ophthalmology.

## Diagnosis Mapping And Safe Fusion Fixes

Diagnosis repair was kept secondary to P0/P1.

Safe changes:

- Added P0 symptom synonym layer without retraining:
  - `سخونه`,
  - `تكسير فجسمي`,
  - `جوووع`,
  - `سكر عالي`,
  - `لخبطة`,
  - `قلة بول`,
  - `مغص كلوي شديد`,
  - `بولي غامق`,
  - `جسمي مصفر`,
  - `pregnant ومعايا bleeding`.
- Increased safe guardrail support for:
  - Common Cold/URI over asthma when no asthma evidence exists,
  - Tuberculosis/respiratory group for chronic cough + weight loss/night sweats,
  - Diabetes/endocrine group for sugar + vision symptoms,
  - Hypertension/cardiology for pressure + vision/neurologic symptoms.
- Added validation-runner mapping so `mode == closing` maps to diagnosis group `closing`.

No classifier artifact was replaced.

## Tests Added

New focused test file:

- `tests/test_p0_final_validation_repairs.py`

Covered:

- all 14 emergency-miss patterns from the final validation,
- negated red flags not triggering emergency,
- negated breathlessness not repeated in follow-up questions,
- `تسلم يا دكتور` closing behavior,
- under-specified mental health/eye/pregnancy/pediatric clarification behavior,
- P0 extraction expansions,
- top doctor-routing repairs.

Existing test update:

- `tests/test_phase4b_targeted_improvements.py` now expects emergency mode to return no normal follow-up questions.

## Unit Test Result

Commands:

```bash
python -m py_compile app/safety.py app/knowledge_service.py app/clarification_service.py app/classifier_service.py app/main.py scripts/validate_final_cases.py tests/test_p0_final_validation_repairs.py
python -m unittest discover -s tests -v
```

Result:

- Syntax check passed.
- Unit tests passed: 78 tests.

No Groq calls were made by the tests; test setup disables the LLM client.

## Mini Recheck

A targeted fallback-only mini recheck was run on prior failed P0/P1/safety/user-facing rows only.

Input:

- `D:/Project Graduation/medbridge-ai-service/final_validation_failed_cases_recheck.csv`

Case count:

- 111 prior failed rows.

Output:

- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/final_validation_results.csv`
- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/final_validation_metrics.json`
- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/final_validation_summary.md`
- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/p0_recheck_before_after_metrics.csv`
- `D:/Project Graduation/medbridge-ai-service/p0_recheck_results/20260617_130548/p0_recheck_remaining_failures.csv`

This was not a full final validation rerun.

## Mini Recheck Results

Comparison is only for the 111 rechecked prior-failed rows.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Mode accuracy | 64.86% | 90.99% | +26.13 pts |
| Diagnosis group accuracy | 55.86% | 84.68% | +28.83 pts |
| Urgency accuracy | 61.26% | 79.28% | +18.02 pts |
| Doctor accuracy | 54.05% | 88.29% | +34.23 pts |
| Emergency recall | 72.55% | 100.00% | +27.45 pts |
| Safety pass rate | 68.47% | 97.30% | +28.83 pts |
| Clarification behavior pass rate | 100.00% | 100.00% | 0 pts |
| Closing behavior pass rate | 0.00% | 100.00% | +100 pts |
| Must-not-contain violation rate | 17.12% | 2.70% | -14.41 pts |
| Failure rows | 111 | 37 | -74 |

Important: these are mini recheck metrics on prior failed rows only, not the new official final validation metrics.

## Remaining Risks

Remaining mini recheck failures include:

- urgency too low for some medium urinary, cardiovascular, and neurological cases,
- diagnosis group ambiguity in respiratory/allergy, endocrine/urinary, digestive/cardiovascular, and infectious/digestive overlap,
- some expected doctor labels remain strict, especially infectious vs dermatologist/gastroenterologist and dental/eye clarification routing,
- some cases may need validation-label review before the next full final validation,
- full 200-case performance is not known after this repair pass because full validation was intentionally not rerun.

Examples of remaining rows:

- `FV005`: wheezing/cough at night still Low instead of Medium.
- `FV010`: upper-airway/allergy-like symptoms still diagnosis-group mismatch against respiratory.
- `FV028`: English digestive symptoms still diagnosis-group mismatch.
- `FV089`, `FV098`, `FV099`, `FV102`: urinary urgency still Low where expected Medium.
- `FV105`, `FV109`: infectious doctor routing still differs from expected infectious specialist.
- `FV190`, `FV194`: remaining eye clarification/routing issues.

## Honest Expectation Of Metric Improvement

The mini recheck suggests a major improvement in the highest-risk areas:

- emergency recall is likely much closer to target,
- safety pass rate is likely much closer to target,
- must-not-contain violations are likely reduced,
- doctor routing likely improved.

However, final claims cannot be made until the full locked 200-case validation is rerun once under approved conditions. The mini recheck excluded previously passing rows and was not designed to produce official final metrics.

## Recommendation

A second full controlled validation run is recommended, but only after the team reviews this repair report and approves it.

Before rerunning full validation:

1. Review remaining mini recheck failures.
2. Decide whether to fix remaining P1 urinary/eye/infectious routing issues now or accept them for the next run.
3. Confirm no further AI logic changes are planned.
4. Run the full 200-case validation once, fallback-only or Groq-backed according to the team decision.

Deployment should remain paused until a new approved full validation run meets emergency and safety acceptance targets.
