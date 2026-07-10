# Final Validation Results

Generated: 2026-06-17T13:18:25

Groq client available from local .env: True

Groq calls during validation: 0

Validation LLM mode: fallback-only (`llm_service.client = None` during case execution)

Groq model configured locally: llama-3.1-8b-instant

## Metrics

| metric | value |
| --- | --- |
| case_count | 200 |
| groq_rows | 0 |
| fallback_rows | 200 |
| mode_accuracy | 0.9350 |
| diagnosis_group_accuracy | 0.8750 |
| urgency_accuracy | 0.8750 |
| doctor_accuracy | 0.9250 |
| emergency_recall | 1.0000 |
| safety_pass_rate | 0.9800 |
| clarification_behavior_pass_rate | 0.9783 |
| closing_behavior_pass_rate | 1.0000 |
| must_not_contain_violation_rate | 0.0200 |
| average_confidence | 0.3565 |
| average_latency_seconds | 0.4449 |

## Specialty Breakdown

| specialty_area | case_count | mode_accuracy | urgency_accuracy | doctor_accuracy | safety_pass_rate |
| --- | --- | --- | --- | --- | --- |
| cardiovascular | 12 | 91.67% | 75.00% | 91.67% | 100.00% |
| closing | 8 | 100.00% | 100.00% | 100.00% | 100.00% |
| dental | 6 | 100.00% | 100.00% | 83.33% | 100.00% |
| digestive | 15 | 100.00% | 86.67% | 100.00% | 100.00% |
| endocrine | 16 | 81.25% | 81.25% | 87.50% | 93.75% |
| eye | 6 | 83.33% | 100.00% | 66.67% | 83.33% |
| general | 14 | 92.86% | 85.71% | 92.86% | 92.86% |
| infectious | 16 | 81.25% | 81.25% | 68.75% | 100.00% |
| mental_health | 8 | 100.00% | 100.00% | 100.00% | 100.00% |
| multi_turn | 14 | 100.00% | 85.71% | 100.00% | 100.00% |
| neuro_ent | 15 | 100.00% | 93.33% | 100.00% | 100.00% |
| pediatric | 10 | 100.00% | 100.00% | 100.00% | 100.00% |
| poisoning | 3 | 100.00% | 100.00% | 100.00% | 100.00% |
| pregnancy_gynecology | 10 | 90.00% | 80.00% | 100.00% | 100.00% |
| respiratory | 15 | 93.33% | 93.33% | 93.33% | 93.33% |
| skin | 15 | 93.33% | 100.00% | 93.33% | 100.00% |
| trauma | 3 | 100.00% | 100.00% | 100.00% | 100.00% |
| urinary | 14 | 92.86% | 57.14% | 92.86% | 100.00% |

## Notes

- Review Groq and fallback rows separately before making deployment claims.
- Emergency recall and safety pass rate are higher priority than diagnosis group accuracy.
- This remains engineering validation and is not clinician-certified clinical validation.
