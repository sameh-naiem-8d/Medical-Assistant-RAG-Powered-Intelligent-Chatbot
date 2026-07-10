# Phase 4B Targeted Improvements Report

Generated: 2026-06-17

This phase made targeted coverage and safety improvements based on Phase 4A audits.

No deployment, Groq call, final validation, 50-case regression, 160-case validation, model training, artifact rebuild, dataset modification, classifier replacement, report removal, notebook removal, artifact removal, or baseline overwrite was performed.

## Audit Documents Reviewed

- `PHASE4A_DISEASE_COVERAGE_AUDIT.md`
- `PHASE4A_SYMPTOM_COVERAGE_AUDIT.md`
- `PHASE4A_WEAKNESS_MAP.md`
- `PHASE4_IMPROVEMENT_PLAN.md`
- `PHASE4A_AUDIT_SUMMARY.md`

Main practical fixes selected:

- Expand high-confidence Arabic/Egyptian symptom phrases for weak areas.
- Add context detection for unsupported specialties without pretending to diagnose unsupported diseases.
- Strengthen P0 safety rules for self-harm, pregnancy, pediatrics, urinary retention, trauma, and vision loss.
- Improve doctor routing for clear specialty contexts.
- Improve follow-up questions for fever/infection, urinary, endocrine, neuro/ENT, child, pregnancy, mental health, dental, eye, and trauma cases.

## Files Changed

- `app/classifier_service.py`
- `app/safety.py`
- `app/clarification_service.py`
- `app/knowledge_service.py`
- `app/main.py`
- `app/llm_service.py`
- `tests/test_phase4b_targeted_improvements.py`

## Symptom Phrases Added

Added a separate `PHASE4B_TARGETED_SYNONYMS` layer in `classifier_service.py`.

Areas expanded:

- Infectious/fever:
  - `سخونية شديدة`
  - `حرارة شديدة`
  - `حرارة بقالها أيام`
  - `تعرق مع حرارة`
  - `تكسير في الجسم`
  - `كحة مع بلغم`
  - `ألم حلق مع حرارة`
  - `طفح مع حرارة`

- Endocrine/diabetes-like:
  - `تبول كتير`
  - `بول كتير`
  - `جوع شديد`
  - `هبوط سكر`
  - `ارتفاع سكر`
  - `السكر واطي`
  - `السكر عالي`
  - `ريحة نفس غريبة`
  - `دوخة مع سكر`

- Urinary/kidney:
  - `حرقان بول`
  - `دم في البول`
  - `ألم جنب`
  - `ألم في الخاصرة`
  - `مغص كلوي`
  - `احتباس بول`
  - `صعوبة التبول`
  - `مش عارف اتبول`

- Neuro/ENT overlap:
  - `الدنيا بتلف`
  - `عدم اتزان`
  - `مش متزن`
  - `تنميل`
  - `ضعف في ناحية`
  - `تيبس رقبة`
  - `رقبة ناشفة`

- Eye/vision:
  - `احمرار عين`
  - `مش شايف كويس فجأة`
  - `تغير في النظر`
  - `فقدان نظر`

- Mental health and trauma:
  - `قلق`
  - `توتر`
  - `نوبات هلع`
  - `خوف شديد`
  - `اكتئاب`
  - `كدمة`
  - `إصابة`

## Safety Guardrails Added

Added Phase 4B red-flag phrases and context detection in `safety.py`.

New high-risk areas:

- Suicidal/self-harm ideation.
- Pregnancy bleeding or severe pregnancy pain.
- Child not feeding, severe lethargy, convulsions, or breathing difficulty.
- Urinary retention.
- Severe headache with neck stiffness, fever, or confusion wording.
- Sudden vision loss.
- Head injury.
- Deep wound or persistent/severe bleeding.
- Poisoning/overdose wording.
- One-sided leg swelling/pain phrase.

Safety behavior:

- Red flags return `High` urgency.
- `High` urgency still routes to `Emergency care`.
- No medication or OTC advice was added.
- Emergency mode still comes before routine clarification/diagnosis.

## Routing Improvements

Added context-aware doctor routing in `knowledge_service.py`.

New routing behavior:

- Dental pain/swelling -> `Dentist`.
- Eye pain/redness/vision change -> `Ophthalmologist`.
- Pregnancy context -> `Gynecologist`, or `Emergency care` if urgent.
- Child context -> `Pediatrician`, or `Emergency care` if urgent.
- Self-harm -> `Emergency care`.
- Anxiety/depression/panic without self-harm -> `Psychiatrist`.
- Trauma/injury without emergency -> `Orthopedic doctor`.
- Urinary burning/frequency/flank/retention context -> `Urologist`, or `Emergency care` if urgent.
- Diabetes/sugar context -> `Endocrinologist`.
- Dizziness with ear/tinnitus context -> `ENT specialist` unless neurological red flags are present.
- Vague fever/infectious context -> `General Practitioner` when not severe.

Clarification mode can now show a specialist when specialty context is clear, while still keeping `possible_diagnosis: null`.

## Follow-Up Question Improvements

Added targeted question groups for:

- Fever/infection.
- Diabetes/endocrine-like symptoms.
- Urinary symptoms.
- Dizziness/ENT overlap.
- Pediatric symptoms.
- Pregnancy symptoms.
- Mental health symptoms.
- Dental symptoms.
- Eye symptoms.
- Trauma/injury.

High urgency follow-ups were improved for:

- Self-harm crisis.
- Pregnancy red flags.
- Pediatric red flags.
- Trauma red flags.

Existing repeated-question filtering remains active through assistant history.

## User-Facing Cleanup Preserved

Preserved or improved:

- No `غير محدد` as patient-facing diagnosis for no-diagnosis emergency/context cases.
- No confidence `0.0` displayed as a failure in clarification.
- No irrelevant RAG cases in clarification.
- No raw English advice added.
- No medication/OTC advice added.
- Internal misspelled disease labels remain internal.
- Emergency mode still places urgent advice first.

`llm_service.py` now displays no-diagnosis fallback as:

```text
حالة تحتاج تقييم طبي
```

instead of:

```text
غير محدد
```

## Focused Tests Added

New test file:

- `tests/test_phase4b_targeted_improvements.py`

Covered cases:

- Self-harm phrase triggers emergency and does not show `غير محدد`.
- Pregnancy bleeding triggers emergency.
- Child not feeding/lethargy triggers emergency.
- Urinary retention triggers emergency.
- Flank pain + fever + urinary burning routes safely to urology/emergency.
- Dizziness + tinnitus routes ENT/neurology.
- Eye pain + vision change routes ophthalmology.
- Dental pain + swelling routes dentist.
- Severe trauma/bleeding triggers emergency.
- Anxiety without self-harm routes psychiatry and does not become fake emergency.
- Vague fever asks infection-focused follow-up questions.

## Unit Test Result

Command:

```bash
python -m unittest discover -s tests -v
```

Result:

- 56 tests passed.

No Groq call was made. The API tests disable the LLM client and stub RAG retrieval.

## Risks and Limitations

- These changes improve routing, extraction, clarification, and safety guardrails, but they do not add new trained disease labels.
- Dental, eye, pregnancy, pediatric, mental-health, and trauma cases are triage/routing contexts, not classifier-supported definitive diagnoses.
- Some broad phrases still need future validation on a larger locked set to measure false positives.
- Infectious disease differentiation still needs better data and evaluation; Phase 4B improves context and follow-up, not true disease-class expansion.
- Final metrics were not recalculated because final validation and 50-case regression were explicitly postponed.

## Final Validation Status

Final validation remains postponed.

Not run:

- 50-case regression.
- 160-case validation.
- Groq-backed validation.
- Model training.
- Artifact rebuild.

## Recommended Next Phase

Recommended Phase 4C:

Build a 300-500 case Arabic/Egyptian validation set with:

- diagnosis group
- urgency
- doctor type
- expected mode
- safety expectation
- multi-turn examples
- category breakdowns for new Phase 4B contexts

Only after that set is stable should the project run broader regression/validation and decide whether model comparison is justified.

