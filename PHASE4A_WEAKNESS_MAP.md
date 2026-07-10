# Phase 4A Weakness Map

Generated: 2026-06-17

No AI logic, symptom extraction, model training, artifacts, datasets, deployment settings, or prompts were changed.

## Evidence Used

Existing project evidence:

- `PHASE3_3_POLISH_REPORT.md`
- `phase3_3_regression/evaluation_summary.md`
- `FINAL_VALIDATION_REPORT.md`
- `VALIDATION_AUDIT.md`
- `ACCURACY_UPGRADE_PHASE1_REPORT.md`
- `ACCURACY_UPGRADE_PHASE2_REPORT.md`
- `ERROR_ANALYSIS.md`
- `PHASE2_ERROR_ANALYSIS.md`

Accepted baseline for this phase:

- Phase 3.3 50-case regression:
  - Diagnosis accuracy: 96%.
  - Urgency accuracy: 94%.
  - Doctor accuracy: 98%.
  - Emergency recall: 100%.
  - Safety pass rate: 100%.
- Final handoff sprint:
  - 44 tests passed.
  - Multi-turn history support.
  - Closing mode.

## Current Status After Phase 3.3

Phase 3.3 improved the AI service in the areas that were blocking handoff:

- Vague and body-area-only inputs now use clarification mode.
- Clear medium-risk cases no longer over-clarify when enough symptoms exist.
- Negative or weak RAG cases are filtered out.
- Doctor routing improved for urinary, skin, digestive, neuro/vestibular, and emergency examples.
- Stroke-like numbness, allergic airway risk, chest pain/breathlessness, dehydration, and bleeding patterns are safer.
- Internal labels are mostly hidden from patient-facing answers through Arabic diagnosis/doctor layers.
- Multi-turn support and closing mode were added in the final handoff sprint.

## Known Weak Categories

### Infectious

Current risk:

- Final validation showed infectious diagnosis as the weakest category.
- Fever/chills/headache/body pain patterns overlap across malaria, dengue, typhoid, respiratory infection, and general evaluation.
- Travel/exposure details are often missing from short user messages.

Likely failure cases:

- `حرارة وتكسير في الجسم`
- `سخونية وصداع`
- `حرارة بعد سفر`
- `حرارة وطفح`
- `عرق بالليل وكحة`

Main likely causes:

- Symptom overlap.
- Too few direct disease labels.
- Need for exposure/travel/duration follow-up.
- Classifier trained on structured symptom matrices, not natural Arabic cases.

### Endocrine

Current risk:

- Hypoglycemia and diabetes-like symptoms improved, but urgency can still be difficult.
- Sweating, tremor, hunger, dizziness, fatigue, palpitations, and anxiety overlap with cardiac and general categories.

Likely failure cases:

- `بعرق وبرتعش وجعان`
- `عطشان وبدخل الحمام كتير`
- `دايخ ومش مركز`
- `وزني بينزل وباكل كتير`
- `رقبتي وارمة وتعبان`

Main likely causes:

- Missing glucose measurement context.
- Overlap with anxiety/cardiac/neurologic symptoms.
- Limited endocrine label set.

### Urinary

Current risk:

- UTI routing improved, but urinary bleeding, flank pain, dehydration, and low urine output require careful urgency.
- Nephrology conditions are not direct classifier labels.

Likely failure cases:

- `دم في البول`
- `وجع في جنبي وحرقان`
- `مش عارف أتبول`
- `البول قليل ولونه غامق`
- `حرقان بسيط بس متكرر`

Main likely causes:

- UTI is the only direct urology label.
- Kidney stones, pyelonephritis, and renal failure are not represented.
- Urgency depends heavily on wording.

### Vague / General Symptoms

Current risk:

- Clarification mode is safer than forced diagnosis, but vague symptoms still reduce diagnosis accuracy.
- General/vague urgency in locked validation was weak because symptoms such as fatigue, appetite loss, dizziness, and weight loss can be low, medium, or serious depending on duration and context.

Likely failure cases:

- `مش مرتاح`
- `تعبان بقالي فترة`
- `نفسي مسدودة`
- `وزني بينزل`
- `جسمي مكسر`

Main likely causes:

- Under-specified messages.
- Need for duration, severity, chronic disease, and red-flag follow-up.
- No universal internal medicine classifier label.

### Neuro / ENT Overlap

Current risk:

- Vertigo, migraine, cervical spondylosis, hypertension, ear symptoms, and stroke-style red flags overlap.
- Ear-specific symptoms are not first-class structured columns.

Likely failure cases:

- `دوخة وزغللة`
- `الدنيا بتلف بيا`
- `ودني مسدودة ودايخ`
- `صداع مع رقبة ناشفة`
- `تنميل بسيط في إيدي`

Main likely causes:

- Ear/ENT labels are missing.
- Neurologic red flags need high sensitivity.
- Hypertension overlap remains possible if pressure context is unclear.

### Emergency Grouping Limitations

Current risk:

- Emergency recall is strong in accepted Phase 3.3 metrics, but diagnosis grouping can be misleading.
- Some emergency cases may be counted as diagnosis mismatches even when urgency and doctor routing are correct.

Examples:

- Severe breathlessness or cyanosis may map to `Heart attack` because emergency routing is correct but exact disease group is broad.
- Anaphylaxis-like symptoms may map to `Allergy`, which is clinically acceptable as a direction but not a definitive diagnosis.
- Stroke-like symptoms may map to `Paralysis (brain hemorrhage)`, a source label that is not ideal wording.

Main likely causes:

- Original label set does not include clean emergency syndrome labels.
- Evaluation expected groups may be stricter than safe triage behavior.

## What Improved

High-impact improvements already completed:

- Respiratory cough + fever + fatigue no longer defaults to asthma without asthma context.
- Chest pain + breathlessness routes as high urgency/emergency.
- Diarrhea + vomiting + abdominal pain routes as digestive/gastroenteritis.
- Rash + itching routes dermatology/fungal/allergy.
- Allergy/anaphylaxis pattern with swelling and breathlessness routes high urgency.
- Hypoglycemia pattern with sweating, tremor, hunger routes endocrine.
- Vague fatigue + appetite loss + weight loss no longer automatically over-selects hepatitis.
- Doctor routing fixes for hypertension, varicose veins, cervical spondylosis.
- Multi-turn conversation can move from clarification to diagnosis.
- Closing messages no longer trigger diagnosis.

## What Remains Risky

Safety-sensitive:

- Hidden emergency symptoms inside long vague messages.
- Pediatric red flags.
- Pregnancy red flags.
- Self-harm statements.
- Head injury.
- Poisoning/overdose.
- Severe dehydration in children or older adults.

Accuracy-sensitive:

- Infectious disease differentiation.
- Endocrine urgency.
- Urinary vs kidney stone/renal problems.
- ENT vs neuro dizziness.
- General symptoms with no duration/context.

UX-sensitive:

- Avoiding overuse of General Practitioner when specialist routing is possible.
- Avoiding scary language for low-risk cases.
- Avoiding false certainty for unsupported labels.
- Avoiding repeated questions in longer chats.

## Priority Ranking

### P0: Safety / Emergency

Must not regress:

- Chest pain + breathlessness.
- Stroke-like facial droop/slurred speech/one-sided weakness.
- Severe allergic reaction.
- Severe dehydration.
- Vomiting blood or blood in stool.
- Severe headache with stiff neck.
- Fainting/seizure-like symptoms.

New P0 gaps to address later:

- Pregnancy bleeding/severe pain.
- Infant fever/poor feeding/fewer wet diapers.
- Self-harm or suicidal wording.
- Head injury with vomiting/confusion.
- Severe burns.
- Poisoning/overdose.
- Sudden vision loss.
- One-sided leg swelling/pain suggesting clot risk.

### P1: Common User Cases

High-impact everyday cases:

- Flu/COVID-like symptoms.
- Sinusitis/tonsillitis/ear infection.
- Food poisoning.
- Kidney stones.
- Menstrual/pregnancy symptoms.
- Panic/anxiety-like symptoms with chest tightness.
- Back pain/sciatica.
- Dental pain.
- Eye infection.

### P2: Diagnosis Accuracy

Improve after evaluation expansion:

- Infectious disease differentiation.
- Endocrine classification and urgency.
- Urinary subtype handling.
- Neuro/ENT overlap.
- Respiratory doctor routing.
- Cardiovascular doctor routing.

### P3: Polish

Quality polish after safety and accuracy:

- More natural Arabic follow-up wording.
- More specific doctor labels where appropriate.
- Cleaner internal label translations.
- Better answer length control.
- More robust fallback wording.
- Additional frontend display rules.

## Recommendation

Do not train or replace the classifier yet.

Recommended sequence:

1. Preserve Phase 3 baseline.
2. Expand evaluation to 300-500 Arabic/Egyptian cases.
3. Add targeted symptom/urgency/routing improvements with unit tests.
4. Run controlled regression.
5. Compare tabular models only after the evaluation set is stable.
6. Consider transformer experiments only after enough labeled Arabic data exists.

