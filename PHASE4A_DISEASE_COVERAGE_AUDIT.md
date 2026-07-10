# Phase 4A Disease Coverage Audit

Generated: 2026-06-17

No deployment, Groq call, model training, artifact rebuild, dataset modification, or classifier replacement was performed.

## Source Used

Current structured classifier labels were read from:

```text
D:/Project Graduation/medbridge-ai-service/artifacts/disease_label_encoder.pkl
```

The symptom columns were read from:

```text
D:/Project Graduation/medbridge-ai-service/artifacts/symptom_columns.pkl
```

## 1. Total Structured Disease Labels

Total structured classifier disease labels: **41**

Important note:

- `General medical evaluation` is used by the fusion/guardrail logic, but it is not one of the original 41 classifier labels.
- Some dataset labels contain spelling issues, such as `Peptic ulcer diseae`, `Osteoarthristis`, and `(vertigo) Paroymsal  Positional Vertigo`. These should remain internal unless safely translated for the patient.

## 2. Full Disease Label List

1. `(vertigo) Paroymsal  Positional Vertigo`
2. `AIDS`
3. `Acne`
4. `Alcoholic hepatitis`
5. `Allergy`
6. `Arthritis`
7. `Bronchial Asthma`
8. `Cervical spondylosis`
9. `Chicken pox`
10. `Chronic cholestasis`
11. `Common Cold`
12. `Dengue`
13. `Diabetes`
14. `Dimorphic hemmorhoids(piles)`
15. `Drug Reaction`
16. `Fungal infection`
17. `GERD`
18. `Gastroenteritis`
19. `Heart attack`
20. `Hepatitis B`
21. `Hepatitis C`
22. `Hepatitis D`
23. `Hepatitis E`
24. `Hypertension`
25. `Hyperthyroidism`
26. `Hypoglycemia`
27. `Hypothyroidism`
28. `Impetigo`
29. `Jaundice`
30. `Malaria`
31. `Migraine`
32. `Osteoarthristis`
33. `Paralysis (brain hemorrhage)`
34. `Peptic ulcer diseae`
35. `Pneumonia`
36. `Psoriasis`
37. `Tuberculosis`
38. `Typhoid`
39. `Urinary tract infection`
40. `Varicose veins`
41. `hepatitis A`

## 3. Labels Grouped by Medical Specialty

Labels can overlap multiple specialties. This grouping is for coverage analysis, not strict clinical taxonomy.

### General/Internal Medicine

Direct classifier coverage is limited.

Related labels:

- `Allergy`
- `Drug Reaction`
- `Hypertension`
- `Diabetes`
- `Hypoglycemia`
- `Hypothyroidism`
- `Hyperthyroidism`

Gap:

- There is no true broad internal-medicine label in the classifier. The service uses `General medical evaluation` as a fusion-layer safety direction when symptoms are vague or constitutional.

### Respiratory

- `Bronchial Asthma`
- `Common Cold`
- `Pneumonia`
- `Tuberculosis`

Notes:

- Respiratory basics are well represented.
- Influenza and COVID-like illness are not direct labels.
- COPD, bronchitis, sinusitis, and allergic rhinitis are not direct labels.

### Gastroenterology / Hepatology

- `Alcoholic hepatitis`
- `Chronic cholestasis`
- `Dimorphic hemmorhoids(piles)`
- `GERD`
- `Gastroenteritis`
- `hepatitis A`
- `Hepatitis B`
- `Hepatitis C`
- `Hepatitis D`
- `Hepatitis E`
- `Jaundice`
- `Peptic ulcer diseae`

Notes:

- Hepatitis/jaundice coverage is broad for the original dataset.
- Common abdominal emergencies such as appendicitis, gallbladder disease, pancreatitis, bowel obstruction, and severe dehydration are not direct labels.

### Neurology

- `(vertigo) Paroymsal  Positional Vertigo`
- `Cervical spondylosis`
- `Migraine`
- `Paralysis (brain hemorrhage)`

Notes:

- Vertigo/headache/stroke-style red flags have some support.
- Seizures, meningitis, neuropathy, Bell palsy, and many headache causes are not direct labels.

### Cardiology / Vascular

- `Heart attack`
- `Hypertension`
- `Varicose veins`

Notes:

- Emergency chest pain is covered mainly through safety rules and `Heart attack`.
- Arrhythmia, heart failure, angina, pulmonary embolism, and vascular insufficiency are not direct labels.

### Dermatology

- `Acne`
- `Allergy`
- `Chicken pox`
- `Drug Reaction`
- `Fungal infection`
- `Impetigo`
- `Psoriasis`

Notes:

- Rash/itching/skin infection patterns are among the better covered areas.
- Eczema, cellulitis, scabies, urticaria, burns, wounds, and skin abscess are not direct labels.

### Urology

- `Urinary tract infection`

Notes:

- UTI is represented.
- Kidney stones, kidney infection, renal failure, prostate symptoms, incontinence, and STI syndromes are not direct labels.

### Endocrinology

- `Diabetes`
- `Hypoglycemia`
- `Hyperthyroidism`
- `Hypothyroidism`

Notes:

- Thyroid and sugar-related symptoms are covered.
- Diabetic ketoacidosis, adrenal disorders, pregnancy diabetes, and chronic diabetes complications are not direct labels.

### ENT

Direct ENT classifier coverage is weak.

Related/overlapping labels:

- `Common Cold`
- `Allergy`
- `(vertigo) Paroymsal  Positional Vertigo`

Gaps:

- Otitis media, ear infection, sinusitis, tonsillitis, allergic rhinitis, hearing loss, tinnitus, and throat infections are not direct classifier labels.

### Infectious Diseases

- `AIDS`
- `Chicken pox`
- `Dengue`
- `Fungal infection`
- `hepatitis A`
- `Hepatitis B`
- `Hepatitis C`
- `Hepatitis D`
- `Hepatitis E`
- `Malaria`
- `Tuberculosis`
- `Typhoid`

Notes:

- The dataset includes several infectious labels, but validation showed infectious diagnosis is still one of the weakest categories.
- Real infectious disease presentations overlap heavily with respiratory, digestive, skin, and general symptoms.

### Orthopedics / Rheumatology

- `Arthritis`
- `Cervical spondylosis`
- `Osteoarthristis`

Gaps:

- Sprains, fractures, back strain, sciatica, gout, rheumatoid arthritis detail, and trauma are not direct labels.

### Gynecology / Obstetrics

No direct classifier disease labels.

Related symptom column:

- `abnormal_menstruation`

Gaps:

- Pregnancy symptoms, miscarriage red flags, ectopic pregnancy, pelvic inflammatory disease, vaginal bleeding, menstrual pain, PCOS, menopause, and pregnancy hypertension are not covered as direct labels.

### Emergency / Red-Flag Conditions

Direct or related labels:

- `Heart attack`
- `Paralysis (brain hemorrhage)`
- `Hypoglycemia`
- `Allergy` for severe allergic reaction/anaphylaxis through guardrails
- `Gastroenteritis` or digestive labels when dehydration/bleeding phrases are present

Important:

- Emergency handling is not only classifier-driven. It depends heavily on `safety.py`, urgency scoring, and guardrails.
- This is correct because emergency triage must override normal diagnosis confidence.

### Other / Immunology

- `Allergy`
- `Drug Reaction`
- `AIDS`

Notes:

- Immunology coverage is narrow and mostly symptom-triggered.

## 4. Missing or Weak Specialties

Specialties with weak or no direct classifier coverage:

- Pediatrics.
- Gynecology and obstetrics.
- Psychiatry and mental health.
- Dental and oral medicine beyond tongue ulcers.
- Ophthalmology beyond red/watery eyes and blurred vision.
- Nephrology beyond UTI-like symptoms.
- Oncology and hematology.
- Trauma and emergency surgery.
- ENT beyond common cold/allergy/vertigo overlap.
- Pulmonology beyond asthma, pneumonia, cold, and tuberculosis.
- Cardiology beyond heart attack, hypertension, and varicose veins.

## 5. Common Diseases Users May Ask About But Are Not Directly Covered

Respiratory/ENT:

- Influenza.
- COVID-like illness.
- Acute bronchitis.
- Sinusitis.
- Tonsillitis.
- Ear infection.
- Allergic rhinitis.
- COPD.

Digestive:

- Appendicitis.
- Gallbladder disease.
- Pancreatitis.
- Food poisoning as a direct label.
- Irritable bowel syndrome.
- Inflammatory bowel disease.
- Kidney stone abdominal/flank pain overlap.

Cardiovascular:

- Angina.
- Arrhythmia.
- Heart failure.
- Pulmonary embolism.
- Deep vein thrombosis.

Neurology:

- Stroke as a clean label.
- Seizure disorder.
- Meningitis.
- Bell palsy.
- Peripheral neuropathy.
- Anxiety-related dizziness/panic symptoms.

Endocrine/metabolic:

- Diabetic ketoacidosis.
- Prediabetes.
- Electrolyte imbalance.
- Adrenal problems.

Urology/renal:

- Kidney stones.
- Pyelonephritis.
- Prostate enlargement.
- Urinary incontinence.
- Kidney failure.

Skin:

- Eczema.
- Urticaria/hives.
- Cellulitis.
- Scabies.
- Burns.
- Wounds/abscesses.

Gynecology/obstetrics:

- Pregnancy-related nausea.
- Ectopic pregnancy.
- Miscarriage warning signs.
- Vaginal bleeding.
- Menstrual cramps.
- PCOS.
- Vaginal infection.

Pediatrics:

- Child fever.
- Dehydration in children.
- Febrile seizures.
- Poor feeding in infants.
- Pediatric rash.

Mental health:

- Panic attack.
- Depression.
- Anxiety.
- Insomnia.
- Self-harm risk.

## 6. What Existing System Layers Can Still Handle

### Clarification Mode

Can help when:

- Symptoms are vague.
- The user gives only a body area, such as "بطني" or "صدري".
- The message is too short for safe diagnosis.
- The disease is not directly covered but the system can ask better questions.

Limit:

- Clarification mode does not create disease coverage. It only buys safety and more information.

### Doctor Routing

Can help when:

- The exact diagnosis is uncertain but the specialty is clear.
- Examples: urinary symptoms to urology, rash to dermatology, red flags to emergency care.

Limit:

- If diagnosis group is wrong, doctor routing often becomes wrong downstream.

### Emergency Rules

Can help when:

- The user reports chest pain, breathlessness, fainting, stroke-like symptoms, severe dehydration, bleeding, severe allergic reaction, or other red flags.

Limit:

- Emergency rules must be expanded carefully. Over-escalation can reduce trust, but under-escalation is higher risk.

### RAG

Can help when:

- The Arabic MAQA knowledge base contains similar patient questions.
- The classifier is uncertain but retrieval supports a specialty or explanation.

Limit:

- RAG should not be trusted as a standalone diagnosis engine.
- Retrieved cases can be noisy and must remain filtered.

## 7. What Needs Real Dataset / Model Expansion

Needs real expansion:

- New disease labels for missing common conditions.
- Natural-language Arabic symptom-to-disease cases.
- Clinician-reviewed Arabic evaluation data.
- Multi-turn medical dialogue cases.
- Specialty-specific datasets for ENT, gynecology, pediatrics, nephrology, ophthalmology, mental health, and trauma.

Do not add weak hardcoded mappings as a substitute for disease coverage.

Safe next step:

- Expand evaluation first, then compare models and targeted dictionary/routing improvements against the preserved Phase 3 baseline.

