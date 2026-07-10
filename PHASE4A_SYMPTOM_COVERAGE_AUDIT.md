# Phase 4A Symptom Coverage Audit

Generated: 2026-06-17

No symptom dictionary changes were made in this audit.

## Source Used

Symptom columns were read from:

```text
D:/Project Graduation/medbridge-ai-service/artifacts/symptom_columns.pkl
```

Arabic/Egyptian coverage was reviewed from:

```text
app/arabic_symptom_dictionary.py
app/classifier_service.py
app/clarification_service.py
app/safety.py
```

## Summary

| item | count |
| --- | ---: |
| Structured symptom columns | 132 |
| Symptom columns with dictionary coverage | 132 |
| Symptom columns without dictionary entry | 0 |
| Body-area topic groups | 6 |
| Body-area terms | 39 |
| Body-area-only terms | 26 |
| Follow-up question groups | 7 |
| Red-flag phrases | 73 |
| Closing phrases | 11 |

Important interpretation:

- All 132 structured symptom columns have at least one synonym entry.
- This does not mean every Egyptian Arabic phrase is covered.
- Coverage is strongest for the graduation evaluation set and known weak areas fixed in Phases 1 to 3.3.
- Phase 4 should expand phrase diversity, especially for real-world wording and underrepresented specialties.

## Current 132 Symptom Columns

```text
itching
skin_rash
nodal_skin_eruptions
continuous_sneezing
shivering
chills
joint_pain
stomach_pain
acidity
ulcers_on_tongue
muscle_wasting
vomiting
burning_micturition
spotting_ urination
fatigue
weight_gain
anxiety
cold_hands_and_feets
mood_swings
weight_loss
restlessness
lethargy
patches_in_throat
irregular_sugar_level
cough
high_fever
sunken_eyes
breathlessness
sweating
dehydration
indigestion
headache
yellowish_skin
dark_urine
nausea
loss_of_appetite
pain_behind_the_eyes
back_pain
constipation
abdominal_pain
diarrhoea
mild_fever
yellow_urine
yellowing_of_eyes
acute_liver_failure
fluid_overload
swelling_of_stomach
swelled_lymph_nodes
malaise
blurred_and_distorted_vision
phlegm
throat_irritation
redness_of_eyes
sinus_pressure
runny_nose
congestion
chest_pain
weakness_in_limbs
fast_heart_rate
pain_during_bowel_movements
pain_in_anal_region
bloody_stool
irritation_in_anus
neck_pain
dizziness
cramps
bruising
obesity
swollen_legs
swollen_blood_vessels
puffy_face_and_eyes
enlarged_thyroid
brittle_nails
swollen_extremeties
excessive_hunger
extra_marital_contacts
drying_and_tingling_lips
slurred_speech
knee_pain
hip_joint_pain
muscle_weakness
stiff_neck
swelling_joints
movement_stiffness
spinning_movements
loss_of_balance
unsteadiness
weakness_of_one_body_side
loss_of_smell
bladder_discomfort
foul_smell_of urine
continuous_feel_of_urine
passage_of_gases
internal_itching
toxic_look_(typhos)
depression
irritability
muscle_pain
altered_sensorium
red_spots_over_body
belly_pain
abnormal_menstruation
dischromic _patches
watering_from_eyes
increased_appetite
polyuria
family_history
mucoid_sputum
rusty_sputum
lack_of_concentration
visual_disturbances
receiving_blood_transfusion
receiving_unsterile_injections
coma
stomach_bleeding
distention_of_abdomen
history_of_alcohol_consumption
fluid_overload.1
blood_in_sputum
prominent_veins_on_calf
palpitations
painful_walking
pus_filled_pimples
blackheads
scurring
skin_peeling
silver_like_dusting
small_dents_in_nails
inflammatory_nails
blister
red_sore_around_nose
yellow_crust_ooze
```

## Symptoms Well Covered

These areas have strong current support from synonym expansion, tests, and guardrails:

- Respiratory basics: cough, fever, phlegm, runny nose, congestion, throat irritation, breathlessness.
- Emergency respiratory/cardio phrases: severe chest pain, breathlessness, choking, blue lips.
- Digestive symptoms: vomiting, diarrhea, abdominal pain, nausea, constipation, acidity, blood in stool/vomit.
- Skin symptoms: rash, itching, acne-like pimples, blackheads, fungal/allergy patterns, swelling with breathing difficulty.
- Urinary symptoms: burning urination, frequency, blood in urine, bladder discomfort, flank/back pain.
- Endocrine-like symptoms: sweating, tremor/shivering, excessive hunger, thirst/polyuria, weight loss/gain, thyroid swelling.
- Jaundice/liver clues: yellow skin, yellow eyes, dark urine, fatigue.
- Neurological red flags: severe sudden headache, one-sided numbness/weakness, slurred speech, stiff neck, imbalance/vertigo.
- Vague general symptoms: fatigue, malaise, loss of appetite, weight loss, "مش مرتاح".
- Varicose/vascular leg symptoms: swollen legs, prominent calf veins, painful walking.

## Symptoms Partially Covered

These are present in the 132-column schema but need more Arabic/Egyptian phrase diversity:

| area | current issue |
| --- | --- |
| ENT and ear symptoms | vertigo and cold/allergy overlap exist, but ear pain, tinnitus, hearing loss, blocked ear, and ear discharge are not first-class columns |
| Eye symptoms | red/watery eyes and blurred vision exist, but eye pain, light sensitivity, discharge, and vision loss need more phrases |
| Gynecology | `abnormal_menstruation` exists, but pregnancy, pelvic pain, vaginal bleeding, and discharge are not well represented |
| Pediatrics | child-specific danger signs are not separated from adult symptom phrases |
| Mental health | anxiety, depression, mood swings, restlessness exist, but panic attack, self-harm, insomnia, and severe agitation need policy-level handling |
| Chronic disease context | diabetes, blood pressure, asthma history, heart disease history need richer phrase extraction |
| Medication/allergy context | allergy after food/drug exists, but specific exposure timing and medication reaction phrasing need expansion |
| Trauma/injury | pain and swelling exist, but fall, fracture, wound, burn, and head injury are not represented well |
| Nephrology | UTI phrases exist, but kidney stones, low urine, renal failure, swelling/edema context need more coverage |
| Infectious disease context | travel/fever/chills exist, but exposure history, outbreak context, and persistent fever phrases need more coverage |

## Missing or Weak Symptom Areas

Some user complaints are common but not naturally represented by the current 132 columns:

- Ear pain.
- Tinnitus.
- Hearing loss.
- Sore tonsils or visible pus on tonsils as a clean ENT signal.
- Dental pain, gum swelling, mouth abscess.
- Eye pain, sudden vision loss, eye discharge.
- Pregnancy, missed period, positive pregnancy test.
- Vaginal bleeding, pelvic pain, vaginal discharge.
- Pediatric poor feeding, fewer wet diapers, persistent crying, fever in infant.
- Trauma, fall, fracture, wound, burn.
- Severe anxiety/panic with chest tightness, while still preserving emergency safety.
- Self-harm or suicidal statements.
- Kidney stone colic.
- Leg swelling with one-sided calf pain as DVT-style red flag.
- Cancer warning symptoms beyond weight loss/fatigue.
- Medication overdose or poisoning.

## Body-Area-Only Phrase Coverage

Current body-area topic groups:

- throat
- abdomen
- chest
- head
- urinary
- skin

Current behavior is good for:

- `زوري واجعني`
- `بطني تعبانة`
- `صدري واجعني`
- `راسي واجعاني`
- urinary-only statements
- skin-only statements

Weak body-area-only areas to add later:

- `وداني واجعاني`
- `عيني واجعاني`
- `سناني واجعاني`
- `رجلي واجعاني`
- `ضهري واجعني`
- `حوضي واجعني`
- `كتفي واجعني`
- `ركبتي واجعاني`

Recommended later behavior:

- Add targeted follow-up topics for ear, eye, dental, limb/joint, back, pelvic/gynecology, and injury.

## Emergency Phrase Coverage

Current red-flag phrase coverage is strong for:

- Severe chest pain.
- Breathlessness/choking.
- Fainting/loss of consciousness.
- Seizure-like wording.
- Vomiting blood.
- Blood in stool.
- Blood in urine with severe pain.
- Severe dehydration.
- Stroke-like facial droop/slurred speech/one-sided weakness.
- Severe headache with stiff neck.
- Severe allergic reaction.

Emergency gaps to expand later:

- Pregnancy bleeding or severe abdominal pain in pregnancy.
- Infant fever or poor feeding.
- Head injury with vomiting/confusion.
- Severe burns.
- Poisoning/overdose.
- Severe one-sided leg swelling/pain.
- Sudden vision loss.
- Suicidal or self-harm statements.
- Severe asthma attack in a known asthma patient.

## Pediatric Phrase Gaps

Recommended phrases to add later:

- `ابني عنده حرارة`
- `بنتي سخنة`
- `الطفل مش بيرضع`
- `البيبي مش بيبل حفاضات`
- `عنده تشنجات مع الحرارة`
- `نايم ومش بيرد`
- `بيعيط جامد ومش بيهدى`
- `طفل أقل من ٣ شهور وعنده حرارة`

These should be handled carefully because pediatric triage has lower thresholds for urgent care.

## Pregnancy / Gynecology Phrase Gaps

Recommended phrases to add later:

- `أنا حامل وعندي نزيف`
- `حامل وبطني بتوجعني جامد`
- `الدورة متأخرة`
- `نزيف مهبلي`
- `إفرازات مهبلية`
- `وجع في الحوض`
- `دوخة مع حمل`
- `ضغط عالي وأنا حامل`

These should trigger either gynecology routing or emergency escalation depending on severity.

## Chronic Disease Context Gaps

Recommended phrases to add later:

- `عندي سكر`
- `مريض سكر`
- `عندي ضغط`
- `عندي ربو`
- `عندي حساسية صدر`
- `عندي مرض قلب`
- `باخد أدوية سيولة`
- `مناعتى ضعيفة`
- `عندي فشل كلوي`
- `بعمل غسيل كلى`

These phrases may not always be symptoms, but they strongly affect urgency and doctor routing.

## Recommended Arabic/Egyptian Phrases To Add Later

Respiratory:

- `نفسي بيصفر`
- `صدرى مكتوم`
- `كحة ناشفة`
- `كحة ببلغم أخضر`
- `مش قادر آخد نفسي`
- `بنهج من أقل مجهود`

Digestive:

- `مغص جامد`
- `بطني بتعصرني`
- `ترجيع مستمر`
- `إسهال مائي`
- `مش قادر أشرب`
- `بقالي يومين مش بتبول كويس`

Skin/allergy:

- `ارتيكاريا`
- `حساسية منتشرة`
- `شفايفي وارمة`
- `وشي وارم`
- `حكة في كل جسمي`
- `طلعتلي بقع حمرا`

Neurology:

- `الدنيا بتلف بيا`
- `مش متزن`
- `كلامي تقيل`
- `وشي اتعوج`
- `إيدي مش حاسس بيها`
- `صداع مفاجئ جامد`

Urinary/renal:

- `حرقان وأنا بتبول`
- `بروح الحمام كل شوية`
- `البول لونه أحمر`
- `وجع في جنبي`
- `مش عارف أتبول`

Endocrine:

- `جعان جدًا`
- `بعرق وبتنفض`
- `بقيس السكر لقيته واطي`
- `عطشان طول الوقت`
- `بنزل في الوزن`

ENT/eye/dental:

- `ودني بتوجعني`
- `ودني مسدودة`
- `في صفير في ودني`
- `عيني بتوجعني`
- `مش شايف كويس فجأة`
- `سناني واجعاني`
- `لثتي وارمة`

Pediatric/pregnancy/chronic:

- `ابني حرارته عالية`
- `الطفل مش بيرضع`
- `حامل وعندي نزيف`
- `عندي سكر وضغطي واطي`
- `عندي ربو ومش قادر أتنفس`

## Recommendation

Do not modify the dictionary during Phase 4A.

Recommended next step for Phase 4B:

1. Add targeted phrases only after reviewing validation errors.
2. Add unit tests for every new phrase.
3. Avoid broad phrases that cause false positives.
4. Preserve Phase 3 baseline behavior.

