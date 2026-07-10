# Final AI Handoff Report

Generated: 2026-06-17

No deployment, artifact rebuild, dataset modification, heavy training, or Groq smoke test was performed during this handoff sprint.

## Project Purpose

MedBridge AI is an Arabic-first medical chatbot module for symptom triage, likely diagnosis direction, urgency assessment, doctor routing, and safe patient-facing explanations.

The service is intended to integrate with the real MedBridge backend and frontend as a separate AI microservice. It does not replace doctors and does not store patient data.

## Final Architecture

The AI service is implemented as a stateless FastAPI service.

Main flow:

1. `POST /chat` receives current user message plus optional recent history.
2. `classifier_service.py` extracts Arabic/Egyptian symptom expressions.
3. The structured disease classifier produces disease probabilities.
4. The fusion layer combines classifier candidates, extracted symptoms, symptom severity, RAG evidence, medical knowledge, and safety guardrails.
5. `rag_service.py` retrieves Arabic MAQA cases using SentenceTransformer embeddings and FAISS.
6. `knowledge_service.py` provides descriptions, precautions, severity, follow-up questions, and doctor routing.
7. `safety.py` applies red-flag and urgency rules.
8. `llm_service.py` generates constrained Arabic answers using Groq when configured, or safe fallback wording if unavailable.
9. The API returns structured JSON for the backend/frontend.

The service supports:

- `GET /health`
- `POST /chat`

## `/chat` Response Schema Snapshot

The full API response includes:

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

Backend systems may store the full response for audit and QA. Patient-facing frontend screens should prefer the Arabic display fields and should hide raw retrieved cases, internal symptom codes, raw RAG scores, and confidence values when the mode is `clarification`.

## Datasets Used

Arabic MAQA RAG dataset:

- `D:/Project Graduation/First Dataset/MAQA_Train.xlsx`
- `D:/Project Graduation/First Dataset/MAQA_Test.xlsx`
- `D:/Project Graduation/First Dataset/Train.xlsx`
- `D:/Project Graduation/First Dataset/Test.xlsx`

Disease classifier dataset:

- `D:/Project Graduation/Training.csv`
- `D:/Project Graduation/Testing.csv`
- `D:/Project Graduation/dataset.csv`

Medical knowledge layer:

- `D:/Project Graduation/symptom_Description.csv`
- `D:/Project Graduation/symptom_precaution.csv`
- `D:/Project Graduation/Symptom_severity.csv`

## Current Accepted Metrics

Accepted Phase 3.3 status before this handoff sprint:

- Unit tests: 40 passed.
- 50-case regression:
  - Diagnosis accuracy: 96%.
  - Urgency accuracy: 94%.
  - Doctor accuracy: 98%.
  - Emergency recall: 100%.
  - Safety pass rate: 100%.

Additional focused handoff-sprint unit test result:

- Command: `python -m unittest discover -s tests -v`
- Result: 44 tests passed.

No new 50-case regression or 160-case validation was run in this sprint.

Supplemental Phase 4C polish result:

- Added patient-facing Arabic display labels for internal diagnosis and doctor labels.
- Added optional `display_diagnosis_ar` and `display_doctor_ar` response fields.
- Added safer RAG filtering before retrieved cases influence classifier fusion, answer precautions, or LLM prompt context.
- Added a frontend-safe response helper for teams that want to strip debug/internal fields before returning data to the patient UI.
- Command: `python -m unittest discover -s tests -v`
- Result: 64 tests passed.
- No Groq call, final validation, 50-case regression, 160-case validation, artifact rebuild, training, or deployment was run during Phase 4C.

## Final Controlled Validation Update

The first full controlled validation run on 2026-06-17 did not meet acceptance targets, mainly because emergency recall and safety were too weak. A targeted P0 repair pass was completed afterward and unit tests reached 78 passed.

A second approved full 200-case validation was then run once under the same deterministic fallback-only conditions.

Run conditions:

- Groq generation disabled by the validation runner.
- All 200 rows used deterministic fallback mode.
- No deployment, training, artifact rebuild, dataset modification, or classifier replacement.
- No repeated reruns to chase metrics.
- Runtime/API errors: 0.

Second validation output:

- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_131656/`

Main result:

| Metric | Result |
|---|---:|
| Mode accuracy | 93.50% |
| Broad diagnosis group accuracy | 87.50% |
| Urgency accuracy | 87.50% |
| Doctor accuracy | 92.50% |
| Emergency recall | 100.00% |
| Safety pass rate | 98.00% |
| Clarification behavior pass rate | 97.83% |
| Closing behavior pass rate | 100.00% |
| Must-not-contain violation rate | 2.00% |

Acceptance status:

- Passed P0 safety/emergency targets.
- Passed 7 of 8 acceptance targets.
- Urgency accuracy remains below the 90% target and is documented as a P1 quality limitation.

Detailed files:

- `D:/Project Graduation/medbridge-ai-service/FINAL_VALIDATION_RESULTS_REPORT.md`
- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_131656/final_validation_results.csv`
- `D:/Project Graduation/medbridge-ai-service/final_validation_results/20260617_131656/failure_analysis.md`

Recommendation: accept the AI module for backend/frontend handoff and integration testing as a production-like graduation prototype. It is not ready for unsupervised public medical production.

## Phase 3.3 Improvements

Phase 3.3 resolved the final known medical UX and routing problems before handoff:

- Reduced over-clarification for medium-clear symptom clusters.
- Prevented irrelevant/negative RAG cases from appearing in weak answers.
- Improved doctor routing for urinary, skin, neuro/vestibular, digestive, and emergency patterns.
- Prevented raw English medical advice from leaking into patient answers.
- Improved safety handling around emergency and red-flag symptoms.
- Preserved clarification mode for vague or body-area-only inputs.

## Final AI Behavior Modes

### `clarification`

Used when the message is vague, too short, body-area-only, or too low-confidence.

Behavior:

- No diagnosis is returned.
- `possible_diagnosis` is `null`.
- `confidence` is `0.0`.
- `suggested_doctor` is `Needs more information`.
- The answer asks targeted follow-up questions.
- RAG cases are hidden.

### `diagnosis`

Used when there is enough symptom evidence for a fused diagnosis direction.

Behavior:

- Returns likely diagnosis direction.
- Returns urgency and doctor routing.
- May still include follow-up questions.
- Patient-facing answer uses Arabic diagnosis and doctor labels.

### `emergency`

Used when red flags or high urgency are detected.

Behavior:

- Emergency warning comes first.
- Doctor routing is `Emergency care`.
- The answer does not encourage waiting or routine clinic follow-up.

### `closing`

Added in this sprint for messages such as `شكرا`, `تمام`, `خلاص`, `كده كفاية`, `مش محتاج حاجة تاني`, or `thank you`.

Behavior:

- Returns a polite closing answer.
- Does not diagnose again.
- Does not retrieve RAG cases.
- Does not ask follow-up questions.
- Backend may mark the session closed if desired.

## Multi-Turn Conversation Support

The `/chat` endpoint accepts optional `history`.

Final behavior:

- Recent user messages are combined with the current message for symptom extraction and triage.
- Assistant messages are not used as symptom evidence, preventing false extraction from questions like "هل عندك كحة أو حرارة؟".
- Assistant messages are used to avoid repeating the exact same follow-up questions.
- The chatbot can move from clarification to diagnosis after enough user-provided symptoms are collected.
- Emergency override can happen at any turn because urgency checks use current plus recent user context.

The AI service still remains stateless. The backend must store and pass recent history.

## What Backend and Frontend Teams Need To Do

Backend team:

- Store sessions and messages in MongoDB.
- Send recent history to `/chat`.
- Save the AI response and metadata.
- Handle timeouts and invalid responses safely.
- Keep secrets outside source code.
- Decide whether `mode: closing` should mark the session closed.

Frontend team:

- Display `answer`.
- Display `follow_up_questions` as quick replies or prompts.
- Highlight `urgency_level`, especially `High`.
- Show `suggested_doctor` in diagnosis/emergency contexts.
- Hide `retrieved_cases`, raw RAG scores, internal symptom codes, and confidence 0 in clarification mode.

## Files The Team Should Care About

Core app:

- `app/main.py`
- `app/schemas.py`
- `app/classifier_service.py`
- `app/clarification_service.py`
- `app/knowledge_service.py`
- `app/rag_service.py`
- `app/llm_service.py`
- `app/display_labels.py`
- `app/response_utils.py`
- `app/safety.py`
- `app/arabic_symptom_dictionary.py`

Artifacts:

- `artifacts/faiss.index`
- `artifacts/knowledge_base.pkl`
- `artifacts/maqa_clean_data.pkl`
- `artifacts/maqa_embeddings.pkl`
- `artifacts/disease_classifier.pkl`
- `artifacts/disease_label_encoder.pkl`
- `artifacts/symptom_columns.pkl`
- `artifacts/medical_knowledge.pkl`

Documentation:

- `README.md`
- `TEAM_CHATBOT_INTEGRATION_GUIDE.md`
- `FINAL_AI_HANDOFF_REPORT.md`
- `MODEL_IMPROVEMENT_DECISION.md`
- `FINAL_COMPLETION_SUMMARY.md`

Testing:

- `tests/test_symptom_extraction.py`
- `tests/test_phase3_clarification.py`
- `tests/test_phase4b_targeted_improvements.py`
- `tests/test_phase4c_rag_labels_polish.py`
- `tests/test_p0_final_validation_repairs.py`

## How To Run Locally

```bash
cd "D:/Project Graduation/medbridge-ai-service"
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Required environment:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
ARTIFACTS_DIR=artifacts
CORS_ORIGINS=http://localhost:5173
```

## Quick Test

Health:

```bash
curl http://localhost:8000/health
```

Chat:

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"عندي كحة وسخونية وتعب\",\"history\":[]}"
```

Closing:

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"شكرا\",\"history\":[]}"
```

## Known Limitations

- This is not a replacement for doctors.
- It cannot honestly claim to cover all diseases.
- It is best described as a broad Arabic symptom triage and doctor-routing assistant.
- The second full 200-case validation passed emergency and safety targets, but urgency accuracy remained below target at 87.50%.
- Remaining weak areas include urinary urgency calibration, infectious doctor routing, endocrine vague-symptom handling, eye routing, and some strict Medium-vs-Low urgency labels.
- Groq rate limits/timeouts can affect answer generation and should be monitored in deployment.
- The structured `Testing.csv` classifier result is useful but not enough evidence for real-world performance.
- Future production deployment needs stricter operational monitoring, clinician-reviewed test cases, and a stable LLM quota setup.

## Future Improvement Plan

Recommended next improvements:

- Compare RandomForest, Logistic Regression, Linear SVM, XGBoost, and LightGBM on structured symptoms.
- Research and test Arabic/multilingual transformer classifiers only with enough high-quality data.
- Expand Arabic medical QA and symptom-to-disease datasets carefully.
- Build larger clinician-reviewed Arabic evaluation sets.
- Run fully consistent Groq-backed validation with fallback mode separated.
- Improve weak categories before claiming production readiness.
