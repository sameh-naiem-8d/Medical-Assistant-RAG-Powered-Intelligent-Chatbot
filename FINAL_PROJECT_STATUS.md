# Final Project Status

Generated: 2026-06-06

This document summarizes the current status of the MedBridge AI Medical Chatbot project after notebook preparation, artifact building, FastAPI implementation, accuracy upgrade phases, evaluation, and the final locked validation run.

No deployment has been performed.

## Final Architecture

MedBridge AI is implemented as a stateless FastAPI AI service. It does not store patient data and does not use MongoDB internally.

Request flow:

1. The frontend or backend sends a user message and optional chat history to `/chat`.
2. `classifier_service.py` extracts Arabic/Egyptian symptom expressions and maps them to the structured symptom columns.
3. The disease classifier predicts top disease candidates from the extracted symptoms.
4. The fusion layer combines classifier output, extracted symptoms, symptom severity, RAG retrieval, medical descriptions, precautions, and safety guardrails.
5. `rag_service.py` retrieves Arabic MAQA medical Q&A cases using SentenceTransformer embeddings and FAISS.
6. `knowledge_service.py` provides symptom severity, disease descriptions, precautions, doctor routing, and follow-up questions.
7. `safety.py` applies urgency/red-flag rules.
8. `llm_service.py` uses Groq through `GROQ_API_KEY` from `.env` to generate the final Arabic answer under strict prompt rules.
9. The API returns structured JSON with answer, symptoms, diagnosis, confidence, urgency, doctor, precautions, follow-up questions, and retrieved cases.

Main API endpoints:

- `GET /health`
- `POST /chat`

The `/chat` input supports:

- `user_id`
- `session_id`
- `message`
- `history`

The AI service remains stateless. Chat history storage is reserved for the backend team and future MongoDB integration.

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

The full MAQA RAG build indexed 341,218 rows with embedding dimension 384.

## Notebooks Created

Clean final notebook:

- `D:/Project Graduation/MedBridge_AI_Clean_Notebook.ipynb`

This notebook is the discussion-ready notebook. It documents the pipeline from dataset loading and EDA through preprocessing, classifier training, RAG construction, evaluation, artifact saving, and FastAPI connection.

Older notebooks exist but are not the final project notebook:

- `D:/Project Graduation/Chatbot_Medical.ipynb`
- `D:/Project Graduation/Medbridge(1).ipynb`

These older notebooks should be treated as historical work only. The project was intentionally rebuilt around the clean notebook and production service.

## Artifacts Created

Artifact directory:

- `D:/Project Graduation/medbridge-ai-service/artifacts`

Current artifacts:

| artifact | purpose |
| --- | --- |
| `faiss.index` | FAISS vector index for MAQA retrieval |
| `maqa_clean_data.pkl` | cleaned MAQA rows |
| `maqa_embeddings.pkl` | multilingual sentence embeddings |
| `knowledge_base.pkl` | RAG knowledge base records |
| `disease_classifier.pkl` | trained disease classifier |
| `disease_label_encoder.pkl` | disease label encoder |
| `symptom_columns.pkl` | structured symptom column list |
| `medical_knowledge.pkl` | descriptions, precautions, and severity knowledge |
| `artifact_metrics.json` | artifact summary and classifier metrics |

Artifact metrics:

- RAG rows: 341,218
- Embedding dimension: 384
- Medical description rows: 41
- Medical precaution rows: 41
- Symptom severity rows: 132
- Official structured classifier test labels: 41

## Evaluation Results

Official `Testing.csv` disease classifier evaluation:

- Accuracy: 100.00%
- Weighted precision: 100.00%
- Weighted recall: 100.00%
- Weighted F1: 100.00%

Important interpretation:

The 100% classifier result does not prove real-world chatbot performance. `Testing.csv` is a small structured symptom-matrix benchmark with one row per disease label, so it is much easier than real Arabic free-text patient messages.

Final 50-case chatbot evaluation after Phase 2:

- Diagnosis group accuracy: 96.00%
- Urgency accuracy: 94.00%
- Doctor recommendation accuracy: 100.00%
- Emergency recall: 100.00%
- Average confidence: 0.5631
- Follow-up question rate: 72.00%

The final Phase 2 safety score from that run was not considered production evidence because Groq quota exhaustion caused fallback-dominated answer generation.

## Final Locked Validation Results

Final locked validation file:

- `D:/Project Graduation/medbridge-ai-service/FINAL_VALIDATION_REPORT.md`

Locked validation results:

- Validation cases: 160
- Groq rows: 156
- Fallback rows: 4
- Diagnosis accuracy: 81.25%
- Urgency accuracy: 81.88%
- Doctor recommendation accuracy: 84.38%
- Emergency recall: 97.78%
- Safety pass rate: 86.88%
- Average confidence: 0.5380
- Average latency: 14.0739 seconds
- Median latency: 14.2386 seconds
- P95 latency: 20.9496 seconds
- Follow-up question rate: 71.25%

The final validation run was mixed-mode because four rows used fallback:

- Three Groq token-per-minute rate-limit failures
- One Groq timeout

Category breakdown:

| category | cases | groq | fallback | diagnosis | urgency | doctor | safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cardiovascular | 16 | 15 | 1 | 75.00% | 87.50% | 68.75% | 93.75% |
| digestive | 16 | 16 | 0 | 87.50% | 93.75% | 93.75% | 81.25% |
| emergency | 16 | 16 | 0 | 81.25% | 100.00% | 100.00% | 75.00% |
| endocrine | 16 | 15 | 1 | 81.25% | 62.50% | 81.25% | 93.75% |
| general | 16 | 16 | 0 | 87.50% | 43.75% | 87.50% | 100.00% |
| infectious | 16 | 16 | 0 | 62.50% | 100.00% | 81.25% | 81.25% |
| neurological | 16 | 14 | 2 | 81.25% | 81.25% | 87.50% | 75.00% |
| respiratory | 16 | 16 | 0 | 87.50% | 87.50% | 68.75% | 81.25% |
| skin | 16 | 16 | 0 | 87.50% | 75.00% | 87.50% | 100.00% |
| urinary | 16 | 16 | 0 | 81.25% | 87.50% | 87.50% | 87.50% |

## Current Limitations

- Final locked diagnosis accuracy is 81.25%, below the 85% target.
- Final locked safety pass rate is 86.88%, below a production-ready threshold.
- Final validation was mixed-mode because Groq produced four fallback responses.
- Infectious diagnosis remains the weakest category at 62.50%.
- Cardiovascular doctor routing remains weak at 68.75%.
- Respiratory doctor routing remains weak at 68.75%.
- Endocrine urgency remains weak at 62.50%.
- General/vague urgency remains weak at 43.75%, mostly due under-specified symptoms and conservative escalation.
- Emergency recall is strong at 97.78%, but emergency answer safety still needs tighter validation.
- Some medical descriptions in the knowledge layer can still contain English source text, which can affect Arabic answer quality.
- The evaluation sets are useful for graduation discussion, but they are not clinician-certified clinical benchmarks.

## Why Deployment Is Paused

Deployment is paused for honest quality and safety reasons:

1. The final locked validation was not fully Groq-only.
2. The final locked diagnosis accuracy did not reach the 85% target.
3. The safety pass rate is not high enough for production medical triage.
4. Several category-level weaknesses remain.
5. Groq rate limits and timeout behavior need operational handling before real deployment.

The service is suitable for graduation demonstration and backend integration planning, but not yet ready for unsupervised real patient use.

## Ready for Backend Team

The backend team can safely prepare integration around the current stateless API contract.

Ready items:

- FastAPI service structure
- `/health` endpoint
- `/chat` endpoint
- Stateless request/response design
- Optional `user_id`
- Optional `session_id`
- Optional `history`
- No MongoDB dependency inside the AI service
- No patient-data storage inside the AI service
- CORS support for frontend/backend connection
- Dockerfile
- `.env.example`
- README with backend handoff notes
- Artifact-loading service modules
- Structured JSON response schema

Backend/MongoDB responsibility:

- Store users
- Store sessions
- Store chat history
- Pass recent history to `/chat`
- Store returned AI responses if needed
- Manage authentication and production patient-data controls

## What Remains Before Real Production Deployment

Required before production deployment:

- Run a fully Groq-backed locked validation with zero fallback rows.
- Improve diagnosis accuracy above the target threshold, preferably above 85% on locked validation.
- Improve safety pass rate toward 95% or higher.
- Review and tighten weak categories: infectious, cardiovascular routing, respiratory routing, endocrine urgency, and general/vague urgency.
- Add retry/backoff or queueing strategy for Groq rate limits without silently weakening clinical evidence.
- Clean or translate remaining English medical descriptions before they reach patient-facing answers.
- Expand evaluation with clinician-reviewed Arabic cases.
- Add monitoring for latency, fallback rate, safety failures, and emergency routing.
- Define production escalation policy for high-urgency outputs.
- Complete security review for environment variables, CORS origins, logging, and PHI handling.
- Deploy only after backend storage, frontend demo, and infrastructure monitoring are reviewed together.

## Final Status

MedBridge AI is a strong graduation MVP and a well-structured backend-ready AI service. It has a clean notebook, production-style FastAPI structure, full local artifacts, Arabic symptom extraction, disease classification, RAG retrieval, medical knowledge, safety guardrails, Groq-powered answer generation, and evaluation reports.

Deployment remains paused because the final locked validation evidence is mixed-mode and the quality metrics are not yet strong enough for real production medical use.
