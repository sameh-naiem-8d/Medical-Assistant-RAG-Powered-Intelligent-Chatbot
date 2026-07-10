# MedBridge AI Service

MedBridge is an Arabic-first conversational medical-guidance graduation project.

This repository contains the complete project required by the AI, backend, and frontend teams:

- FastAPI medical AI backend.
- Arabic and English multi-turn conversations.
- Disease classification and medical knowledge.
- FAISS and MAQA-based RAG retrieval.
- Embedding and reranker integration.
- Groq primary and fallback LLM integration.
- Emergency detection, negation handling, correction handling, and response consistency checks.
- Local demonstration frontend.
- Integration guides, tests, scripts, and validation reports.

## Complete AI artifacts

The AI runtime artifacts are included in this repository through Git LFS, including the classifier, FAISS index, MAQA knowledge base, and embeddings.

Clone the repository using Git LFS rather than downloading a normal ZIP:

```powershell
git lfs install
git clone https://github.com/Mohamed-515/Medbridge-Ai.git
cd Medbridge-Ai
git lfs pull
```

Verify that all runtime artifacts were downloaded:

```powershell
python scripts/verify_artifacts.py
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the Groq API keys to `.env`. Never commit the real `.env` file.

## Run the backend

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

- Backend: `http://127.0.0.1:8010`
- Health: `http://127.0.0.1:8010/health`
- API documentation: `http://127.0.0.1:8010/docs`

## Run the local frontend

Open another terminal:

```powershell
python -m http.server 5173 --directory local_demo_frontend
```

- Frontend: `http://127.0.0.1:5173`

## Public demonstration

- Frontend: `https://frontend-woad-seven-82.vercel.app`
- Backend: `https://medbridge-ai-backend-production.up.railway.app`
- Health: `https://medbridge-ai-backend-production.up.railway.app/health`

The current public Railway demonstration runs the Groq LLM integration, disease classifier, medical knowledge, and safety logic. Full RAG is available in this repository and in the verified local setup, while it is currently disabled on the public Railway deployment because of hosting build and resource constraints.

## Team integration

Start with:

- `QUICK_START.md`
- `API_CONTRACT.md`
- `BACKEND_INTEGRATION_GUIDE.md`
- `TEAM_CHATBOT_INTEGRATION_GUIDE.md`
- `MONGODB_CHAT_SCHEMA.md`
- `ARTIFACT_SETUP.md`
- `DEPLOYMENT.md`

## Security

This repository does not contain Groq keys, `.env`, Railway credentials, Vercel credentials, or private tokens.
