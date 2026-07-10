# Deployment

## Repository contents

This repository contains the complete MedBridge source, frontend, runtime artifacts, Docker configuration, tests, and integration documentation.

Large AI artifacts are stored in the same GitHub repository through Git LFS.

## Local execution

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/verify_artifacts.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Add real Groq keys only to the local `.env` file.

## Docker

```powershell
git lfs pull
docker build -t medbridge-ai .
docker run --env-file .env -p 8000:8000 medbridge-ai
```

The Docker build installs dependencies and preloads the embedding and reranker models before enabling offline transformer loading.

## Public services

- Frontend: `https://frontend-woad-seven-82.vercel.app`
- Backend: `https://medbridge-ai-backend-production.up.railway.app`
- Health: `https://medbridge-ai-backend-production.up.railway.app/health`

The current public Railway version runs the Groq integration, classifier, medical knowledge, and safety logic.

Full RAG is present and functional in the repository/local setup, but is currently disabled in the public Railway deployment because the full model-caching build was not completed within the current hosting setup.
