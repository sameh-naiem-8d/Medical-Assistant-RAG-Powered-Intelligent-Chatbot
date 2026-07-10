# Environment Local Setup Guide

This guide explains local LLM key setup for MedBridge AI.

Do not commit `.env`. Do not share real API keys in chat, screenshots, GitHub, reports, or deployment logs.

## 1. Create `.env`

From PowerShell:

```powershell
cd "D:\Project Graduation\medbridge-ai-service"
Copy-Item .env.example .env
notepad .env
```

## 2. Paste Two Groq Keys

Preferred setup:

```text
GROQ_API_KEYS=PASTE_FIRST_GROQ_KEY_HERE,PASTE_SECOND_GROQ_KEY_HERE
GROQ_MODEL=llama-3.1-8b-instant
ARTIFACTS_DIR=artifacts
CORS_ORIGINS=http://localhost:5173
```

Replace the placeholder values with real Groq keys only in your local `.env`.

## 3. Legacy Single-Key Setup Still Works

If you only have one key:

```text
GROQ_API_KEY=PASTE_SINGLE_GROQ_KEY_HERE
```

The service remains backward compatible with this old format.

## 4. Alternative Primary/Secondary Setup

This also works:

```text
GROQ_API_KEY_PRIMARY=PASTE_PRIMARY_GROQ_KEY_HERE
GROQ_API_KEY_SECONDARY=PASTE_BACKUP_GROQ_KEY_HERE
```

If more than one format is set, the service deduplicates keys and keeps the first configured order.

## 5. How Failover Works

For each LLM answer:

1. The service tries the first configured Groq key.
2. If that call succeeds, the answer is used.
3. If the call fails because of rate limit, quota, timeout, temporary API failure, or network/provider error, the service tries the next configured key.
4. Each configured key is tried at most once per answer generation.
5. If all configured Groq keys fail, the service returns the existing safe fallback answer.

Failover only affects LLM answer generation reliability. It does not change diagnosis, classifier, fusion, safety, urgency, or doctor routing logic.

## 6. Check `/health`

Start the service:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/health
```

Expected with two real Groq keys configured:

```json
{
  "status": "ok",
  "service": "MedBridge AI Service",
  "llm_configured": true,
  "llm_key_count": 2,
  "artifacts": {}
}
```

The `artifacts` object will contain local artifact readiness details.

The health response never exposes actual API key values.

## 7. If One Key Fails

If the primary key hits quota, rate limit, timeout, or temporary provider failure, the service tries the backup key automatically.

The user receives the normal AI answer if the backup key succeeds.

## 8. If Both Keys Fail

If all configured Groq keys fail, the service returns the existing safe fallback answer.

The fallback answer is deterministic, conservative, and does not call external LLM providers.

## 9. Gemini Note

`.env.example` includes an optional commented `GEMINI_API_KEY` placeholder only for future experiments.

Gemini is not used automatically. Do not expect Gemini failover unless a future Gemini provider implementation is built and tested.

## 10. Security Rules

- Never commit `.env`.
- Never paste real API keys into documentation.
- Never print real API keys in logs.
- Use deployment platform secret managers for hosted environments.
- Rotate a key immediately if it is exposed.
