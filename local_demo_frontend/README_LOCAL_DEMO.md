# MedBridge AI Local Demo

This is a local-only patient-facing demo UI for manual testing. It is not the official team frontend and should stay ignored by Git.

## Run Locally

Terminal 1: start the FastAPI backend.

```powershell
cd "D:\Project Graduation\MEDBRIDGE_AI_TEAM_DELIVERY\medbridge-ai-service"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Terminal 2: start the static local demo frontend.

```powershell
cd "D:\Project Graduation\MEDBRIDGE_AI_TEAM_DELIVERY\medbridge-ai-service\local_demo_frontend"
python -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173
```

## Expected Checks

Backend health:

```text
http://127.0.0.1:8010/health
```

Expected: `status: ok`, `llm_configured: true`, `llm_key_count: 2`, and all artifacts loaded.

Frontend status:

```text
http://127.0.0.1:5173
```

Expected: the page shows `متصل بخدمة MedBridge`. If it shows `غير متصل بالخدمة`, press `إعادة فحص الاتصال` after starting the backend.

## What The Demo Shows

The UI shows patient-facing content only:

- assistant answer
- emergency badge when needed

It intentionally hides internal/debug fields such as mode, urgency metadata, extracted symptoms, raw English diagnosis/doctor labels, confidence, retrieved cases, RAG data, follow-up metadata, and API error internals. The assistant `answer` is the single patient-visible chat response.

## Troubleshooting

If port `8010` is busy:

```powershell
netstat -ano | findstr :8010
taskkill /PID <PID> /F
```

If the frontend cannot connect:

1. Open `http://127.0.0.1:8010/health` first.
2. Confirm it returns `status: ok`.
3. Open `http://127.0.0.1:5173`.
4. Press `إعادة فحص الاتصال`.

If the backend is off, the frontend will show a friendly connection error and will not show raw API fields.

## Notes

- The demo sends the last 10 local chat messages as `history`.
- The reset button clears only browser-side demo history.
- The AI service remains stateless and does not store patient data.
