# Team Final Handoff Checklist

## AI Service

- Run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8010`.
- Open `http://127.0.0.1:8010/health`.
- Confirm `status: ok`.
- Confirm `llm_configured: true` for Groq-powered manual testing.
- Confirm artifacts are loaded.

## Backend

- Add `AI_SERVICE_URL`.
- Add chat session/message MongoDB models.
- Create or replace chatbot route.
- Load last 6-10 messages and pass them as `history`.
- Save user message, assistant answer, and `case_state_update`.
- Do not store secrets or hidden prompts.

## Frontend

- Show `answer`.
- Do not render `follow_up_questions` as a separate patient-visible block; the assistant question should already be inside `answer`.
- Highlight emergency answers.
- Hide internal fields such as confidence, raw diagnosis, retrieved cases, and case state.

## Final Accepted AI Status

- Emergency recall: 100%
- Safety pass rate: 98%
- Mode accuracy: 93.50%
- Diagnosis group accuracy: 87.50%
- Doctor accuracy: 92.50%
- Clarification behavior: 97.83%
- Closing behavior: 100%
- Urgency accuracy: 87.50%

This is accepted for backend/frontend handoff as a production-like graduation prototype. It is not approved for unsupervised public medical production.
