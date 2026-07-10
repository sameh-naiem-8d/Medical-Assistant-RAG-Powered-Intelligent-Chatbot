# Backend Integration Guide

The inspected backend ZIP `D:\medback-master.zip` is a Node/Express/Mongoose project. It already has:

- `src/app.js`
- `src/routes/chatbot.routes.js`
- `src/controllers/chatbot.controller.js`
- `src/models/chatbot.model.js`
- `src/models/message.model.js`
- auth middleware: `protect`, `isUser`
- current route: `POST /api/chatbot/analyze`

The current chatbot route is a placeholder. The backend team can either replace that handler or add a cleaner new route such as `POST /api/ai/chat`.

## Environment Variables In Backend

```env
AI_SERVICE_URL=http://127.0.0.1:8010
AI_SERVICE_TIMEOUT_MS=30000
```

Never expose `GROQ_API_KEY` to the frontend or main backend clients.

## Recommended Route

`POST /api/ai/chat`

Auth required.

Request from frontend to backend:

```json
{
  "message": "عندي كحة وسخونية",
  "conversationId": "optional-session-id",
  "source": "web"
}
```

Backend sends to AI service:

```json
{
  "user_id": "authenticated-user-id",
  "conversation_id": "mongo-session-id",
  "language": "ar",
  "source": "web",
  "message": "عندي كحة وسخونية",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

## Important Rules

- The main backend owns MongoDB persistence.
- The AI service owns medical conversation reasoning only.
- Send only recent history, not the full chat forever.
- Users must only access their own sessions.
- Frontend should receive patient-facing fields, not internal debug metadata.

## Suggested Implementation Order

1. Add the two Mongoose models from `examples/medback_ai_model_example.js`.
2. Add the controller from `examples/medback_ai_controller_example.js`.
3. Add the route from `examples/medback_ai_routes_example.js`.
4. Mount the route in `src/app.js`.
5. Test local backend to local AI service on port `8010`.
6. After deployment, set `AI_SERVICE_URL` to the deployed FastAPI URL.
