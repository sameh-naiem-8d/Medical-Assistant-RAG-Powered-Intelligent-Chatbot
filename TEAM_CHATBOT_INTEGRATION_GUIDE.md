# MedBridge AI Team Chatbot Integration Guide

Generated: 2026-06-17

This guide explains how the real MedBridge backend and frontend teams should connect their existing system to the MedBridge AI chatbot service.

No deployment is performed by this document.

## 1. What This Service Is

MedBridge AI is an independent FastAPI AI microservice for Arabic symptom triage and doctor routing.

It is:

- Stateless.
- Arabic-first.
- Designed to receive a message plus optional recent history.
- Responsible for AI reasoning, triage mode, urgency, doctor routing, and patient-facing answer text.
- Not responsible for authentication, users, sessions, MongoDB, or long-term storage.

The real backend team should store users, sessions, and chat messages in MongoDB. The AI service only receives recent history and returns one structured response.

## 2. Endpoints

### `GET /health`

Use this for readiness checks.

Expected shape:

```json
{
  "status": "ok",
  "service": "MedBridge AI Service",
  "llm_configured": true,
  "artifacts": {
    "faiss.index": true,
    "knowledge_base.pkl": true,
    "disease_classifier.pkl": true
  }
}
```

### `POST /chat`

Use this for every user message.

## 3. Request Body

```json
{
  "message": "عندي كحة وسخونية",
  "history": [
    {
      "role": "user",
      "content": "زوري واجعني"
    },
    {
      "role": "assistant",
      "content": "هل يوجد حرارة أو كحة؟"
    }
  ],
  "user_id": "optional-user-id",
  "session_id": "optional-session-id"
}
```

Field notes:

- `message`: required current user message.
- `history`: optional recent chat history. Send the last 6 to 10 messages when available.
- `user_id`: optional external backend user id. The AI service does not store it.
- `session_id`: optional external backend session id. The AI service does not store it.

Important history rule:

- The AI service uses recent user messages as symptom context.
- Assistant messages are used to avoid repeating exact same follow-up questions.
- The backend remains the source of truth for full conversation history.

## 4. Response Body

```json
{
  "mode": "diagnosis",
  "answer": "...",
  "extracted_symptoms": ["cough", "high_fever"],
  "possible_diagnosis": "Common Cold",
  "display_diagnosis_ar": "نزلة برد / التهاب في الجهاز التنفسي العلوي",
  "confidence": 0.59,
  "urgency_level": "Medium",
  "suggested_doctor": "General Practitioner",
  "display_doctor_ar": "طبيب عام / باطنة",
  "precautions": [],
  "needs_follow_up": true,
  "follow_up_questions": [
    "منذ كم يوم بدأت الكحة والحرارة؟"
  ],
  "retrieved_cases": []
}
```

Field explanation:

- `mode`: chatbot behavior mode. Values: `clarification`, `diagnosis`, `emergency`, `closing`.
- `answer`: patient-facing Arabic answer. This is the main text to show in the chat UI.
- `extracted_symptoms`: internal symptom codes extracted from the Arabic message. Store for analytics if needed, but do not show to patients.
- `possible_diagnosis`: internal English diagnosis label for backend logic. Do not show this directly to patients.
- `display_diagnosis_ar`: optional patient-facing Arabic diagnosis label. Prefer this in UI badges when present.
- `confidence`: numeric model confidence from 0.0 to 1.0. Do not show `0.0` in clarification mode as a failure.
- `urgency_level`: `Low`, `Medium`, or `High`.
- `suggested_doctor`: internal English doctor type for routing and booking.
- `display_doctor_ar`: optional patient-facing Arabic doctor label. Prefer this in UI labels when present.
- `precautions`: structured precaution list when available.
- `needs_follow_up`: whether the AI has follow-up metadata for conversation continuity.
- `follow_up_questions`: backend metadata for storage/suggested reply chips. Do not render it as a separate patient-visible block because the question is already included naturally inside `answer`.
- `retrieved_cases`: internal RAG evidence snippets. Hide from patients.

Important display rule:

- Show `answer` as the single patient-visible chat message. Optional Arabic doctor/diagnosis badges may be shown only if they do not duplicate the answer.
- Keep `possible_diagnosis` and `suggested_doctor` for backend logic, booking flows, and analytics.
- Never show internal English/misspelled labels such as `Peptic ulcer diseae` directly in the chat UI.
- Never show `retrieved_cases` or raw RAG scores to patients.

## 5. Integration Flow

1. Frontend user sends a message.
2. Team backend saves the user message in MongoDB.
3. Team backend loads recent session history.
4. Team backend calls the AI service `POST /chat`.
5. AI service returns structured response.
6. Team backend saves assistant response plus useful metadata.
7. Team backend returns a clean response to the frontend.
8. Frontend displays the answer once, with optional emergency styling or non-duplicating badges.

## 6. Suggested MongoDB Collections

### `chat_sessions`

Recommended fields:

- `_id`
- `user_id`
- `title`
- `created_at`
- `updated_at`
- `status`: `active` or `closed`
- `last_mode`
- `last_urgency`
- `last_possible_diagnosis`
- `last_suggested_doctor`

### `chat_messages`

Recommended fields:

- `_id`
- `session_id`
- `user_id`
- `role`: `user` or `assistant`
- `content`
- `timestamp`
- `mode`
- `possible_diagnosis`
- `urgency_level`
- `suggested_doctor`
- `needs_follow_up`
- `follow_up_questions`
- `safety_flags`

Do not store Groq API keys, raw environment variables, or deployment secrets in MongoDB.

## 7. Backend Pseudo-Code

### Python/FastAPI-style

```python
@router.post("/chat/sessions/{session_id}/messages")
async def send_chat_message(session_id: str, payload: UserMessage, current_user: User):
    await mongo.chat_messages.insert_one({
        "session_id": session_id,
        "user_id": current_user.id,
        "role": "user",
        "content": payload.message,
        "timestamp": utc_now(),
    })

    recent_messages = await load_recent_messages(session_id, limit=10)
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in recent_messages
    ]

    ai_response = await http.post(
        f"{AI_SERVICE_URL}/chat",
        json={
            "user_id": str(current_user.id),
            "session_id": session_id,
            "message": payload.message,
            "history": history,
        },
        timeout=30,
    )

    data = ai_response.json()

    await mongo.chat_messages.insert_one({
        "session_id": session_id,
        "user_id": current_user.id,
        "role": "assistant",
        "content": data["answer"],
        "timestamp": utc_now(),
        "mode": data["mode"],
        "possible_diagnosis": data["possible_diagnosis"],
        "urgency_level": data["urgency_level"],
        "suggested_doctor": data["suggested_doctor"],
        "needs_follow_up": data["needs_follow_up"],
        "follow_up_questions": data["follow_up_questions"],
    })

    if data["mode"] == "closing":
        await mongo.chat_sessions.update_one(
            {"_id": session_id},
            {"$set": {"status": "closed", "updated_at": utc_now()}},
        )

    return frontend_safe_response(data)
```

### Node/Express-style

```javascript
app.post("/chat/sessions/:sessionId/messages", async (req, res) => {
  const { message } = req.body;
  const { sessionId } = req.params;
  const userId = req.user.id;

  await ChatMessage.create({
    session_id: sessionId,
    user_id: userId,
    role: "user",
    content: message,
    timestamp: new Date()
  });

  const recentMessages = await ChatMessage.find({ session_id: sessionId })
    .sort({ timestamp: -1 })
    .limit(10)
    .lean();

  const history = recentMessages
    .reverse()
    .map((item) => ({ role: item.role, content: item.content }));

  const aiResult = await fetch(`${process.env.AI_SERVICE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: String(userId),
      session_id: String(sessionId),
      message,
      history
    })
  });

  const data = await aiResult.json();

  await ChatMessage.create({
    session_id: sessionId,
    user_id: userId,
    role: "assistant",
    content: data.answer,
    timestamp: new Date(),
    mode: data.mode,
    possible_diagnosis: data.possible_diagnosis,
    urgency_level: data.urgency_level,
    suggested_doctor: data.suggested_doctor,
    needs_follow_up: data.needs_follow_up,
    follow_up_questions: data.follow_up_questions
  });

  res.json(frontendSafeResponse(data));
});
```

## 8. Frontend Rules

Show to the patient:

- `answer`
- `urgency_level`
- `display_doctor_ar` when available
- `display_diagnosis_ar` only in `diagnosis` or `emergency` mode

Hide from the patient:

- `retrieved_cases`
- raw RAG scores
- internal extracted symptom codes
- internal misspelled labels
- internal English diagnosis labels
- `confidence: 0.0` in clarification mode

The frontend should treat `answer` as the only chat bubble text. The other fields are for UI badges, routing, storage, analytics, suggested reply chips, or booking flows. Do not show a separate “أسئلة توضيحية” / “Clarifying Questions” section.

If the team wants a strict patient-safe subset, mirror `app.response_utils.to_frontend_safe_response(data)`:

```python
{
    "mode": data["mode"],
    "answer": data["answer"],
    "display_diagnosis_ar": data.get("display_diagnosis_ar"),
    "display_doctor_ar": data.get("display_doctor_ar"),
    "urgency_level": data["urgency_level"],
    "suggested_doctor": data["suggested_doctor"],
    "needs_follow_up": data["needs_follow_up"],
}
```

## 9. UI Behavior by Mode

### `clarification`

- Show the answer only.
- Keep the chat open.
- Encourage the user to reply with more details.
- Do not show a failed diagnosis state.
- Do not show confidence.

### `diagnosis`

- Show the answer.
- Show urgency.
- Show suggested doctor.
- Keep the chat open unless the user closes it.

### `emergency`

- Highlight the urgent warning strongly.
- Show emergency care routing.
- Do not display it like routine advice.
- Provide a clear call to seek urgent medical care.
- Do not encourage waiting or monitoring only.

### `closing`

- Show the polite closing answer.
- Do not ask more medical questions.
- The backend may mark the session as `closed`.
- The user can start a new session later.

## 10. Error Handling

### AI service down

Backend should return a friendly message:

```text
الخدمة غير متاحة حاليًا. لو عندك أعراض شديدة أو طارئة، توجه للطوارئ أو اتصل بالإسعاف.
```

Do not invent a diagnosis from the frontend or backend.

### Timeout

Use a backend timeout such as 30 seconds. If timeout happens, return the same safe unavailable message and log the event.

### Invalid response

If required fields are missing, treat it as service failure. Do not show partial internal data.

### Missing Groq key

The AI service can still return a fallback answer, but the deployment should not be considered fully configured unless `/health` returns `llm_configured: true`.

### Fallback behavior

Fallback answers should be logged internally by the backend if possible. The frontend should not show technical fallback labels to the patient.

## 11. Local Run Instructions

From:

```bash
D:/Project Graduation/medbridge-ai-service
```

Install dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
CORS_ORIGINS=http://localhost:5173
ARTIFACTS_DIR=artifacts
```

Run:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
http://localhost:8000/health
```

Test:

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"عندي كحة وسخونية\",\"history\":[]}"
```

## 12. Important Limitations

- This is a decision-support and triage assistant, not a replacement for doctors.
- It does not guarantee coverage of all diseases.
- The honest claim is: broad Arabic symptom triage and doctor-routing assistant.
- Emergency advice must tell the user to seek urgent medical care.
- The backend team must own user identity, access control, retention policy, and MongoDB storage.
- The frontend should never expose internal RAG cases, raw scores, or symptom codes to patients.
