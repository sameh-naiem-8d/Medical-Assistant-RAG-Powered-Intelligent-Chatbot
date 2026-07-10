# MongoDB Chat Schema Recommendation

MongoDB belongs to the main backend, not the AI service. The AI service stays stateless and only returns `case_state_update` for the backend to save.

## ai_chat_sessions

Recommended fields:

```js
{
  _id: ObjectId,
  userId: ObjectId,
  title: String,
  language: String,
  source: "mobile" | "web" | "local_demo",
  status: "active" | "archived",
  summary: String,
  lastKnownCaseState: {
    mode: String,
    urgency_level: String,
    possible_diagnosis: String,
    display_diagnosis_ar: String,
    suggested_doctor: String,
    display_doctor_ar: String,
    confidence: Number,
    known_symptoms: [String],
    denied_symptoms: [String],
    denied_concepts: [String],
    medical_domains: [String],
    follow_up_questions: [String],
    safety_flags: [String],
    temperature_c: Number,
    has_high_temperature: Boolean,
    duration_known: Boolean,
    previous_diagnosis: String,
    asked_temperature: Boolean,
    asked_duration: Boolean,
    asked_red_flags: Boolean
  },
  createdAt: Date,
  updatedAt: Date,
  lastMessageAt: Date
}
```

## ai_chat_messages

Recommended fields:

```js
{
  _id: ObjectId,
  sessionId: ObjectId,
  userId: ObjectId,
  role: "user" | "assistant" | "system",
  content: String,
  language: String,
  isVisibleToUser: Boolean,
  aiMetadata: {
    mode: String,
    urgency: String,
    possibleDiagnosis: String,
    displayDiagnosisAr: String,
    doctorSuggestion: String,
    displayDoctorAr: String,
    followUpQuestions: [String],
    safetyFlags: [String],
    confidence: Number,
    modelProvider: String
  },
  createdAt: Date,
  updatedAt: Date
}
```

## Backend Flow

1. Authenticate the user.
2. Create an `ai_chat_sessions` row if `conversation_id` is missing.
3. Load the last 6-10 visible messages from `ai_chat_messages`.
4. Call AI `POST /chat`.
5. Save the user message.
6. Save the assistant answer and safe metadata.
7. Update `lastKnownCaseState` from `case_state_update`.
8. Return the answer to the frontend.

Do not store API keys, hidden prompts, or large retrieved RAG chunks in MongoDB.
