# Api Contract


`GET /health` returns status, build identifiers, LLM configuration status without secret values, and artifact flags.

`POST /chat` accepts `user_id`, `conversation_id`, `session_id`, `language`, `source`, `message`, and `history`. It returns `mode`, `answer`, symptoms, diagnosis labels, confidence, urgency, doctor labels, precautions, follow-up questions, retrieved cases, and `case_state_update`.

The patient-facing UI should render `answer` as the single chat message and prefer display labels for visible diagnosis/doctor text.
