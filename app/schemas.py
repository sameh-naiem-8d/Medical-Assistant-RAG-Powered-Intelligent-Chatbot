from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="Conversation role: user or assistant.")
    content: str = Field(default="", description="Message text supplied by the backend or frontend.")


class ChatRequest(BaseModel):
    user_id: str | None = Field(
        default=None,
        description="Optional external user identifier. The AI service does not store it.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional backend conversation/session identifier. Returned unchanged when supplied.",
    )
    session_id: str | None = Field(
        default=None,
        description="Legacy optional external session identifier. The AI service remains stateless.",
    )
    language: str | None = Field(
        default=None,
        description="Optional backend language hint. The AI still detects language from message/history.",
    )
    source: str | None = Field(
        default=None,
        description="Optional caller source such as web, mobile, backend, or local_demo. Not stored.",
    )
    message: str = Field(..., min_length=1, description="Current user message.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Optional recent chat history supplied by the main backend. Not stored by this service.",
    )


class RetrievedCase(BaseModel):
    q_body: str
    a_body: str
    category: str | None = None
    score: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        description="Passthrough conversation_id/session_id supplied by the backend, if any.",
    )
    mode: str = Field(
        default="diagnosis",
        description="Response mode: diagnosis, clarification, emergency, or closing.",
    )
    answer: str
    extracted_symptoms: list[str] = Field(default_factory=list)
    possible_diagnosis: str | None = None
    display_diagnosis_ar: str | None = Field(
        default=None,
        description="Optional patient-facing Arabic diagnosis label. Internal diagnosis remains unchanged.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_level: str
    suggested_doctor: str
    display_doctor_ar: str | None = Field(
        default=None,
        description="Optional patient-facing Arabic doctor label. Internal doctor label remains unchanged.",
    )
    precautions: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False
    follow_up_questions: list[str] = Field(default_factory=list)
    retrieved_cases: list[RetrievedCase] = Field(default_factory=list)
    case_state_update: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe stateless metadata for the backend to store on the chat session.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    app_version: str | None = None
    build_id: str | None = None
    frontend_build_id: str | None = None
    llm_configured: bool
    llm_key_count: int = Field(
        default=0,
        description="Number of configured Groq keys. Secret values are never exposed.",
    )
    llm_model: str | None = Field(
        default=None,
        description="Configured primary LLM model name. No secrets are exposed.",
    )
    llm_fallback_model: str | None = Field(
        default=None,
        description="Configured fallback LLM model name, if any.",
    )
    artifacts: dict[str, Any]
