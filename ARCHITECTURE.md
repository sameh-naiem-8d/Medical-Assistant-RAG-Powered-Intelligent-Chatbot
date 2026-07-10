# Architecture

FastAPI receives `/chat`, builds conversation state, applies emergency safety checks, extracts symptoms, consults classifier/knowledge/RAG services, uses Groq or deterministic fallback, and applies final consistency guardrails.
