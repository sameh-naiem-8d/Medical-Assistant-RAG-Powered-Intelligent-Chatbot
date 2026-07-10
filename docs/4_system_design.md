# System Design

## System Overview

MedBridge AI is an AI-powered medical assistant that combines Machine Learning, Retrieval-Augmented Generation (RAG), and a Large Language Model (LLM) to provide preliminary medical guidance based on user symptoms.

## System Architecture

```
User
   │
   ▼
FastAPI Backend
   │
   ▼
Conversation Orchestrator
   │
   ├── Disease Classifier
   ├── RAG Service
   ├── Knowledge Base
   └── LLM Service
   │
   ▼
AI Response
```

## Main Components

### Backend

- Handles API requests.
- Validates user input.
- Manages chat sessions.
- Coordinates all system services.

### Disease Classifier

- Receives extracted symptoms.
- Predicts the most likely diseases using a Machine Learning model.

### Knowledge Base

- Stores trusted medical information.
- Provides medical context for the AI.

### RAG Service

- Searches the knowledge base for relevant information.
- Supplies the retrieved content to the LLM.

### LLM Service

- Generates natural and easy-to-understand responses.
- Uses retrieved medical knowledge to improve response quality.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Process user symptoms and generate a response. |
| `/health` | GET | Check system health and artifact status. |

## Workflow

1. The user submits symptoms.
2. The backend validates the request.
3. The classifier predicts possible diseases.
4. The RAG service retrieves related medical knowledge.
5. The LLM generates the final response.
6. The response is returned to the user.

## Technologies

- Python
- FastAPI
- Scikit-learn
- FAISS
- Groq LLM
- Git & GitHub

## Future Improvements

- Voice interaction.
- Mobile application support.
- Hospital information system integration.
- Enhanced Arabic language understanding.
