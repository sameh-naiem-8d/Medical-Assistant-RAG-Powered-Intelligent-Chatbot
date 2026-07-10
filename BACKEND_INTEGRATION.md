# Backend Integration

The AI service is stateless for production. Main backend should store sessions and message history, then pass recent messages in `history`.
