FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/app/model_cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache/sentence_transformers \
    TOKENIZERS_PARALLELISM=false \
    RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    RAG_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
    ENABLE_RERANKER=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY scripts/preload_models.py ./scripts/preload_models.py

# Download and cache the embedding and reranker models during the image build.
RUN python scripts/preload_models.py

COPY app ./app
COPY artifacts ./artifacts

ENV ARTIFACTS_DIR=/app/artifacts \
    ENABLE_RAG_RETRIEVAL=true \
    RAG_LOAD_ON_STARTUP=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
