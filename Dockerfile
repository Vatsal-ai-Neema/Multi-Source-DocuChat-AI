# ═══════════════════════════════════════════════════════════════
# DocuChat AI v2 — Dockerfile
# ═══════════════════════════════════════════════════════════════
# Build:  docker build -t docuchat-ai-v2 .
# Run:    docker run -p 8501:8501 -e GEMINI_API_KEY=your_key docuchat-ai-v2
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (MiniLM for RAG)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-download NLI model for [07] Contradiction Detector (~85MB, runs locally)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-small')"

COPY . .
RUN mkdir -p data faiss_index

EXPOSE 10000
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0"]
# OPTIONAL - comment for CI speed
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-small')"
