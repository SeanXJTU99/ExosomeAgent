# Exosome CRO Agent — Docker Image
# Single-container deployment for the full agent stack:
#   vLLM (INT4 model serving) + FastAPI (LangGraph runtime)
#
# Base: nvidia/cuda:12.1-runtime-ubuntu22.04
# Target: RTX 4090 24G or RTX 3090 24G

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL description="Exosome CRO Agent — LangGraph + vLLM + FAISS RAG"
LABEL hardware="RTX 4090 24G (or RTX 3090 24G)"
LABEL data_notice="All knowledge base data is fictional for demonstration"

# ─── System dependencies ──────────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

# ─── Python dependencies ──────────────────────────────────────────────────

WORKDIR /app

# Install PyTorch (CUDA 12.1)
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ─────────────────────────────────────────────────────

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# ─── Runtime ──────────────────────────────────────────────────────────────

# Expose vLLM API port
EXPOSE 8000

# Default: start vLLM API server
# Override with docker compose for full stack
CMD ["bash", "src/deploy/vllm_serve.sh"]
