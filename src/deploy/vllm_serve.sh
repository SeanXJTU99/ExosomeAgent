#!/usr/bin/env bash
# =============================================================================
# vLLM API Server Launch Script
# Serves the AWQ INT4 quantized exosome CRO agent model behind an OpenAI-
# compatible API endpoint for LangGraph to consume.
#
# Target hardware: Single RTX 4090 24G
# Model: Qwen2.5-14B-Instruct AWQ INT4
# =============================================================================

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-checkpoints/awq_int4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
DTYPE="${DTYPE:-auto}"

# ─── Launch ─────────────────────────────────────────────────────────────────

echo "============================================"
echo "vLLM API Server — Exosome CRO Agent"
echo "============================================"
echo "  Model:       ${MODEL_PATH}"
echo "  Endpoint:    http://${HOST}:${PORT}"
echo "  Max len:     ${MAX_MODEL_LEN}"
echo "  GPU util:    ${GPU_MEMORY_UTIL}"
echo "  Max seqs:    ${MAX_NUM_SEQS}"
echo "  Dtype:       ${DTYPE}"
echo "============================================"

# In production:
#   python -m vllm.entrypoints.openai.api_server \
#       --model "${MODEL_PATH}" \
#       --host "${HOST}" \
#       --port "${PORT}" \
#       --max-model-len "${MAX_MODEL_LEN}" \
#       --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
#       --max-num-seqs "${MAX_NUM_SEQS}" \
#       --dtype "${DTYPE}" \
#       --quantization awq \
#       --enable-prefix-caching \
#       --trust-remote-code

echo "[Mock] vLLM server would start at http://${HOST}:${PORT}"
echo "[Mock] Press Ctrl+C to stop"
