#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# ORION — vLLM Server Launcher
# ═══════════════════════════════════════════════════════════════
# Starts a vLLM OpenAI-compatible server in Podman with
# Qwen 2.5 72B GPTQ-Int4 for local zero-cost inference.
#
# Usage:
#   ./scripts/start_vllm.sh              # Start with defaults
#   ./scripts/start_vllm.sh --help       # Show options
#   VLLM_PORT=8001 ./scripts/start_vllm.sh  # Custom port
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configurable parameters ──────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-qwen2.5-72b}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-2}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-orion-vllm}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ── Help ─────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
ORION vLLM Server Launcher

  Starts Qwen 2.5 72B (GPTQ-Int4) in a Podman container with
  OpenAI-compatible API on port $VLLM_PORT.

  Environment Variables:
    VLLM_MODEL           Model HuggingFace ID   (default: $VLLM_MODEL)
    VLLM_SERVED_NAME     Served model name       (default: $VLLM_SERVED_NAME)
    VLLM_PORT            API port                (default: $VLLM_PORT)
    VLLM_MAX_MODEL_LEN   Max context length      (default: $VLLM_MAX_MODEL_LEN)
    VLLM_TENSOR_PARALLEL Tensor parallel GPUs    (default: $VLLM_TENSOR_PARALLEL)
    VLLM_GPU_MEM_UTIL    GPU memory utilization  (default: $VLLM_GPU_MEM_UTIL)
    HF_CACHE             HuggingFace cache dir   (default: \$HOME/.cache/huggingface)

  Examples:
    ./scripts/start_vllm.sh
    VLLM_PORT=8001 ./scripts/start_vllm.sh
    VLLM_TENSOR_PARALLEL=4 ./scripts/start_vllm.sh
EOF
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────
echo "═══ ORION vLLM Launcher ═══"
echo ""

if ! command -v podman &>/dev/null; then
    echo "ERROR: Podman is not installed or not in PATH."
    exit 1
fi

if ! podman info &>/dev/null; then
    echo "ERROR: Podman daemon is not running."
    exit 1
fi

# Check for NVIDIA runtime
if ! podman info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Podman runtime not detected."
    echo "  vLLM requires GPU access. Install nvidia-container-toolkit."
    echo ""
fi

# ── Stop existing container ──────────────────────────────────
if podman ps -a --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER_NAME}$"; then
    echo "  → Stopping existing container: ${VLLM_CONTAINER_NAME}"
    podman rm -f "${VLLM_CONTAINER_NAME}" &>/dev/null || true
fi

# ── Launch vLLM ──────────────────────────────────────────────
echo "  → Launching vLLM server..."
echo "    Model:    ${VLLM_MODEL}"
echo "    Served:   ${VLLM_SERVED_NAME}"
echo "    Port:     ${VLLM_PORT}"
echo "    Context:  ${VLLM_MAX_MODEL_LEN}"
echo "    TP Size:  ${VLLM_TENSOR_PARALLEL}"
echo "    GPU Mem:  ${VLLM_GPU_MEM_UTIL}"
echo ""

podman run -d \
    --name "${VLLM_CONTAINER_NAME}" \
    --runtime nvidia \
    --gpus all \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -p "${VLLM_PORT}:8000" \
    "${VLLM_IMAGE}" \
    --model "${VLLM_MODEL}" \
    --served-model-name "${VLLM_SERVED_NAME}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL}" \
    --quantization gptq \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

echo ""
echo "  → Container started: ${VLLM_CONTAINER_NAME}"

# ── Wait for health ──────────────────────────────────────────
echo "  → Waiting for vLLM to become healthy (max 120s)..."

HEALTH_URL="http://localhost:${VLLM_PORT}/health"
MAX_WAIT=120
POLL_INTERVAL=5
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    if curl -sf "${HEALTH_URL}" &>/dev/null; then
        echo "  ✓ vLLM is healthy!"
        echo "  ✓ API endpoint: http://localhost:${VLLM_PORT}/v1"
        echo ""
        echo "  Test with:"
        echo "    curl http://localhost:${VLLM_PORT}/v1/models"
        exit 0
    fi
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
    echo "    Waiting... (${elapsed}s / ${MAX_WAIT}s)"
done

echo "  ✗ vLLM did not become healthy within ${MAX_WAIT}s."
echo "  Check logs: podman logs ${VLLM_CONTAINER_NAME}"
exit 1
