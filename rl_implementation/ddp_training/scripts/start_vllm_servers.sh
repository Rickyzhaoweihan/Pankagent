#!/bin/bash
# =============================================================================
# Start vLLM OpenAI-compatible Servers (3-GPU Setup)
# 
# GPU Allocation:
#   - GPU 0: Orchestrator -> port 8001
#   - GPU 1: Cypher Generator -> port 8002
#   - GPU 2: Reserved for training
#
# Usage:
#   ./start_vllm_servers.sh [orchestrator_model] [cypher_model]
#   
# Example:
#   ./start_vllm_servers.sh \
#       /path/to/models/qwen2.5-14b \
#       /path/to/models/qwen2.5-coder-14b
#
# To stop all servers:
#   pkill -f "vllm.entrypoints.openai.api_server"
# =============================================================================

set -e

# =============================================================================
# Conda Environment
# =============================================================================

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python path: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# =============================================================================
# Cache and Temporary Directories
# =============================================================================

SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"

# Redirect ~/.cache to scratch (fixes vLLM P2P cache issues on NFS)
export XDG_CACHE_HOME="$SCRATCH/cache"

mkdir -p /tmp/r /tmp/t
export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/t

ulimit -n 1048576 2>/dev/null || echo "Note: Could not increase file descriptor limit"

# =============================================================================
# vLLM and PyTorch Configuration
# =============================================================================

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# Configuration
# =============================================================================

# Default model paths
PANKLLM_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"
ORCHESTRATOR_MODEL="${1:-${ORCHESTRATOR_MODEL:-$PANKLLM_DIR/models/qwen2.5-14b}}"
CYPHER_MODEL="${2:-${CYPHER_MODEL:-$PANKLLM_DIR/models/qwen2.5-coder-14b}}"

# Ports
ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8001}"
CYPHER_INFERENCE_PORT="${CYPHER_INFERENCE_PORT:-8002}"

# vLLM settings
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

# Log directory
LOG_DIR="${LOG_DIR:-$PANKLLM_DIR/outputs/stage1_ddp/logs}"
mkdir -p "$LOG_DIR"

# =============================================================================
# Start Servers
# =============================================================================

echo "=============================================="
echo "Starting vLLM servers (3-GPU Setup)"
echo "=============================================="
echo ""
echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""
echo "GPU Allocation:"
echo "  Orchestrator (GPU 0): port $ORCHESTRATOR_PORT"
echo "  CypherGen (GPU 1): port $CYPHER_INFERENCE_PORT"
echo "  Training: GPU 2 (reserved)"
echo ""

# Clean up existing servers
echo "Stopping any existing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Start Orchestrator (GPU 0)
echo "Starting Orchestrator on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$ORCHESTRATOR_MODEL" \
    --port $ORCHESTRATOR_PORT \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-stats \
    > "$LOG_DIR/vllm_orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "  PID: $ORCH_PID"

# Start Cypher Generator (GPU 1)
echo "Starting CypherGen on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$CYPHER_MODEL" \
    --port $CYPHER_INFERENCE_PORT \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-stats \
    > "$LOG_DIR/vllm_cypher.log" 2>&1 &
CYPHER_PID=$!
echo "  PID: $CYPHER_PID"

echo ""
echo "Waiting for servers to be ready..."
echo "(This may take a few minutes)"
echo ""

# Wait for servers
wait_for_server() {
    local port=$1
    local name=$2
    local max_attempts=120  # 10 minutes
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $name ready (port $port)"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    
    echo "  ✗ $name failed to start (port $port)"
    return 1
}

wait_for_server $ORCHESTRATOR_PORT "Orchestrator" || exit 1
wait_for_server $CYPHER_INFERENCE_PORT "CypherGen" || exit 1

echo ""
echo "=============================================="
echo "All vLLM servers ready!"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  http://localhost:$ORCHESTRATOR_PORT/v1/completions  (Orchestrator)"
echo "  http://localhost:$CYPHER_INFERENCE_PORT/v1/completions  (CypherGen)"
echo ""
echo "GPU 2 is free for training."
echo ""
echo "To stop servers: pkill -f 'vllm.entrypoints.openai.api_server'"
echo "Or press Ctrl+C"
echo ""
echo "Server logs:"
echo "  $LOG_DIR/vllm_orchestrator.log"
echo "  $LOG_DIR/vllm_cypher.log"
echo ""

# Keep script running so Ctrl+C stops both servers
trap "echo 'Stopping servers...'; kill $ORCH_PID $CYPHER_PID 2>/dev/null; exit 0" INT TERM
wait
