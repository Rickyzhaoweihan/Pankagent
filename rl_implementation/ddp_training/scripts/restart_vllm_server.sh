#!/bin/bash
# =============================================================================
# Restart Individual vLLM Server
#
# Stops and restarts a single vLLM server with a new model path.
# Useful for updating models after training without restarting all servers.
#
# Usage:
#   ./restart_vllm_server.sh --server orchestrator --model /path/to/trained/model
#   ./restart_vllm_server.sh --server cypher --model /path/to/trained/model
#   ./restart_vllm_server.sh --server orchestrator  # Uses default model path
#
# Options:
#   --server    Server to restart: 'orchestrator' or 'cypher'
#   --model     Path to new model (optional, uses default if not specified)
#   --port      Override default port (optional)
#   --gpu       Override default GPU (optional)
#
# =============================================================================

set -e

# =============================================================================
# Parse arguments
# =============================================================================
SERVER=""
MODEL_PATH=""
PORT=""
GPU=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server|-s)
            SERVER="$2"
            shift 2
            ;;
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --server <orchestrator|cypher> [--model /path/to/model] [--port PORT] [--gpu GPU]"
            echo ""
            echo "Options:"
            echo "  --server, -s    Server to restart: 'orchestrator' or 'cypher' (required)"
            echo "  --model, -m     Path to new model (optional, uses default if not specified)"
            echo "  --port, -p      Override default port (optional)"
            echo "  --gpu, -g       Override default GPU (optional)"
            echo ""
            echo "Examples:"
            echo "  $0 --server orchestrator --model /path/to/trained/orchestrator"
            echo "  $0 --server cypher --model /path/to/trained/cypher"
            echo "  $0 --server orchestrator  # Uses default model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate server argument
if [[ -z "$SERVER" ]]; then
    echo "ERROR: --server is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ "$SERVER" != "orchestrator" && "$SERVER" != "cypher" ]]; then
    echo "ERROR: --server must be 'orchestrator' or 'cypher'"
    exit 1
fi

# =============================================================================
# Conda Environment
# =============================================================================
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
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
PANKLLM_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"

# Default model paths
DEFAULT_ORCHESTRATOR_MODEL="$PANKLLM_DIR/models/qwen2.5-14b"
DEFAULT_CYPHER_MODEL="$PANKLLM_DIR/models/qwen2.5-coder-14b"

# Default ports
DEFAULT_ORCHESTRATOR_PORT=8001
DEFAULT_CYPHER_PORT=8002

# Default GPUs
DEFAULT_ORCHESTRATOR_GPU=0
DEFAULT_CYPHER_GPU=1

# vLLM settings
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

# Log directory
LOG_DIR="${LOG_DIR:-$PANKLLM_DIR/outputs/stage1_ddp/logs}"
mkdir -p "$LOG_DIR"

# =============================================================================
# Set server-specific configuration
# =============================================================================
if [[ "$SERVER" == "orchestrator" ]]; then
    SERVER_NAME="Orchestrator"
    TARGET_PORT="${PORT:-$DEFAULT_ORCHESTRATOR_PORT}"
    TARGET_GPU="${GPU:-$DEFAULT_ORCHESTRATOR_GPU}"
    TARGET_MODEL="${MODEL_PATH:-$DEFAULT_ORCHESTRATOR_MODEL}"
    LOG_FILE="$LOG_DIR/vllm_orchestrator.log"
else
    SERVER_NAME="Cypher Generator"
    TARGET_PORT="${PORT:-$DEFAULT_CYPHER_PORT}"
    TARGET_GPU="${GPU:-$DEFAULT_CYPHER_GPU}"
    TARGET_MODEL="${MODEL_PATH:-$DEFAULT_CYPHER_MODEL}"
    LOG_FILE="$LOG_DIR/vllm_cypher.log"
fi

# =============================================================================
# Validate model path
# =============================================================================
if [[ ! -d "$TARGET_MODEL" ]]; then
    echo "ERROR: Model path does not exist: $TARGET_MODEL"
    exit 1
fi

# =============================================================================
# Restart Server
# =============================================================================
echo ""
echo "=============================================="
echo "Restarting $SERVER_NAME Server"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: $TARGET_MODEL"
echo "  Port: $TARGET_PORT"
echo "  GPU: $TARGET_GPU"
echo "  Log: $LOG_FILE"
echo ""

# Stop existing server on this port
echo "Stopping existing $SERVER_NAME server on port $TARGET_PORT..."
# Find and kill process listening on the target port
PID=$(lsof -ti:$TARGET_PORT 2>/dev/null || true)
if [[ -n "$PID" ]]; then
    echo "  Killing PID $PID..."
    kill $PID 2>/dev/null || true
    sleep 3
    # Force kill if still running
    if kill -0 $PID 2>/dev/null; then
        echo "  Force killing PID $PID..."
        kill -9 $PID 2>/dev/null || true
    fi
else
    echo "  No existing server found on port $TARGET_PORT"
fi

# Wait for port to be free
sleep 2

# Verify port is free
if lsof -ti:$TARGET_PORT >/dev/null 2>&1; then
    echo "ERROR: Port $TARGET_PORT is still in use!"
    exit 1
fi

# Start new server
echo ""
echo "Starting $SERVER_NAME on GPU $TARGET_GPU..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --port $TARGET_PORT \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-stats \
    > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "  Started with PID: $NEW_PID"

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready..."
wait_for_server() {
    local port=$1
    local max_attempts=120  # 10 minutes
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
        
        # Check if process is still running
        if ! kill -0 $NEW_PID 2>/dev/null; then
            echo "ERROR: Server process died!"
            echo "Check log: $LOG_FILE"
            tail -20 "$LOG_FILE"
            return 1
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 5
    done
    
    echo ""
    echo "ERROR: Server failed to start within timeout"
    return 1
}

if wait_for_server $TARGET_PORT; then
    echo ""
    echo "=============================================="
    echo "✓ $SERVER_NAME server restarted successfully!"
    echo "=============================================="
    echo ""
    echo "Endpoint: http://localhost:$TARGET_PORT/v1/completions"
    echo "Model: $TARGET_MODEL"
    echo "Log: $LOG_FILE"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "✗ Failed to restart $SERVER_NAME server"
    echo "=============================================="
    echo "Check log: $LOG_FILE"
    exit 1
fi

