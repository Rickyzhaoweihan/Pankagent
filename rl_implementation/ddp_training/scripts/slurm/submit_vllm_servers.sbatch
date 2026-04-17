#!/bin/bash
#SBATCH --job-name=pankllm_vllm
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/vllm_servers_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/vllm_servers_%j.err

# =============================================================================
# PanKLLM vLLM Servers Job (2 GPUs)
# =============================================================================
#
# This job runs vLLM inference servers for:
#   - GPU 0: Orchestrator (port 8001)
#   - GPU 1: Cypher Generator (port 8002)
#
# Usage:
#   sbatch submit_vllm_servers.sbatch
#
# After servers are ready, submit training job:
#   sbatch submit_training_only.sbatch
#
# Monitor:
#   tail -f outputs/stage1_ddp/logs/vllm_servers_*.out
#   tail -f outputs/stage1_ddp/logs/vllm_orchestrator.log
#   tail -f outputs/stage1_ddp/logs/vllm_cypher.log
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM vLLM Servers Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# =============================================================================
# Cache Directories
# =============================================================================

SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

# Temp directories
mkdir -p /tmp/ray_$SLURM_JOB_ID /tmp/tmp_$SLURM_JOB_ID
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
export TMPDIR=/tmp/tmp_$SLURM_JOB_ID

# Increase file descriptor limit
ulimit -n 1048576 2>/dev/null || echo "Note: Could not increase file limit"

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
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-$PANKLLM_DIR/models/qwen2.5-14b}"
CYPHER_MODEL="${CYPHER_MODEL:-$PANKLLM_DIR/models/qwen2.5-coder-14b}"

# Ports
ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8001}"
CYPHER_INFERENCE_PORT="${CYPHER_INFERENCE_PORT:-8002}"

# vLLM settings
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

# Log directory
LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"
mkdir -p "$LOG_DIR"

echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""
echo "Ports:"
echo "  Orchestrator: $ORCHESTRATOR_PORT"
echo "  Cypher: $CYPHER_INFERENCE_PORT"
echo ""

# =============================================================================
# GPU Status
# =============================================================================

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# =============================================================================
# Start vLLM Servers
# =============================================================================

declare -a VLLM_PIDS

cleanup() {
    echo ""
    echo "=============================================="
    echo "Shutting down vLLM servers..."
    echo "=============================================="
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping vLLM server (PID $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Clean up temp directories
    rm -rf /tmp/ray_$SLURM_JOB_ID /tmp/tmp_$SLURM_JOB_ID 2>/dev/null || true
    echo "Cleanup complete"
    echo "End Time: $(date)"
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "Starting vLLM Servers"
echo "=============================================="

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
VLLM_PIDS+=($!)
echo "  PID: ${VLLM_PIDS[-1]}"

# Start Cypher Generator (GPU 1)
echo "Starting Cypher Generator on GPU 1..."
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
VLLM_PIDS+=($!)
echo "  PID: ${VLLM_PIDS[-1]}"

# =============================================================================
# Wait for Servers to be Ready
# =============================================================================

echo ""
echo "Waiting for vLLM servers to be ready..."
echo "(This may take a few minutes)"

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
echo "Server logs:"
echo "  $LOG_DIR/vllm_orchestrator.log"
echo "  $LOG_DIR/vllm_cypher.log"
echo ""
echo "You can now submit training jobs:"
echo "  sbatch submit_training_only.sbatch"
echo ""
echo "=============================================="
echo "Servers running. Waiting for job timeout or cancellation..."
echo "=============================================="

# Keep the job running - servers will stay alive until job ends
# Check server health periodically
while true; do
    sleep 300  # Check every 5 minutes
    
    if ! curl -s "http://localhost:$ORCHESTRATOR_PORT/health" > /dev/null 2>&1; then
        echo "WARNING: Orchestrator server unhealthy!"
    fi
    
    if ! curl -s "http://localhost:$CYPHER_INFERENCE_PORT/health" > /dev/null 2>&1; then
        echo "WARNING: CypherGen server unhealthy!"
    fi
    
    echo "[$(date)] Servers healthy"
done

