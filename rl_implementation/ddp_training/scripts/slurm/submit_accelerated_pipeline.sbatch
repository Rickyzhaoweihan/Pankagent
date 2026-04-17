#!/bin/bash
#SBATCH --job-name=pankllm_accel
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/accel_pipeline_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/accel_pipeline_%j.err

# =============================================================================
# Accelerated Dynamic Training Pipeline (4 GPU SLURM Job)
# =============================================================================
#
# NOTE: You can also use submit_full_pipeline.sbatch with ACCELERATED=1
#       which provides more options and is the recommended entry point.
#       This script is kept for convenience.
#
# Optimized pipeline with overlapping rollout collection and training:
#   - While training iteration N, start collecting rollouts for N+1
#   - Saves 30-50% time compared to sequential pipeline
#
# GPU Allocation:
#   GPU 0: vLLM Orchestrator server (port 8001)
#   GPU 1: vLLM Cypher Generator server (port 8002)
#   GPU 2: Cypher Generator training
#   GPU 3: Orchestrator training
#
# Submit with:
#   sbatch submit_accelerated_pipeline.sbatch
#
# Options (via --export):
#   sbatch --export=ITERATIONS=20 submit_accelerated_pipeline.sbatch
#   sbatch --export=QUESTIONS=128,DIFFICULTY=medium submit_accelerated_pipeline.sbatch
#   sbatch --export=AUTO_MODE=1 submit_accelerated_pipeline.sbatch  # Run until convergence
#   sbatch --export=CLEAR_HISTORY=1 submit_accelerated_pipeline.sbatch  # Fresh start
#   sbatch --export=SKIP_FIRST_COLLECT=1 submit_accelerated_pipeline.sbatch  # Use existing rollouts
#
# =============================================================================

set -e

echo "=============================================="
echo "Accelerated Training Pipeline (4 GPU)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Configuration (can be overridden via --export)
# =============================================================================

ITERATIONS="${ITERATIONS:-10}"
QUESTIONS="${QUESTIONS:-64}"
DIFFICULTY="${DIFFICULTY:-easy}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
AUTO_MODE="${AUTO_MODE:-0}"
SKIP_FIRST_COLLECT="${SKIP_FIRST_COLLECT:-0}"
CLEAR_HISTORY="${CLEAR_HISTORY:-0}"

echo "Configuration:"
echo "  Max Iterations: $ITERATIONS"
echo "  Questions/Iter: $QUESTIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Epochs/Iter: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Auto Converge: $AUTO_MODE"
echo "  Skip First Collect: $SKIP_FIRST_COLLECT"
echo "  Clear History: $CLEAR_HISTORY"
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

mkdir -p /tmp/r /tmp/t
export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/t

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

ulimit -n 1048576 2>/dev/null || true

# =============================================================================
# PYTHONPATH Setup
# =============================================================================

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"
SCRIPT_DIR="$PANKLLM_DIR/rl_implementation/ddp_training/scripts"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "Directories:"
echo "  RLLM: $RLLM_DIR"
echo "  PanKLLM: $PANKLLM_DIR"
echo ""

# Create output directories
LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$PANKLLM_DIR/outputs/cypher_trained"
mkdir -p "$PANKLLM_DIR/outputs/orchestrator_trained"

# =============================================================================
# Model Paths
# =============================================================================

ORCHESTRATOR_MODEL="$PANKLLM_DIR/models/qwen2.5-14b"
CYPHER_MODEL="$PANKLLM_DIR/models/qwen2.5-coder-14b"

# Verify models
for MODEL in "$ORCHESTRATOR_MODEL" "$CYPHER_MODEL"; do
    if [ ! -d "$MODEL" ]; then
        echo "ERROR: Model not found: $MODEL"
        exit 1
    fi
done

# =============================================================================
# vLLM Settings
# =============================================================================

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

ORCHESTRATOR_PORT=8001
CYPHER_PORT=8002

# =============================================================================
# Start vLLM Servers
# =============================================================================

echo "=============================================="
echo "Starting vLLM Servers"
echo "=============================================="

# Orchestrator on GPU 0
echo "Starting Orchestrator on GPU 0 (port $ORCHESTRATOR_PORT)..."
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
    > "$LOG_DIR/vllm_orch_${SLURM_JOB_ID}.log" 2>&1 &
ORCH_PID=$!

# Cypher on GPU 1
echo "Starting Cypher on GPU 1 (port $CYPHER_PORT)..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$CYPHER_MODEL" \
    --port $CYPHER_PORT \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-stats \
    > "$LOG_DIR/vllm_cypher_${SLURM_JOB_ID}.log" 2>&1 &
CYPHER_PID=$!

echo "  Orchestrator PID: $ORCH_PID"
echo "  Cypher PID: $CYPHER_PID"

# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
wait_for_server() {
    local port=$1
    local name=$2
    local attempt=0
    while [ $attempt -lt 120 ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $name ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "  ✗ $name timeout"
    return 1
}

wait_for_server $ORCHESTRATOR_PORT "Orchestrator" || exit 1
wait_for_server $CYPHER_PORT "Cypher" || exit 1

echo ""
echo "✓ All vLLM servers ready!"
echo ""

# Export PIDs for pipeline scripts
export VLLM_ORCH_PID=$ORCH_PID
export VLLM_CYPHER_PID=$CYPHER_PID

# =============================================================================
# Optional: Clear History for Fresh Start
# =============================================================================

if [ "$CLEAR_HISTORY" = "1" ]; then
    HISTORY_FILE="$PANKLLM_DIR/outputs/stage1_ddp/training_history.json"
    if [ -f "$HISTORY_FILE" ]; then
        echo "Clearing training history for fresh start..."
        rm -f "$HISTORY_FILE"
        echo "  ✓ History cleared"
    fi
    echo ""
fi

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    echo ""
    echo "Cleaning up vLLM servers..."
    kill $ORCH_PID $CYPHER_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "Done"
}
trap cleanup EXIT

# =============================================================================
# GPU Status (Training GPUs only - GPU 2,3)
# =============================================================================

echo "Checking training GPUs (2,3)..."
python3 -c "
import torch

# Only check training GPUs to avoid conflict with vLLM servers on GPU 0,1
training_gpus = [2, 3]
total_gpus = torch.cuda.device_count()
print(f'Total GPUs visible: {total_gpus}')

for i in training_gpus:
    if i < total_gpus:
        try:
            props = torch.cuda.get_device_properties(i)
            print(f'  GPU {i}: {props.name} (available for training)')
        except Exception as e:
            print(f'  GPU {i}: Available (details unavailable)')
    else:
        print(f'  GPU {i}: Not visible')
" 2>/dev/null || echo "  (GPU check skipped - vLLM servers using GPUs)"
echo ""

# =============================================================================
# Run Accelerated Pipeline
# =============================================================================

echo "=============================================="
echo "Running Accelerated Pipeline"
echo "=============================================="
echo "Time: $(date)"
echo ""

cd "$SCRIPT_DIR"

# Build arguments
ARGS="--iterations $ITERATIONS --questions $QUESTIONS --difficulty $DIFFICULTY --epochs $EPOCHS --batch-size $BATCH_SIZE"
[ "$AUTO_MODE" = "1" ] && ARGS="$ARGS --auto"
[ "$SKIP_FIRST_COLLECT" = "1" ] && ARGS="$ARGS --skip-collect"

echo "Command: ./run_accelerated_pipeline.sh $ARGS"
echo ""

./run_accelerated_pipeline.sh $ARGS 2>&1 | tee "$LOG_DIR/accel_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo ""
echo "Trained Models:"
echo "  Cypher: $PANKLLM_DIR/outputs/cypher_trained/best_model"
echo "  Orchestrator: $PANKLLM_DIR/outputs/orchestrator_trained/best_model"
echo ""

# Show training summary
HISTORY="$PANKLLM_DIR/outputs/stage1_ddp/training_history.json"
if [ -f "$HISTORY" ]; then
    echo "Training Summary:"
    python3 -c "
import json
from pathlib import Path
path = Path('$HISTORY')
h = []
if path.exists() and path.stat().st_size > 0:
    try:
        h = json.load(open('$HISTORY'))
    except:
        h = []
if h and isinstance(h, list):
    m = h[-1].get('metrics', {})
    print(f\"  Cypher Reward: {m.get('cypher_reward', 0):.3f}\")
    print(f\"  Orch Reward: {m.get('orch_avg_reward', 0):.3f}\")
    print(f\"  Iterations: {len(h)}\")
    ct = sum(1 for x in h if x.get('trained',{}).get('cypher'))
    ot = sum(1 for x in h if x.get('trained',{}).get('orchestrator'))
    print(f\"  Cypher Trained: {ct} times\")
    print(f\"  Orch Trained: {ot} times\")
else:
    print('  No history available')
"
fi

exit $EXIT_CODE

