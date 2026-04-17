#!/bin/bash
#SBATCH --job-name=pankllm_rl_stage1
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=3
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/slurm_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/slurm_%j.err

# =============================================================================
# PanKLLM RL Stage 1 Training - SLURM Submission Script
# =============================================================================
#
# GPU Allocation (3 GPUs):
#   GPU 0: Orchestrator vLLM server -> port 8001
#   GPU 1: Cypher Generator vLLM server -> port 8002
#   GPU 2: Cypher Generator training (LoRA, rank=128)
#
# Submit with:
#   sbatch rl_implementation/ddp_training/scripts/submit_stage1.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f outputs/stage1_ddp/logs/slurm_*.out
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM RL Stage 1 Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Project directory
PROJECT_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"
cd "$PROJECT_DIR"

# Cache directories
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"

# Redirect ~/.cache to scratch (fixes vLLM P2P cache write issues on NFS)
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
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# Model Paths
# =============================================================================

ORCHESTRATOR_MODEL="$PROJECT_DIR/models/qwen2.5-14b"
CYPHER_MODEL="$PROJECT_DIR/models/qwen2.5-coder-14b"

# =============================================================================
# Ports and Settings
# =============================================================================

ORCHESTRATOR_PORT=8001
CYPHER_INFERENCE_PORT=8002

# vLLM settings
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTIL=0.90
VLLM_MAX_NUM_SEQS=64
VLLM_MAX_BATCHED_TOKENS=131072

# No tensor parallelism for 3-GPU setup
ORCHESTRATOR_TP=1  # GPU 0
CYPHER_TP=1        # GPU 1

# =============================================================================
# Output Directories
# =============================================================================

OUTPUT_DIR="$PROJECT_DIR/outputs/stage1_ddp"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_PATH="$PROJECT_DIR/rl_implementation/ddp_training/config/stage1_config.yaml"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Training settings (small for quick iteration)
NUM_EPOCHS=5
QUESTIONS_PER_EPOCH=8

# =============================================================================
# GPU Status
# =============================================================================

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# =============================================================================
# Start vLLM Servers
# =============================================================================

declare -a VLLM_PIDS

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping vLLM server (PID $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Clean up temp directories
    rm -rf /tmp/ray_$SLURM_JOB_ID /tmp/tmp_$SLURM_JOB_ID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

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
# Wait for Servers
# =============================================================================

echo ""
echo "Waiting for vLLM servers to be ready..."

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
echo "All vLLM servers ready!"
echo ""

# GPU status after loading
echo "GPU Status after loading models:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# =============================================================================
# Start Training
# =============================================================================

echo "=============================================="
echo "Starting Training"
echo "=============================================="

# Set environment for training ports
export ORCH_QUESTION_PORT=$ORCHESTRATOR_PORT
export ORCH_DATA_EVAL_PORT=$ORCHESTRATOR_PORT
export ORCH_SYNTHESIS_PORT=$ORCHESTRATOR_PORT
export ORCH_ANSWER_EVAL_PORT=$ORCHESTRATOR_PORT
export CYPHER_INFERENCE_PORT=$CYPHER_INFERENCE_PORT

# Training on GPU 2 only
export CUDA_VISIBLE_DEVICES=2

# Verify GPU 2 is free
echo "Verifying training GPU allocation..."
python3 -c "
import os
import torch
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f'cuda:0 = {props.name}')
    print(f'Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total')
    if free/1024**3 < 70:
        print('WARNING: GPU has less than 70GB free!')
        exit(1)
    print('GPU allocation verified ✓')
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU allocation verification failed!"
    exit 1
fi

echo ""
echo "Starting training script..."
echo "Time: $(date)"
echo ""

# Run training
python3 -m rl_implementation.ddp_training.train_stage1 \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --questions_per_epoch "$QUESTIONS_PER_EPOCH" \
    2>&1 | tee "$LOG_DIR/training_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "End Time: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"

