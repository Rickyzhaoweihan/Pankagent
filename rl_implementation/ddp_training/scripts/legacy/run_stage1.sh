#!/bin/bash
# =============================================================================
# Stage 1 Training Launch Script
# Cypher Generator Only (Single GPU + LoRA)
# 
# 3-GPU Setup:
#    - GPU 0: Orchestrator vLLM server
#    - GPU 1: Cypher Generator vLLM server
#    - GPU 2: Cypher Generator training (LoRA)
#
# =============================================================================

set -e

# =============================================================================
# Job Info
# =============================================================================

echo "=============================================="
echo "Stage 1 Training: Cypher Generator (DDP + LoRA)"
echo "=============================================="
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Node: $(hostname)"

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

# Redirect ~/.cache to scratch (fixes vLLM P2P cache write issues on NFS)
export XDG_CACHE_HOME="$SCRATCH/cache"

mkdir -p /tmp/r /tmp/t
export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/t

# System limits (only works in SLURM jobs, ignore if interactive)
ulimit -n 1048576 2>/dev/null || echo "Note: Could not increase file descriptor limit (requires SLURM job)"

# =============================================================================
# vLLM and PyTorch Configuration
# =============================================================================

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

# Single GPU training - no NCCL needed
# (Keeping these for potential future DDP use)
# export NCCL_TIMEOUT=10800
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=WARN

# =============================================================================
# Resolve RLLM Path
# =============================================================================

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
echo "RLLM Directory: $RLLM_DIR"

PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"
RL_IMPL_DIR="$PANKLLM_DIR/rl_implementation"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"
echo "PanKLLM Dir: $PANKLLM_DIR"

# Show GPU status
nvidia-smi

# =============================================================================
# Model Paths
# =============================================================================

# Orchestrator: Qwen2.5-14B (general-purpose LLM)
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-$PANKLLM_DIR/models/qwen2.5-14b}"
# Cypher Generator: Qwen2.5-Coder-14B-SFT (code-specialized, fine-tuned)
CYPHER_MODEL="${CYPHER_MODEL:-$PANKLLM_DIR/models/qwen2.5-coder-14b}"

# =============================================================================
# vLLM Server Configuration
# =============================================================================

# Ports for vLLM servers
ORCHESTRATOR_PORT=8001
CYPHER_INFERENCE_PORT=8002

# Legacy port variables (all point to same orchestrator server)
ORCH_QUESTION_PORT=$ORCHESTRATOR_PORT
ORCH_DATA_EVAL_PORT=$ORCHESTRATOR_PORT
ORCH_SYNTHESIS_PORT=$ORCHESTRATOR_PORT
ORCH_ANSWER_EVAL_PORT=$ORCHESTRATOR_PORT

# vLLM server settings
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

# Tensor parallelism settings (3-GPU setup - no TP needed)
ORCHESTRATOR_TP=1  # GPU 0
CYPHER_TP=1        # GPU 1

# =============================================================================
# Training Configuration
# =============================================================================

CONFIG_PATH="${1:-$RL_IMPL_DIR/ddp_training/config/stage1_config.yaml}"
CHECKPOINT_DIR="${2:-$PANKLLM_DIR/outputs/stage1_ddp/checkpoints}"
LOG_DIR="${3:-$PANKLLM_DIR/outputs/stage1_ddp/logs}"
NUM_EPOCHS="${4:-5}"
QUESTIONS_PER_EPOCH="${5:-8}"  # Small for quick iteration

echo ""
echo "Configuration:"
echo "  Config: $CONFIG_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Questions/Epoch: $QUESTIONS_PER_EPOCH"
echo ""
echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""

# Create output directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup_servers() {
    echo ""
    echo "Stopping vLLM servers..."
    # Kill any vLLM servers started by this script
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping server PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a bit for graceful shutdown
    sleep 2
    # Force kill if still running
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Force stopping server PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "All vLLM servers stopped."
}

# Trap to ensure cleanup on exit
trap cleanup_servers EXIT INT TERM

# =============================================================================
# Start vLLM Servers
# =============================================================================

# Change to PanKLLM directory
cd "$PANKLLM_DIR"

# Clean up any leftover processes
echo "Cleaning up any leftover processes..."
pkill -9 -f "train_stage1" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 3

# Show GPU status
echo ""
echo "GPU Status before starting servers:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Array to store server PIDs
declare -a VLLM_PIDS

echo "=============================================="
echo "Starting vLLM OpenAI-compatible servers..."
echo "=============================================="

# Function to start a vLLM server with tensor parallelism
start_vllm_server_tp() {
    local gpu_ids=$1      # e.g., "0,1,2,3"
    local port=$2
    local model_path=$3
    local name=$4
    local tp_size=$5      # tensor parallelism size
    local log_file="$LOG_DIR/vllm_${name}.log"
    
    echo "Starting $name server on GPUs [$gpu_ids], port $port, TP=$tp_size..."
    echo "  Model: $model_path"
    echo "  Log: $log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids python3 -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port $port \
        --tensor-parallel-size $tp_size \
        --max-model-len $VLLM_MAX_MODEL_LEN \
        --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
        --max-num-seqs $VLLM_MAX_NUM_SEQS \
        --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
        --dtype bfloat16 \
        --trust-remote-code \
        --disable-log-stats \
        > "$log_file" 2>&1 &
    
    local pid=$!
    VLLM_PIDS+=($pid)
    echo "  Started with PID $pid"
}

# Start Orchestrator server (GPU 0)
echo ""
echo "Starting Orchestrator on GPU 0..."
start_vllm_server_tp "0" $ORCHESTRATOR_PORT "$ORCHESTRATOR_MODEL" "orchestrator" $ORCHESTRATOR_TP

# Start Cypher Generator server (GPU 1)
echo ""
echo "Starting Cypher Generator on GPU 1..."
start_vllm_server_tp "1" $CYPHER_INFERENCE_PORT "$CYPHER_MODEL" "cypher_inference" $CYPHER_TP

echo ""
echo "Waiting for all servers to be ready..."
echo "This may take a few minutes as models are loaded..."

# Function to wait for a server to be ready
wait_for_server() {
    local port=$1
    local name=$2
    local max_attempts=120  # 10 minutes max wait
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $name server ready on port $port"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    
    echo "  ✗ $name server failed to start on port $port"
    return 1
}

# Wait for all servers (now just 2 servers)
echo ""
wait_for_server $ORCHESTRATOR_PORT "Orchestrator" || exit 1
wait_for_server $CYPHER_INFERENCE_PORT "CypherGen" || exit 1

echo ""
echo "=============================================="
echo "All vLLM servers are ready!"
echo "=============================================="
echo ""

# Show GPU status after loading models
echo "GPU Status after loading models:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training..."
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Set environment variables for training script to know the ports
# All orchestrator modes use the same server (same model, different prompts)
export ORCH_QUESTION_PORT=$ORCHESTRATOR_PORT
export ORCH_DATA_EVAL_PORT=$ORCHESTRATOR_PORT
export ORCH_SYNTHESIS_PORT=$ORCHESTRATOR_PORT
export ORCH_ANSWER_EVAL_PORT=$ORCHESTRATOR_PORT
export CYPHER_INFERENCE_PORT=$CYPHER_INFERENCE_PORT

# For training, use GPU 2 (GPUs 0-1 are used by vLLM servers)
# Single GPU training - simpler and avoids NCCL timeout issues
export CUDA_VISIBLE_DEVICES=2

# Verify GPU 2 is available and free
echo "Verifying GPU 2 allocation..."
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
        print('WARNING: GPU has less than 70GB free, may be shared with vLLM!')
        exit(1)
    print('GPU allocation verified ✓')
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU allocation verification failed!"
    exit 1
fi

echo ""

# Run training script with single GPU
# No torchrun needed - just regular python
python3 -m rl_implementation.ddp_training.train_stage1 \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --questions_per_epoch "$QUESTIONS_PER_EPOCH" \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training complete!"
echo "End Time: $(date)"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs saved to: $LOG_DIR"

# Servers will be stopped by the cleanup trap
