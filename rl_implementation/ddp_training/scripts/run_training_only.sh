#!/bin/bash
# =============================================================================
# Run Training Only (3-GPU Setup)
# 
# Prerequisites: vLLM servers must already be running
#   - Start with: ./start_vllm_servers.sh
#
# GPU Allocation:
#   - GPU 0: Orchestrator (vLLM) - must be running
#   - GPU 1: Cypher Generator (vLLM) - must be running
#   - GPU 2: Training (this script)
#
# Usage:
#   ./run_training_only.sh [num_epochs] [questions_per_epoch]
#
# Example:
#   ./run_training_only.sh 5 8
# =============================================================================

set -e

echo "=============================================="
echo "Stage 1 Training (vLLM servers must be running)"
echo "=============================================="
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Verify vLLM Servers
# =============================================================================

ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8001}"
CYPHER_PORT="${CYPHER_INFERENCE_PORT:-8002}"

echo "Checking vLLM servers..."

if ! curl -s "http://localhost:$ORCHESTRATOR_PORT/health" > /dev/null 2>&1; then
    echo "❌ ERROR: Orchestrator not running on port $ORCHESTRATOR_PORT"
    echo ""
    echo "Start vLLM servers first:"
    echo "  ./start_vllm_servers.sh"
    exit 1
fi
echo "  ✓ Orchestrator ready (port $ORCHESTRATOR_PORT)"

if ! curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1; then
    echo "❌ ERROR: Cypher Generator not running on port $CYPHER_PORT"
    echo ""
    echo "Start vLLM servers first:"
    echo "  ./start_vllm_servers.sh"
    exit 1
fi
echo "  ✓ Cypher Generator ready (port $CYPHER_PORT)"

echo ""

# =============================================================================
# Conda Environment
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

# Memory optimization for large models (avoid expandable_segments - causes PyTorch bug)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# =============================================================================
# PYTHONPATH Setup
# =============================================================================

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "RLLM Dir: $RLLM_DIR"
echo "PanKLLM Dir: $PANKLLM_DIR"

# =============================================================================
# Training Configuration
# =============================================================================

NUM_EPOCHS="${1:-5}"
QUESTIONS_PER_EPOCH="${2:-8}"

CONFIG_PATH="$PANKLLM_DIR/rl_implementation/ddp_training/config/stage1_config.yaml"
CHECKPOINT_DIR="$PANKLLM_DIR/outputs/stage1_ddp/checkpoints"
LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo ""
echo "Training Configuration:"
echo "  Config: $CONFIG_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Questions/Epoch: $QUESTIONS_PER_EPOCH"
echo ""

# =============================================================================
# GPU Setup - Use whatever GPU is allocated to this process
# =============================================================================

# If running with separate srun (1 GPU allocated), don't override CUDA_VISIBLE_DEVICES
# SLURM already sets it up correctly
# Only set it if not already set (e.g., running directly without srun)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=2
    echo "Setting CUDA_VISIBLE_DEVICES=2 (not in SLURM job)"
else
    echo "Using SLURM-allocated GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Verify GPU is available and free
echo "Verifying GPU allocation..."
python3 -c "
import os
import torch
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f'cuda:0 = {props.name}')
    print(f'Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total')
    if free/1024**3 < 60:
        print('WARNING: GPU has less than 60GB free!')
    print('✓ GPU available')
else:
    print('ERROR: No GPU available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed!"
    exit 1
fi

# =============================================================================
# Port Environment Variables
# =============================================================================

export ORCH_QUESTION_PORT=$ORCHESTRATOR_PORT
export ORCH_DATA_EVAL_PORT=$ORCHESTRATOR_PORT
export ORCH_SYNTHESIS_PORT=$ORCHESTRATOR_PORT
export ORCH_ANSWER_EVAL_PORT=$ORCHESTRATOR_PORT
export CYPHER_INFERENCE_PORT=$CYPHER_PORT

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Time: $(date)"
echo ""

cd "$PANKLLM_DIR"

python3 -m rl_implementation.ddp_training.train_stage1 \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --questions_per_epoch "$QUESTIONS_PER_EPOCH" \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "End Time: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "Note: vLLM servers are still running."
echo "To stop them: pkill -f 'vllm.entrypoints.openai.api_server'"

