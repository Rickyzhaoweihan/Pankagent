#!/bin/bash
# =============================================================================
# FSDP Training Script (2 GPUs)
# 
# Uses PyTorch FSDP to shard the Cypher Generator model across 2 GPUs.
# This enables training with larger LoRA rank, longer sequences, and entropy.
#
# Prerequisites:
#   - vLLM servers running on separate GPUs (start_vllm_servers.sh)
#   - 2 GPUs available for training (CUDA_VISIBLE_DEVICES should have 2 GPUs)
#
# Usage:
#   ./run_fsdp_training.sh [num_epochs] [questions_per_epoch]
#
# Example:
#   # With vLLM on GPUs 0-1, training on GPUs 2-3
#   CUDA_VISIBLE_DEVICES=2,3 ./run_fsdp_training.sh 5 8
# =============================================================================

set -e

echo "=============================================="
echo "FSDP Training (2 GPUs)"
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
    echo "Start vLLM servers first: ./start_vllm_servers.sh"
    exit 1
fi
echo "  ✓ Orchestrator ready (port $ORCHESTRATOR_PORT)"

if ! curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1; then
    echo "❌ ERROR: Cypher Generator not running on port $CYPHER_PORT"
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

# Memory optimization
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
# GPU Verification
# =============================================================================

echo ""
echo "Verifying GPU allocation..."
python3 -c "
import os
import torch
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.device_count() < 2:
    print('ERROR: Need at least 2 GPUs for FSDP training!')
    exit(1)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  cuda:{i} = {props.name}, {free/1024**3:.1f}/{total/1024**3:.1f} GB free')
print('✓ GPU allocation verified')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed!"
    exit 1
fi

# =============================================================================
# Training Configuration
# =============================================================================

NUM_EPOCHS="${1:-5}"
QUESTIONS_PER_EPOCH="${2:-8}"

CONFIG_PATH="$PANKLLM_DIR/rl_implementation/ddp_training/config/stage1_config.yaml"
CHECKPOINT_DIR="$PANKLLM_DIR/outputs/stage1_fsdp/checkpoints"
LOG_DIR="$PANKLLM_DIR/outputs/stage1_fsdp/logs"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo ""
echo "Training Configuration:"
echo "  Config: $CONFIG_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Questions/Epoch: $QUESTIONS_PER_EPOCH"
echo "  GPUs: 2 (FSDP)"
echo ""

# =============================================================================
# Port Environment Variables
# =============================================================================

export ORCH_QUESTION_PORT=$ORCHESTRATOR_PORT
export ORCH_DATA_EVAL_PORT=$ORCHESTRATOR_PORT
export ORCH_SYNTHESIS_PORT=$ORCHESTRATOR_PORT
export ORCH_ANSWER_EVAL_PORT=$ORCHESTRATOR_PORT
export CYPHER_INFERENCE_PORT=$CYPHER_PORT

# =============================================================================
# NCCL Configuration for FSDP
# =============================================================================

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30 minutes timeout
export TORCH_NCCL_BLOCKING_WAIT=1

# =============================================================================
# Run FSDP Training with torchrun
# =============================================================================

echo "=============================================="
echo "Starting FSDP Training (2 GPUs)"
echo "=============================================="
echo "Time: $(date)"
echo ""

cd "$PANKLLM_DIR"

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    -m rl_implementation.ddp_training.train_stage1_fsdp \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --questions_per_epoch "$QUESTIONS_PER_EPOCH" \
    2>&1 | tee "$LOG_DIR/fsdp_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=============================================="
echo "FSDP Training Complete"
echo "=============================================="
echo "End Time: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"

