#!/bin/bash
#SBATCH --job-name=pankllm_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/train_both_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/train_both_%j.err

# =============================================================================
# Train Both Cypher Generator and Orchestrator (SLURM Job)
# =============================================================================
#
# Trains BOTH models in PARALLEL using 2 GPUs:
#   - GPU 0: Cypher Generator
#   - GPU 1: Orchestrator
#
# Uses pre-collected rollouts (offline RL).
# No vLLM servers needed - uses stored prompts from JSONL.
#
# Submit with:
#   sbatch submit_train_both.sbatch
#
# Or sequentially on 1 GPU (takes longer but uses less resources):
#   sbatch --export=SEQUENTIAL=1 --gpus=1 --mem=64G submit_train_both.sbatch
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM Training - Cypher + Orchestrator"
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
echo ""

# =============================================================================
# Directories
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DDP_DIR="$SCRIPT_DIR/.."
CONFIG_DIR="$DDP_DIR/config"

echo "Directories:"
echo "  Script: $SCRIPT_DIR"
echo "  Project: $PROJECT_DIR"
echo ""

# Create output directories
mkdir -p "$PROJECT_DIR/outputs/cypher_trained/logs"
mkdir -p "$PROJECT_DIR/outputs/orchestrator_trained/logs"
mkdir -p "$PROJECT_DIR/outputs/stage1_ddp/logs"

# =============================================================================
# Cache Directories (Scratch for faster I/O)
# =============================================================================

SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

# Memory optimization for 14B models
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# =============================================================================
# PYTHONPATH Setup
# =============================================================================

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "RLLM Dir: $RLLM_DIR"
echo "PanKLLM Dir: $PANKLLM_DIR"
echo ""

# =============================================================================
# Configuration
# =============================================================================

ROLLOUTS_PATH="${ROLLOUTS:-$PROJECT_DIR/outputs/stage1_ddp/rollouts_collect_base_model.jsonl}"

CYPHER_CONFIG="${CYPHER_CONFIG:-$CONFIG_DIR/train_cypher_config.yaml}"
CYPHER_OUTPUT="${CYPHER_OUTPUT:-$PROJECT_DIR/outputs/cypher_trained}"

ORCH_CONFIG="${ORCH_CONFIG:-$CONFIG_DIR/train_orchestrator_config.yaml}"
ORCH_OUTPUT="${ORCH_OUTPUT:-$PROJECT_DIR/outputs/orchestrator_trained}"

echo "Configuration:"
echo "  Rollouts: $ROLLOUTS_PATH"
echo "  Cypher Config: $CYPHER_CONFIG"
echo "  Cypher Output: $CYPHER_OUTPUT"
echo "  Orchestrator Config: $ORCH_CONFIG"
echo "  Orchestrator Output: $ORCH_OUTPUT"
echo ""

# Verify rollouts exist
if [ -f "$ROLLOUTS_PATH" ]; then
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_PATH")
    echo "✓ Found rollouts: $ROLLOUT_COUNT entries"
else
    echo "❌ ERROR: Rollouts not found at $ROLLOUTS_PATH"
    echo "Run collect_rollouts.py first!"
    exit 1
fi
echo ""

# =============================================================================
# GPU Verification
# =============================================================================

echo "Verifying GPU allocation..."
python3 -c "
import os
import torch
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  cuda:{i} = {props.name}, {free/1024**3:.1f}/{total/1024**3:.1f} GB free')
print('✓ GPUs available')
"
echo ""

# =============================================================================
# Training Mode Selection
# =============================================================================

SEQUENTIAL="${SEQUENTIAL:-0}"

if [ "$SEQUENTIAL" == "1" ] || [ "$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)" -lt 2 ]; then
    echo "Mode: SEQUENTIAL (one GPU)"
    echo "  Training Cypher Generator first, then Orchestrator"
    
    # =============================================================================
    # Sequential Training
    # =============================================================================
    
    echo ""
    echo "=============================================="
    echo "Step 1/2: Training Cypher Generator"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES=0 python3 train_cypher_from_rollouts.py \
        --config "$CYPHER_CONFIG" \
        --rollouts "$ROLLOUTS_PATH" \
        --checkpoint-dir "$CYPHER_OUTPUT" \
        --log-dir "$CYPHER_OUTPUT/logs" \
        2>&1 | tee "$CYPHER_OUTPUT/logs/train_${SLURM_JOB_ID}.log"
    
    echo ""
    echo "=============================================="
    echo "Step 2/2: Training Orchestrator"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    CUDA_VISIBLE_DEVICES=0 python3 train_orchestrator_from_rollouts.py \
        --config "$ORCH_CONFIG" \
        --rollouts "$ROLLOUTS_PATH" \
        --checkpoint-dir "$ORCH_OUTPUT" \
        --log-dir "$ORCH_OUTPUT/logs" \
        2>&1 | tee "$ORCH_OUTPUT/logs/train_${SLURM_JOB_ID}.log"

else
    echo "Mode: PARALLEL (two GPUs)"
    echo "  GPU 0: Cypher Generator"
    echo "  GPU 1: Orchestrator"
    
    # =============================================================================
    # Parallel Training
    # =============================================================================
    
    echo ""
    echo "=============================================="
    echo "Starting Parallel Training"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    cd "$SCRIPT_DIR"
    
    # Start Cypher Generator training on GPU 0
    echo "Starting Cypher Generator on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python3 train_cypher_from_rollouts.py \
        --config "$CYPHER_CONFIG" \
        --rollouts "$ROLLOUTS_PATH" \
        --checkpoint-dir "$CYPHER_OUTPUT" \
        --log-dir "$CYPHER_OUTPUT/logs" \
        2>&1 | tee "$CYPHER_OUTPUT/logs/train_${SLURM_JOB_ID}.log" &
    CYPHER_PID=$!
    
    # Start Orchestrator training on GPU 1
    echo "Starting Orchestrator on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python3 train_orchestrator_from_rollouts.py \
        --config "$ORCH_CONFIG" \
        --rollouts "$ROLLOUTS_PATH" \
        --checkpoint-dir "$ORCH_OUTPUT" \
        --log-dir "$ORCH_OUTPUT/logs" \
        2>&1 | tee "$ORCH_OUTPUT/logs/train_${SLURM_JOB_ID}.log" &
    ORCH_PID=$!
    
    echo ""
    echo "Training processes started:"
    echo "  Cypher Generator PID: $CYPHER_PID"
    echo "  Orchestrator PID: $ORCH_PID"
    echo ""
    echo "Waiting for both to complete..."
    
    # Wait for both processes
    CYPHER_STATUS=0
    ORCH_STATUS=0
    
    wait $CYPHER_PID || CYPHER_STATUS=$?
    wait $ORCH_PID || ORCH_STATUS=$?
    
    echo ""
    echo "Training Results:"
    if [ $CYPHER_STATUS -eq 0 ]; then
        echo "  ✓ Cypher Generator: SUCCESS"
    else
        echo "  ✗ Cypher Generator: FAILED (exit code: $CYPHER_STATUS)"
    fi
    
    if [ $ORCH_STATUS -eq 0 ]; then
        echo "  ✓ Orchestrator: SUCCESS"
    else
        echo "  ✗ Orchestrator: FAILED (exit code: $ORCH_STATUS)"
    fi
fi

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo ""
echo "Outputs:"
echo "  Cypher Generator: $CYPHER_OUTPUT/best_model"
echo "  Orchestrator: $ORCH_OUTPUT/best_model"
echo ""
echo "Logs:"
echo "  Cypher: $CYPHER_OUTPUT/logs/"
echo "  Orchestrator: $ORCH_OUTPUT/logs/"

