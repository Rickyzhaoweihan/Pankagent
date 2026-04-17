#!/bin/bash
#SBATCH --job-name=orch_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/orchestrator_trained/logs/train_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/orchestrator_trained/logs/train_%j.err

# =============================================================================
# Orchestrator Training from Rollouts (SLURM Job)
# =============================================================================
#
# Trains Orchestrator using pre-collected rollouts (offline RL).
# Trains TWO roles:
#   1. Question Generation (orch_qgen_reward)
#   2. Answer Synthesis (orch_synth_reward)
#
# No vLLM servers needed - uses stored prompts from JSONL.
#
# Submit with:
#   sbatch submit_train_orchestrator.sbatch
#
# Or with custom config:
#   sbatch --export=CONFIG=/path/to/config.yaml submit_train_orchestrator.sbatch
#
# =============================================================================

set -e

echo "=============================================="
echo "Orchestrator Training (SLURM)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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

# Memory optimization for 14B model
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

# Use provided config or default
CONFIG_PATH="${CONFIG:-$CONFIG_DIR/train_orchestrator_config.yaml}"
ROLLOUTS_PATH="${ROLLOUTS:-$PROJECT_DIR/outputs/stage1_ddp/rollouts_collect_base_model.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/orchestrator_trained}"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

echo "Configuration:"
echo "  Config: $CONFIG_PATH"
echo "  Rollouts: $ROLLOUTS_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Verify rollouts exist
if [ -f "$ROLLOUTS_PATH" ]; then
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_PATH")
    echo "✓ Found rollouts: $ROLLOUT_COUNT entries"
    
    # Check for orchestrator prompts
    if grep -q "orch_qgen_prompt" "$ROLLOUTS_PATH" 2>/dev/null; then
        echo "✓ Rollouts contain stored orchestrator prompts"
    else
        echo "⚠ Warning: Rollouts may not have stored prompts"
    fi
else
    echo "❌ ERROR: Rollouts not found at $ROLLOUTS_PATH"
    echo "Run collect_rollouts.py first!"
    exit 1
fi
echo ""

# =============================================================================
# Training Roles Info
# =============================================================================

echo "Orchestrator Training Roles:"
echo "  Role 1 (Question Generation) - learns to generate diverse questions"
echo "  Role 3 (Answer Synthesis) - learns to synthesize accurate answers"
echo "  (Roles 2 & 4 are inference-only)"
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
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f'cuda:0 = {props.name}')
    print(f'Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total')
    print('✓ GPU available')
else:
    print('ERROR: No GPU available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed!"
    exit 1
fi
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "=============================================="
echo "Starting Orchestrator Training"
echo "=============================================="
echo "Time: $(date)"
echo ""

cd "$SCRIPT_DIR"

# Run training
python3 train_orchestrator_from_rollouts.py \
    --config "$CONFIG_PATH" \
    --rollouts "$ROLLOUTS_PATH" \
    --checkpoint-dir "$OUTPUT_DIR" \
    --log-dir "$OUTPUT_DIR/logs" \
    2>&1 | tee "$OUTPUT_DIR/logs/train_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo "Checkpoints: $OUTPUT_DIR"
echo "Best Model: $OUTPUT_DIR/best_model"
echo "Logs: $OUTPUT_DIR/logs"

