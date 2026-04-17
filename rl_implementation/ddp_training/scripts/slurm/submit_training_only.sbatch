#!/bin/bash
#SBATCH --job-name=pankllm_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/train_only_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/train_only_%j.err

# =============================================================================
# PanKLLM RL Training-Only Job (Single GPU)
# =============================================================================
#
# PREREQUISITES:
#   vLLM servers must be running BEFORE this job starts!
#   Start them with: ./start_vllm_servers.sh
#
# This job:
#   - Connects to existing vLLM servers on ports 8001, 8002
#   - Runs PPO training on 1 GPU
#   - Saves best model by reward
#
# Submit with:
#   sbatch submit_training_only.sbatch
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM Training-Only Job"
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
# Verify vLLM Servers
# =============================================================================

ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8001}"
CYPHER_PORT="${CYPHER_INFERENCE_PORT:-8002}"

echo ""
echo "Checking vLLM servers..."

MAX_WAIT=60  # Wait up to 60 seconds for servers
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$ORCHESTRATOR_PORT/health" > /dev/null 2>&1 && \
       curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ Orchestrator ready (port $ORCHESTRATOR_PORT)"
        echo "  ✓ Cypher Generator ready (port $CYPHER_PORT)"
        break
    fi
    
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -lt $MAX_WAIT ]; then
        echo "  Waiting for vLLM servers... ($WAIT_COUNT s)"
        sleep 5
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM servers not responding after $MAX_WAIT seconds"
    echo ""
    echo "Make sure vLLM servers are running:"
    echo "  ./start_vllm_servers.sh"
    exit 1
fi

# =============================================================================
# Training Configuration
# =============================================================================

NUM_EPOCHS="${NUM_EPOCHS:-10}"
QUESTIONS_PER_EPOCH="${QUESTIONS_PER_EPOCH:-64}"

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
    2>&1 | tee "$LOG_DIR/training_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "End Time: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Best Model: $CHECKPOINT_DIR/best_model"
echo "Logs: $LOG_DIR"

