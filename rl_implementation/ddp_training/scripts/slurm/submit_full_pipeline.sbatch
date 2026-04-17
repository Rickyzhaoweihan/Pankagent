#!/bin/bash
#SBATCH --job-name=pankllm_pipeline
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_%j.err

# =============================================================================
# Full Dynamic Training Pipeline (4 GPU SLURM Job)
# =============================================================================
#
# This is the MAIN ENTRY POINT for training. It handles everything:
#   1. Starts vLLM servers on GPU 0-1 (Orchestrator + Cypher Generator)
#   2. Runs the dynamic training pipeline with adaptive thresholds
#   3. Collects rollouts, analyzes metrics, trains adaptively
#   4. Updates vLLM servers after training
#   5. Stops on convergence or max iterations
#   6. Cleans up servers on exit
#
# GPU Allocation (4 GPUs Required):
#   GPU 0: vLLM Orchestrator server (port 8001) - inference
#   GPU 1: vLLM Cypher Generator server (port 8002) - inference  
#   GPU 2: Cypher Generator training
#   GPU 3: Orchestrator training
#
# Submit with:
#   sbatch submit_full_pipeline.sbatch
#
# Options (via --export):
#   # Basic usage (uses defaults: coverage-based with 128 target usable rollouts)
#   sbatch submit_full_pipeline.sbatch
#   
#   # Custom coverage-based collection (target 256 usable rollouts)
#   sbatch --export=TARGET_FILTERED=256,MIN_USABLE_REWARD=0.15 submit_full_pipeline.sbatch
#   
#   # Fixed count mode (old behavior, disable coverage-based)
#   sbatch --export=TARGET_FILTERED=0,QUESTIONS=64 submit_full_pipeline.sbatch
#   
#   # Accelerated mode (overlapping, but NO coverage-based/adaptive sampling)
#   # Only use for fast iteration with fixed rollout count
#   sbatch --export=ACCELERATED=1,TARGET_FILTERED=0,QUESTIONS=64 submit_full_pipeline.sbatch
#   
#   # Use existing rollouts, skip first collection
#   sbatch --export=SKIP_FIRST_COLLECT=1 submit_full_pipeline.sbatch
#   
#   # Fresh start (clear history)
#   sbatch --export=CLEAR_HISTORY=1 submit_full_pipeline.sbatch
#   
#   # Full custom configuration
#   sbatch --export=ITERATIONS=20,TARGET_FILTERED=128,DIFFICULTY=medium,EPOCHS=5,ACCELERATED=1,CLEAR_HISTORY=1 submit_full_pipeline.sbatch
#   
#   
#   # Disable adaptive sampling (use uniform entity selection)
#   sbatch --export=NO_ADAPTIVE_SAMPLING=1 submit_full_pipeline.sbatch
#
# Available Options:
#   ITERATIONS        - Max training iterations (default: 10)
#   QUESTIONS         - Questions per iteration (default: 64, used if TARGET_FILTERED=0)
#   DIFFICULTY        - easy/medium/hard (default: easy)
#   EPOCHS            - Training epochs per iteration (default: 3)
#   BATCH_SIZE        - Collection batch size (default: 8)
#   ACCELERATED       - 0=standard (default), 1=overlapping (no coverage/adaptive)
#   SKIP_FIRST_COLLECT - Skip first rollout collection (default: 0)
#   CLEAR_HISTORY     - Clear training history for fresh start (default: 0)
#   FORCE_DEGREE_EXTRACTION - Re-extract entity degrees even if exists (default: 0)
#
# Coverage-Based Rollout Collection (NEW):
#   TARGET_FILTERED   - Target usable rollouts per iteration (default: 128)
#   MIN_USABLE_REWARD - Min reward for rollout to count as usable (default: 0.1)
#   NO_ADAPTIVE_SAMPLING - Set to 1 to disable adaptive entity sampling (default: 0)
#
# Adaptive Entity Sampling:
#   The pipeline automatically extracts entity degrees from Neo4j to enable
#   adaptive sampling. This biases entity selection toward entities with more
#   connections (higher degree = more likely to produce answerable questions).
#   The sampler learns from answerability feedback using Thompson Sampling.
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM Full Training Pipeline (4 GPU)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Configuration (can be overridden via --export)
# =============================================================================

# Pipeline mode: 0 = standard (with all features), 1 = accelerated (overlapping, fixed count only)
# Default to standard pipeline which supports coverage-based stopping and adaptive sampling
ACCELERATED="${ACCELERATED:-0}"

# Training parameters
ITERATIONS="${ITERATIONS:-10}"
QUESTIONS="${QUESTIONS:-64}"
DIFFICULTY="${DIFFICULTY:-easy}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Coverage-based rollout collection (default: enabled)
# Collect until we have TARGET_FILTERED usable rollouts (reward >= MIN_USABLE_REWARD)
# Set TARGET_FILTERED=0 to use fixed QUESTIONS count instead
TARGET_FILTERED="${TARGET_FILTERED:-128}"
MIN_USABLE_REWARD="${MIN_USABLE_REWARD:-0.1}"

# Adaptive entity sampling (enabled by default)
NO_ADAPTIVE_SAMPLING="${NO_ADAPTIVE_SAMPLING:-0}"

# Skip first collection (use existing rollouts)
SKIP_FIRST_COLLECT="${SKIP_FIRST_COLLECT:-0}"

# Clear history for fresh start
CLEAR_HISTORY="${CLEAR_HISTORY:-0}"

# Force re-extraction of entity degrees
FORCE_DEGREE_EXTRACTION="${FORCE_DEGREE_EXTRACTION:-0}"

echo "Configuration:"
echo "  Accelerated Mode: $ACCELERATED"
echo "  Max Iterations: $ITERATIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Epochs/Iter: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Skip First Collect: $SKIP_FIRST_COLLECT"
echo "  Clear History: $CLEAR_HISTORY"
echo "  Force Degree Extraction: $FORCE_DEGREE_EXTRACTION"
echo ""
echo "Rollout Collection:"
if [ "$TARGET_FILTERED" -gt 0 ]; then
    echo "  Mode: COVERAGE-BASED"
    echo "  Target Usable Rollouts: $TARGET_FILTERED"
    echo "  Min Usable Reward: $MIN_USABLE_REWARD"
else
    echo "  Mode: FIXED COUNT"
    echo "  Questions/Iter: $QUESTIONS"
fi
echo "  Adaptive Sampling: $([ "$NO_ADAPTIVE_SAMPLING" = "1" ] && echo "DISABLED" || echo "ENABLED")"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

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

# Temporary directories
mkdir -p /tmp/r /tmp/t
export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/t

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# vLLM settings
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

ulimit -n 1048576 2>/dev/null || echo "Note: Could not increase file descriptor limit"

# =============================================================================
# PYTHONPATH Setup
# =============================================================================

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"
SCRIPT_DIR="$PANKLLM_DIR/rl_implementation/ddp_training/scripts"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "RLLM Dir: $RLLM_DIR"
echo "PanKLLM Dir: $PANKLLM_DIR"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# =============================================================================
# Create Job-Specific Output Directories
# =============================================================================
# Each pipeline run gets its own rollouts directory to avoid mixing data

RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"
ROLLOUTS_DIR="$PANKLLM_DIR/outputs/stage1_ddp/rollouts_run_${RUN_ID}"

mkdir -p "$PANKLLM_DIR/outputs/stage1_ddp/logs"
mkdir -p "$ROLLOUTS_DIR"
mkdir -p "$PANKLLM_DIR/outputs/cypher_trained"
mkdir -p "$PANKLLM_DIR/outputs/orchestrator_trained"

# Export for pipeline scripts to use
export ROLLOUTS_DIR
export RUN_ID

echo "Run ID: $RUN_ID"
echo "Rollouts Dir: $ROLLOUTS_DIR"

# =============================================================================
# Model Paths
# =============================================================================

ORCHESTRATOR_MODEL="$PANKLLM_DIR/models/qwen2.5-14b"
CYPHER_MODEL="$PANKLLM_DIR/models/qwen2.5-coder-14b"

echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""

# Verify models exist
if [ ! -d "$ORCHESTRATOR_MODEL" ]; then
    echo "ERROR: Orchestrator model not found: $ORCHESTRATOR_MODEL"
    exit 1
fi
if [ ! -d "$CYPHER_MODEL" ]; then
    echo "ERROR: Cypher model not found: $CYPHER_MODEL"
    exit 1
fi

# =============================================================================
# vLLM Server Settings
# =============================================================================

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-131072}"

ORCHESTRATOR_PORT=8001
CYPHER_PORT=8002

LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"

# =============================================================================
# Start vLLM Servers
# =============================================================================

echo "=============================================="
echo "Starting vLLM Servers"
echo "=============================================="
echo ""

# Start Orchestrator on GPU 0
echo "Starting Orchestrator vLLM on GPU 0 (port $ORCHESTRATOR_PORT)..."
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
    > "$LOG_DIR/vllm_orchestrator_${SLURM_JOB_ID}.log" 2>&1 &
ORCH_PID=$!
echo "  PID: $ORCH_PID"

# Start Cypher Generator on GPU 1
echo "Starting Cypher vLLM on GPU 1 (port $CYPHER_PORT)..."
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
echo "  PID: $CYPHER_PID"

# Wait for servers to be ready
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
    
    echo "  ✗ $name failed to start"
    return 1
}

wait_for_server $ORCHESTRATOR_PORT "Orchestrator" || exit 1
wait_for_server $CYPHER_PORT "Cypher Generator" || exit 1

echo ""
echo "✓ All vLLM servers ready!"
echo ""

# Export PIDs for pipeline scripts to use when restarting servers
export VLLM_ORCH_PID=$ORCH_PID
export VLLM_CYPHER_PID=$CYPHER_PID
export VLLM_ORCH_PORT=$ORCHESTRATOR_PORT
export VLLM_CYPHER_PORT=$CYPHER_PORT
export VLLM_ORCH_MODEL=$ORCHESTRATOR_MODEL
export VLLM_CYPHER_MODEL=$CYPHER_MODEL

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
# Extract Entity Degrees for Adaptive Sampling
# =============================================================================
# The adaptive sampler needs entity degrees to bias sampling toward entities
# that have more connections (more likely to produce answerable questions).
# This step extracts degrees from Neo4j if not already done.

ENTITY_DEGREES_PATH="$PANKLLM_DIR/outputs/entity_degrees.json"

echo "=============================================="
echo "Checking Entity Degrees for Adaptive Sampling"
echo "=============================================="

if [ ! -f "$ENTITY_DEGREES_PATH" ] || [ "${FORCE_DEGREE_EXTRACTION:-0}" = "1" ]; then
    echo "Extracting entity degrees from Neo4j..."
    echo "This provides prior knowledge for adaptive entity sampling."
    echo ""
    
    python3 "$PANKLLM_DIR/rl_implementation/utils/degree_extractor.py" \
        --output "$ENTITY_DEGREES_PATH" \
        --timeout 180
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Entity degrees extracted to $ENTITY_DEGREES_PATH"
        
        # Show summary
        python3 -c "
import json
with open('$ENTITY_DEGREES_PATH') as f:
    data = json.load(f)
degrees = data.get('degrees', {})
print(f'  Summary:')
for node_type, entities in degrees.items():
    print(f'    {node_type}: {len(entities)} entities')
"
    else
        echo "  ⚠ Warning: Degree extraction failed. Adaptive sampling will fall back to uniform."
    fi
else
    echo "Entity degrees already exist: $ENTITY_DEGREES_PATH"
    echo "  (Use FORCE_DEGREE_EXTRACTION=1 to re-extract)"
fi
echo ""

# Export path for training scripts
export ENTITY_DEGREES_PATH

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    echo ""
    echo "=============================================="
    echo "Cleaning up..."
    echo "=============================================="
    
    # Kill vLLM servers
    if [ -n "$ORCH_PID" ] && kill -0 $ORCH_PID 2>/dev/null; then
        echo "Stopping Orchestrator vLLM (PID: $ORCH_PID)..."
        kill $ORCH_PID 2>/dev/null || true
    fi
    
    if [ -n "$CYPHER_PID" ] && kill -0 $CYPHER_PID 2>/dev/null; then
        echo "Stopping Cypher vLLM (PID: $CYPHER_PID)..."
        kill $CYPHER_PID 2>/dev/null || true
    fi
    
    # Kill any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    
    echo "Cleanup complete"
}

trap cleanup EXIT

# =============================================================================
# GPU Status (Training GPUs only - GPU 2,3)
# =============================================================================

echo "Checking training GPUs (2,3)..."
python3 -c "
import torch
import os

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
# Run Training Pipeline
# =============================================================================

echo "=============================================="
echo "Starting Training Pipeline"
echo "=============================================="
echo "Time: $(date)"
echo ""

cd "$SCRIPT_DIR"

# Build command arguments
PIPELINE_ARGS="--iterations $ITERATIONS --difficulty $DIFFICULTY --epochs $EPOCHS --batch-size $BATCH_SIZE"

# Coverage-based stopping (default) vs fixed count
if [ "$TARGET_FILTERED" -gt 0 ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --target-filtered $TARGET_FILTERED --min-usable-reward $MIN_USABLE_REWARD"
else
    PIPELINE_ARGS="$PIPELINE_ARGS --questions $QUESTIONS"
fi

# Adaptive sampling (enabled by default)
if [ "$NO_ADAPTIVE_SAMPLING" = "1" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --no-adaptive-sampling"
fi

# Note: Pipeline always runs for ITERATIONS count (no auto-convergence)

if [ "$SKIP_FIRST_COLLECT" = "1" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-collect"
fi

# Choose pipeline script
if [ "$ACCELERATED" = "1" ]; then
    echo "Running ACCELERATED pipeline (overlapping collection + training)..."
    PIPELINE_SCRIPT="./run_accelerated_pipeline.sh"
else
    echo "Running STANDARD pipeline (sequential)..."
    PIPELINE_SCRIPT="./run_full_training_pipeline.sh"
fi

echo "Command: $PIPELINE_SCRIPT $PIPELINE_ARGS"
echo ""

# Run pipeline
$PIPELINE_SCRIPT $PIPELINE_ARGS 2>&1 | tee "$LOG_DIR/pipeline_${SLURM_JOB_ID}.log"

PIPELINE_EXIT=$?

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo "Exit Code: $PIPELINE_EXIT"
echo ""

echo "Outputs:"
echo "  Cypher Model (LoRA): $PANKLLM_DIR/outputs/cypher_trained/best_model"
echo "  Cypher Model (Merged for vLLM): $PANKLLM_DIR/outputs/cypher_trained/best_model_merged"
echo "  Orchestrator Model (LoRA): $PANKLLM_DIR/outputs/orchestrator_trained/best_model"
echo "  Orchestrator Model (Merged for vLLM): $PANKLLM_DIR/outputs/orchestrator_trained/best_model_merged"
echo ""

echo "Logs:"
echo "  Pipeline: $LOG_DIR/pipeline_${SLURM_JOB_ID}.log"
echo "  vLLM Orchestrator: $LOG_DIR/vllm_orchestrator_${SLURM_JOB_ID}.log"
echo "  vLLM Cypher: $LOG_DIR/vllm_cypher_${SLURM_JOB_ID}.log"
echo ""

HISTORY_FILE="$PANKLLM_DIR/outputs/stage1_ddp/training_history.json"
if [ -f "$HISTORY_FILE" ]; then
    echo "Training Summary:"
    python3 -c "
import json
from pathlib import Path
path = Path('$HISTORY_FILE')
history = []
if path.exists() and path.stat().st_size > 0:
    try:
        history = json.load(open('$HISTORY_FILE'))
    except:
        history = []
if history and isinstance(history, list):
    first = history[0]
    last = history[-1]
    
    # Starting metrics
    first_metrics = first.get('metrics', {})
    print(f\"  Starting Rewards:\")
    print(f\"    Cypher: {first_metrics.get('cypher_reward', 0):.3f}\")
    print(f\"    Orchestrator: {first_metrics.get('orch_avg_reward', 0):.3f}\")
    
    # Final metrics
    last_metrics = last.get('metrics', {})
    print(f\"  Final Rewards:\")
    print(f\"    Cypher: {last_metrics.get('cypher_reward', 0):.3f}\")
    print(f\"    Orchestrator: {last_metrics.get('orch_avg_reward', 0):.3f}\")
    
    # Improvement
    cypher_improvement = last_metrics.get('cypher_reward', 0) - first_metrics.get('cypher_reward', 0)
    orch_improvement = last_metrics.get('orch_avg_reward', 0) - first_metrics.get('orch_avg_reward', 0)
    print(f\"  Improvement:\")
    print(f\"    Cypher: {'+' if cypher_improvement >= 0 else ''}{cypher_improvement:.3f}\")
    print(f\"    Orchestrator: {'+' if orch_improvement >= 0 else ''}{orch_improvement:.3f}\")
    
    # Final thresholds (adaptive)
    thresholds = last.get('thresholds', {})
    if thresholds:
        print(f\"  Final Adaptive Thresholds:\")
        print(f\"    Cypher: {thresholds.get('cypher_effective', 0):.2f} (max: {thresholds.get('cypher_max', 0):.2f})\")
        print(f\"    Orchestrator: {thresholds.get('orchestrator_effective', 0):.2f} (max: {thresholds.get('orchestrator_max', 0):.2f})\")
    
    # Training stats
    print(f\"  Training Stats:\")
    print(f\"    Total Iterations: {len(history)}\")
    cypher_trains = sum(1 for h in history if h.get('trained', {}).get('cypher', False))
    orch_trains = sum(1 for h in history if h.get('trained', {}).get('orchestrator', False))
    print(f\"    Cypher Trained: {cypher_trains} times\")
    print(f\"    Orchestrator Trained: {orch_trains} times\")
else:
    print('  No training history available')
"
fi

echo ""
echo "=============================================="
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "✓ Pipeline completed successfully!"
else
    echo "✗ Pipeline failed with exit code: $PIPELINE_EXIT"
fi
echo "=============================================="

exit $PIPELINE_EXIT

