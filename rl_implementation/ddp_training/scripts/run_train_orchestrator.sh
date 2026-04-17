#!/bin/bash
# =============================================================================
# Run Orchestrator Training from Rollouts
#
# Trains the Orchestrator model using pre-collected rollouts.
# Uses GRPO (Group Relative Policy Optimization) for advantage estimation.
#
# Orchestrator has TWO trainable roles:
#   1. Question Generation (Role 1) - trained with orch_qgen_reward
#   2. Answer Synthesis (Role 3) - trained with orch_synth_reward
#
# Prerequisites:
#   - Rollouts collected via collect_rollouts.py
#   - Rollouts stored in JSONL format with orchestrator prompts
#
# Usage:
#   ./run_train_orchestrator.sh [config_file]
#   ./run_train_orchestrator.sh --rollouts /path/to/rollouts.jsonl --num-epochs 10
#
# Examples:
#   ./run_train_orchestrator.sh                                         # Use default config
#   ./run_train_orchestrator.sh ../config/train_orchestrator_config.yaml # Custom config
#   ./run_train_orchestrator.sh --train-qgen-only                       # Train only question generation
#   ./run_train_orchestrator.sh --train-synth-only                      # Train only answer synthesis
# =============================================================================

set -e

echo "=============================================="
echo "Orchestrator Training from Rollouts"
echo "=============================================="
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DDP_DIR="$SCRIPT_DIR/.."
CONFIG_DIR="$DDP_DIR/config"

# Conda environment
CONDA_ENV="${CONDA_ENV:-vllm}"

echo "Directories:"
echo "  Script: $SCRIPT_DIR"
echo "  Project: $PROJECT_DIR"
echo "  Config: $CONFIG_DIR"
echo ""

# Activate conda
if [ -f "/home/rickyhan/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "/home/rickyhan/.conda/envs/$CONDA_ENV/bin/activate"
    echo "✓ Activated conda environment: $CONDA_ENV"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || echo "Warning: Could not activate $CONDA_ENV"
fi
echo ""

# =============================================================================
# Default Configuration
# =============================================================================

# Default config file
DEFAULT_CONFIG="$CONFIG_DIR/train_orchestrator_config.yaml"

# Default paths
DEFAULT_ROLLOUTS="$PROJECT_DIR/outputs/stage1_ddp/rollouts_collect_base_model.jsonl"
DEFAULT_ORCHESTRATOR_MODEL="$PROJECT_DIR/models/qwen2.5-14b"
DEFAULT_OUTPUT_DIR="$PROJECT_DIR/outputs/orchestrator_trained"

# Training defaults
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# =============================================================================
# Parse Arguments
# =============================================================================

CONFIG_FILE=""
EXTRA_ARGS=""

# Check if first argument is a config file
if [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
    CONFIG_FILE="$1"
    shift
fi

# Collect remaining arguments
EXTRA_ARGS="$@"

# Use default config if not specified
if [ -z "$CONFIG_FILE" ]; then
    if [ -f "$DEFAULT_CONFIG" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
        echo "Using default config: $CONFIG_FILE"
    else
        echo "Warning: No config file found, using command line args only"
    fi
else
    echo "Using config file: $CONFIG_FILE"
fi
echo ""

# =============================================================================
# GPU Setup
# =============================================================================

# Use specified GPU or default to GPU 0
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null | head -5 || echo "  (nvidia-smi not available)"
echo ""

# =============================================================================
# Verify Rollouts Exist
# =============================================================================

# Check if rollouts file is specified in args
ROLLOUTS_PATH="$DEFAULT_ROLLOUTS"
for arg in $EXTRA_ARGS; do
    if [[ "$prev_arg" == "--rollouts" ]]; then
        ROLLOUTS_PATH="$arg"
    fi
    prev_arg="$arg"
done

echo "Checking rollouts..."
if [ -f "$ROLLOUTS_PATH" ]; then
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_PATH")
    echo "  ✓ Found rollouts: $ROLLOUTS_PATH ($ROLLOUT_COUNT entries)"
    
    # Check if rollouts have orchestrator prompts
    if grep -q "orch_qgen_prompt" "$ROLLOUTS_PATH" 2>/dev/null; then
        echo "  ✓ Rollouts contain stored orchestrator prompts"
    else
        echo "  ⚠ Warning: Rollouts may not have stored prompts (will rebuild)"
    fi
else
    echo "  ⚠ Warning: Rollouts not found at $ROLLOUTS_PATH"
    echo "    Run collect_rollouts.py first, or specify --rollouts /path/to/rollouts.jsonl"
fi
echo ""

# =============================================================================
# Training Roles Info
# =============================================================================

echo "Orchestrator Training Roles:"
echo "  Role 1 (Question Generation):"
echo "    - Learns to generate diverse, answerable questions"
echo "    - Reward: orch_qgen_reward"
echo ""
echo "  Role 3 (Answer Synthesis):"
echo "    - Learns to synthesize accurate answers from retrieved data"
echo "    - Reward: orch_synth_reward"
echo ""
echo "  (Roles 2 & 4 are inference-only, not trained)"
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training..."
echo "=============================================="
echo ""

cd "$SCRIPT_DIR"

# Build command
CMD="python3 train_orchestrator_from_rollouts.py"

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Command: $CMD"
echo ""

# Run training
$CMD

echo ""
echo "=============================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=============================================="

