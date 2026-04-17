#!/bin/bash
#SBATCH --job-name=orch_focus
#SBATCH --partition=spgpu
#SBATCH --account=drjieliu99
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/orch_focus_%j.log
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/orch_focus_%j.err

# =============================================================================
# ORCHESTRATOR-FOCUSED TRAINING PIPELINE (SLURM)
# =============================================================================
#
# This script:
# 1. Starts vLLM inference servers (GPU 0: Orchestrator, GPU 1: Cypher)
# 2. Runs the orchestrator-focused training pipeline (GPU 3)
# 3. Cleans up on completion
#
# Key differences from full pipeline:
# - ONLY trains Orchestrator (Cypher training disabled)
# - Uses the new reward design that strongly penalizes unanswerable questions
# - Includes entity grounding to help Orchestrator learn valid entities
# - Runs for more iterations (20) with more epochs per iteration (8)
#
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"
OUTPUT_DIR="$PROJECT_DIR/outputs/stage1_ddp"
LOG_DIR="$OUTPUT_DIR/logs"
MAIN_SCRIPTS_DIR="$PROJECT_DIR/rl_implementation/ddp_training/scripts"

# Models
ORCHESTRATOR_MODEL="$PROJECT_DIR/models/qwen2.5-14b"
CYPHER_MODEL="$PROJECT_DIR/models/qwen2.5-coder-14b"

# Training parameters (can be overridden)
MAX_ITERATIONS=${MAX_ITERATIONS:-20}
EPOCHS_PER_ITER=${EPOCHS_PER_ITER:-8}
QUESTIONS_PER_ITER=${QUESTIONS_PER_ITER:-64}
SKIP_COLLECTION=${SKIP_COLLECTION:-false}
CLEAR_HISTORY=${CLEAR_HISTORY:-false}

# Create directories
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "ORCHESTRATOR-FOCUSED TRAINING PIPELINE (SLURM)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS"
echo "Time: $(date)"
echo ""
echo "Configuration:"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Epochs per iteration: $EPOCHS_PER_ITER"
echo "  Questions per iteration: $QUESTIONS_PER_ITER"
echo "  Skip first collection: $SKIP_COLLECTION"
echo "  Clear history: $CLEAR_HISTORY"
echo ""

# =============================================================================
# Step 1: Start vLLM Servers
# =============================================================================
echo "============================================================"
echo "Step 1: Starting vLLM Inference Servers"
echo "============================================================"

# Function to start vLLM server
start_vllm_server() {
    local model_type=$1
    local model_path=$2
    local port=$3
    local gpu=$4
    local log_file="$LOG_DIR/vllm_${model_type}_${SLURM_JOB_ID}.log"
    
    echo "Starting $model_type server on GPU $gpu, port $port..."
    
    CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port $port \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --enable-prefix-caching \
        > "$log_file" 2>&1 &
    
    echo "  PID: $!"
    echo "  Log: $log_file"
}

# Start servers
start_vllm_server "orchestrator" "$ORCHESTRATOR_MODEL" 8001 0
start_vllm_server "cypher" "$CYPHER_MODEL" 8002 1

# Wait for servers to be ready
echo ""
echo "Waiting for servers to start..."
MAX_WAIT=300  # 5 minutes
WAIT_INTERVAL=10
TOTAL_WAIT=0

while true; do
    ORCH_READY=$(curl -s http://localhost:8001/health >/dev/null 2>&1 && echo "true" || echo "false")
    CYPHER_READY=$(curl -s http://localhost:8002/health >/dev/null 2>&1 && echo "true" || echo "false")
    
    if [[ "$ORCH_READY" == "true" && "$CYPHER_READY" == "true" ]]; then
        echo "  ✓ Both servers are ready!"
        break
    fi
    
    if [[ $TOTAL_WAIT -ge $MAX_WAIT ]]; then
        echo "  ✗ Timeout waiting for servers"
        echo "    Orchestrator: $ORCH_READY"
        echo "    Cypher: $CYPHER_READY"
        exit 1
    fi
    
    echo "  Waiting... ($TOTAL_WAIT s) Orch=$ORCH_READY, Cypher=$CYPHER_READY"
    sleep $WAIT_INTERVAL
    TOTAL_WAIT=$((TOTAL_WAIT + WAIT_INTERVAL))
done

# =============================================================================
# Step 2: Run Orchestrator-Focused Training
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Running Orchestrator-Focused Training"
echo "============================================================"
echo ""

# Build command
CMD="$MAIN_SCRIPTS_DIR/run_orchestrator_focus_training.sh"
CMD="$CMD --iterations $MAX_ITERATIONS"
CMD="$CMD --epochs $EPOCHS_PER_ITER"
CMD="$CMD --questions $QUESTIONS_PER_ITER"

if [[ "$SKIP_COLLECTION" == "true" ]]; then
    CMD="$CMD --skip-collection"
fi

if [[ "$CLEAR_HISTORY" == "true" ]]; then
    CMD="$CMD --clear-history"
fi

echo "Command: $CMD"
echo ""

# Run training
$CMD

TRAIN_EXIT_CODE=$?

# =============================================================================
# Step 3: Cleanup
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: Cleanup"
echo "============================================================"

# Kill vLLM servers
echo "Stopping vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" || true

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "ORCHESTRATOR-FOCUSED TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Time: $(date)"
echo ""

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Outputs:"
    echo "  Trained model: $PROJECT_DIR/outputs/orchestrator_trained/best_model_merged"
    echo "  Logs: $LOG_DIR"
    echo "  History: $OUTPUT_DIR/training_history.json"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "  Check logs in: $LOG_DIR"
fi

exit $TRAIN_EXIT_CODE

