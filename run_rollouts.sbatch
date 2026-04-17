#!/bin/bash
#SBATCH --job-name=rollouts_eval
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/rollouts_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/rollouts_%j.err


# =============================================================================
# Rollouts Evaluation Job
# 
# This script:
#   1. Starts a vLLM server hosting the cypher-writer model on GPU
#   2. Waits for the server to be ready
#   3. Runs all 680 rollout questions through the multi-agent system
#   4. Saves summaries to rollouts_results.json
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "ROLLOUTS EVALUATION JOB"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "============================================================"

# ----- Configuration -----
WORKDIR=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation
MODEL_PATH=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/orchestrator_trained_hints_test_0118_2223/best_model_merged
VLLM_PORT=8002
QUESTIONS_FILE=rollouts_iter_009_to_013_questions.txt
OUTPUT_FILE=rollouts_results.json

# =============================================
# >>> SET YOUR API KEYS HERE <<<
# =============================================
# Anthropic API Key (used by Planner, FormatAgent, PankBaseAgent, HIRN Literature, TemplateToolAgent)
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"

# OpenAI API Key (kept for backward compat, e.g. vLLM text2cypher uses OpenAI-compatible API)
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
# =============================================

cd "$WORKDIR"

# Create output logs dir if it doesn't exist
mkdir -p outputs/logs
mkdir -p logs

# ----- Environment Setup -----
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Export API keys for sub-agents
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
echo "ANTHROPIC_API_KEY set"
echo "OPENAI_API_KEY set"

# Export vLLM port so Python code (text2cypher_agent.py) picks it up
export VLLM_PORT="$VLLM_PORT"
echo "VLLM_PORT=$VLLM_PORT"

# ----- Step 1: Start vLLM Server -----
echo ""
echo "============================================================"
echo "STEP 1: Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $VLLM_PORT"
echo "============================================================"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name cypher-writer \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    > logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# ----- Step 2: Wait for vLLM to be Ready -----
echo ""
echo "Waiting for vLLM server to be ready..."

MAX_WAIT=600  # 10 minutes
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if vLLM process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs/vllm_server_${SLURM_JOB_ID}.log"
        cat logs/vllm_server_${SLURM_JOB_ID}.log | tail -50
        exit 1
    fi
    
    # Check if the server is responding
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM server is ready! (took ${ELAPSED}s)"
        break
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ----- Step 3: Run Rollout Questions -----
echo ""
echo "============================================================"
echo "STEP 3: Running rollout questions..."
echo "  Input:  $QUESTIONS_FILE"
echo "  Output: $OUTPUT_FILE"
echo "============================================================"
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate agent
python3 run_rollouts.py \
    --input "$QUESTIONS_FILE" \
    --output "$OUTPUT_FILE" \
    --skip-vllm-check \
    --resume \
    --timeout 300 \
    --agent-log "logs/agent_debug_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

# ----- Cleanup: Stop vLLM Server -----
echo ""
echo "============================================================"
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM server stopped."

echo ""
echo "============================================================"
echo "JOB COMPLETE"
echo "Exit code: $EXIT_CODE"
echo "Finished:  $(date)"
echo "Output:    $WORKDIR/$OUTPUT_FILE"
echo "============================================================"

exit $EXIT_CODE

