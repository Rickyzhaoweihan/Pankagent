#!/bin/bash
# =============================================================================
# Accelerated Dynamic Training Pipeline
# =============================================================================
#
# Optimized version that overlaps rollout collection with training:
#   - While training iteration N, start collecting rollouts for N+1
#   - vLLM updates happen AFTER training, BEFORE next collection uses new model
#   - Can save 30-50% of total time compared to sequential pipeline
#
# Key insight: Rollouts collected with "old" model are still valid training data
# for GRPO (off-policy is fine). We just need to update vLLM before we need
# the NEW model's behavior.
#
# Timeline for iteration N (when training needed):
#   [Analyze N] → [Train N + Collect N+1 in parallel] → [Update vLLM] → [Next]
#
# GPU Allocation (4 GPUs required):
#   GPU 0: vLLM Orchestrator server (port 8001) - inference only
#   GPU 1: vLLM Cypher server (port 8002) - inference only
#   GPU 2: Cypher Generator training
#   GPU 3: Orchestrator training
#
# Prerequisites:
#   - vLLM servers running on GPU 0-1
#   - Neo4j API accessible
#
# Usage:
#   ./run_accelerated_pipeline.sh                        # Default: 10 iterations
#   ./run_accelerated_pipeline.sh --iterations 20        # Custom iterations
#   ./run_accelerated_pipeline.sh --questions 128        # Questions per iteration
#   ./run_accelerated_pipeline.sh --auto                 # Run until convergence
#
# =============================================================================

set -e

echo "=============================================="
echo "Accelerated Dynamic Training Pipeline"
echo "=============================================="
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Parse Arguments
# =============================================================================

MAX_ITERATIONS=10
QUESTIONS_PER_ITER=64
DIFFICULTY="easy"
BATCH_SIZE=8
AUTO_MODE=false
SKIP_FIRST_COLLECT=false
CONFIG_FILE=""
TRAINING_EPOCHS=3
OVERLAP_COLLECTION=true  # Enable overlapping by default

while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations|-n)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --questions|-q)
            QUESTIONS_PER_ITER="$2"
            shift 2
            ;;
        --difficulty|-d)
            DIFFICULTY="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs|-e)
            TRAINING_EPOCHS="$2"
            shift 2
            ;;
        --auto|-a)
            AUTO_MODE=true
            MAX_ITERATIONS=100
            shift
            ;;
        --skip-collect|--skip-first)
            SKIP_FIRST_COLLECT=true
            shift
            ;;
        --no-overlap)
            OVERLAP_COLLECTION=false
            shift
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --iterations, -n N    Max training iterations (default: 10)"
            echo "  --questions, -q N     Questions per iteration (default: 64)"
            echo "  --difficulty, -d D    Difficulty: easy/medium/hard (default: easy)"
            echo "  --batch-size, -b N    Collection batch size (default: 8)"
            echo "  --epochs, -e N        Training epochs per iteration (default: 3)"
            echo "  --auto, -a            Run until convergence"
            echo "  --skip-collect        Skip first rollout collection"
            echo "  --no-overlap          Disable overlapping (sequential mode)"
            echo "  --config, -c FILE     Custom config YAML file"
            echo "  --help, -h            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DDP_DIR="$SCRIPT_DIR/.."
CONFIG_DIR="$DDP_DIR/config"

if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$CONFIG_DIR/dynamic_training_config.yaml"
fi

# Conda environment
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Environment:"
echo "  Python: $(which python3)"
echo "  Conda: $CONDA_DEFAULT_ENV"
echo "  Overlap Collection: $OVERLAP_COLLECTION"
echo ""

# Cache directories
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

# Output directories
# Use job-specific ROLLOUTS_DIR if exported from SLURM script, else default
ROLLOUTS_DIR="${ROLLOUTS_DIR:-$PROJECT_DIR/outputs/stage1_ddp}"
LOG_DIR="$PROJECT_DIR/outputs/stage1_ddp/logs"
HISTORY_FILE="$ROLLOUTS_DIR/training_history.json"

# Create rollouts dir if not exists
mkdir -p "$ROLLOUTS_DIR"

echo "Rollouts Directory: $ROLLOUTS_DIR"
CYPHER_OUTPUT="$PROJECT_DIR/outputs/cypher_trained"
ORCH_OUTPUT="$PROJECT_DIR/outputs/orchestrator_trained"

mkdir -p "$ROLLOUTS_DIR" "$LOG_DIR" "$CYPHER_OUTPUT" "$ORCH_OUTPUT"

echo "Training Parameters:"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Questions/Iter: $QUESTIONS_PER_ITER"
echo "  Difficulty: $DIFFICULTY"
echo "  Epochs/Iter: $TRAINING_EPOCHS"
echo ""

# =============================================================================
# Check vLLM Servers
# =============================================================================

check_vllm_servers() {
    local SILENT=${1:-false}
    
    if [ "$SILENT" = false ]; then
        echo "Checking vLLM servers..."
    fi
    
    if ! curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
        [ "$SILENT" = false ] && echo "  ✗ Orchestrator server not running!"
        return 1
    fi
    
    if ! curl -s "http://localhost:8002/health" > /dev/null 2>&1; then
        [ "$SILENT" = false ] && echo "  ✗ Cypher Generator server not running!"
        return 1
    fi
    
    if [ "$SILENT" = false ]; then
        echo "  ✓ Both vLLM servers running"
    fi
    return 0
}

wait_for_vllm_servers() {
    local MAX_WAIT=${1:-120}
    local WAITED=0
    
    echo "Waiting for vLLM servers to be ready..."
    while [ $WAITED -lt $MAX_WAIT ]; do
        if check_vllm_servers true; then
            echo "  ✓ Servers ready after ${WAITED}s"
            return 0
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        echo "  Waiting... ${WAITED}s"
    done
    
    echo "  ✗ Timeout waiting for servers"
    return 1
}

if ! check_vllm_servers; then
    echo "Please start vLLM servers first: ./start_vllm_servers.sh"
    exit 1
fi
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

get_rollouts_path() {
    local ITER=$1
    echo "$ROLLOUTS_DIR/rollouts_iter_$(printf '%03d' $ITER).jsonl"
}

# Collect rollouts (can run in background)
collect_rollouts() {
    local ITER=$1
    local OUTPUT_FILE=$(get_rollouts_path $ITER)
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_collect.log"
    
    echo "[Collect $ITER] Starting... (output: $OUTPUT_FILE)"
    
    python3 "$SCRIPT_DIR/collect_rollouts.py" \
        --num-questions "$QUESTIONS_PER_ITER" \
        --difficulty "$DIFFICULTY" \
        --batch-size "$BATCH_SIZE" \
        --output "$OUTPUT_FILE" \
        --schema-path "$PROJECT_DIR/legacy/PankBaseAgent/text_to_cypher/data/input/kg_schema.json" \
        --entity-samples "$PROJECT_DIR/outputs/entity_samples.json" \
        --orchestrator-model "$PROJECT_DIR/models/qwen2.5-14b" \
        --cypher-model "$PROJECT_DIR/models/qwen2.5-coder-14b" \
        --orchestrator-port 8001 \
        --cypher-port 8002 \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Collect $ITER] ✓ Complete ($(wc -l < "$OUTPUT_FILE") entries)"
    else
        echo "[Collect $ITER] ✗ Failed (exit code: $EXIT_CODE)"
    fi
    return $EXIT_CODE
}

analyze_and_decide() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    
    # Run decision engine - verbose output goes to stderr, JSON to stdout
    # We capture stdout (JSON) and let stderr print to console
    python3 "$SCRIPT_DIR/training_decision.py" \
        --rollouts "$ROLLOUTS_FILE" \
        --history "$HISTORY_FILE" \
        --config "$CONFIG_FILE" \
        --iteration "$ITER" \
        --update-history \
        --verbose
}

train_cypher() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_cypher.log"
    
    echo "[Train Cypher $ITER] Starting on GPU 2..."
    
    CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/run_train_cypher.sh" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$TRAINING_EPOCHS" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Train Cypher $ITER] ✓ Complete"
    else
        echo "[Train Cypher $ITER] ✗ Failed (exit code: $EXIT_CODE)"
    fi
    return $EXIT_CODE
}

train_orchestrator() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_orchestrator.log"
    
    echo "[Train Orch $ITER] Starting on GPU 3..."
    
    CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/run_train_orchestrator.sh" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$TRAINING_EPOCHS" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Train Orch $ITER] ✓ Complete"
    else
        echo "[Train Orch $ITER] ✗ Failed (exit code: $EXIT_CODE)"
    fi
    return $EXIT_CODE
}

update_vllm_server() {
    local SERVER=$1
    local MODEL_PATH=$2
    
    echo "[Update vLLM] Updating $SERVER with $MODEL_PATH"
    
    if [ -f "$SCRIPT_DIR/restart_vllm_server.sh" ] && [ -d "$MODEL_PATH" ]; then
        bash "$SCRIPT_DIR/restart_vllm_server.sh" \
            --server "$SERVER" \
            --model "$MODEL_PATH" 2>&1 | while read line; do echo "  $line"; done
        return $?
    else
        echo "  ⚠ Skipping (script or model not found)"
        return 0
    fi
}

update_history() {
    local CYPHER_TRAINED=$1
    local ORCH_TRAINED=$2
    
    python3 -c "
import json
from pathlib import Path

history_file = '$HISTORY_FILE'
path = Path(history_file)
if path.exists() and path.stat().st_size > 0:
    try:
        history = json.load(open(history_file))
        if history and isinstance(history, list):
            history[-1]['trained'] = {
                'cypher': $CYPHER_TRAINED,
                'orchestrator': $ORCH_TRAINED,
            }
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
    except:
        pass
"
}

# =============================================================================
# Main Training Loop with Overlapping
# =============================================================================

echo "=============================================="
echo "Starting Accelerated Training Loop"
echo "=============================================="
echo ""

CONVERGED=false
CONSECUTIVE_NO_TRAIN=0
CONVERGENCE_THRESHOLD=3
BG_COLLECT_PID=""
NEXT_ROLLOUTS_READY=false

for ITER in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "=============================================="
    echo "ITERATION $ITER / $MAX_ITERATIONS"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    ROLLOUTS_FILE=$(get_rollouts_path $ITER)
    NEXT_ROLLOUTS_FILE=$(get_rollouts_path $((ITER + 1)))
    
    # =========================================================================
    # Step 1: Get rollouts for this iteration
    # =========================================================================
    
    if [ "$SKIP_FIRST_COLLECT" = true ] && [ $ITER -eq 1 ]; then
        echo "Step 1: Using existing rollouts (--skip-collect)"
        LATEST_ROLLOUTS=$(ls -t "$ROLLOUTS_DIR"/rollouts_*.jsonl 2>/dev/null | head -1)
        if [ -z "$LATEST_ROLLOUTS" ]; then
            echo "ERROR: No existing rollouts found!"
            exit 1
        fi
        ROLLOUTS_FILE="$LATEST_ROLLOUTS"
        echo "  Using: $ROLLOUTS_FILE"
        
    elif [ -f "$ROLLOUTS_FILE" ] && [ "$NEXT_ROLLOUTS_READY" = true ]; then
        # Rollouts were collected in background during previous iteration
        echo "Step 1: Using pre-collected rollouts (from background)"
        echo "  Using: $ROLLOUTS_FILE"
        NEXT_ROLLOUTS_READY=false
        
    else
        # Need to collect now (first iteration or no overlap)
        echo "Step 1: Collecting rollouts"
        collect_rollouts $ITER
    fi
    
    # Verify rollouts
    if [ ! -f "$ROLLOUTS_FILE" ]; then
        echo "ERROR: Rollouts not found: $ROLLOUTS_FILE"
        exit 1
    fi
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_FILE")
    echo "  Rollouts: $ROLLOUT_COUNT entries"
    echo ""
    
    # =========================================================================
    # Step 2: Analyze and decide
    # =========================================================================
    
    echo "Step 2: Analyzing & Making Decision"
    echo "-------------------------------------------"
    DECISION=$(analyze_and_decide "$ROLLOUTS_FILE" $ITER)
    
    # Debug: Check if DECISION is valid JSON
    if [ -z "$DECISION" ]; then
        echo "WARNING: Empty decision received, defaulting to no training"
        TRAIN_CYPHER="false"
        TRAIN_ORCH="false"
    else
        # Parse decision with robust error handling
        TRAIN_CYPHER=$(python3 -c "
import json, sys
try:
    d = json.loads('''$DECISION''')
    print('true' if d.get('train_cypher') else 'false')
except:
    print('false')
" 2>/dev/null || echo "false")
        TRAIN_ORCH=$(python3 -c "
import json, sys
try:
    d = json.loads('''$DECISION''')
    print('true' if d.get('train_orchestrator') else 'false')
except:
    print('false')
" 2>/dev/null || echo "false")
    fi
    
    echo ""
    echo "Decision:"
    echo "  Train Cypher: $TRAIN_CYPHER"
    echo "  Train Orchestrator: $TRAIN_ORCH"
    echo ""
    
    # Check convergence
    if [ "$TRAIN_CYPHER" = "false" ] && [ "$TRAIN_ORCH" = "false" ]; then
        CONSECUTIVE_NO_TRAIN=$((CONSECUTIVE_NO_TRAIN + 1))
        echo "No training needed. Consecutive no-train iterations: $CONSECUTIVE_NO_TRAIN"
        
        if [ $CONSECUTIVE_NO_TRAIN -ge $CONVERGENCE_THRESHOLD ]; then
            echo ""
            echo "🎯 Convergence detected! Models above threshold for $CONVERGENCE_THRESHOLD iterations."
            CONVERGED=true
            break
        fi
        
        # Even if no training, collect next rollouts for monitoring
        if [ "$OVERLAP_COLLECTION" = true ] && [ $ITER -lt $MAX_ITERATIONS ]; then
            echo "Collecting next iteration rollouts for monitoring..."
            collect_rollouts $((ITER + 1))
        fi
        continue
    else
        CONSECUTIVE_NO_TRAIN=0
    fi
    
    # =========================================================================
    # Step 3: Train + Overlap Collection
    # =========================================================================
    
    echo "Step 3: Training (with overlapped collection)"
    echo "-------------------------------------------"
    
    CYPHER_TRAINED=false
    ORCH_TRAINED=false
    BG_COLLECT_PID=""
    
    # Start background collection for NEXT iteration (if enabled and not last)
    if [ "$OVERLAP_COLLECTION" = true ] && [ $ITER -lt $MAX_ITERATIONS ]; then
        echo ""
        echo ">>> Starting BACKGROUND collection for iteration $((ITER + 1))..."
        echo "    (Uses current vLLM models - rollouts still valid for training)"
        collect_rollouts $((ITER + 1)) &
        BG_COLLECT_PID=$!
        echo "    Background PID: $BG_COLLECT_PID"
        echo ""
    fi
    
    # Start training based on decision
    if [ "$TRAIN_CYPHER" = "true" ] && [ "$TRAIN_ORCH" = "true" ]; then
        echo "Training BOTH models in parallel..."
        echo ""
        
        train_cypher "$ROLLOUTS_FILE" $ITER &
        CYPHER_PID=$!
        
        train_orchestrator "$ROLLOUTS_FILE" $ITER &
        ORCH_PID=$!
        
        # Wait for training to complete
        wait $CYPHER_PID && CYPHER_TRAINED=true
        wait $ORCH_PID && ORCH_TRAINED=true
        
    elif [ "$TRAIN_CYPHER" = "true" ]; then
        echo "Training Cypher Generator only..."
        echo ""
        train_cypher "$ROLLOUTS_FILE" $ITER && CYPHER_TRAINED=true
        
    elif [ "$TRAIN_ORCH" = "true" ]; then
        echo "Training Orchestrator only..."
        echo ""
        train_orchestrator "$ROLLOUTS_FILE" $ITER && ORCH_TRAINED=true
    fi
    
    echo ""
    echo "Training Results:"
    echo "  Cypher trained: $CYPHER_TRAINED"
    echo "  Orchestrator trained: $ORCH_TRAINED"
    
    # =========================================================================
    # Step 4: Wait for background collection (if running)
    # =========================================================================
    
    if [ -n "$BG_COLLECT_PID" ]; then
        echo ""
        echo "Step 4a: Waiting for background collection..."
        if wait $BG_COLLECT_PID; then
            echo "  ✓ Background collection complete"
            NEXT_ROLLOUTS_READY=true
        else
            echo "  ✗ Background collection failed"
            NEXT_ROLLOUTS_READY=false
        fi
    fi
    
    # =========================================================================
    # Step 5: Update vLLM servers AFTER training completes
    # =========================================================================
    # 
    # Key timing: Update AFTER training, BEFORE next iteration uses new model
    # The background-collected rollouts used OLD models, which is fine for training.
    # But next iteration's NEW collection should use UPDATED models.
    #
    
    if [ "$CYPHER_TRAINED" = true ] || [ "$ORCH_TRAINED" = true ]; then
        echo ""
        echo "Step 5: Updating vLLM Servers"
        echo "-------------------------------------------"
        echo "(Updates AFTER training, BEFORE next fresh collection)"
        echo ""
        
        if [ "$CYPHER_TRAINED" = true ]; then
            # Use merged model for vLLM (not LoRA adapter)
            CYPHER_MERGED="$CYPHER_OUTPUT/best_model_merged"
            if [ -d "$CYPHER_MERGED" ]; then
                update_vllm_server "cypher" "$CYPHER_MERGED"
            else
                echo "Warning: Merged model not found at $CYPHER_MERGED"
            fi
        fi
        
        if [ "$ORCH_TRAINED" = true ]; then
            # Use merged model for vLLM (not LoRA adapter)
            ORCH_MERGED="$ORCH_OUTPUT/best_model_merged"
            if [ -d "$ORCH_MERGED" ]; then
                update_vllm_server "orchestrator" "$ORCH_MERGED"
            else
                echo "Warning: Merged model not found at $ORCH_MERGED"
            fi
        fi
        
        # Wait for servers to be ready
        echo ""
        wait_for_vllm_servers 120 || echo "Warning: vLLM check failed, continuing..."
    fi
    
    # Update history with training results
    update_history "$CYPHER_TRAINED" "$ORCH_TRAINED"
    
    echo ""
    echo "✓ Iteration $ITER complete!"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Accelerated Pipeline Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo ""

if [ "$CONVERGED" = true ]; then
    echo "Status: CONVERGED after $ITER iterations"
else
    echo "Status: Completed $MAX_ITERATIONS iterations"
fi

echo ""
echo "Results:"
echo "  Cypher Model: $CYPHER_OUTPUT/best_model"
echo "  Orchestrator Model: $ORCH_OUTPUT/best_model"
echo ""

# Show final metrics
if [ -f "$HISTORY_FILE" ]; then
    echo "Training Summary:"
    python3 -c "
import json
from pathlib import Path

history_file = '$HISTORY_FILE'
path = Path(history_file)
history = []
if path.exists() and path.stat().st_size > 0:
    try:
        history = json.load(open(history_file))
    except:
        history = []

if history and isinstance(history, list):
    last = history[-1]
    metrics = last.get('metrics', {})
    print(f\"  Final Cypher Reward: {metrics.get('cypher_reward', 0):.3f}\")
    print(f\"  Final Orch Reward: {metrics.get('orch_avg_reward', 0):.3f}\")
    print(f\"  Total Iterations: {len(history)}\")
    
    cypher_trains = sum(1 for h in history if h.get('trained', {}).get('cypher', False))
    orch_trains = sum(1 for h in history if h.get('trained', {}).get('orchestrator', False))
    print(f\"  Cypher Training Runs: {cypher_trains}\")
    print(f\"  Orchestrator Training Runs: {orch_trains}\")
    
    # Estimate time saved
    if len(history) > 1:
        print(f\"  Estimated Time Saved: ~{len(history) * 30}s (overlapped collection)\")
else:
    print('  No training history available')
"
fi

echo ""
echo "✓ Pipeline finished successfully!"

