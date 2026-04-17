#!/bin/bash
# =============================================================================
# Dynamic Training Pipeline
# =============================================================================
#
# Adaptive training loop that:
#   1. Collects rollouts with current models
#   2. Analyzes metrics and decides which models need training
#   3. Trains only the models that need it
#   4. Updates vLLM servers for trained models only
#   5. Repeats for the specified number of iterations
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
#   ./run_full_training_pipeline.sh                        # Default: 10 iterations
#   ./run_full_training_pipeline.sh --iterations 20        # Custom iterations
#   ./run_full_training_pipeline.sh --questions 128        # Questions per iteration
#   ./run_full_training_pipeline.sh --iterations 20        # Run 20 iterations
#   ./run_full_training_pipeline.sh --skip-collect         # Skip first collection
#   ./run_full_training_pipeline.sh --config custom.yaml   # Custom config
#
# =============================================================================

set -e

echo "=============================================="
echo "Dynamic Training Pipeline"
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

# Coverage-based stopping (set to enable, 0 to disable and use QUESTIONS_PER_ITER)
TARGET_FILTERED_ROLLOUTS=128  # Collect until we have this many usable rollouts
MIN_USABLE_REWARD=0.1         # Minimum reward for a rollout to count as usable
MAX_COLLECTION_ATTEMPTS=500   # Safety limit for total collection attempts

# Adaptive entity sampling
USE_ADAPTIVE_SAMPLING=true
ENTITY_DEGREES_PATH="$PROJECT_DIR/outputs/entity_degrees.json"
ADAPTIVE_SAMPLER_PATH="$PROJECT_DIR/outputs/adaptive_sampler_state.json"
SKIP_FIRST_COLLECT=false
CONFIG_FILE=""
TRAINING_EPOCHS=3

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
            # DEPRECATED: Auto-convergence removed. Use --iterations instead.
            echo "Warning: --auto is deprecated. Pipeline always runs for specified iterations."
            echo "Use --iterations N to set number of iterations."
            shift
            ;;
        --skip-collect|--skip-first)
            SKIP_FIRST_COLLECT=true
            shift
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --target-filtered)
            TARGET_FILTERED_ROLLOUTS="$2"
            shift 2
            ;;
        --min-usable-reward)
            MIN_USABLE_REWARD="$2"
            shift 2
            ;;
        --no-adaptive-sampling)
            USE_ADAPTIVE_SAMPLING=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --iterations, -n N       Max training iterations (default: 10)"
            echo "  --questions, -q N        Questions per iteration (default: 64, used if --target-filtered=0)"
            echo "  --difficulty, -d D       Difficulty: easy/medium/hard (default: easy)"
            echo "  --batch-size, -b N       Collection batch size (default: 8)"
            echo "  --epochs, -e N           Training epochs per iteration (default: 3)"
            echo "  --auto, -a               (deprecated, no longer used)"
            echo "  --skip-collect           Skip first rollout collection"
            echo "  --config, -c FILE        Custom config YAML file"
            echo ""
            echo "Coverage-based stopping (default):"
            echo "  --target-filtered N      Target usable rollouts (default: 128)"
            echo "  --min-usable-reward R    Min reward for usable (default: 0.1)"
            echo ""
            echo "Adaptive sampling:"
            echo "  --no-adaptive-sampling   Disable adaptive entity sampling"
            echo ""
            echo "  --help, -h               Show this help"
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

# Default config file
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$CONFIG_DIR/dynamic_training_config.yaml"
fi

# Conda environment
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Environment:"
echo "  Python: $(which python3)"
echo "  Conda: $CONDA_DEFAULT_ENV"
echo "  Script Dir: $SCRIPT_DIR"
echo "  Project Dir: $PROJECT_DIR"
echo "  Config: $CONFIG_FILE"
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

# Experiment-specific trained model directories (isolated per RUN_ID)
# This ensures each experiment starts fresh from base models
# Within an experiment, iterations build on each other
RUN_ID="${RUN_ID:-default}"
CYPHER_OUTPUT="$PROJECT_DIR/outputs/cypher_trained_${RUN_ID}"
ORCH_OUTPUT="$PROJECT_DIR/outputs/orchestrator_trained_${RUN_ID}"

mkdir -p "$ROLLOUTS_DIR" "$LOG_DIR" "$CYPHER_OUTPUT" "$ORCH_OUTPUT"

echo "Output Directories:"
echo "  Rollouts: $ROLLOUTS_DIR"
echo "  Logs: $LOG_DIR"
echo "  Cypher Models: $CYPHER_OUTPUT (experiment-specific)"
echo "  Orchestrator Models: $ORCH_OUTPUT (experiment-specific)"
echo ""

echo "Training Parameters:"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs/Iter: $TRAINING_EPOCHS"
# Note: Pipeline always runs for MAX_ITERATIONS (no auto-convergence)
echo ""
echo "Rollout Collection Strategy:"
if [ "$TARGET_FILTERED_ROLLOUTS" -gt 0 ]; then
    echo "  Mode: COVERAGE-BASED"
    echo "  Target Usable Rollouts: $TARGET_FILTERED_ROLLOUTS"
    echo "  Min Usable Reward: $MIN_USABLE_REWARD"
    echo "  Max Attempts: $MAX_COLLECTION_ATTEMPTS"
else
    echo "  Mode: FIXED COUNT"
    echo "  Questions/Iter: $QUESTIONS_PER_ITER"
fi
echo ""
echo "Adaptive Sampling:"
echo "  Enabled: $USE_ADAPTIVE_SAMPLING"
if [ "$USE_ADAPTIVE_SAMPLING" = true ]; then
    echo "  Entity Degrees: $ENTITY_DEGREES_PATH"
    echo "  Sampler State: $ADAPTIVE_SAMPLER_PATH"
fi
echo ""

# =============================================================================
# Check vLLM Servers
# =============================================================================

check_vllm_servers() {
    echo "Checking vLLM servers..."
    
    if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
        echo "  ✓ Orchestrator server (port 8001) is running"
    else
        echo "  ✗ Orchestrator server not running!"
        echo "    Please start vLLM servers first: ./start_vllm_servers.sh"
        return 1
    fi
    
    if curl -s "http://localhost:8002/health" > /dev/null 2>&1; then
        echo "  ✓ Cypher Generator server (port 8002) is running"
    else
        echo "  ✗ Cypher Generator server not running!"
        echo "    Please start vLLM servers first: ./start_vllm_servers.sh"
        return 1
    fi
    
    return 0
}

if ! check_vllm_servers; then
    exit 1
fi
echo ""

# =============================================================================
# Training Functions
# =============================================================================

collect_rollouts() {
    local ITER=$1
    local OUTPUT_FILE="$ROLLOUTS_DIR/rollouts_iter_$(printf '%03d' $ITER).jsonl"
    
    echo "Collecting rollouts for iteration $ITER..."
    echo "  Output: $OUTPUT_FILE"
    
    # Build collect command with coverage-based stopping
    local COLLECT_CMD="python3 $SCRIPT_DIR/collect_rollouts.py"
    
    # Coverage-based stopping vs fixed count
    if [ "$TARGET_FILTERED_ROLLOUTS" -gt 0 ]; then
        echo "  Mode: Coverage-based (target: $TARGET_FILTERED_ROLLOUTS usable, min reward: $MIN_USABLE_REWARD)"
        COLLECT_CMD="$COLLECT_CMD --target-filtered-rollouts $TARGET_FILTERED_ROLLOUTS"
        COLLECT_CMD="$COLLECT_CMD --min-usable-reward $MIN_USABLE_REWARD"
        COLLECT_CMD="$COLLECT_CMD --max-collection-attempts $MAX_COLLECTION_ATTEMPTS"
    else
        echo "  Mode: Fixed count ($QUESTIONS_PER_ITER questions)"
        COLLECT_CMD="$COLLECT_CMD --num-questions $QUESTIONS_PER_ITER"
    fi
    
    # Adaptive sampling
    if [ "$USE_ADAPTIVE_SAMPLING" = true ] && [ -f "$ENTITY_DEGREES_PATH" ]; then
        echo "  Adaptive sampling: ENABLED"
        COLLECT_CMD="$COLLECT_CMD --use-adaptive-sampling"
        COLLECT_CMD="$COLLECT_CMD --entity-degrees $ENTITY_DEGREES_PATH"
        COLLECT_CMD="$COLLECT_CMD --adaptive-sampler-state $ADAPTIVE_SAMPLER_PATH"
    else
        echo "  Adaptive sampling: DISABLED"
    fi
    
    # Common arguments
    COLLECT_CMD="$COLLECT_CMD --difficulty $DIFFICULTY"
    COLLECT_CMD="$COLLECT_CMD --batch-size $BATCH_SIZE"
    COLLECT_CMD="$COLLECT_CMD --output $OUTPUT_FILE"
    COLLECT_CMD="$COLLECT_CMD --schema-path $PROJECT_DIR/legacy/PankBaseAgent/text_to_cypher/data/input/kg_schema.json"
    COLLECT_CMD="$COLLECT_CMD --entity-samples $PROJECT_DIR/outputs/entity_samples.json"
    COLLECT_CMD="$COLLECT_CMD --orchestrator-model $PROJECT_DIR/models/qwen2.5-14b"
    COLLECT_CMD="$COLLECT_CMD --cypher-model $PROJECT_DIR/models/qwen2.5-coder-14b"
    COLLECT_CMD="$COLLECT_CMD --orchestrator-port 8001"
    COLLECT_CMD="$COLLECT_CMD --cypher-port 8002"
    
    # Execute
    eval $COLLECT_CMD
    
    echo "$OUTPUT_FILE"
}

analyze_and_decide() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    
    echo "Analyzing rollouts and making training decision..."
    
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
    
    echo "Training Cypher Generator on GPU 2..."
    echo "  Log: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/run_train_cypher.sh" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$TRAINING_EPOCHS" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Cypher Generator training completed"
        return 0
    else
        echo "  ✗ Cypher Generator training failed (exit code: $EXIT_CODE)"
        return 1
    fi
}

train_orchestrator() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_orchestrator.log"
    
    echo "Training Orchestrator on GPU 3..."
    echo "  Log: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/run_train_orchestrator.sh" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$TRAINING_EPOCHS" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Orchestrator training completed"
        return 0
    else
        echo "  ✗ Orchestrator training failed (exit code: $EXIT_CODE)"
        return 1
    fi
}

update_vllm_server() {
    local SERVER=$1  # 'cypher' or 'orchestrator'
    local MODEL_PATH=$2
    
    echo "Updating vLLM server for $SERVER..."
    echo "  Model: $MODEL_PATH"
    
    if [ -f "$SCRIPT_DIR/restart_vllm_server.sh" ]; then
        bash "$SCRIPT_DIR/restart_vllm_server.sh" \
            --server "$SERVER" \
            --model "$MODEL_PATH"
        echo "  ✓ vLLM server updated"
    else
        echo "  ⚠ restart_vllm_server.sh not found, skipping server update"
    fi
}

# =============================================================================
# Main Training Loop
# =============================================================================

echo "=============================================="
echo "Starting Dynamic Training Loop"
echo "=============================================="
echo ""

# Track rewards for logging improvement (no automatic convergence/early stopping)
LAST_CYPHER_REWARD=0.0
LAST_ORCH_REWARD=0.0
TOTAL_CYPHER_TRAINS=0
TOTAL_ORCH_TRAINS=0

for ITER in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "=============================================="
    echo "ITERATION $ITER / $MAX_ITERATIONS"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    # Skip first collection if requested
    if [ "$SKIP_FIRST_COLLECT" = true ] && [ $ITER -eq 1 ]; then
        echo "Skipping first rollout collection..."
        # Use existing rollouts
        LATEST_ROLLOUTS=$(ls -t "$ROLLOUTS_DIR"/rollouts_*.jsonl 2>/dev/null | head -1)
        if [ -z "$LATEST_ROLLOUTS" ]; then
            echo "ERROR: No existing rollouts found!"
            exit 1
        fi
        echo "  Using existing: $LATEST_ROLLOUTS"
        ROLLOUTS_FILE="$LATEST_ROLLOUTS"
    else
        # Step 1: Collect rollouts
        echo "Step 1: Collecting Rollouts"
        echo "-------------------------------------------"
        ROLLOUTS_FILE="$ROLLOUTS_DIR/rollouts_iter_$(printf '%03d' $ITER).jsonl"
        collect_rollouts $ITER
        echo ""
    fi
    
    # Verify rollouts exist
    if [ ! -f "$ROLLOUTS_FILE" ]; then
        echo "ERROR: Rollouts file not found: $ROLLOUTS_FILE"
        exit 1
    fi
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_FILE")
    echo "Rollouts collected: $ROLLOUT_COUNT entries"
    echo ""
    
    # Step 2: Analyze and decide
    echo "Step 2: Analyzing & Making Decision"
    echo "-------------------------------------------"
    DECISION=$(analyze_and_decide "$ROLLOUTS_FILE" $ITER)
    echo ""
    
    # Parse decision with robust error handling
    # NOTE: DECISION is single-line JSON from training_decision.py
    if [ -z "$DECISION" ]; then
        echo "WARNING: Empty decision received, defaulting to no training"
        TRAIN_CYPHER="false"
        TRAIN_ORCH="false"
        CURRENT_CYPHER_REWARD=0.0
        CURRENT_ORCH_REWARD=0.0
    else
        # Use a temp file to avoid shell quoting issues with JSON
        DECISION_FILE=$(mktemp)
        echo "$DECISION" > "$DECISION_FILE"
        
        PARSED=$(python3 -c "
import json
import sys
try:
    with open('$DECISION_FILE', 'r') as f:
        d = json.load(f)
    train_cypher = 'true' if d.get('train_cypher') else 'false'
    train_orch = 'true' if d.get('train_orchestrator') else 'false'
    cypher_reward = d.get('metrics', {}).get('cypher_reward', 0.0)
    orch_reward = d.get('metrics', {}).get('orch_avg_reward', 0.0)
    print(f'{train_cypher}|{train_orch}|{cypher_reward}|{orch_reward}')
except Exception as e:
    print(f'false|false|0.0|0.0', file=sys.stderr)
    print('false|false|0.0|0.0')
" 2>/dev/null)
        
        rm -f "$DECISION_FILE"
        
        TRAIN_CYPHER=$(echo "$PARSED" | cut -d'|' -f1)
        TRAIN_ORCH=$(echo "$PARSED" | cut -d'|' -f2)
        CURRENT_CYPHER_REWARD=$(echo "$PARSED" | cut -d'|' -f3)
        CURRENT_ORCH_REWARD=$(echo "$PARSED" | cut -d'|' -f4)
        
        # Fallback if parsing failed
        [ -z "$TRAIN_CYPHER" ] && TRAIN_CYPHER="false"
        [ -z "$TRAIN_ORCH" ] && TRAIN_ORCH="false"
        [ -z "$CURRENT_CYPHER_REWARD" ] && CURRENT_CYPHER_REWARD="0.0"
        [ -z "$CURRENT_ORCH_REWARD" ] && CURRENT_ORCH_REWARD="0.0"
    fi
    
    echo "Decision:"
    echo "  Train Cypher: $TRAIN_CYPHER"
    echo "  Train Orchestrator: $TRAIN_ORCH"
    echo "  Current Rewards: cypher=$CURRENT_CYPHER_REWARD, orch=$CURRENT_ORCH_REWARD"
    echo ""
    
    # Step 3: Train models (if needed)
    CYPHER_TRAINED=false
    ORCH_TRAINED=false
    
    if [ "$TRAIN_CYPHER" = "true" ] || [ "$TRAIN_ORCH" = "true" ]; then
        echo "Step 3: Training Models"
        echo "-------------------------------------------"
        
        # Train in parallel if both need training
        if [ "$TRAIN_CYPHER" = "true" ] && [ "$TRAIN_ORCH" = "true" ]; then
            echo "Training both models in parallel..."
            
            train_cypher "$ROLLOUTS_FILE" $ITER &
            CYPHER_PID=$!
            
            train_orchestrator "$ROLLOUTS_FILE" $ITER &
            ORCH_PID=$!
            
            # Wait for both
            wait $CYPHER_PID && CYPHER_TRAINED=true
            wait $ORCH_PID && ORCH_TRAINED=true
            
        elif [ "$TRAIN_CYPHER" = "true" ]; then
            train_cypher "$ROLLOUTS_FILE" $ITER && CYPHER_TRAINED=true
            
        elif [ "$TRAIN_ORCH" = "true" ]; then
            train_orchestrator "$ROLLOUTS_FILE" $ITER && ORCH_TRAINED=true
        fi
        
        echo ""
    fi
    
    # Step 4: Update vLLM servers for trained models
    if [ "$CYPHER_TRAINED" = true ] || [ "$ORCH_TRAINED" = true ]; then
        echo "Step 4: Updating vLLM Servers"
        echo "-------------------------------------------"
        
        if [ "$CYPHER_TRAINED" = true ]; then
            # Use merged model for vLLM (not LoRA adapter)
            CYPHER_BEST="$CYPHER_OUTPUT/best_model_merged"
            if [ -d "$CYPHER_BEST" ]; then
                update_vllm_server "cypher" "$CYPHER_BEST"
            else
                echo "Warning: Merged model not found at $CYPHER_BEST"
            fi
        fi
        
        if [ "$ORCH_TRAINED" = true ]; then
            # Use merged model for vLLM (not LoRA adapter)
            ORCH_BEST="$ORCH_OUTPUT/best_model_merged"
            if [ -d "$ORCH_BEST" ]; then
                update_vllm_server "orchestrator" "$ORCH_BEST"
            else
                echo "Warning: Merged model not found at $ORCH_BEST"
            fi
        fi
        
        echo ""
        
        # Wait for servers to be ready
        echo "Waiting for vLLM servers to restart..."
        sleep 30
        check_vllm_servers || echo "Warning: Server check failed, continuing anyway"
        
        # =====================================================================
        # POST-TRAINING VALIDATION
        # =====================================================================
        echo ""
        echo "=== POST-TRAINING VALIDATION ==="
        echo "Validating model improvement on sample of rollouts..."
        
        VALIDATION_RESULT=$(python3 "$SCRIPT_DIR/validate_model_improvement.py" \
            --rollouts "$ROLLOUTS_FILE" \
            --cypher-url "http://localhost:$VLLM_CYPHER_PORT/v1" \
            --orchestrator-url "http://localhost:$VLLM_ORCH_PORT/v1" \
            --sample-size 20 \
            2>/dev/null || echo '{"success": false}')
        
        # Parse and display results
        python3 -c "
import json
try:
    r = json.loads('''$VALIDATION_RESULT''')
    if r.get('success'):
        orig = r.get('original_cypher_reward', 0)
        new = r.get('new_cypher_reward', 0)
        imp = r.get('cypher_improvement', 0)
        improved = r.get('improved_count', 0)
        total = r.get('num_samples', 0)
        
        if imp > 0:
            print(f'  ✓ IMPROVED: {orig:.3f} → {new:.3f} (+{imp:.3f})')
        elif imp < 0:
            print(f'  ⚠ REGRESSED: {orig:.3f} → {new:.3f} ({imp:.3f})')
        else:
            print(f'  → No change: {orig:.3f} → {new:.3f}')
        print(f'  {improved}/{total} questions improved')
    else:
        print('  Validation skipped or failed')
except:
    print('  Validation result unavailable')
"
        echo ""
    fi
    
    # Update history with actual training results
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

# Find and update the last entry with actual outcomes
if history and isinstance(history, list):
    history[-1]['actually_trained'] = {
        'cypher': $( [ "$CYPHER_TRAINED" = true ] && echo 'True' || echo 'False' ),
        'orchestrator': $( [ "$ORCH_TRAINED" = true ] && echo 'True' || echo 'False' ),
    }
    # Keep 'trained' for backward compatibility (decision)
    # 'actually_trained' is what really happened
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
"
    
    # =========================================================================
    # Progress Tracking (NO automatic convergence/early stopping)
    # =========================================================================
    # The pipeline runs for exactly MAX_ITERATIONS - it only decides train or not,
    # never when to stop. User controls the iteration count.
    
    if [ "$CYPHER_TRAINED" = true ] || [ "$ORCH_TRAINED" = true ]; then
        # Training happened - log improvement for information only
        IMPROVED=$(python3 -c "
cypher_improved = float('$CURRENT_CYPHER_REWARD') > float('$LAST_CYPHER_REWARD') + 0.01
orch_improved = float('$CURRENT_ORCH_REWARD') > float('$LAST_ORCH_REWARD') + 0.01
any_improved = cypher_improved or orch_improved
print('true' if any_improved else 'false')
" 2>/dev/null || echo "false")
        
        if [ "$IMPROVED" = "true" ]; then
            echo "  ✓ Improvement detected!"
        else
            echo "  ⚠ No improvement this iteration"
        fi
        
        # Update last rewards for next comparison
        LAST_CYPHER_REWARD="$CURRENT_CYPHER_REWARD"
        LAST_ORCH_REWARD="$CURRENT_ORCH_REWARD"
        
        # Track total training runs
        [ "$CYPHER_TRAINED" = true ] && TOTAL_CYPHER_TRAINS=$((TOTAL_CYPHER_TRAINS + 1))
        [ "$ORCH_TRAINED" = true ] && TOTAL_ORCH_TRAINS=$((TOTAL_ORCH_TRAINS + 1))
    else
        echo "  No training needed this iteration (above thresholds)"
    fi
    
    echo ""
    echo "Iteration $ITER / $MAX_ITERATIONS complete!"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Training Pipeline Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo ""

echo "Status: Completed $MAX_ITERATIONS iterations"
echo "  Cypher trained: $TOTAL_CYPHER_TRAINS times"
echo "  Orchestrator trained: $TOTAL_ORCH_TRAINS times"

echo ""
echo "Results (experiment-specific, RUN_ID=$RUN_ID):"
echo "  Cypher Generator: $CYPHER_OUTPUT/best_model"
echo "  Cypher Merged: $CYPHER_OUTPUT/best_model_merged"
echo "  Orchestrator: $ORCH_OUTPUT/best_model"
echo "  Orchestrator Merged: $ORCH_OUTPUT/best_model_merged"
echo ""
echo "History: $HISTORY_FILE"
echo "Logs: $LOG_DIR"
echo ""

# Show final metrics
if [ -f "$HISTORY_FILE" ]; then
    echo "Final Metrics:"
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
    print(f\"  Cypher Reward: {metrics.get('cypher_reward', 0):.3f}\")
    print(f\"  Orch QGen Reward: {metrics.get('orch_qgen_reward', 0):.3f}\")
    print(f\"  Orch Synth Reward: {metrics.get('orch_synth_reward', 0):.3f}\")
    print(f\"  Total Iterations: {len(history)}\")
    
    cypher_trained = sum(1 for h in history if h.get('trained', {}).get('cypher', False))
    orch_trained = sum(1 for h in history if h.get('trained', {}).get('orchestrator', False))
    print(f\"  Cypher Training Runs: {cypher_trained}\")
    print(f\"  Orchestrator Training Runs: {orch_trained}\")
else:
    print('  No training history available')
"
fi

echo ""
echo "✓ Pipeline finished successfully!"
