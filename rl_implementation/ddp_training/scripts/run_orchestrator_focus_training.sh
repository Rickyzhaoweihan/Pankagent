#!/bin/bash
# =============================================================================
# ORCHESTRATOR-FOCUSED TRAINING PIPELINE
# =============================================================================
#
# This script trains ONLY the Orchestrator to generate better questions.
# 
# Strategy:
# - Cypher Generator training is DISABLED (base model works well)
# - Focus entirely on making Orchestrator ask ANSWERABLE questions
# - Uses the new reward design that strongly penalizes unanswerable questions
#
# Prerequisites:
# - vLLM servers must be running (GPU 0: Orchestrator, GPU 1: Cypher)
# - Neo4j database must be accessible
#
# GPU Allocation:
# - GPU 0: vLLM Orchestrator inference
# - GPU 1: vLLM Cypher inference  
# - GPU 2: (unused in this mode)
# - GPU 3: Orchestrator training
#
# Usage:
#   ./run_orchestrator_focus_training.sh [OPTIONS]
#
# Options:
#   --iterations N     Number of training iterations (default: 20)
#   --epochs N         Epochs per iteration (default: 8)
#   --questions N      Questions per iteration (default: 64)
#   --skip-collection  Skip first rollout collection
#   --clear-history    Clear training history before starting
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CONFIG_DIR="$SCRIPT_DIR/../config"

# Defaults
MAX_ITERATIONS=20
EPOCHS_PER_ITER=8
QUESTIONS_PER_ITER=64
SKIP_COLLECTION=false
CLEAR_HISTORY=false

# Output directories
OUTPUT_DIR="$PROJECT_DIR/outputs/stage1_ddp"
LOG_DIR="$OUTPUT_DIR/logs"
HISTORY_FILE="$OUTPUT_DIR/training_history.json"

# Training GPU (Orchestrator only)
ORCH_TRAIN_GPU=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS_PER_ITER="$2"
            shift 2
            ;;
        --questions)
            QUESTIONS_PER_ITER="$2"
            shift 2
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
            shift
            ;;
        --clear-history)
            CLEAR_HISTORY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================================"
echo "ORCHESTRATOR-FOCUSED TRAINING PIPELINE"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Epochs per iteration: $EPOCHS_PER_ITER"
echo "  Questions per iteration: $QUESTIONS_PER_ITER"
echo "  Training GPU: $ORCH_TRAIN_GPU"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Clear history if requested
if [[ "$CLEAR_HISTORY" == "true" ]]; then
    echo "Clearing training history..."
    rm -f "$HISTORY_FILE"
fi

# =============================================================================
# Pre-flight Checks
# =============================================================================
echo "Checking vLLM servers..."

# Check Orchestrator server
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "ERROR: Orchestrator vLLM server not running on port 8001"
    echo "Start it with: ./start_vllm_servers.sh"
    exit 1
fi

# Check Cypher server
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "ERROR: Cypher vLLM server not running on port 8002"
    echo "Start it with: ./start_vllm_servers.sh"
    exit 1
fi

echo "  ✓ Both vLLM servers are running"
echo ""

# =============================================================================
# Training Loop
# =============================================================================
CONSECUTIVE_NO_IMPROVEMENT=0
BEST_ORCH_REWARD=0.0

for ((ITER=1; ITER<=MAX_ITERATIONS; ITER++)); do
    echo "=============================================="
    echo "ITERATION $ITER / $MAX_ITERATIONS"
    echo "=============================================="
    echo "Time: $(date)"
    
    ITER_PADDED=$(printf "%03d" $ITER)
    ROLLOUT_FILE="$OUTPUT_DIR/rollouts_iter_${ITER_PADDED}.jsonl"
    ORCH_LOG="$LOG_DIR/iter_${ITER_PADDED}_orchestrator.log"
    
    # -------------------------------------------------------------------------
    # Step 1: Collect Rollouts
    # -------------------------------------------------------------------------
    if [[ "$SKIP_COLLECTION" == "true" && $ITER -eq 1 ]]; then
        # Find most recent rollout file
        LATEST_ROLLOUT=$(ls -t "$OUTPUT_DIR"/rollouts_iter_*.jsonl 2>/dev/null | head -1)
        if [[ -n "$LATEST_ROLLOUT" ]]; then
            echo "Skipping first rollout collection..."
            echo "  Using existing: $LATEST_ROLLOUT"
            ROLLOUT_FILE="$LATEST_ROLLOUT"
        else
            echo "No existing rollouts found, collecting new ones..."
            SKIP_COLLECTION=false
        fi
    fi
    
    if [[ "$SKIP_COLLECTION" != "true" || $ITER -gt 1 ]]; then
        echo ""
        echo "Step 1: Collecting Rollouts"
        echo "-------------------------------------------"
        
        python3 "$SCRIPT_DIR/collect_rollouts.py" \
            --output "$ROLLOUT_FILE" \
            --num-questions $QUESTIONS_PER_ITER \
            --difficulty easy \
            --batch-size 8
        
        echo "Rollouts saved to: $ROLLOUT_FILE"
    fi
    
    # Count rollouts
    NUM_ROLLOUTS=$(wc -l < "$ROLLOUT_FILE" 2>/dev/null || echo "0")
    echo "Rollouts collected: $NUM_ROLLOUTS entries"
    
    # -------------------------------------------------------------------------
    # Step 2: Analyze Rollouts & Compute Metrics
    # -------------------------------------------------------------------------
    echo ""
    echo "Step 2: Analyzing Rollouts"
    echo "-------------------------------------------"
    
    METRICS=$(python3 -c "
import json
import sys

rollouts = []
with open('$ROLLOUT_FILE') as f:
    for line in f:
        rollouts.append(json.loads(line))

cypher_rewards = [r['trajectory'].get('cypher_reward', 0) for r in rollouts]
qgen_rewards = [r['trajectory'].get('orch_qgen_reward', 0) for r in rollouts]
synth_rewards = [r['trajectory'].get('orch_synth_reward', 0) for r in rollouts]

# Count answerable questions
num_answerable = sum(1 for r in rollouts 
                     if r['trajectory'].get('cypher_reward', 0) > 0.3)
pct_answerable = 100 * num_answerable / len(rollouts) if rollouts else 0

avg_cypher = sum(cypher_rewards) / len(cypher_rewards) if cypher_rewards else 0
avg_qgen = sum(qgen_rewards) / len(qgen_rewards) if qgen_rewards else 0
avg_synth = sum(synth_rewards) / len(synth_rewards) if synth_rewards else 0
avg_orch = (avg_qgen + avg_synth) / 2

print(f'cypher={avg_cypher:.3f}')
print(f'qgen={avg_qgen:.3f}')
print(f'synth={avg_synth:.3f}')
print(f'orch_avg={avg_orch:.3f}')
print(f'answerable_pct={pct_answerable:.1f}')
print(f'num_answerable={num_answerable}')
print(f'total={len(rollouts)}')
")
    
    # Parse metrics
    CYPHER_REWARD=$(echo "$METRICS" | grep "cypher=" | cut -d= -f2)
    QGEN_REWARD=$(echo "$METRICS" | grep "qgen=" | cut -d= -f2)
    SYNTH_REWARD=$(echo "$METRICS" | grep "synth=" | cut -d= -f2)
    ORCH_AVG=$(echo "$METRICS" | grep "orch_avg=" | cut -d= -f2)
    ANSWERABLE_PCT=$(echo "$METRICS" | grep "answerable_pct=" | cut -d= -f2)
    NUM_ANSWERABLE=$(echo "$METRICS" | grep "num_answerable=" | cut -d= -f2)
    TOTAL_ROLLOUTS=$(echo "$METRICS" | grep "total=" | cut -d= -f2)
    
    echo "Metrics:"
    echo "  Cypher Reward: $CYPHER_REWARD"
    echo "  Orchestrator QGen: $QGEN_REWARD"
    echo "  Orchestrator Synth: $SYNTH_REWARD"
    echo "  Orchestrator Avg: $ORCH_AVG"
    echo ""
    echo "  Answerable Questions: $NUM_ANSWERABLE / $TOTAL_ROLLOUTS ($ANSWERABLE_PCT%)"
    echo ""
    
    # Check for improvement
    IMPROVED=$(python3 -c "print('true' if $ORCH_AVG > $BEST_ORCH_REWARD else 'false')")
    if [[ "$IMPROVED" == "true" ]]; then
        BEST_ORCH_REWARD=$ORCH_AVG
        CONSECUTIVE_NO_IMPROVEMENT=0
        echo "  ★ New best Orchestrator reward: $BEST_ORCH_REWARD"
    else
        CONSECUTIVE_NO_IMPROVEMENT=$((CONSECUTIVE_NO_IMPROVEMENT + 1))
        echo "  No improvement ($CONSECUTIVE_NO_IMPROVEMENT consecutive)"
    fi
    
    # -------------------------------------------------------------------------
    # Step 3: Train Orchestrator
    # -------------------------------------------------------------------------
    echo ""
    echo "Step 3: Training Orchestrator"
    echo "-------------------------------------------"
    
    # Update config with current rollout file
    TRAIN_CONFIG="$CONFIG_DIR/train_orchestrator_config.yaml"
    
    echo "Training Orchestrator on GPU $ORCH_TRAIN_GPU..."
    echo "Log: $ORCH_LOG"
    
    CUDA_VISIBLE_DEVICES=$ORCH_TRAIN_GPU python3 \
        "$SCRIPT_DIR/train_orchestrator_from_rollouts.py" \
        --config "$TRAIN_CONFIG" \
        --rollouts "$ROLLOUT_FILE" \
        --epochs $EPOCHS_PER_ITER \
        2>&1 | tee "$ORCH_LOG"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "  ✓ Orchestrator training complete"
    else
        echo "  ✗ Orchestrator training failed"
        echo "  Check log: $ORCH_LOG"
    fi
    
    # -------------------------------------------------------------------------
    # Step 4: Update vLLM Server (if training was successful)
    # -------------------------------------------------------------------------
    MERGED_MODEL="$PROJECT_DIR/outputs/orchestrator_trained/best_model_merged"
    
    if [[ -d "$MERGED_MODEL" ]]; then
        echo ""
        echo "Step 4: Updating vLLM Orchestrator Server"
        echo "-------------------------------------------"
        
        "$SCRIPT_DIR/restart_vllm_server.sh" orchestrator "$MERGED_MODEL" 0
        
        # Wait for server to be ready
        echo "Waiting for server to restart..."
        sleep 30
        
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "  ✓ Orchestrator vLLM server updated and ready"
        else
            echo "  ✗ Warning: Server may not be ready yet"
        fi
    else
        echo "No merged model found at $MERGED_MODEL"
        echo "Skipping vLLM update"
    fi
    
    # -------------------------------------------------------------------------
    # Step 5: Update History
    # -------------------------------------------------------------------------
    echo ""
    echo "Step 5: Updating Training History"
    echo "-------------------------------------------"
    
    python3 -c "
import json
from pathlib import Path

history_file = Path('$HISTORY_FILE')
if history_file.exists() and history_file.stat().st_size > 0:
    try:
        history = json.loads(history_file.read_text())
    except:
        history = []
else:
    history = []

history.append({
    'iteration': $ITER,
    'cypher_reward': $CYPHER_REWARD,
    'orch_qgen_reward': $QGEN_REWARD,
    'orch_synth_reward': $SYNTH_REWARD,
    'orch_avg_reward': $ORCH_AVG,
    'answerable_pct': $ANSWERABLE_PCT,
    'num_rollouts': $TOTAL_ROLLOUTS,
    'trained_orchestrator': True,
    'trained_cypher': False,
})

history_file.write_text(json.dumps(history, indent=2))
print(f'History updated: {len(history)} iterations')
"
    
    echo ""
    echo "Iteration $ITER complete!"
    echo ""
    
    # Check for early stopping (no improvement for 5 iterations)
    if [[ $CONSECUTIVE_NO_IMPROVEMENT -ge 5 ]]; then
        echo "=============================================="
        echo "EARLY STOPPING: No improvement for 5 iterations"
        echo "=============================================="
        break
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "ORCHESTRATOR-FOCUSED TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Final Metrics:"
echo "  Best Orchestrator Reward: $BEST_ORCH_REWARD"
echo "  Total Iterations: $ITER"
echo ""

# Print history summary
python3 -c "
import json
from pathlib import Path

history_file = Path('$HISTORY_FILE')
if history_file.exists():
    history = json.loads(history_file.read_text())
    
    print('Training Progress:')
    print('-' * 60)
    print(f'{'Iter':<6} {'Cypher':<10} {'Orch Avg':<10} {'Answerable %':<15}')
    print('-' * 60)
    
    for h in history:
        print(f\"{h['iteration']:<6} {h['cypher_reward']:<10.3f} {h['orch_avg_reward']:<10.3f} {h['answerable_pct']:<15.1f}\")
    
    # Check for improvement
    if len(history) >= 2:
        first_orch = history[0]['orch_avg_reward']
        last_orch = history[-1]['orch_avg_reward']
        first_ans = history[0]['answerable_pct']
        last_ans = history[-1]['answerable_pct']
        
        print()
        print('Improvement:')
        print(f'  Orchestrator Reward: {first_orch:.3f} → {last_orch:.3f} ({100*(last_orch-first_orch)/max(first_orch,0.001):.1f}% change)')
        print(f'  Answerable Questions: {first_ans:.1f}% → {last_ans:.1f}%')
"

echo ""
echo "Trained model: $PROJECT_DIR/outputs/orchestrator_trained/best_model_merged"
echo "Training logs: $LOG_DIR"
echo "History: $HISTORY_FILE"

