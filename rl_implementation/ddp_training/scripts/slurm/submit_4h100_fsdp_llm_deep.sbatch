#!/bin/bash
#SBATCH --job-name=pankllm_4h100_deep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_h100_deep_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_h100_deep_%j.err
#SBATCH --mail-user=rickyhan@umich.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# 4x H100 FSDP Training Pipeline with LLM Deep Think Controller
# =============================================================================
#
# This pipeline uses an LLM with DEEP THINK mode to make intelligent
# training decisions. When training is struggling (reward drops, oscillation),
# the LLM analyzes actual rollout data to identify root causes.
#
# H100 Configuration:
#   - 80GB VRAM per GPU (vs 48GB on L40S)
#   - Increased context length: 8192 tokens (vs 4096 on L40S)
#   - Higher memory allocation: 256GB RAM
#
# Deep Think Mode Options:
#   - auto: Trigger automatically when issues detected (default)
#   - always: Always perform deep analysis (for debugging)
#   - never: Disable deep think, use standard analysis
#
# Submit with:
#   # Standard deep think (auto-trigger)
#   sbatch --export=LLM_PROVIDER=claude,LLM_API_KEY=$ANTHROPIC_API_KEY,RUN_ID=h100_deep \
#       submit_4h100_fsdp_llm_deep.sbatch
#
#   # Force deep think every iteration
#   sbatch --export=LLM_PROVIDER=claude,LLM_API_KEY=$ANTHROPIC_API_KEY,RUN_ID=h100_debug,DEEP_THINK=always \
#       submit_4h100_fsdp_llm_deep.sbatch
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM 4x H100 FSDP + LLM Deep Think Controller"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

ITERATIONS="${ITERATIONS:-10}"
QUESTIONS="${QUESTIONS:-64}"
DIFFICULTY="${DIFFICULTY:-easy}"
DEFAULT_EPOCHS="${DEFAULT_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Coverage-based rollout collection
TARGET_FILTERED="${TARGET_FILTERED:-128}"
MIN_USABLE_REWARD="${MIN_USABLE_REWARD:-0.1}"

# Adaptive entity sampling
NO_ADAPTIVE_SAMPLING="${NO_ADAPTIVE_SAMPLING:-0}"

# Other options
SKIP_FIRST_COLLECT="${SKIP_FIRST_COLLECT:-0}"
CLEAR_HISTORY="${CLEAR_HISTORY:-0}"
FORCE_DEGREE_EXTRACTION="${FORCE_DEGREE_EXTRACTION:-0}"

# Prompt hints management
CLEAR_PROMPT_HINTS="${CLEAR_PROMPT_HINTS:-0}"

# Checkpoint management: keep only last N checkpoints (0 = keep all)
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"

# =============================================================================
# LLM Deep Think Controller Configuration
# =============================================================================

LLM_PROVIDER="${LLM_PROVIDER:-claude}"
LLM_API_KEY="${LLM_API_KEY:-}"

# Deep Think Mode: auto (trigger on issues), always, or never
DEEP_THINK="${DEEP_THINK:-auto}"

# Number of rollouts to sample for deep analysis
DEEP_THINK_SAMPLE_SIZE="${DEEP_THINK_SAMPLE_SIZE:-20}"

# Export API key to environment if provided
if [ -n "$LLM_API_KEY" ]; then
    if [ "$LLM_PROVIDER" = "claude" ]; then
        export ANTHROPIC_API_KEY="$LLM_API_KEY"
    elif [ "$LLM_PROVIDER" = "openai" ]; then
        export OPENAI_API_KEY="$LLM_API_KEY"
    fi
fi

# Validate LLM configuration
if [ "$LLM_PROVIDER" = "claude" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Claude selected but ANTHROPIC_API_KEY not set"
    echo "  Set with: --export=LLM_API_KEY=\$ANTHROPIC_API_KEY"
    exit 1
fi
if [ "$LLM_PROVIDER" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OpenAI selected but OPENAI_API_KEY not set"
    echo "  Set with: --export=LLM_API_KEY=\$OPENAI_API_KEY"
    exit 1
fi

echo "Configuration:"
echo "  GPUs: 4x H100 (80GB each)"
echo "  Training: FSDP FULL_SHARD (2 GPUs per model)"
echo "  Max Iterations: $ITERATIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Default Epochs: $DEFAULT_EPOCHS"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
if [ "$KEEP_LAST_CHECKPOINTS" -gt 0 ]; then
    echo "  Checkpoints: Keep last $KEEP_LAST_CHECKPOINTS iterations (~$((KEEP_LAST_CHECKPOINTS * 3))GB)"
else
    echo "  Checkpoints: Keep all (⚠️ ~2.6GB per iteration)"
fi
echo ""
echo "🧠 LLM Deep Think Controller:"
echo "  Provider: $LLM_PROVIDER"
echo "  Deep Think Mode: $DEEP_THINK"
echo "  Sample Size: $DEEP_THINK_SAMPLE_SIZE rollouts"
if [ "$LLM_PROVIDER" = "claude" ]; then
    echo "  Model: Claude Opus 4.5 (claude-opus-4-5-20251101)"
else
    echo "  Model: GPT-4o (gpt-4o)"
fi
if [ -n "$LLM_API_KEY" ]; then
    echo "  API Key: ****${LLM_API_KEY: -4}"
else
    echo "  API Key: (from environment)"
fi
echo ""

# =============================================================================
# H100-Specific vLLM Settings (LARGER CONTEXT for 80GB VRAM)
# =============================================================================

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"  # H100: 8192 (vs 4096 on L40S)
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"  # H100: higher utilization
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"  # H100: more concurrent sequences
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-65536}"  # H100: larger batch

echo "H100 vLLM Settings (GPUs 0-1 for inference):"
echo "  Max Model Length: $VLLM_MAX_MODEL_LEN (H100 extended context)"
echo "  GPU Memory Utilization: $VLLM_GPU_MEMORY_UTIL"
echo "  Max Num Seqs: $VLLM_MAX_NUM_SEQS"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Cache directories
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

mkdir -p /tmp/r /tmp/t
export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/t

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # H100: larger splits
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export TOKENIZERS_PARALLELISM=false

# FSDP/NCCL settings
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

ulimit -n 1048576 2>/dev/null || echo "Note: Could not increase file descriptor limit"

# =============================================================================
# PYTHONPATH Setup
# =============================================================================

# Hardcoded paths (rllm.__file__ returns None in namespace packages)
RLLM_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm"
PANKLLM_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"

SCRIPT_DIR="$PANKLLM_DIR/rl_implementation/ddp_training/scripts"
DDP_DIR="$PANKLLM_DIR/rl_implementation/ddp_training"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "RLLM Dir: $RLLM_DIR"
echo "PanKLLM Dir: $PANKLLM_DIR"
echo ""

# =============================================================================
# Output Directories
# =============================================================================

RUN_ID="${RUN_ID:-h100_deep_$SLURM_JOB_ID}"
ROLLOUTS_DIR="$PANKLLM_DIR/outputs/stage1_ddp/rollouts_run_${RUN_ID}"
LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"
HISTORY_FILE="$ROLLOUTS_DIR/training_history.json"
LLM_DECISIONS_LOG="$LOG_DIR/llm_decisions_${RUN_ID}.log"
DEEP_ANALYSIS_DIR="$ROLLOUTS_DIR/deep_analysis"

mkdir -p "$LOG_DIR"
mkdir -p "$ROLLOUTS_DIR"
mkdir -p "$DEEP_ANALYSIS_DIR"

export ROLLOUTS_DIR
export RUN_ID

echo "Run ID: $RUN_ID"
echo "Rollouts Dir: $ROLLOUTS_DIR"
echo "LLM Decisions Log: $LLM_DECISIONS_LOG"
echo "Deep Analysis Dir: $DEEP_ANALYSIS_DIR"
echo ""

# =============================================================================
# Model Paths
# =============================================================================

ORCHESTRATOR_BASE="$PANKLLM_DIR/models/qwen2.5-14b"
CYPHER_BASE="$PANKLLM_DIR/models/qwen2.5-coder-14b-sft"

ORCHESTRATOR_OUTPUT="$PANKLLM_DIR/outputs/orchestrator_trained_${RUN_ID}"
CYPHER_OUTPUT="$PANKLLM_DIR/outputs/cypher_trained_${RUN_ID}"
ORCHESTRATOR_TRAINED="$ORCHESTRATOR_OUTPUT/best_model_merged"
CYPHER_TRAINED="$CYPHER_OUTPUT/best_model_merged"

mkdir -p "$ORCHESTRATOR_OUTPUT"
mkdir -p "$CYPHER_OUTPUT"

ORCHESTRATOR_MODEL="$ORCHESTRATOR_BASE"
CYPHER_MODEL="$CYPHER_BASE"

echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""

# Verify models exist
if [ ! -d "$ORCHESTRATOR_BASE" ]; then
    echo "ERROR: Base Orchestrator model not found: $ORCHESTRATOR_BASE"
    exit 1
fi
if [ ! -d "$CYPHER_BASE" ]; then
    echo "ERROR: Base Cypher model not found: $CYPHER_BASE"
    exit 1
fi

# =============================================================================
# Ports
# =============================================================================

ORCHESTRATOR_PORT=8001
CYPHER_PORT=8002

# =============================================================================
# Entity Degrees Extraction
# =============================================================================

ENTITY_DEGREES_PATH="$PANKLLM_DIR/outputs/entity_degrees.json"

if [ ! -f "$ENTITY_DEGREES_PATH" ] || [ "${FORCE_DEGREE_EXTRACTION:-0}" = "1" ]; then
    echo "Extracting entity degrees from Neo4j..."
    python3 "$PANKLLM_DIR/rl_implementation/utils/degree_extractor.py" \
        --output "$ENTITY_DEGREES_PATH" \
        --timeout 180 || echo "Warning: Degree extraction failed"
fi
export ENTITY_DEGREES_PATH

# =============================================================================
# Clear History (if requested)
# =============================================================================

if [ "$CLEAR_HISTORY" = "1" ] && [ -f "$HISTORY_FILE" ]; then
    echo "Clearing training history..."
    rm -f "$HISTORY_FILE"
fi

# =============================================================================
# Clear Prompt Hints (if requested)
# =============================================================================

PROMPT_HINTS_FILE="$ROLLOUTS_DIR/prompt_hints.json"
if [ "$CLEAR_PROMPT_HINTS" = "1" ] && [ -f "$PROMPT_HINTS_FILE" ]; then
    echo "Clearing prompt hints..."
    rm -f "$PROMPT_HINTS_FILE"
fi

# =============================================================================
# Helper Functions
# =============================================================================

start_vllm_servers() {
    echo "Starting vLLM servers on GPUs 0-1 (H100 with extended context)..."
    
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
    
    echo "  Waiting for servers to be ready..."
    local max_attempts=150
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        orch_ready=$(curl -s "http://localhost:$ORCHESTRATOR_PORT/health" > /dev/null 2>&1 && echo "1" || echo "0")
        cypher_ready=$(curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1 && echo "1" || echo "0")
        
        if [ "$orch_ready" = "1" ] && [ "$cypher_ready" = "1" ]; then
            echo "  ✓ Both vLLM servers ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 5
    done
    
    echo "  ✗ vLLM servers failed to start"
    return 1
}

stop_vllm_servers() {
    echo "Stopping vLLM servers..."
    
    [ -n "$ORCH_PID" ] && kill $ORCH_PID 2>/dev/null || true
    [ -n "$CYPHER_PID" ] && kill $CYPHER_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    
    sleep 15
    echo "  ✓ vLLM servers stopped"
}

collect_rollouts() {
    local ITER=$1
    local ROLLOUTS_FILE="$ROLLOUTS_DIR/rollouts_iter_$(printf '%03d' $ITER).jsonl"
    
    echo "Collecting rollouts..." >&2
    
    local COLLECT_ARGS="--output $ROLLOUTS_FILE"
    COLLECT_ARGS="$COLLECT_ARGS --difficulty $DIFFICULTY"
    COLLECT_ARGS="$COLLECT_ARGS --batch-size $BATCH_SIZE"
    COLLECT_ARGS="$COLLECT_ARGS --orchestrator-port $ORCHESTRATOR_PORT"
    COLLECT_ARGS="$COLLECT_ARGS --cypher-port $CYPHER_PORT"
    
    if [ "$TARGET_FILTERED" -gt 0 ]; then
        COLLECT_ARGS="$COLLECT_ARGS --target-filtered-rollouts $TARGET_FILTERED"
        COLLECT_ARGS="$COLLECT_ARGS --min-usable-reward $MIN_USABLE_REWARD"
    else
        COLLECT_ARGS="$COLLECT_ARGS --num-questions $QUESTIONS"
    fi
    
    if [ "$NO_ADAPTIVE_SAMPLING" != "1" ]; then
        COLLECT_ARGS="$COLLECT_ARGS --use-adaptive-sampling"
        COLLECT_ARGS="$COLLECT_ARGS --entity-degrees $ENTITY_DEGREES_PATH"
        COLLECT_ARGS="$COLLECT_ARGS --adaptive-sampler-state $PANKLLM_DIR/outputs/adaptive_sampler_state.json"
    fi
    
    # Pass prompt hints file if it exists (from Deep Think analysis)
    local PROMPT_HINTS_FILE="$ROLLOUTS_DIR/prompt_hints.json"
    if [ -f "$PROMPT_HINTS_FILE" ]; then
        COLLECT_ARGS="$COLLECT_ARGS --prompt-hints $PROMPT_HINTS_FILE"
        echo "  📝 Loading prompt hints from: $PROMPT_HINTS_FILE" >&2
    fi
    
    python3 "$SCRIPT_DIR/collect_rollouts.py" $COLLECT_ARGS >&2
    
    echo "$ROLLOUTS_FILE"
}

analyze_and_update_history() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    
    python3 "$SCRIPT_DIR/training_decision.py" \
        --rollouts "$ROLLOUTS_FILE" \
        --history "$HISTORY_FILE" \
        --iteration "$ITER" \
        --update-history \
        --verbose
}

get_llm_decision_deep() {
    local ITER=$1
    local ROLLOUTS_FILE=$2
    local LLM_DECISION_FILE=$(mktemp)
    local ANALYSIS_FILE="$DEEP_ANALYSIS_DIR/deep_analysis_iter_$(printf '%03d' $ITER).md"
    
    # All status messages go to stderr (>&2) so they don't contaminate JSON output
    echo "" >&2
    echo "🧠 Querying LLM ($LLM_PROVIDER) with DEEP THINK mode..." >&2
    echo "   Iteration: $ITER" >&2
    echo "   History: $HISTORY_FILE" >&2
    echo "   Rollouts Dir: $ROLLOUTS_DIR" >&2
    echo "   Deep Think Mode: $DEEP_THINK" >&2
    echo "   Sample Size: $DEEP_THINK_SAMPLE_SIZE" >&2
    
    # Call Deep Think LLM controller
    python3 "$SCRIPT_DIR/llm_training_controller_deep.py" \
        --history "$HISTORY_FILE" \
        --rollouts-dir "$ROLLOUTS_DIR" \
        --provider "$LLM_PROVIDER" \
        --deep-think "$DEEP_THINK" \
        --sample-size "$DEEP_THINK_SAMPLE_SIZE" \
        --output "$LLM_DECISION_FILE" \
        --analysis-output "$ANALYSIS_FILE" \
        --fallback-on-error \
        --verbose >> "$LLM_DECISIONS_LOG" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ] && [ -f "$LLM_DECISION_FILE" ] && [ -s "$LLM_DECISION_FILE" ]; then
        # Log the decision
        echo "" >> "$LLM_DECISIONS_LOG"
        echo "=== Iteration $ITER Decision (Deep Think) ===" >> "$LLM_DECISIONS_LOG"
        echo "Timestamp: $(date)" >> "$LLM_DECISIONS_LOG"
        cat "$LLM_DECISION_FILE" >> "$LLM_DECISIONS_LOG"
        echo "" >> "$LLM_DECISIONS_LOG"
        
        # Check if deep analysis was generated
        if [ -f "$ANALYSIS_FILE" ]; then
            echo "   📄 Deep Analysis Report: $ANALYSIS_FILE" >&2
        fi
        
        # Output ONLY the JSON to stdout
        cat "$LLM_DECISION_FILE"
        rm -f "$LLM_DECISION_FILE"
    else
        echo "ERROR: LLM controller failed (exit code: $EXIT_CODE)" >&2
        echo '{"decision": {"train_cypher": true, "train_orchestrator": true, "cypher_epochs": '$DEFAULT_EPOCHS', "orchestrator_epochs": '$DEFAULT_EPOCHS', "reasoning": "LLM failed, using default"}, "confidence": 0.0, "deep_think_mode": false}'
        rm -f "$LLM_DECISION_FILE" 2>/dev/null || true
    fi
}

train_cypher_fsdp() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local NUM_EPOCHS=${3:-$DEFAULT_EPOCHS}
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_cypher_fsdp.log"
    
    echo "Training Cypher Generator with FSDP on GPUs 0-1..."
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Output: $CYPHER_OUTPUT"
    
    # Use H100-optimized config if available, otherwise fall back to L40S config
    local CONFIG_FILE="$DDP_DIR/config/train_cypher_config_fsdp_h100.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        CONFIG_FILE="$DDP_DIR/config/train_cypher_config_fsdp_l40s.yaml"
        echo "  Note: Using L40S config (H100 config not found)"
    fi
    
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        "$SCRIPT_DIR/train_cypher_from_rollouts.py" \
        --config "$CONFIG_FILE" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$NUM_EPOCHS" \
        --checkpoint-dir "$CYPHER_OUTPUT" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Cypher training completed ($NUM_EPOCHS epochs)"
        [ -d "$CYPHER_TRAINED" ] && CYPHER_MODEL="$CYPHER_TRAINED"
        return 0
    else
        echo "  ✗ Cypher training failed (check $LOG_FILE)"
        return 1
    fi
}

train_orchestrator_fsdp() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local NUM_EPOCHS=${3:-$DEFAULT_EPOCHS}
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_orchestrator_fsdp.log"
    
    echo "Training Orchestrator with FSDP on GPUs 2-3..."
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Output: $ORCHESTRATOR_OUTPUT"
    
    # Use H100-optimized config if available, otherwise fall back to L40S config
    local CONFIG_FILE="$DDP_DIR/config/train_orchestrator_config_fsdp_h100.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        CONFIG_FILE="$DDP_DIR/config/train_orchestrator_config_fsdp_l40s.yaml"
        echo "  Note: Using L40S config (H100 config not found)"
    fi
    
    CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        --master_port=29501 \
        "$SCRIPT_DIR/train_orchestrator_from_rollouts.py" \
        --config "$CONFIG_FILE" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$NUM_EPOCHS" \
        --checkpoint-dir "$ORCHESTRATOR_OUTPUT" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Orchestrator training completed ($NUM_EPOCHS epochs)"
        [ -d "$ORCHESTRATOR_TRAINED" ] && ORCHESTRATOR_MODEL="$ORCHESTRATOR_TRAINED"
        return 0
    else
        echo "  ✗ Orchestrator training failed (check $LOG_FILE)"
        return 1
    fi
}

# =============================================================================
# Cleanup
# =============================================================================

cleanup() {
    echo ""
    echo "Cleaning up..."
    stop_vllm_servers
    echo "Cleanup complete"
}

trap cleanup EXIT

# =============================================================================
# GPU Check
# =============================================================================

echo "Checking GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.device_count() < 4:
    print('ERROR: Need 4 GPUs!')
    exit(1)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f'  GPU {i}: {props.name}, {mem_gb:.1f} GB')
"
echo ""

# =============================================================================
# Main Training Loop with LLM Deep Think Controller
# =============================================================================

echo ""
echo "=============================================="
echo "Starting LLM Deep Think Controlled Training (H100)"
echo "=============================================="
echo ""

# Initialize LLM decisions log
echo "LLM Deep Think Training Controller Decisions Log (H100)" > "$LLM_DECISIONS_LOG"
echo "Run ID: $RUN_ID" >> "$LLM_DECISIONS_LOG"
echo "Provider: $LLM_PROVIDER" >> "$LLM_DECISIONS_LOG"
echo "Deep Think Mode: $DEEP_THINK" >> "$LLM_DECISIONS_LOG"
echo "Max Model Length: $VLLM_MAX_MODEL_LEN (H100 extended)" >> "$LLM_DECISIONS_LOG"
echo "Started: $(date)" >> "$LLM_DECISIONS_LOG"
echo "========================================" >> "$LLM_DECISIONS_LOG"
echo "" >> "$LLM_DECISIONS_LOG"

TOTAL_CYPHER_TRAINS=0
TOTAL_ORCH_TRAINS=0
TOTAL_SKIPPED=0
TOTAL_DEEP_THINKS=0

for ITER in $(seq 1 $ITERATIONS); do
    echo ""
    echo "=============================================="
    echo "ITERATION $ITER / $ITERATIONS"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    # =========================================================================
    # Phase 1: Collect Rollouts
    # =========================================================================
    
    if [ "$SKIP_FIRST_COLLECT" = "1" ] && [ $ITER -eq 1 ]; then
        echo "Skipping first rollout collection..."
        LATEST_ROLLOUTS=$(ls -t "$ROLLOUTS_DIR"/rollouts_*.jsonl 2>/dev/null | head -1)
        if [ -z "$LATEST_ROLLOUTS" ]; then
            echo "ERROR: No existing rollouts found!"
            exit 1
        fi
        ROLLOUTS_FILE="$LATEST_ROLLOUTS"
    else
        echo "=== PHASE 1: INFERENCE ==="
        start_vllm_servers || exit 1
        ROLLOUTS_FILE=$(collect_rollouts $ITER)
        stop_vllm_servers
    fi
    
    echo ""
    
    if [ ! -f "$ROLLOUTS_FILE" ]; then
        echo "ERROR: Rollouts file not found: $ROLLOUTS_FILE"
        exit 1
    fi
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_FILE")
    echo "Rollouts collected: $ROLLOUT_COUNT entries"
    echo ""
    
    # =========================================================================
    # Phase 2: Analyze and Update History
    # =========================================================================
    
    echo "Analyzing rollouts and updating history..."
    analyze_and_update_history "$ROLLOUTS_FILE" $ITER
    echo ""
    
    # =========================================================================
    # Phase 3: LLM Deep Think Training Decision
    # =========================================================================
    
    echo "=== PHASE 2: LLM DEEP THINK DECISION ==="
    
    LLM_DECISION=$(get_llm_decision_deep $ITER "$ROLLOUTS_FILE")
    LLM_DECISION_FILE=$(mktemp)
    echo "$LLM_DECISION" > "$LLM_DECISION_FILE"
    
    # Debug: Check if we got valid JSON
    if ! echo "$LLM_DECISION" | head -c 1 | grep -q '{'; then
        echo "WARNING: LLM decision does not start with '{'" >&2
        echo "First 200 chars: ${LLM_DECISION:0:200}" >&2
    fi
    
    # Extract and save prompt hints if present (NEW: Deep Think prompt hints feature)
    PROMPT_HINTS_FILE="$ROLLOUTS_DIR/prompt_hints.json"
    python3 -c "
import json
import sys
sys.path.insert(0, '$PANKLLM_DIR')

try:
    with open('$LLM_DECISION_FILE', 'r') as f:
        content = f.read()
    
    d = json.loads(content)
    prompt_hints = d.get('prompt_hints', {})
    
    if prompt_hints:
        # Use the PromptHintsManager to accumulate hints
        from rl_implementation.utils.prompt_hints_manager import PromptHintsManager
        manager = PromptHintsManager('$PROMPT_HINTS_FILE')
        manager.add_hints(prompt_hints, iteration=$ITER)
        
        # Count hints added
        cg_count = len(prompt_hints.get('cypher_generator', []))
        orch_hints = prompt_hints.get('orchestrator', {})
        orch_count = sum(len(v) for v in orch_hints.values() if isinstance(v, list))
        
        print(f'  📝 Saved prompt hints: {cg_count} for Cypher, {orch_count} for Orchestrator', file=sys.stderr)
    else:
        print('  (No prompt hints from Deep Think this iteration)', file=sys.stderr)
        
except Exception as e:
    print(f'  WARNING: Failed to save prompt hints: {e}', file=sys.stderr)
"
    
    # Parse LLM decision (including deep think fields)
    PARSED=$(python3 -c "
import json
import sys

try:
    with open('$LLM_DECISION_FILE', 'r') as f:
        content = f.read()
    
    d = json.loads(content)
    decision = d.get('decision', {})
    analysis = d.get('analysis', {})
    early_stop = d.get('early_stopping', {})
    rollback = d.get('rollback', {})
    deep_analysis = d.get('deep_analysis', {})
    
    train_cypher = 'true' if decision.get('train_cypher') else 'false'
    train_orch = 'true' if decision.get('train_orchestrator') else 'false'
    cypher_epochs = int(decision.get('cypher_epochs', $DEFAULT_EPOCHS))
    orch_epochs = int(decision.get('orchestrator_epochs', $DEFAULT_EPOCHS))
    reasoning = decision.get('reasoning', 'N/A')[:100]
    confidence = d.get('confidence', 0.0)
    
    cypher_gap = analysis.get('cypher_gap_percent', 0)
    orch_gap = analysis.get('orch_gap_percent', 0)
    health = analysis.get('training_health', 'unknown')
    
    recommend_stop = 'true' if early_stop.get('recommend_stop') else 'false'
    stop_reason = early_stop.get('reason', 'N/A')[:80]
    recommend_rollback = 'true' if rollback.get('recommend_rollback') else 'false'
    rollback_iter = rollback.get('target_iteration', 0) or 0
    
    # Deep think specific fields
    deep_think_mode = 'true' if d.get('deep_think_mode') else 'false'
    trigger_reason = d.get('trigger_reason', 'N/A')
    root_cause = deep_analysis.get('root_cause', 'N/A')[:60]
    
    print(f'{train_cypher}|{train_orch}|{cypher_epochs}|{orch_epochs}|{reasoning}|{confidence}|{cypher_gap}|{orch_gap}|{recommend_stop}|{stop_reason}|{recommend_rollback}|{rollback_iter}|{health}|{deep_think_mode}|{trigger_reason}|{root_cause}')
except Exception as e:
    print(f'ERROR parsing LLM decision: {e}', file=sys.stderr)
    print(f'false|false|0|0|Parse error: {e}|0.0|0|0|false|N/A|false|0|error|false|error|N/A')
")
    
    rm -f "$LLM_DECISION_FILE"
    
    TRAIN_CYPHER=$(echo "$PARSED" | cut -d'|' -f1)
    TRAIN_ORCH=$(echo "$PARSED" | cut -d'|' -f2)
    CYPHER_EPOCHS=$(echo "$PARSED" | cut -d'|' -f3)
    ORCH_EPOCHS=$(echo "$PARSED" | cut -d'|' -f4)
    LLM_REASONING=$(echo "$PARSED" | cut -d'|' -f5)
    LLM_CONFIDENCE=$(echo "$PARSED" | cut -d'|' -f6)
    CYPHER_GAP=$(echo "$PARSED" | cut -d'|' -f7)
    ORCH_GAP=$(echo "$PARSED" | cut -d'|' -f8)
    RECOMMEND_STOP=$(echo "$PARSED" | cut -d'|' -f9)
    STOP_REASON=$(echo "$PARSED" | cut -d'|' -f10)
    RECOMMEND_ROLLBACK=$(echo "$PARSED" | cut -d'|' -f11)
    ROLLBACK_ITER=$(echo "$PARSED" | cut -d'|' -f12)
    TRAINING_HEALTH=$(echo "$PARSED" | cut -d'|' -f13)
    DEEP_THINK_ACTIVE=$(echo "$PARSED" | cut -d'|' -f14)
    TRIGGER_REASON=$(echo "$PARSED" | cut -d'|' -f15)
    ROOT_CAUSE=$(echo "$PARSED" | cut -d'|' -f16)
    
    echo ""
    echo "📋 LLM Deep Think Decision Summary:"
    echo "  ┌─────────────────────────────────────────────────────────────────────┐"
    if [ "$DEEP_THINK_ACTIVE" = "true" ]; then
        echo "  │ 🔍 DEEP THINK MODE ACTIVE"
        echo "  │ Triggered By: $TRIGGER_REASON"
        echo "  │ Root Cause: $ROOT_CAUSE"
        echo "  │"
        TOTAL_DEEP_THINKS=$((TOTAL_DEEP_THINKS + 1))
    fi
    echo "  │ Train Cypher:      $TRAIN_CYPHER ($CYPHER_EPOCHS epochs) - gap: ${CYPHER_GAP}%"
    echo "  │ Train Orchestrator: $TRAIN_ORCH ($ORCH_EPOCHS epochs) - gap: ${ORCH_GAP}%"
    echo "  │ Confidence: $LLM_CONFIDENCE | Health: $TRAINING_HEALTH"
    echo "  │ Reasoning: $LLM_REASONING"
    echo "  └─────────────────────────────────────────────────────────────────────┘"
    
    # Log the enforcement decision
    echo ""
    echo "🎯 ENFORCING LLM DECISION:"
    if [ "$TRAIN_CYPHER" = "true" ]; then
        echo "  ✓ Will train Cypher Generator for $CYPHER_EPOCHS epochs"
    else
        echo "  ✗ Will SKIP Cypher Generator training"
    fi
    if [ "$TRAIN_ORCH" = "true" ]; then
        echo "  ✓ Will train Orchestrator for $ORCH_EPOCHS epochs"
    else
        echo "  ✗ Will SKIP Orchestrator training"
    fi
    
    # Handle early stopping recommendation
    SHOULD_STOP=false
    if [ "$RECOMMEND_STOP" = "true" ]; then
        echo ""
        echo "  🛑 EARLY STOPPING RECOMMENDED BY LLM"
        echo "  │ Reason: $STOP_REASON"
        echo "  │ Health: $TRAINING_HEALTH"
        
        if [ "$TRAINING_HEALTH" = "critical" ]; then
            echo "  │ ❌ Training health is CRITICAL - STOPPING TRAINING LOOP"
            SHOULD_STOP=true
        else
            echo "  │ ⚠️ Continuing with caution (health not critical)"
        fi
    fi
    
    # Handle rollback recommendation
    if [ "$RECOMMEND_ROLLBACK" = "true" ] && [ -n "$ROLLBACK_ITER" ] && [ "$ROLLBACK_ITER" != "0" ]; then
        echo ""
        echo "  🔄 ROLLBACK RECOMMENDED BY LLM"
        echo "  │ Target iteration: $ROLLBACK_ITER"
        
        ROLLBACK_CKPT_DIR="$ROLLOUTS_DIR/checkpoints_iter_$(printf '%03d' $ROLLBACK_ITER)"
        ROLLBACK_CYPHER_CKPT="$ROLLBACK_CKPT_DIR/cypher_best_model"
        ROLLBACK_ORCH_CKPT="$ROLLBACK_CKPT_DIR/orchestrator_best_model"
        
        if [ -d "$ROLLBACK_CYPHER_CKPT" ] || [ -d "$ROLLBACK_ORCH_CKPT" ]; then
            echo "  │ ✓ Found checkpoints from iteration $ROLLBACK_ITER"
            echo "  │ To rollback, run:"
            if [ -d "$ROLLBACK_CYPHER_CKPT" ]; then
                echo "  │   cp -r $ROLLBACK_CYPHER_CKPT $CYPHER_OUTPUT/best_model"
            fi
            if [ -d "$ROLLBACK_ORCH_CKPT" ]; then
                echo "  │   cp -r $ROLLBACK_ORCH_CKPT $ORCHESTRATOR_OUTPUT/best_model"
            fi
        else
            echo "  │ ⚠️ Checkpoints not found for iteration $ROLLBACK_ITER"
        fi
    fi
    echo ""
    
    # If early stopping was triggered, break out of the training loop
    if [ "$SHOULD_STOP" = "true" ]; then
        echo ""
        echo "=============================================="
        echo "🛑 EARLY STOPPING TRIGGERED BY LLM CONTROLLER"
        echo "=============================================="
        echo "The LLM has determined that continued training is counterproductive."
        echo "Best results were likely achieved in an earlier iteration."
        if [ "$DEEP_THINK_ACTIVE" = "true" ]; then
            echo ""
            echo "Deep Think Analysis identified root cause: $ROOT_CAUSE"
            echo "See analysis report: $DEEP_ANALYSIS_DIR/deep_analysis_iter_$(printf '%03d' $ITER).md"
        fi
        echo ""
        break
    fi
    
    # =========================================================================
    # Phase 4: FSDP Training (based on LLM decision)
    # =========================================================================
    
    CYPHER_TRAINED_THIS_ITER=false
    ORCH_TRAINED_THIS_ITER=false
    
    if [ "$TRAIN_CYPHER" = "true" ] || [ "$TRAIN_ORCH" = "true" ]; then
        echo "=== PHASE 3: FSDP TRAINING ==="
        
        if [ "$TRAIN_CYPHER" = "true" ] && [ "$TRAIN_ORCH" = "true" ]; then
            echo "Training BOTH models in parallel..."
            
            train_cypher_fsdp "$ROLLOUTS_FILE" $ITER "$CYPHER_EPOCHS" &
            CYPHER_TRAIN_PID=$!
            
            train_orchestrator_fsdp "$ROLLOUTS_FILE" $ITER "$ORCH_EPOCHS" &
            ORCH_TRAIN_PID=$!
            
            wait $CYPHER_TRAIN_PID && CYPHER_TRAINED_THIS_ITER=true
            wait $ORCH_TRAIN_PID && ORCH_TRAINED_THIS_ITER=true
            
        elif [ "$TRAIN_CYPHER" = "true" ]; then
            train_cypher_fsdp "$ROLLOUTS_FILE" $ITER "$CYPHER_EPOCHS" && CYPHER_TRAINED_THIS_ITER=true
            
        elif [ "$TRAIN_ORCH" = "true" ]; then
            train_orchestrator_fsdp "$ROLLOUTS_FILE" $ITER "$ORCH_EPOCHS" && ORCH_TRAINED_THIS_ITER=true
        fi
        
        [ "$CYPHER_TRAINED_THIS_ITER" = true ] && TOTAL_CYPHER_TRAINS=$((TOTAL_CYPHER_TRAINS + 1))
        [ "$ORCH_TRAINED_THIS_ITER" = true ] && TOTAL_ORCH_TRAINS=$((TOTAL_ORCH_TRAINS + 1))
        
        # Save per-iteration checkpoints for potential rollback
        ITER_CKPT_DIR="$ROLLOUTS_DIR/checkpoints_iter_$(printf '%03d' $ITER)"
        mkdir -p "$ITER_CKPT_DIR"
        
        if [ "$CYPHER_TRAINED_THIS_ITER" = true ] && [ -d "$CYPHER_OUTPUT/best_model" ]; then
            echo "  Saving Cypher checkpoint for iteration $ITER..."
            cp -r "$CYPHER_OUTPUT/best_model" "$ITER_CKPT_DIR/cypher_best_model" 2>/dev/null || true
        fi
        
        if [ "$ORCH_TRAINED_THIS_ITER" = true ] && [ -d "$ORCHESTRATOR_OUTPUT/best_model" ]; then
            echo "  Saving Orchestrator checkpoint for iteration $ITER..."
            cp -r "$ORCHESTRATOR_OUTPUT/best_model" "$ITER_CKPT_DIR/orchestrator_best_model" 2>/dev/null || true
        fi
        
        # Clean up old checkpoints to save space (keep only last N)
        if [ "$KEEP_LAST_CHECKPOINTS" -gt 0 ] && [ $ITER -gt $KEEP_LAST_CHECKPOINTS ]; then
            OLD_ITER=$((ITER - KEEP_LAST_CHECKPOINTS))
            OLD_CKPT_DIR="$ROLLOUTS_DIR/checkpoints_iter_$(printf '%03d' $OLD_ITER)"
            if [ -d "$OLD_CKPT_DIR" ]; then
                echo "  🗑️ Removing old checkpoint: iter $OLD_ITER (keeping last $KEEP_LAST_CHECKPOINTS)"
                rm -rf "$OLD_CKPT_DIR"
            fi
        fi
        
        echo ""
    else
        echo "⏸️  LLM decided to SKIP training this iteration (stabilization)"
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
    fi
    
    # Update history with training details
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
    history[-1]['actually_trained'] = {
        'cypher': $( [ "$CYPHER_TRAINED_THIS_ITER" = true ] && echo 'True' || echo 'False' ),
        'orchestrator': $( [ "$ORCH_TRAINED_THIS_ITER" = true ] && echo 'True' || echo 'False' ),
    }
    history[-1]['training_type'] = 'fsdp'
    history[-1]['training_strategy'] = 'llm_deep_think_$LLM_PROVIDER'
    history[-1]['gpu_type'] = 'h100'
    history[-1]['llm_decision'] = {
        'cypher_epochs': $CYPHER_EPOCHS if $( [ "$CYPHER_TRAINED_THIS_ITER" = true ] && echo 'True' || echo 'False' ) else 0,
        'orchestrator_epochs': $ORCH_EPOCHS if $( [ "$ORCH_TRAINED_THIS_ITER" = true ] && echo 'True' || echo 'False' ) else 0,
        'confidence': $LLM_CONFIDENCE,
        'reasoning': '''$LLM_REASONING''',
        'deep_think_active': $( [ "$DEEP_THINK_ACTIVE" = true ] && echo 'True' || echo 'False' ),
        'trigger_reason': '''$TRIGGER_REASON''',
        'root_cause': '''$ROOT_CAUSE''',
    }
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
"
    
    echo ""
    echo "✓ Iteration $ITER / $ITERATIONS complete!"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "LLM Deep Think Controlled Pipeline Complete! (H100)"
echo "=============================================="
echo "End Time: $(date)"
echo ""

echo "Training Summary:"
echo "  Total Iterations: $ITERATIONS"
echo "  Cypher trained: $TOTAL_CYPHER_TRAINS times"
echo "  Orchestrator trained: $TOTAL_ORCH_TRAINS times"
echo "  Skipped (stabilization): $TOTAL_SKIPPED times"
echo "  Deep Think activations: $TOTAL_DEEP_THINKS"
echo ""

echo "LLM Deep Think Controller:"
echo "  Provider: $LLM_PROVIDER"
echo "  Mode: $DEEP_THINK"
echo "  Decisions logged: $LLM_DECISIONS_LOG"
echo "  Analysis reports: $DEEP_ANALYSIS_DIR/"
echo ""

echo "H100 Configuration:"
echo "  Max Model Length: $VLLM_MAX_MODEL_LEN tokens (extended context)"
echo "  GPU Memory Util: $VLLM_GPU_MEMORY_UTIL"
echo ""

echo "Results:"
echo "  Cypher: $CYPHER_OUTPUT/best_model"
echo "  Orchestrator: $ORCHESTRATOR_OUTPUT/best_model"
echo ""

if [ -f "$HISTORY_FILE" ]; then
    echo "Final Metrics:"
    python3 -c "
import json
from pathlib import Path

history_file = '$HISTORY_FILE'
path = Path(history_file)
if path.exists() and path.stat().st_size > 0:
    history = json.load(open(history_file))
    if history:
        last = history[-1]
        metrics = last.get('metrics', {})
        print(f\"  Cypher Reward: {metrics.get('cypher_reward', 0):.3f}\")
        print(f\"  Orch Reward: {metrics.get('orch_avg_reward', 0):.3f}\")
        
        # Find best
        best_cypher = max(h['metrics'].get('cypher_reward', 0) for h in history)
        best_orch = max(h['metrics'].get('orch_avg_reward', 0) for h in history)
        print(f\"  Best Cypher: {best_cypher:.3f}\")
        print(f\"  Best Orch: {best_orch:.3f}\")
        
        # Count deep think activations
        deep_thinks = sum(1 for h in history if h.get('llm_decision', {}).get('deep_think_active'))
        print(f\"  Deep Think Sessions: {deep_thinks}\")
"
fi

echo ""
echo "✓ LLM Deep Think Controlled Pipeline finished! (H100)"

