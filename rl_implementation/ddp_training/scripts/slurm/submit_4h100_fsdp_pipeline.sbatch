#!/bin/bash
#SBATCH --job-name=pankllm_4h100_fsdp
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_4h100_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_4h100_%j.err
#SBATCH --mail-user=rickyhan@umich.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# 4x H100 FSDP Training Pipeline (80GB VRAM each)
# =============================================================================
#
# Uses FSDP (Fully Sharded Data Parallel) for memory-efficient distributed training.
# H100's 80GB VRAM allows larger batch sizes and faster training compared to L40S.
#
# Resource Allocation:
#   - INFERENCE: 2 vLLM servers on GPUs 0-1 (time-shared with training)
#   - TRAINING: Cypher uses FSDP on GPUs 0-1, Orchestrator uses FSDP on GPUs 2-3
#               (both can train in parallel since they use separate GPU pairs)
#
# Memory per GPU with FSDP FULL_SHARD (2 GPUs):
#   - Model (14B bf16): 28GB → 14GB per GPU (sharded)
#   - Optimizer states: 56GB → 28GB per GPU (sharded)
#   - Gradients: 28GB → 14GB per GPU (sharded)
#   - Activations (batch=32, seq=4096): ~15-20GB
#   - Total per GPU: ~55-65GB (fits comfortably in 80GB)
#
# Submit with:
#   sbatch submit_4h100_fsdp_pipeline.sbatch
#   sbatch --export=ITERATIONS=10,RUN_ID=h100_fsdp_exp submit_4h100_fsdp_pipeline.sbatch
#
# =============================================================================

set -e

echo "=============================================="
echo "PanKLLM 4x H100 FSDP Training Pipeline"
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
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"  # Batch size per GPU for FSDP

# Coverage-based rollout collection
TARGET_FILTERED="${TARGET_FILTERED:-128}"
MIN_USABLE_REWARD="${MIN_USABLE_REWARD:-0.1}"

# Adaptive entity sampling
NO_ADAPTIVE_SAMPLING="${NO_ADAPTIVE_SAMPLING:-0}"

# Other options
SKIP_FIRST_COLLECT="${SKIP_FIRST_COLLECT:-0}"
CLEAR_HISTORY="${CLEAR_HISTORY:-0}"
FORCE_DEGREE_EXTRACTION="${FORCE_DEGREE_EXTRACTION:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

# =============================================================================
# Training Strategy Options
# =============================================================================
# TRAINING_STRATEGY: 
#   - "default": Use threshold-based training decision (original behavior)
#   - "force": Force train both models every iteration (same as FORCE_TRAIN=1)
#   - "llm": Use LLM (Claude/OpenAI) to make intelligent training decisions
#
# LLM_PROVIDER: "claude" or "openai" (only used when TRAINING_STRATEGY=llm)
# LLM_API_KEY: API key for the selected provider (or set ANTHROPIC_API_KEY/OPENAI_API_KEY)
#
# Example usage:
#   sbatch --export=TRAINING_STRATEGY=llm,LLM_PROVIDER=claude,LLM_API_KEY=$ANTHROPIC_API_KEY ...
#   sbatch --export=TRAINING_STRATEGY=llm,LLM_PROVIDER=openai,LLM_API_KEY=$OPENAI_API_KEY ...
# =============================================================================

TRAINING_STRATEGY="${TRAINING_STRATEGY:-default}"
LLM_PROVIDER="${LLM_PROVIDER:-claude}"
LLM_API_KEY="${LLM_API_KEY:-}"

# Export API key to environment if provided
if [ -n "$LLM_API_KEY" ]; then
    if [ "$LLM_PROVIDER" = "claude" ]; then
        export ANTHROPIC_API_KEY="$LLM_API_KEY"
    elif [ "$LLM_PROVIDER" = "openai" ]; then
        export OPENAI_API_KEY="$LLM_API_KEY"
    fi
fi

echo "Configuration:"
echo "  GPUs: 4x H100 (80GB each)"
echo "  Training: FSDP FULL_SHARD (2 GPUs per model)"
echo "  Max Iterations: $ITERATIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Epochs/Iter: $EPOCHS (default, may be overridden by LLM)"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo ""
echo "Training Strategy: $TRAINING_STRATEGY"
if [ "$TRAINING_STRATEGY" = "llm" ]; then
    echo "  LLM Provider: $LLM_PROVIDER"
    if [ -n "$LLM_API_KEY" ]; then
        echo "  API Key: ****${LLM_API_KEY: -4}"  # Show last 4 chars only
    else
        echo "  API Key: (using environment variable)"
    fi
fi
echo ""

# =============================================================================
# H100-Specific vLLM Settings (80GB allows larger settings)
# =============================================================================

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-65536}"

echo "H100 vLLM Settings (GPUs 0-1 for inference):"
echo "  Max Model Length: $VLLM_MAX_MODEL_LEN"
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
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
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

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
PANKLLM_DIR="$RLLM_DIR/examples/PanKLLM_RL_post-training"
SCRIPT_DIR="$PANKLLM_DIR/rl_implementation/ddp_training/scripts"
DDP_DIR="$PANKLLM_DIR/rl_implementation/ddp_training"

export PYTHONPATH="$RLLM_DIR:$PANKLLM_DIR:${PYTHONPATH:-}"

echo "RLLM Dir: $RLLM_DIR"
echo "PanKLLM Dir: $PANKLLM_DIR"
echo ""

# =============================================================================
# Output Directories
# =============================================================================

RUN_ID="${RUN_ID:-h100_fsdp_$SLURM_JOB_ID}"
ROLLOUTS_DIR="$PANKLLM_DIR/outputs/stage1_ddp/rollouts_run_${RUN_ID}"
LOG_DIR="$PANKLLM_DIR/outputs/stage1_ddp/logs"
HISTORY_FILE="$ROLLOUTS_DIR/training_history.json"

mkdir -p "$LOG_DIR"
mkdir -p "$ROLLOUTS_DIR"

export ROLLOUTS_DIR
export RUN_ID

echo "Run ID: $RUN_ID"
echo "Rollouts Dir: $ROLLOUTS_DIR"
echo ""

# =============================================================================
# Model Paths
# =============================================================================

ORCHESTRATOR_BASE="$PANKLLM_DIR/models/qwen2.5-14b"
CYPHER_BASE="$PANKLLM_DIR/models/qwen2.5-coder-14b"

# Experiment-specific trained model directories
ORCHESTRATOR_OUTPUT="$PANKLLM_DIR/outputs/orchestrator_trained_${RUN_ID}"
CYPHER_OUTPUT="$PANKLLM_DIR/outputs/cypher_trained_${RUN_ID}"
ORCHESTRATOR_TRAINED="$ORCHESTRATOR_OUTPUT/best_model_merged"
CYPHER_TRAINED="$CYPHER_OUTPUT/best_model_merged"

mkdir -p "$ORCHESTRATOR_OUTPUT"
mkdir -p "$CYPHER_OUTPUT"

# Start from base models
ORCHESTRATOR_MODEL="$ORCHESTRATOR_BASE"
CYPHER_MODEL="$CYPHER_BASE"

echo "Models:"
echo "  Orchestrator: $ORCHESTRATOR_MODEL"
echo "  Cypher: $CYPHER_MODEL"
echo ""
echo "Training Outputs (FSDP):"
echo "  Cypher: $CYPHER_OUTPUT"
echo "  Orchestrator: $ORCHESTRATOR_OUTPUT"
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
# Helper Functions
# =============================================================================

start_vllm_servers() {
    echo "Starting vLLM servers on GPUs 0-1..."
    
    # Start Orchestrator on GPU 0
    echo "  Starting Orchestrator vLLM on GPU 0..."
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
    
    # Start Cypher Generator on GPU 1
    echo "  Starting Cypher vLLM on GPU 1..."
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
    
    # Wait for servers (H100 loads faster)
    echo "  Waiting for servers to be ready..."
    local max_attempts=120
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        orch_ready=$(curl -s "http://localhost:$ORCHESTRATOR_PORT/health" > /dev/null 2>&1 && echo "1" || echo "0")
        cypher_ready=$(curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1 && echo "1" || echo "0")
        
        if [ "$orch_ready" = "1" ] && [ "$cypher_ready" = "1" ]; then
            echo "  ✓ Both vLLM servers ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 3
    done
    
    echo "  ✗ vLLM servers failed to start"
    echo "  Check logs: $LOG_DIR/vllm_*.log"
    return 1
}

stop_vllm_servers() {
    echo "Stopping vLLM servers to free GPUs for FSDP training..."
    
    if [ -n "$ORCH_PID" ] && kill -0 $ORCH_PID 2>/dev/null; then
        kill $ORCH_PID 2>/dev/null || true
        wait $ORCH_PID 2>/dev/null || true
    fi
    
    if [ -n "$CYPHER_PID" ] && kill -0 $CYPHER_PID 2>/dev/null; then
        kill $CYPHER_PID 2>/dev/null || true
        wait $CYPHER_PID 2>/dev/null || true
    fi
    
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    
    # Wait for GPU memory to be freed
    sleep 10
    
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
    
    python3 "$SCRIPT_DIR/collect_rollouts.py" $COLLECT_ARGS >&2
    
    echo "$ROLLOUTS_FILE"
}

analyze_and_decide() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    
    python3 "$SCRIPT_DIR/training_decision.py" \
        --rollouts "$ROLLOUTS_FILE" \
        --history "$HISTORY_FILE" \
        --iteration "$ITER" \
        --update-history \
        --verbose
}

train_cypher_fsdp() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_cypher_fsdp.log"
    
    echo "Training Cypher Generator with FSDP on GPUs 0-1..."
    echo "  Output: $CYPHER_OUTPUT"
    echo "  Config: FSDP FULL_SHARD (2 GPUs, H100 80GB)"
    
    # Use torchrun for FSDP distributed training
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        "$SCRIPT_DIR/train_cypher_from_rollouts.py" \
        --config "$DDP_DIR/config/train_cypher_config_fsdp_h100.yaml" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$EPOCHS" \
        --checkpoint-dir "$CYPHER_OUTPUT" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Cypher FSDP training completed"
        if [ -d "$CYPHER_TRAINED" ]; then
            CYPHER_MODEL="$CYPHER_TRAINED"
            echo "  Using trained model: $CYPHER_TRAINED"
        fi
        return 0
    else
        echo "  ✗ Cypher FSDP training failed (check $LOG_FILE)"
        return 1
    fi
}

train_orchestrator_fsdp() {
    local ROLLOUTS_FILE=$1
    local ITER=$2
    local LOG_FILE="$LOG_DIR/iter_$(printf '%03d' $ITER)_orchestrator_fsdp.log"
    
    echo "Training Orchestrator with FSDP on GPUs 2-3..."
    echo "  Output: $ORCHESTRATOR_OUTPUT"
    echo "  Config: FSDP FULL_SHARD (2 GPUs, H100 80GB)"
    
    # Use torchrun for FSDP distributed training
    CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        --master_port=29501 \
        "$SCRIPT_DIR/train_orchestrator_from_rollouts.py" \
        --config "$DDP_DIR/config/train_orchestrator_config_fsdp_h100.yaml" \
        --rollouts "$ROLLOUTS_FILE" \
        --num-epochs "$EPOCHS" \
        --checkpoint-dir "$ORCHESTRATOR_OUTPUT" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Orchestrator FSDP training completed"
        if [ -d "$ORCHESTRATOR_TRAINED" ]; then
            ORCHESTRATOR_MODEL="$ORCHESTRATOR_TRAINED"
            echo "  Using trained model: $ORCHESTRATOR_TRAINED"
        fi
        return 0
    else
        echo "  ✗ Orchestrator FSDP training failed (check $LOG_FILE)"
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
    print('ERROR: Need 4 GPUs for FSDP training!')
    exit(1)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f'  GPU {i}: {props.name}, {mem_gb:.1f} GB')
"
echo ""

# =============================================================================
# Main Training Loop
# =============================================================================

echo ""
echo "=============================================="
echo "Starting 4x H100 FSDP Training Loop"
echo "=============================================="
echo ""

TOTAL_CYPHER_TRAINS=0
TOTAL_ORCH_TRAINS=0

for ITER in $(seq 1 $ITERATIONS); do
    echo ""
    echo "=============================================="
    echo "ITERATION $ITER / $ITERATIONS"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    # =========================================================================
    # Phase 1: INFERENCE (vLLM on GPUs 0-1)
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
        echo "=== PHASE 1: INFERENCE (vLLM on GPUs 0-1) ==="
        
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
    # Analyze and Decide
    # =========================================================================
    
    echo "Analyzing rollouts..."
    DECISION=$(analyze_and_decide "$ROLLOUTS_FILE" $ITER)
    
    DECISION_FILE=$(mktemp)
    echo "$DECISION" > "$DECISION_FILE"
    
    PARSED=$(python3 -c "
import json
try:
    with open('$DECISION_FILE', 'r') as f:
        d = json.load(f)
    train_cypher = 'true' if d.get('train_cypher') else 'false'
    train_orch = 'true' if d.get('train_orchestrator') else 'false'
    cypher_reward = d.get('metrics', {}).get('cypher_reward', 0.0)
    orch_reward = d.get('metrics', {}).get('orch_avg_reward', 0.0)
    print(f'{train_cypher}|{train_orch}|{cypher_reward}|{orch_reward}')
except:
    print('false|false|0.0|0.0')
" 2>/dev/null)
    
    rm -f "$DECISION_FILE"
    
    TRAIN_CYPHER=$(echo "$PARSED" | cut -d'|' -f1)
    TRAIN_ORCH=$(echo "$PARSED" | cut -d'|' -f2)
    CURRENT_CYPHER_REWARD=$(echo "$PARSED" | cut -d'|' -f3)
    CURRENT_ORCH_REWARD=$(echo "$PARSED" | cut -d'|' -f4)
    
    echo ""
    echo "Decision:"
    echo "  Train Cypher: $TRAIN_CYPHER"
    echo "  Train Orchestrator: $TRAIN_ORCH"
    echo "  Rewards: cypher=$CURRENT_CYPHER_REWARD, orch=$CURRENT_ORCH_REWARD"
    
    if [ "$FORCE_TRAIN" = "1" ]; then
        echo "  ⚠ FORCE_TRAIN=1: Overriding decision to train both models"
        TRAIN_CYPHER="true"
        TRAIN_ORCH="true"
    fi
    echo ""
    
    # =========================================================================
    # Phase 2: FSDP TRAINING
    # =========================================================================
    
    CYPHER_TRAINED_THIS_ITER=false
    ORCH_TRAINED_THIS_ITER=false
    
    if [ "$TRAIN_CYPHER" = "true" ] || [ "$TRAIN_ORCH" = "true" ]; then
        echo "=== PHASE 2: FSDP TRAINING (H100 80GB) ==="
        echo "  Cypher: GPUs 0-1 (FSDP FULL_SHARD)"
        echo "  Orchestrator: GPUs 2-3 (FSDP FULL_SHARD)"
        echo ""
        
        if [ "$TRAIN_CYPHER" = "true" ] && [ "$TRAIN_ORCH" = "true" ]; then
            echo "Training BOTH models in parallel (using all 4 GPUs)..."
            
            # Train both in parallel - Cypher on GPUs 0-1, Orchestrator on GPUs 2-3
            train_cypher_fsdp "$ROLLOUTS_FILE" $ITER &
            CYPHER_TRAIN_PID=$!
            
            train_orchestrator_fsdp "$ROLLOUTS_FILE" $ITER &
            ORCH_TRAIN_PID=$!
            
            wait $CYPHER_TRAIN_PID && CYPHER_TRAINED_THIS_ITER=true
            wait $ORCH_TRAIN_PID && ORCH_TRAINED_THIS_ITER=true
            
        elif [ "$TRAIN_CYPHER" = "true" ]; then
            train_cypher_fsdp "$ROLLOUTS_FILE" $ITER && CYPHER_TRAINED_THIS_ITER=true
            
        elif [ "$TRAIN_ORCH" = "true" ]; then
            train_orchestrator_fsdp "$ROLLOUTS_FILE" $ITER && ORCH_TRAINED_THIS_ITER=true
        fi
        
        [ "$CYPHER_TRAINED_THIS_ITER" = true ] && TOTAL_CYPHER_TRAINS=$((TOTAL_CYPHER_TRAINS + 1))
        [ "$ORCH_TRAINED_THIS_ITER" = true ] && TOTAL_ORCH_TRAINS=$((TOTAL_ORCH_TRAINS + 1))
        
        # =====================================================================
        # POST-TRAINING VALIDATION
        # =====================================================================
        if [ "$CYPHER_TRAINED_THIS_ITER" = true ] || [ "$ORCH_TRAINED_THIS_ITER" = true ]; then
            echo ""
            echo "=== POST-TRAINING VALIDATION ==="
            
            start_vllm_servers || echo "Warning: Could not start vLLM for validation"
            
            if curl -s "http://localhost:$CYPHER_PORT/health" > /dev/null 2>&1; then
                echo "Validating model improvement..."
                
                VALIDATION_RESULT=$(python3 "$SCRIPT_DIR/validate_model_improvement.py" \
                    --rollouts "$ROLLOUTS_FILE" \
                    --cypher-url "http://localhost:$CYPHER_PORT/v1" \
                    --orchestrator-url "http://localhost:$ORCHESTRATOR_PORT/v1" \
                    --sample-size 20 \
                    2>/dev/null || echo '{"success": false}')
                
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
            fi
            
            stop_vllm_servers
        fi
        
        echo ""
    else
        echo "No training needed this iteration"
    fi
    
    # Update history
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
    history[-1]['training_type'] = 'fsdp_h100'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
"
    
    echo ""
    echo "Iteration $ITER / $ITERATIONS complete!"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "4x H100 FSDP Pipeline Complete!"
echo "=============================================="
echo "End Time: $(date)"
echo ""

echo "Status: Completed $ITERATIONS iterations"
echo "  Cypher trained: $TOTAL_CYPHER_TRAINS times (FSDP)"
echo "  Orchestrator trained: $TOTAL_ORCH_TRAINS times (FSDP)"
echo ""

echo "Results:"
echo "  Cypher: $CYPHER_OUTPUT/best_model"
echo "  Cypher Merged: $CYPHER_TRAINED"
echo "  Orchestrator: $ORCHESTRATOR_OUTPUT/best_model"
echo "  Orchestrator Merged: $ORCHESTRATOR_TRAINED"
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
"
fi

echo ""
echo "✓ H100 FSDP Pipeline finished!"

