#!/bin/bash
# =============================================================================
# Run Rollout Collection
#
# Connects to running vLLM servers and collects rollouts for training.
# Each run creates a NEW file with unique run ID (not append mode).
#
# Prerequisites:
#   - vLLM servers running (./start_vllm_servers.sh)
#   - Neo4j API accessible
#
# Usage:
#   ./run_collect_rollouts.sh                    # Collect 64 easy questions
#   ./run_collect_rollouts.sh 128 medium         # Collect 128 medium questions
#   ./run_collect_rollouts.sh 256 hard 16        # Collect 256 hard, batch 16
#
# Environment Variables:
#   RUN_ID    - Custom run ID (default: timestamp)
#   SLURM_JOB_ID - Used as run ID if available
#
# Output:
#   outputs/stage1_ddp/rollouts_run_<RUN_ID>.jsonl
# =============================================================================

set -e

# =============================================================================
# Parse arguments
# =============================================================================
NUM_QUESTIONS="${1:-64}"
DIFFICULTY="${2:-easy}"
BATCH_SIZE="${3:-8}"

# =============================================================================
# Generate unique run ID
# =============================================================================
if [ -n "$RUN_ID" ]; then
    # Use provided RUN_ID
    :
elif [ -n "$SLURM_JOB_ID" ]; then
    # Use SLURM job ID
    RUN_ID="$SLURM_JOB_ID"
else
    # Generate timestamp-based ID
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DDP_DIR="$(dirname "$SCRIPT_DIR")"
RL_IMPL_DIR="$(dirname "$DDP_DIR")"
PROJECT_DIR="$(dirname "$RL_IMPL_DIR")"

# Output file with run ID
OUTPUT_FILE="$PROJECT_DIR/outputs/stage1_ddp/rollouts_run_${RUN_ID}.jsonl"

# =============================================================================
# Conda Environment
# =============================================================================
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# =============================================================================
# Cache directories (same as vLLM servers)
# =============================================================================
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

# =============================================================================
# Run collection
# =============================================================================
echo ""
echo "=============================================="
echo "Rollout Collection"
echo "=============================================="
echo "  Run ID: $RUN_ID"
echo "  Questions: $NUM_QUESTIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Batch size: $BATCH_SIZE"
echo "  Output: $OUTPUT_FILE"
echo ""

python3 "$SCRIPT_DIR/collect_rollouts.py" \
    --num-questions "$NUM_QUESTIONS" \
    --difficulty "$DIFFICULTY" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT_FILE" \
    --run-id "$RUN_ID" \
    --schema-path "$PROJECT_DIR/legacy/PankBaseAgent/text_to_cypher/data/input/kg_schema.json" \
    --entity-samples "$PROJECT_DIR/outputs/entity_samples.json" \
    --orchestrator-model "$PROJECT_DIR/models/qwen2.5-14b" \
    --cypher-model "$PROJECT_DIR/models/qwen2.5-coder-14b" \
    --orchestrator-port 8001 \
    --cypher-port 8002 \
    "$@"

echo ""
echo "Done! Rollouts saved to: $OUTPUT_FILE"

