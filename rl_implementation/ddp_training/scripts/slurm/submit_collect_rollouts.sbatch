#!/bin/bash
#SBATCH --job-name=collect_rollouts
#SBATCH --account=drjieliu
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/collect_rollouts_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/collect_rollouts_%j.err

# =============================================================================
# Collect Rollouts Job
#
# This job collects rollouts by connecting to already-running vLLM servers.
# Make sure submit_vllm_servers.sbatch is running before submitting this job!
#
# Usage:
#   # Start vLLM servers first (on GPU node)
#   sbatch submit_vllm_servers.sbatch
#   
#   # Wait for servers to be ready, then submit this
#   sbatch submit_collect_rollouts.sbatch
#
# Override defaults:
#   NUM_QUESTIONS=256 DIFFICULTY=medium sbatch submit_collect_rollouts.sbatch
# =============================================================================

set -e

# =============================================================================
# Configuration (can be overridden via environment variables)
# =============================================================================
NUM_QUESTIONS="${NUM_QUESTIONS:-16}"
DIFFICULTY="${DIFFICULTY:-easy}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training"
LOG_DIR="$PROJECT_DIR/outputs/stage1_ddp/logs"

mkdir -p "$LOG_DIR"

# =============================================================================
# Conda Environment
# =============================================================================
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "=============================================="
echo "Rollout Collection Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""
echo "Configuration:"
echo "  Questions: $NUM_QUESTIONS"
echo "  Difficulty: $DIFFICULTY"
echo "  Batch size: $BATCH_SIZE"
echo ""

# =============================================================================
# Cache directories
# =============================================================================
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
mkdir -p "$SCRATCH"/hf_caches/{datasets,hub,transformers,tmp} "$SCRATCH"/torch_cache "$SCRATCH"/cache

export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

# =============================================================================
# Check vLLM servers
# =============================================================================
echo "Checking vLLM servers..."

# Server settings
ORCHESTRATOR_PORT=8001
CYPHER_PORT=8002
ORCHESTRATOR_URL="http://localhost:${ORCHESTRATOR_PORT}"
CYPHER_URL="http://localhost:${CYPHER_PORT}"

check_server() {
    local url=$1
    local name=$2
    local max_attempts=5
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  ✓ $name server ready at $url"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "  ✗ $name server not responding at $url"
    return 1
}

if ! check_server "$ORCHESTRATOR_URL" "Orchestrator"; then
    echo "ERROR: Orchestrator server not available."
    echo "Make sure submit_vllm_servers.sbatch is running first!"
    exit 1
fi

if ! check_server "$CYPHER_URL" "Cypher"; then
    echo "ERROR: Cypher server not available."
    echo "Make sure submit_vllm_servers.sbatch is running first!"
    exit 1
fi

echo ""
echo "All servers ready!"
echo ""

# =============================================================================
# Run collection (job-specific output file)
# =============================================================================
OUTPUT_FILE="$PROJECT_DIR/outputs/stage1_ddp/rollouts_job_${SLURM_JOB_ID}.jsonl"

echo "Starting rollout collection..."
echo "Output file: $OUTPUT_FILE"
echo ""

python3 "$SCRIPT_DIR/../collect_rollouts.py" \
    --num-questions "$NUM_QUESTIONS" \
    --difficulty "$DIFFICULTY" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT_FILE" \
    --schema-path "$PROJECT_DIR/legacy/PankBaseAgent/text_to_cypher/data/input/kg_schema.json" \
    --entity-samples "$PROJECT_DIR/outputs/entity_samples.json" \
    --orchestrator-model "$PROJECT_DIR/models/qwen2.5-14b" \
    --cypher-model "$PROJECT_DIR/models/qwen2.5-coder-14b" \
    --orchestrator-port "$ORCHESTRATOR_PORT" \
    --cypher-port "$CYPHER_PORT" \
    --run-id "job_${SLURM_JOB_ID}"

echo ""
echo "=============================================="
echo "Rollout collection complete!"
echo "=============================================="
echo "Output: $OUTPUT_FILE"

