#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/vllm_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation/outputs/logs/vllm_%j.err

# =============================================================================
# vLLM Server Hosting Job
#
# This script hosts the cypher-writer model via vLLM's OpenAI-compatible API.
# The server will run until the job is cancelled or hits the time limit.
#
# Usage:
#   sbatch host_vllm.sbatch
#
# Then from another terminal (on the same node):
#   curl http://<node>:8002/v1/models
#   curl http://<node>:8002/health
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "vLLM SERVER HOSTING"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "============================================================"

# ----- Configuration -----
MODEL_PATH=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/cypher_trained_hints_test_0118_2223/best_model_merged
VLLM_PORT=8002

# ----- Environment Setup -----
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""
echo "Model:  $MODEL_PATH"
echo "Port:   $VLLM_PORT"
echo ""
echo "Connect to this server at:"
echo "  http://$(hostname):${VLLM_PORT}/v1"
echo ""
# Export the port so Python code (text2cypher_agent.py) picks it up
export VLLM_PORT=$VLLM_PORT

echo "Starting vLLM server..."
echo "============================================================"

# Run vLLM server in the foreground (keeps the job alive)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name cypher-writer \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-seqs 32

