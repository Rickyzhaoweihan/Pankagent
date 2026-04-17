#!/bin/bash
# =============================================================================
# Stage 1 Training Test Script
# Small-scale test with reduced parameters
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PANKLLM_DIR="$PROJECT_ROOT/examples/PanKLLM_RL_post-training"
RL_IMPL_DIR="$PANKLLM_DIR/rl_implementation"

echo "=============================================="
echo "Stage 1 Training TEST"
echo "=============================================="

# Use test config
CONFIG_PATH="$RL_IMPL_DIR/ddp_training/config/stage1_test_config.yaml"

# Run with test settings
bash "$SCRIPT_DIR/run_stage1.sh" \
    "$CONFIG_PATH" \
    "$PANKLLM_DIR/outputs/stage1_ddp_test/checkpoints" \
    "$PANKLLM_DIR/outputs/stage1_ddp_test/logs" \
    2 \
    8

