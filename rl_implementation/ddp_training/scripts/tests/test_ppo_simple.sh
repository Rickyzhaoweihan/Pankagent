#!/bin/bash
# =============================================================================
# Simple PPO Test - Load existing rollouts and test PPO update on GPU 7
# =============================================================================

set -e

cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

# Activate conda environment
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vllm

# Cache directories
SCRATCH="/scratch/drjieliu_root/drjieliu/rickyhan"
export HF_DATASETS_CACHE="$SCRATCH/hf_caches/datasets"
export HF_HOME="$SCRATCH/hf_caches/hub"
export HF_HUB_CACHE="$SCRATCH/hf_caches/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_caches/transformers"
export TORCH_HOME="$SCRATCH/torch_cache"

# PyTorch settings for memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# CRITICAL: Set CUDA_VISIBLE_DEVICES FIRST, BEFORE ANY PYTHON IMPORT
# This makes physical GPU 3 appear as cuda:0 to PyTorch
export CUDA_VISIBLE_DEVICES=3

echo "=============================================="
echo "PPO Test - Using GPU 3 Only"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# First, verify GPU is correct with a fresh Python process
echo "Verifying GPU allocation..."
python3 << 'EOF'
import os
# This should already be set, but double-check
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7', f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES')}, expected '7'"

import torch
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'cuda:0 = {props.name}')
    total = props.total_memory / 1024**3
    print(f'Total memory: {total:.1f} GB')
    # Try to allocate a small tensor to verify
    x = torch.randn(100, 100, device='cuda:0')
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f'Allocated after test tensor: {allocated:.4f} GB')
    del x
    torch.cuda.empty_cache()
    print('GPU test passed!')
EOF

echo ""
echo "Running PPO test with minimal settings..."
echo ""

# Run the PPO test script with minimal memory settings
# - Only 2 rollouts
# - Very short sequences (256 tokens)
# - Single sample per batch
python3 -m rl_implementation.ddp_training.scripts.test_ppo_only \
    --rollouts_path outputs/stage1_ddp/rollouts.jsonl \
    --config rl_implementation/ddp_training/config/stage1_config.yaml \
    --max_rollouts 2 \
    --mini_batch_size 1 \
    --max_prompt_length 256 \
    --max_response_length 64

echo ""
echo "PPO test complete!"

