# GPU Configuration Update: 6 → 8 H100s

## Summary

Updated all GPU-related configurations to use 8 H100 GPUs instead of 6.

## Changes Made

### 1. Cypher Generator PPO Config
**File**: `rl_implementation/config/ppo_collaborative.yaml`

**Changes**:
- Line 261: `n_gpus_per_node: 6` → `n_gpus_per_node: 8`
- Line 276: `n_training_gpus_per_node: 6` → `n_training_gpus_per_node: 8`

### 2. Orchestrator PPO Config
**File**: `rl_implementation/config/orchestrator_ppo.yaml`

**Changes**:
- Line 264: `n_gpus_per_node: 6` → `n_gpus_per_node: 8`
- Line 279: `n_training_gpus_per_node: 6` → `n_training_gpus_per_node: 8`

## GPU Configuration Parameters

### `n_gpus_per_node`
- **Location**: `trainer.n_gpus_per_node`
- **Purpose**: Total number of GPUs available on the node
- **Used by**: Resource pool manager to allocate GPU resources
- **New value**: `8`

### `n_training_gpus_per_node`
- **Location**: `trainer.n_training_gpus_per_node`
- **Purpose**: Number of GPUs to use for training workers
- **Used by**: Worker group initialization
- **New value**: `8`

## Resource Allocation

### Before (6 GPUs)
```
Resource Pool: "global_pool"
├── GPUs per node: [6]
├── Total GPUs: 6
└── Worker allocation:
    ├── Actor/Rollout: 6 workers (1 per GPU)
    └── Critic: Disabled
```

### After (8 GPUs)
```
Resource Pool: "global_pool"
├── GPUs per node: [8]
├── Total GPUs: 8
└── Worker allocation:
    ├── Actor/Rollout: 8 workers (1 per GPU)
    └── Critic: Disabled
```

## Impact on Training

### Parallelism
- **Before**: 6-way data parallelism
- **After**: 8-way data parallelism
- **Benefit**: ~33% more parallel processing capacity

### Batch Size Considerations

With 8 GPUs, you can:

1. **Keep per-GPU batch size the same** (current: 8)
   - Effective global batch size: 8 GPUs × 8 samples = 64 samples
   - More throughput, same convergence

2. **Increase per-GPU batch size** (e.g., to 12)
   - Effective global batch size: 8 GPUs × 12 samples = 96 samples
   - Better GPU utilization, potentially faster convergence

3. **Reduce per-GPU batch size** (e.g., to 6)
   - Effective global batch size: 8 GPUs × 6 samples = 48 samples
   - More memory for larger models

### Current Batch Configuration
```yaml
# Per-GPU batch sizes (unchanged)
actor:
  ppo_micro_batch_size_per_gpu: 8

rollout:
  log_prob_micro_batch_size_per_gpu: 8

critic:
  ppo_micro_batch_size_per_gpu: 8
```

**Effective batch size**: 8 GPUs × 8 samples = **64 samples per batch**

## Memory Considerations

### H100 Specs
- **Memory per GPU**: 80GB HBM3
- **Total memory**: 8 × 80GB = 640GB
- **Memory bandwidth**: 3.35 TB/s per GPU

### Model Size
- **Qwen2.5-7B-Instruct**: ~7B parameters
- **FP16**: ~14GB per model
- **With activations & optimizer states**: ~30-40GB per GPU

### Headroom
- **Available**: 80GB per GPU
- **Used**: ~40GB per GPU
- **Free**: ~40GB per GPU
- **Status**: ✅ Comfortable headroom for batch size 8

## Testing

After this update, test with:

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
python -m rl_implementation.training.test_small_scale
```

Expected behavior:
- Resource pool should allocate 8 GPUs
- Worker groups should create 8 workers
- Training should utilize all 8 GPUs

## Verification Commands

### Check GPU availability
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Expected output:
```
index, name, memory.total [MiB]
0, NVIDIA H100 80GB HBM3, 81920 MiB
1, NVIDIA H100 80GB HBM3, 81920 MiB
2, NVIDIA H100 80GB HBM3, 81920 MiB
3, NVIDIA H100 80GB HBM3, 81920 MiB
4, NVIDIA H100 80GB HBM3, 81920 MiB
5, NVIDIA H100 80GB HBM3, 81920 MiB
6, NVIDIA H100 80GB HBM3, 81920 MiB
7, NVIDIA H100 80GB HBM3, 81920 MiB
```

### Monitor GPU usage during training
```bash
watch -n 1 nvidia-smi
```

All 8 GPUs should show:
- ✅ Process running
- ✅ Memory allocated
- ✅ GPU utilization > 0%

## Future Tuning Recommendations

### Option 1: Increase Throughput
```yaml
# Increase batch size to maximize GPU utilization
ppo_micro_batch_size_per_gpu: 12
log_prob_micro_batch_size_per_gpu: 12
```
- Effective batch: 8 × 12 = 96 samples
- Better GPU utilization
- Faster training

### Option 2: Larger Models
With 8 GPUs and 640GB total memory, you could:
- Use larger models (e.g., 13B, 30B parameters)
- Enable tensor parallelism across multiple GPUs
- Increase sequence length

### Option 3: Multi-Node Scaling
```yaml
nnodes: 2
n_gpus_per_node: 8
```
- Total: 16 GPUs
- Resource pool: [8, 8]
- 2x parallelism

## Status

- ✅ Updated `ppo_collaborative.yaml`
- ✅ Updated `orchestrator_ppo.yaml`
- ✅ Both `n_gpus_per_node` and `n_training_gpus_per_node` set to 8
- ✅ Ready for testing with 8 H100s

---

**Updated**: 2025-11-27  
**Configuration**: 8 × H100 80GB HBM3  
**Batch size**: 8 per GPU (64 effective)  
**Status**: Ready for testing

