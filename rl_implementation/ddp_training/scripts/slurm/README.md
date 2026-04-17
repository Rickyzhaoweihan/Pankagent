# PanKLLM RL Training Pipeline - SLURM Scripts

This directory contains SLURM batch scripts for training the PanKLLM agents (Orchestrator and Cypher Generator) using reinforcement learning.

## Quick Reference

export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"
| Script | GPUs | Best For | Memory/GPU |
|--------|------|----------|------------|
| `submit_2gpu_pipeline.sbatch` | 2x H100 | Production training | 80GB |
| `submit_2l40s_pipeline.sbatch` | 2x L40S | Budget training | 48GB |
| `submit_4l40s_fsdp_pipeline.sbatch` | 4x L40S | Faster L40S training (FSDP) | 48GB |
| `submit_4h100_fsdp_pipeline.sbatch` | 4x H100 | **Fastest FSDP training** | 80GB |
| `submit_full_pipeline.sbatch` | 4x H100 | Full features, parallel | 80GB |

---

## 1. 2-GPU H100 Pipeline (`submit_2gpu_pipeline.sbatch`)

**Best for:** Production training with time-shared GPUs.

Uses 2 H100 GPUs with time-sharing between vLLM inference and model training.

### Basic Usage

```bash
# Default: 10 iterations, coverage-based (128 usable rollouts)
sbatch /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch

# Named experiment
sbatch --export=RUN_ID=my_h100_experiment \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch
```

### Common Configurations

```bash
# Quick test (small rollout count)
sbatch --export=RUN_ID=test_h100,QUESTIONS=16,TARGET_FILTERED=0,ITERATIONS=2,EPOCHS=1,CLEAR_HISTORY=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch

# Full training with more rollouts
sbatch --export=RUN_ID=full_h100_exp,TARGET_FILTERED=128,ITERATIONS=15,EPOCHS=10 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch

# Fixed rollout count (disable coverage-based)
sbatch --export=RUN_ID=fixed_rollouts,TARGET_FILTERED=0,QUESTIONS=64,ITERATIONS=10 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch

# Disable adaptive sampling
sbatch --export=RUN_ID=uniform_sampling,NO_ADAPTIVE_SAMPLING=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2gpu_pipeline.sbatch
```

---

## 2. 2-GPU L40S Pipeline (`submit_2l40s_pipeline.sbatch`)

**Best for:** Training on L40S GPUs (48GB VRAM) with memory optimizations.

Uses reduced context length (4096) and batch sizes for 48GB GPUs.

### Basic Usage

```bash
# Default L40S training
sbatch /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2l40s_pipeline.sbatch

# Named experiment
sbatch --export=RUN_ID=my_l40s_experiment \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2l40s_pipeline.sbatch
```

### Common Configurations

```bash
# Quick test
sbatch --export=RUN_ID=l40s_test,QUESTIONS=8,TARGET_FILTERED=0,ITERATIONS=1,EPOCHS=1,CLEAR_HISTORY=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2l40s_pipeline.sbatch

# Force training (bypass thresholds, for testing)
sbatch --export=RUN_ID=l40s_force,QUESTIONS=8,TARGET_FILTERED=0,ITERATIONS=1,EPOCHS=1,CLEAR_HISTORY=1,FORCE_TRAIN=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2l40s_pipeline.sbatch

# Full L40S training
sbatch --export=RUN_ID=l40s_full,TARGET_FILTERED=128,ITERATIONS=10,EPOCHS=3 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_2l40s_pipeline.sbatch
```

---

## 3. 4-GPU L40S FSDP Pipeline (`submit_4l40s_fsdp_pipeline.sbatch`)

**Best for:** Faster training on L40S with FSDP (Fully Sharded Data Parallel).

Uses 4 L40S GPUs:
- GPUs 0-1: vLLM inference (time-shared)
- GPUs 0-1: Cypher FSDP training
- GPUs 2-3: Orchestrator FSDP training

Both models can train **in parallel** using separate GPU pairs!

### Basic Usage

```bash
# Default FSDP training
sbatch /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4l40s_fsdp_pipeline.sbatch

# Named experiment
sbatch --export=RUN_ID=fsdp_experiment \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4l40s_fsdp_pipeline.sbatch
```

### Common Configurations

```bash
# Quick FSDP test
sbatch --export=RUN_ID=fsdp_test,QUESTIONS=8,TARGET_FILTERED=0,ITERATIONS=1,EPOCHS=1,CLEAR_HISTORY=1,FORCE_TRAIN=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4l40s_fsdp_pipeline.sbatch

# Full FSDP training
sbatch --export=RUN_ID=fsdp_production_1,TARGET_FILTERED=128,ITERATIONS=10,EPOCHS=10 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4l40s_fsdp_pipeline.sbatch

# FSDP with more rollouts
sbatch --export=RUN_ID=fsdp_256,TARGET_FILTERED=256,ITERATIONS=15,EPOCHS=5 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4l40s_fsdp_pipeline.sbatch
```

---

## 4. 4-GPU H100 FSDP Pipeline (`submit_4h100_fsdp_pipeline.sbatch`) ⭐ RECOMMENDED

**Best for:** Fastest training with FSDP on H100 GPUs.

Uses 4 H100 GPUs (80GB each) with FSDP (Fully Sharded Data Parallel):
- GPUs 0-1: vLLM inference (time-shared)
- GPUs 0-1: Cypher FSDP training
- GPUs 2-3: Orchestrator FSDP training

Both models train **in parallel** using separate GPU pairs! H100's 80GB allows larger batch sizes (32) and mini-batch sizes (4) for faster training.

### Basic Usage

```bash
# Default FSDP training on H100
sbatch /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4h100_fsdp_pipeline.sbatch

# Named experiment
sbatch --export=RUN_ID=h100_fsdp_exp \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4h100_fsdp_pipeline.sbatch
```

### Common Configurations

```bash
# Quick H100 FSDP test
sbatch --export=RUN_ID=h100_test,QUESTIONS=8,TARGET_FILTERED=0,ITERATIONS=1,EPOCHS=1,CLEAR_HISTORY=1,FORCE_TRAIN=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4h100_fsdp_pipeline.sbatch

# Full H100 FSDP training (recommended for production)
sbatch --export=RUN_ID=h100_fsdp_v3,TARGET_FILTERED=128,ITERATIONS=10,EPOCHS=10,CLEAR_HISTORY=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4h100_fsdp_pipeline.sbatch

# Extended training
sbatch --export=RUN_ID=h100_extended,TARGET_FILTERED=256,ITERATIONS=20,EPOCHS=5 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_4h100_fsdp_pipeline.sbatch
```

### H100 vs L40S FSDP Comparison

| Aspect | H100 FSDP | L40S FSDP |
|--------|-----------|-----------|
| VRAM per GPU | 80GB | 48GB |
| Batch Size | 32 | 16 |
| Mini-batch Size | 4 | 1 |
| Max Sequence Length | 4096 | 4096 |
| Training Speed | ~2-3x faster | Baseline |

---

## 5. 4-GPU H100 Full Pipeline (`submit_full_pipeline.sbatch`)

**Best for:** Full-featured training with dedicated GPUs for inference and training.

Uses 4 H100 GPUs:
- GPU 0: vLLM Orchestrator (always running)
- GPU 1: vLLM Cypher Generator (always running)
- GPU 2: Cypher training
- GPU 3: Orchestrator training

### Basic Usage

```bash
# Default full pipeline
sbatch /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch

# Named experiment
sbatch --export=RUN_ID=my_full_experiment \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch
```

### Common Configurations

```bash
# Fresh start (clear history)
sbatch --export=RUN_ID=fresh_start,CLEAR_HISTORY=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch

# More iterations with higher quality threshold
sbatch --export=RUN_ID=high_quality,TARGET_FILTERED=256,MIN_USABLE_REWARD=0.2,ITERATIONS=20 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch

# Accelerated mode (overlapping, but no coverage-based)
sbatch --export=RUN_ID=accelerated,ACCELERATED=1,TARGET_FILTERED=0,QUESTIONS=64 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch

# Skip first collection (use existing rollouts)
sbatch --export=RUN_ID=continue_training,SKIP_FIRST_COLLECT=1 \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/ddp_training/scripts/slurm/submit_full_pipeline.sbatch
```

---

## Configuration Options

All scripts support these options via `--export`:

| Option | Default | Description |
|--------|---------|-------------|
| `RUN_ID` | auto | Unique experiment identifier |
| `ITERATIONS` | 10 | Maximum training iterations |
| `EPOCHS` | 3 | Training epochs per iteration |
| `BATCH_SIZE` | 8 (H100), 4 (L40S) | Training batch size |
| `DIFFICULTY` | easy | Question difficulty (easy/medium/hard) |
| `TARGET_FILTERED` | 128 | Target usable rollouts (0 = use QUESTIONS) |
| `MIN_USABLE_REWARD` | 0.1 | Minimum reward for usable rollout |
| `QUESTIONS` | 64 | Fixed question count (if TARGET_FILTERED=0) |
| `NO_ADAPTIVE_SAMPLING` | 0 | Set to 1 to disable Thompson Sampling |
| `CLEAR_HISTORY` | 0 | Set to 1 to clear training history |
| `SKIP_FIRST_COLLECT` | 0 | Set to 1 to skip first rollout collection |
| `FORCE_TRAIN` | 0 | Set to 1 to force training regardless of thresholds |
| `FORCE_DEGREE_EXTRACTION` | 0 | Set to 1 to re-extract entity degrees |

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_*.out

# View errors
tail -f /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/pipeline_*.err

# Check vLLM server logs
tail -f /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/vllm_*.log

# Check training logs for specific iteration
cat /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/stage1_ddp/logs/iter_001_cypher.log

# Cancel job
scancel <JOB_ID>
```

---

## Output Locations

After training, outputs are stored in:

```
outputs/
├── stage1_ddp/
│   ├── logs/                           # All log files
│   │   ├── pipeline_<JOB_ID>.out       # Main pipeline output
│   │   ├── vllm_orchestrator_<JOB_ID>.log
│   │   ├── vllm_cypher_<JOB_ID>.log
│   │   └── iter_XXX_*.log              # Per-iteration training logs
│   └── rollouts_run_<RUN_ID>/          # Experiment-specific data
│       ├── rollouts_iter_001.jsonl     # Collected rollouts
│       ├── rollouts_iter_002.jsonl
│       └── training_history.json       # Training metrics history
├── cypher_trained_<RUN_ID>/            # Experiment-specific trained Cypher model
│   ├── best_model/                     # LoRA adapter
│   └── best_model_merged/              # Merged model (for vLLM)
├── orchestrator_trained_<RUN_ID>/      # Experiment-specific trained Orchestrator
│   ├── best_model/
│   └── best_model_merged/
├── entity_degrees.json                 # Cached entity degrees for adaptive sampling
└── adaptive_sampler_state.json         # Thompson Sampling state
```

---

## Troubleshooting

### Out of Memory (OOM)
- For L40S: Reduce `BATCH_SIZE` (default is 4, try 2)
- For H100: Reduce `BATCH_SIZE` or `VLLM_MAX_MODEL_LEN`

### vLLM Server Not Starting
- Check logs: `cat outputs/stage1_ddp/logs/vllm_*.log`
- May need to wait longer (L40S loads slower than H100)

### Training Not Happening
- Check if reward thresholds are met
- Use `FORCE_TRAIN=1` to bypass threshold checks
- Check `training_history.json` for decision details

### FSDP Errors
- Ensure 4 GPUs are available
- Check for dtype mismatches in logs
- FSDP requires uniform parameter dtypes (bfloat16)

---

## Recommended Workflow

1. **Quick Test** (verify setup):
   ```bash
   sbatch --export=RUN_ID=test,QUESTIONS=8,TARGET_FILTERED=0,ITERATIONS=1,EPOCHS=1,CLEAR_HISTORY=1,FORCE_TRAIN=1 \
       submit_2l40s_pipeline.sbatch
   ```

2. **Initial Training Run**:
   ```bash
   sbatch --export=RUN_ID=exp_v1,CLEAR_HISTORY=1 \
       submit_2gpu_pipeline.sbatch
   ```

3. **Continue Training** (if interrupted):
   ```bash
   sbatch --export=RUN_ID=exp_v1,SKIP_FIRST_COLLECT=1 \
       submit_2gpu_pipeline.sbatch
   ```

4. **Production Training**:
   ```bash
   sbatch --export=RUN_ID=production,TARGET_FILTERED=256,ITERATIONS=20,EPOCHS=5 \
       submit_full_pipeline.sbatch
   ```

