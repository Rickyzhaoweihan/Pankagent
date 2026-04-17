# How to Test the Collaborative Training System

## Prerequisites

1. **Get a GPU node** (8 H100s recommended):
```bash
srun --partition=gpu --gres=gpu:8 --time=4:00:00 --mem=200G --pty bash
```

2. **Activate environment and load CUDA**:
```bash
conda activate vllm
module load cuda/11.8.0
```

3. **Navigate to project directory**:
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
```

---

## Running the Small-Scale Test

### Basic Test Command
```bash
python -m rl_implementation.training.test_small_scale \
    --config rl_implementation/config/training_config_test.yaml
```

### With GPU Monitoring (Recommended)

Use tmux to split terminal and monitor GPU usage:

```bash
# Start tmux
tmux

# Split terminal: Ctrl+b then "

# Top pane - run test:
python -m rl_implementation.training.test_small_scale \
    --config rl_implementation/config/training_config_test.yaml

# Bottom pane (Ctrl+b then ↓) - monitor GPU:
watch -n 1 nvidia-smi
```

### With Output Logging
```bash
python -m rl_implementation.training.test_small_scale \
    --config rl_implementation/config/training_config_test.yaml 2>&1 | tee test_output.log
```

---

## Configuration Files

| Config | Purpose |
|--------|---------|
| `config/training_config_test.yaml` | Small-scale test settings (8 questions, 2 epochs) |
| `config/training_config.yaml` | Full training settings |
| `config/ppo_collaborative.yaml` | Cypher Generator PPO config |
| `config/orchestrator_ppo.yaml` | Orchestrator PPO config |

---

## Key Settings in `training_config_test.yaml`

```yaml
num_epochs: 2                    # Number of training epochs
questions_per_epoch: 8           # Questions per epoch (divisible by 4 workers)
gpu_allocation:
  total_gpus: 8
  mode: "sequential"             # Sequential model swapping
  rollout:
    cypher_gpus: 4               # GPUs for Cypher Generator during rollout
    orch_gpus: 4                 # GPUs for Orchestrator during rollout
```

---

## Key Settings in PPO Configs

### Memory Settings (adjust if OOM)
```yaml
gpu_memory_utilization: 0.4      # Reduce if OOM (0.3-0.5)
tensor_model_parallel_size: 4    # Shard model across 4 GPUs
max_model_len: 6144              # Max context length
```

### Prompt/Response Lengths
```yaml
max_prompt_length: 4096          # Max input tokens
max_response_length: 1024        # Max output tokens
```

---

## Troubleshooting

### OOM (Out of Memory)
1. Reduce `gpu_memory_utilization` to 0.3
2. Reduce `max_model_len` to 4096
3. Set `enforce_eager: True` and `free_cache_engine: True`

### CUDA Library Errors
Make sure CUDA is loaded:
```bash
module load cuda/11.8.0
echo $LD_LIBRARY_PATH  # Should contain cuda path
```

### Slow Generation
- Check if both models are loading on correct GPUs
- Monitor with `nvidia-smi` to see memory distribution

### Prompt Truncation (Cypher not generating)
- Increase `max_prompt_length` in config
- Check debug output for "END OF PROMPT" to verify question is included

---

## Component Tests

### Test Cypher Generator Agent
```bash
python -m rl_implementation.test_cypher_agent
```

### Test Orchestrator Agent
```bash
python -m rl_implementation.test_orchestrator_agent
```

### Test Environment
```bash
python -m rl_implementation.test_environment
```

### Test Rewards
```bash
python -m rl_implementation.test_rewards
```

---

## Output Locations

| Output | Location |
|--------|----------|
| Checkpoints | `outputs/checkpoints/test_small_scale/` |
| Logs | `outputs/logs/test_small_scale/` |
| Rollout Samples | `outputs/logs/test_small_scale/rollout_samples/` |

---

## tmux Quick Reference

| Command | Action |
|---------|--------|
| `tmux` | Start tmux |
| `Ctrl+b "` | Split horizontal |
| `Ctrl+b %` | Split vertical |
| `Ctrl+b ↑↓←→` | Navigate panes |
| `Ctrl+b d` | Detach |
| `tmux attach` | Reattach |

---

## Full Training (After Testing)

Once small-scale test passes:
```bash
python -m rl_implementation.training.train_collaborative_system \
    --config rl_implementation/config/training_config.yaml \
    --schema_path legacy/PankBaseAgent/schemas/kg_schema\ copy.json \
    --checkpoint_dir outputs/checkpoints/full_training \
    --log_dir outputs/logs/full_training
```

