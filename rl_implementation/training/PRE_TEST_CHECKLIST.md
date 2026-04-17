# Pre-Test Checklist

Before running the small-scale training test, verify all prerequisites are met.

## System Requirements

### Hardware
- [ ] **GPUs Available**: At least 1 GPU (preferably 6 H100s)
  ```bash
  nvidia-smi
  # Should show available GPUs
  ```

- [ ] **Disk Space**: At least 100GB free
  ```bash
  df -h /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/
  # Check available space
  ```

- [ ] **Memory**: At least 64GB RAM
  ```bash
  free -h
  # Check available memory
  ```

### Software
- [ ] **Python 3.11+**
  ```bash
  python --version
  # Should be 3.11 or higher
  ```

- [ ] **Conda Environment**: vllm or equivalent
  ```bash
  conda env list
  # Should show vllm environment
  ```

- [ ] **CUDA**: Compatible version installed
  ```bash
  nvcc --version
  # Should show CUDA version
  ```

## Dependencies

### Python Packages
- [ ] **rllm**: Installed in development mode
  ```bash
  cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm
  pip install -e .
  ```

- [ ] **Ray**: For distributed training
  ```bash
  python -c "import ray; print(ray.__version__)"
  # Should print version without error
  ```

- [ ] **PyTorch**: With CUDA support
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  # Should print True
  ```

- [ ] **Transformers**: For model loading
  ```bash
  python -c "import transformers; print(transformers.__version__)"
  # Should print version
  ```

- [ ] **Other Dependencies**
  ```bash
  python -c "import yaml, requests, numpy, pandas"
  # Should not error
  ```

## File Structure

### Required Files
- [ ] **Schema File**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema\ copy.json
  # Should exist
  ```

- [ ] **Model Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b/
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b/
  # Both should exist with model files
  ```

- [ ] **Configuration Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/config/training_config_test.yaml
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/config/ppo_collaborative.yaml
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/config/orchestrator_ppo.yaml
  # All should exist
  ```

- [ ] **Agent Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/agents/cypher_generator_agent.py
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/agents/orchestrator_agent.py
  # Both should exist
  ```

- [ ] **Environment Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/environments/graph_reasoning_env.py
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/environments/neo4j_executor.py
  # Both should exist
  ```

- [ ] **Reward Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/rewards/cypher_reward.py
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/rewards/orchestrator_reward.py
  # Both should exist
  ```

- [ ] **Training Files**
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/training/train_collaborative_system.py
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/training/test_small_scale.py
  # Both should exist
  ```

## Configuration Validation

### PPO Configs
- [ ] **Cypher Generator Config** (`ppo_collaborative.yaml`)
  - [ ] `actor_rollout_ref.model.path` points to qwen2.5-coder-14b
  - [ ] `custom_reward_function.path` points to cypher_reward.py (absolute path)
  - [ ] `n_training_gpus_per_node: 6`
  - [ ] `ppo_micro_batch_size_per_gpu: 8`
  - [ ] `log_prob_micro_batch_size_per_gpu: 8`
  - [ ] `critic.ppo_micro_batch_size_per_gpu: 8`
  - [ ] `actor_rollout_ref.rollout.multi_turn.enable: False`
  - [ ] `data.train_files` points to valid parquet file

- [ ] **Orchestrator Config** (`orchestrator_ppo.yaml`)
  - [ ] `actor_rollout_ref.model.path` points to qwen2.5-14b
  - [ ] `custom_reward_function.path` points to orchestrator_reward.py (absolute path)
  - [ ] Same batch size settings as above
  - [ ] `data.train_files` points to valid parquet file

### Training Config
- [ ] **Test Config** (`training_config_test.yaml`)
  - [ ] `schema_path` is correct
  - [ ] `checkpoint_dir` is writable
  - [ ] `log_dir` is writable
  - [ ] `neo4j_url` is accessible
  - [ ] `num_epochs: 1`
  - [ ] `questions_per_epoch: 3`

## Network Connectivity

- [ ] **Neo4j API**: Accessible
  ```bash
  curl -X POST https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j \
    -H "Content-Type: application/json" \
    -d '{"query": "MATCH (n) RETURN count(n) LIMIT 1"}'
  # Should return JSON response
  ```

- [ ] **Hugging Face**: Can download models (if needed)
  ```bash
  python -c "from transformers import AutoTokenizer; print('OK')"
  # Should print OK
  ```

## Permissions

- [ ] **Write Access**: Can create directories
  ```bash
  mkdir -p /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/test_permissions
  rmdir /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/outputs/test_permissions
  # Should succeed
  ```

- [ ] **Read Access**: Can read model files
  ```bash
  ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b/config.json
  # Should exist and be readable
  ```

## Environment Variables

- [ ] **Ray Temp Dir**: Set if needed
  ```bash
  echo $RAY_TMPDIR
  # Should be /tmp or similar short path
  ```

- [ ] **CUDA Visible Devices**: Set if needed
  ```bash
  echo $CUDA_VISIBLE_DEVICES
  # Should show available GPUs or be empty (all GPUs)
  ```

- [ ] **Tokenizers Parallelism**: Will be set by script
  ```bash
  # No action needed, script handles this
  ```

## Quick Verification Script

Run this to check most requirements at once:

```bash
#!/bin/bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

echo "=== System Check ==="
echo "Python: $(python --version)"
echo "GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "Disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

echo "=== Python Packages ==="
python -c "
import sys
packages = ['ray', 'torch', 'transformers', 'yaml', 'requests', 'numpy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - MISSING')
        sys.exit(1)
"
echo ""

echo "=== File Structure ==="
files=(
    "legacy/PankBaseAgent/schemas/kg_schema copy.json"
    "rl_implementation/config/training_config_test.yaml"
    "rl_implementation/config/ppo_collaborative.yaml"
    "rl_implementation/config/orchestrator_ppo.yaml"
    "rl_implementation/agents/cypher_generator_agent.py"
    "rl_implementation/agents/orchestrator_agent.py"
    "rl_implementation/training/train_collaborative_system.py"
    "rl_implementation/training/test_small_scale.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - MISSING"
    fi
done
echo ""

echo "=== Model Files ==="
if [ -d "models/qwen2.5-coder-14b" ]; then
    echo "✓ qwen2.5-coder-14b"
else
    echo "✗ qwen2.5-coder-14b - MISSING"
fi

if [ -d "models/qwen2.5-14b" ]; then
    echo "✓ qwen2.5-14b"
else
    echo "✗ qwen2.5-14b - MISSING"
fi
echo ""

echo "=== Network ==="
if curl -s -X POST https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j \
    -H "Content-Type: application/json" \
    -d '{"query": "MATCH (n) RETURN count(n) LIMIT 1"}' | grep -q "result"; then
    echo "✓ Neo4j API accessible"
else
    echo "✗ Neo4j API not accessible"
fi
echo ""

echo "=== All checks complete ==="
```

Save this as `check_prerequisites.sh` and run it:

```bash
chmod +x check_prerequisites.sh
./check_prerequisites.sh
```

## Ready to Test?

If all items above are checked ✓, you're ready to run:

```bash
./scripts/run_test_small_scale.sh
```

Or manually:

```bash
python -m rl_implementation.training.test_small_scale
```

## Common Issues

### Issue: Ray socket path too long
**Solution**: Already handled in code (uses /tmp)

### Issue: Model files not found
**Solution**: Verify model paths in PPO config files

### Issue: CUDA out of memory
**Solution**: Reduce batch sizes in PPO configs

### Issue: Import errors
**Solution**: Reinstall rllm: `cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm && pip install -e .`

### Issue: Neo4j API timeout
**Solution**: Check network connectivity, try increasing timeout in code

## Next Steps

After completing this checklist:
1. Run the test: `./scripts/run_test_small_scale.sh`
2. Monitor progress: `python -m rl_implementation.training.monitor_test`
3. Review results: Check `test_small_scale.log`
4. If successful, scale up gradually

