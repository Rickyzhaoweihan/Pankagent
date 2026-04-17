# Small-Scale Training Test Guide

## Overview

This guide explains how to run small-scale tests of the collaborative multi-agent training system before scaling up to full training.

## Test Files

### 1. Configuration
- **`config/training_config_test.yaml`** - Test configuration with minimal resources
  - 3 questions per epoch
  - 1 epoch
  - Debug logging enabled
  - Small batch sizes

### 2. Test Scripts
- **`test_small_scale.py`** - Main test script
  - Runs component tests first
  - Then runs full training loop
  - Comprehensive error reporting

- **`monitor_test.py`** - Monitoring script
  - Tracks GPU usage
  - Shows Ray cluster status
  - Tails log file
  - Detects completion/errors

## Running the Tests

### Prerequisites

1. **Environment Setup**
   ```bash
   cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
   conda activate vllm  # or your environment
   ```

2. **Verify GPU Access**
   ```bash
   nvidia-smi
   # Should show 6 H100 GPUs
   ```

3. **Check Ray**
   ```bash
   ray status
   # If no cluster, Ray will auto-initialize
   ```

### Basic Test Run

```bash
# Run the test (from project root)
python -m rl_implementation.training.test_small_scale
```

This will:
1. Run component tests (imports, config, agents, environment)
2. Initialize the training system
3. Run 1 epoch with 3 questions
4. Save checkpoints and logs

### With Monitoring

Open two terminals:

**Terminal 1 - Run test:**
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
python -m rl_implementation.training.test_small_scale
```

**Terminal 2 - Monitor:**
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
python -m rl_implementation.training.monitor_test
```

### Skip Component Tests

If you've already verified components work:

```bash
python -m rl_implementation.training.test_small_scale --skip-component-tests
```

### Custom Configuration

```bash
python -m rl_implementation.training.test_small_scale --config path/to/custom_config.yaml
```

## Expected Output

### Successful Run

```
================================================================================
SMALL-SCALE TRAINING TEST
================================================================================
Config: rl_implementation/config/training_config_test.yaml
================================================================================

Step 1: Running component tests...
[Test 1] Testing imports...
✓ All imports successful
[Test 2] Testing configuration loading...
✓ Configuration loaded: test_small_scale
[Test 3] Testing file paths...
✓ Schema file exists: ...
[Test 4] Testing agent initialization...
✓ Cypher Generator agent initialized
✓ Orchestrator agent initialized
[Test 5] Testing environment initialization...
✓ Environment initialized

✅ All component tests passed!

Step 2: Running training test...
Loading configuration from rl_implementation/config/training_config_test.yaml
Configuration loaded:
  - Experiment: test_small_scale
  - Questions per epoch: 3
  - Num epochs: 1
  - Schema: ...

Initializing CollaborativeTrainer...
Loading tokenizers...
Creating resource pool manager...
Creating worker groups for Cypher Generator...
Creating worker groups for Orchestrator...
✓ Trainer initialized successfully

Starting training loop...
================================================================================
Epoch 1/1
Step 1: Generating training questions...
Generated 3 questions
Step 2: Collecting rollouts...
Collected 3 complete trajectories
Step 3: Computing rewards...
Step 4: Normalizing rewards...
Step 5: Training Cypher Generator...
Cypher training complete. Avg reward: 0.523
Step 6: Training Orchestrator...
Orchestrator training complete. Gen reward: 0.612
================================================================================
✓ Training completed successfully!

================================================================================
✅ TEST PASSED - Training completed successfully!
================================================================================
```

### Common Errors and Solutions

#### 1. Import Errors
```
✗ Import failed: No module named 'rllm'
```
**Solution**: Install rllm package
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm
pip install -e .
```

#### 2. GPU/CUDA Errors
```
✗ RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in config or free up GPU memory
```bash
# Check GPU usage
nvidia-smi
# Kill processes if needed
```

#### 3. Ray Initialization Errors
```
✗ OSError: AF_UNIX path length cannot exceed 107 bytes
```
**Solution**: Already handled in code (uses /tmp), but if persists:
```bash
export RAY_TMPDIR=/tmp
```

#### 4. Model Loading Errors
```
✗ FileNotFoundError: Model not found at ...
```
**Solution**: Verify model paths in config files
```bash
ls /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/
```

#### 5. Schema File Not Found
```
✗ FileNotFoundError: Schema file not found
```
**Solution**: Check schema path
```bash
ls legacy/PankBaseAgent/schemas/
```

## Output Files

After a successful test run, you'll find:

### Logs
- `test_small_scale.log` - Detailed execution log
- `outputs/logs/test_small_scale/` - Training logs

### Checkpoints
- `outputs/checkpoints/test_small_scale/` - Model checkpoints
  - `epoch_1/` - Checkpoint after epoch 1
  - `best/` - Best checkpoint (if applicable)

### Metrics
- Training metrics in log files
- Reward statistics
- GPU utilization data (if monitored)

## Verification Checklist

After running the test, verify:

- [ ] All component tests passed
- [ ] Training loop completed without errors
- [ ] Checkpoint files were created
- [ ] Log files contain expected information
- [ ] GPU utilization was > 0% (check with `nvidia-smi`)
- [ ] Ray cluster initialized successfully
- [ ] Worker groups were created
- [ ] Execution engines initialized
- [ ] Rollout collection worked
- [ ] PPO updates executed

## Next Steps

### If Test Passes ✅

1. **Review Logs**
   ```bash
   tail -100 test_small_scale.log
   ```

2. **Check Metrics**
   - Look for reward values
   - Verify training progress
   - Check for warnings

3. **Inspect Checkpoints**
   ```bash
   ls -lh outputs/checkpoints/test_small_scale/
   ```

4. **Scale Up Gradually**
   - Try 10 questions, 2 epochs
   - Then 50 questions, 5 epochs
   - Finally full scale (512 questions, 45 epochs)

### If Test Fails ❌

1. **Check Error Messages**
   ```bash
   grep -i error test_small_scale.log
   grep -i traceback test_small_scale.log
   ```

2. **Review Component Tests**
   - Which test failed?
   - Fix that component first

3. **Check System Resources**
   ```bash
   nvidia-smi  # GPU availability
   ray status  # Ray cluster
   df -h       # Disk space
   ```

4. **Enable More Logging**
   - Set `log_level: "DEBUG"` in config
   - Add print statements if needed

5. **Test Components Individually**
   - Test agent initialization separately
   - Test environment separately
   - Test execution engine separately

## Debugging Tips

### Enable Verbose Logging

In `training_config_test.yaml`:
```yaml
log_level: "DEBUG"
debug_mode: true
```

### Test Individual Components

```python
# Test agent initialization
from rl_implementation.agents import CypherGeneratorAgent
agent = CypherGeneratorAgent(schema_path="...", max_steps=5)
print("Agent initialized:", agent)

# Test environment
from rl_implementation.environments import GraphReasoningEnvironment
env = GraphReasoningEnvironment(
    task={'question': 'Test'},
    api_url="...",
    max_turns=5
)
obs, info = env.reset()
print("Environment ready:", env)
```

### Monitor GPU Usage

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Or use the monitor script
python -m rl_implementation.training.monitor_test
```

### Check Ray Logs

```bash
# Ray logs location
ls /tmp/ray/session_*/logs/

# View Ray dashboard (if enabled)
# Open browser to http://localhost:8265
```

## Performance Expectations

For the small-scale test (3 questions, 1 epoch):

- **Duration**: 5-15 minutes (depending on model loading time)
- **GPU Usage**: Should utilize all 6 GPUs during training
- **Memory**: ~20-30GB per GPU (with 14B models)
- **Disk**: ~50GB for checkpoints

## Troubleshooting

### Test Hangs

If the test hangs:
1. Check GPU availability: `nvidia-smi`
2. Check Ray status: `ray status`
3. Look for deadlocks in logs
4. Kill and restart: `Ctrl+C`, then `ray stop`, then retry

### Out of Memory

If you get OOM errors:
1. Reduce `questions_per_epoch` to 2 or 1
2. Check `ppo_micro_batch_size_per_gpu` in PPO configs
3. Free up GPU memory: `ray stop`

### Slow Execution

If execution is very slow:
1. Check GPU utilization: `nvidia-smi`
2. Verify tensor parallelism is working
3. Check network latency to Neo4j API
4. Profile with `cProfile` if needed

## Contact

For issues or questions:
1. Check `CUSTOM_TRAINING_LOOP_STATUS.md` for known issues
2. Review error logs carefully
3. Consult rllm documentation
4. Check Ray documentation for distributed training issues

