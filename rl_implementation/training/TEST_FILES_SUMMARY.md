# Small-Scale Test Files Summary

## Overview

Created comprehensive test infrastructure for small-scale validation of the collaborative multi-agent training system.

## Files Created

### 1. Configuration
**`config/training_config_test.yaml`**
- Small-scale test configuration
- 3 questions per epoch, 1 epoch
- Debug logging enabled
- Test mode flags

### 2. Test Scripts

**`training/test_small_scale.py`**
- Main test script with component tests
- Tests imports, configuration, agents, environment
- Runs full training loop
- Comprehensive error reporting
- Usage:
  ```bash
  python -m rl_implementation.training.test_small_scale
  python -m rl_implementation.training.test_small_scale --skip-component-tests
  ```

**`training/monitor_test.py`**
- Real-time monitoring script
- Tracks GPU usage, Ray status, log tail
- Auto-detects completion/errors
- Usage:
  ```bash
  python -m rl_implementation.training.monitor_test
  python -m rl_implementation.training.monitor_test --interval 10
  ```

### 3. Launch Scripts

**`scripts/run_test_small_scale.sh`**
- Automated test launcher
- Checks prerequisites (GPU, conda, Ray)
- Handles environment setup
- Optional monitoring mode
- Usage:
  ```bash
  ./scripts/run_test_small_scale.sh
  ./scripts/run_test_small_scale.sh --monitor
  ./scripts/run_test_small_scale.sh --skip-component-tests
  ```

**`scripts/check_prerequisites.sh`**
- Comprehensive prerequisite checker
- Validates system, packages, files, network
- Color-coded output
- Usage:
  ```bash
  ./scripts/check_prerequisites.sh
  ```

### 4. Documentation

**`training/TEST_README.md`**
- Complete testing guide
- Running instructions
- Expected output
- Troubleshooting tips
- Performance expectations

**`training/PRE_TEST_CHECKLIST.md`**
- Detailed prerequisite checklist
- System requirements
- File structure validation
- Configuration validation
- Quick verification script

**`training/CUSTOM_TRAINING_LOOP_STATUS.md`**
- Implementation status
- What's completed vs pending
- Technical details
- Known issues

## Quick Start

### Option 1: Automated (Recommended)

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

# Check prerequisites
./scripts/check_prerequisites.sh

# Run test
./scripts/run_test_small_scale.sh
```

### Option 2: With Monitoring

Terminal 1:
```bash
./scripts/run_test_small_scale.sh
```

Terminal 2:
```bash
python -m rl_implementation.training.monitor_test
```

### Option 3: Manual

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
python -m rl_implementation.training.test_small_scale
```

## Test Flow

1. **Component Tests** (if not skipped)
   - Import validation
   - Configuration loading
   - File path checks
   - Agent initialization
   - Environment initialization

2. **Training Initialization**
   - Load configurations
   - Initialize Ray cluster
   - Load tokenizers
   - Create worker groups
   - Create execution engines

3. **Training Loop** (1 epoch, 3 questions)
   - Generate questions (Orchestrator)
   - Collect rollouts (Cypher Generator)
   - Evaluate trajectories (Orchestrator)
   - Compute rewards
   - Train Cypher Generator (PPO)
   - Train Orchestrator (PPO)
   - Save checkpoint

4. **Validation** (optional)
   - Run on fixed validation set
   - Compare metrics

5. **Cleanup**
   - Save final checkpoint
   - Log metrics
   - Shutdown Ray

## Expected Duration

- **Component Tests**: 10-30 seconds
- **Training Initialization**: 2-5 minutes (model loading)
- **Training Loop**: 5-10 minutes (3 questions)
- **Total**: ~10-15 minutes

## Expected Outputs

### Console Output
```
================================================================================
SMALL-SCALE TRAINING TEST
================================================================================
...
✅ All component tests passed!
...
Initializing CollaborativeTrainer...
✓ Trainer initialized successfully
...
Starting training loop...
Epoch 1/1
...
✓ Training completed successfully!
================================================================================
✅ TEST PASSED - Training completed successfully!
================================================================================
```

### Files Created
- `test_small_scale.log` - Detailed execution log
- `outputs/checkpoints/test_small_scale/epoch_1/` - Checkpoint files
- `outputs/logs/test_small_scale/` - Training logs

### Metrics Logged
- Question generation metrics
- Rollout collection metrics
- Reward statistics (mean, std)
- Training metrics (loss, KL divergence)
- GPU utilization
- Timing information

## Success Criteria

- [ ] All component tests pass
- [ ] Trainer initializes without errors
- [ ] 3 questions generated
- [ ] 3 rollouts collected
- [ ] Rewards computed
- [ ] Cypher Generator PPO update completes
- [ ] Orchestrator PPO update completes
- [ ] Checkpoint saved
- [ ] No CUDA errors
- [ ] GPU utilization > 0%

## Common Issues

### 1. Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'rllm'`
**Solution**: 
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm
pip install -e .
```

### 2. Ray Socket Path Error
**Symptom**: `OSError: AF_UNIX path length cannot exceed 107 bytes`
**Solution**: Already handled in code (uses /tmp)

### 3. Model Not Found
**Symptom**: `FileNotFoundError: Model not found`
**Solution**: Check model paths in PPO configs

### 4. CUDA OOM
**Symptom**: `RuntimeError: CUDA out of memory`
**Solution**: Reduce batch sizes or free GPU memory

### 5. Neo4j Timeout
**Symptom**: `requests.exceptions.Timeout`
**Solution**: Check network, increase timeout in code

## Debugging

### Enable Verbose Logging
In `training_config_test.yaml`:
```yaml
log_level: "DEBUG"
debug_mode: true
```

### Check Logs
```bash
# View full log
cat test_small_scale.log

# Search for errors
grep -i error test_small_scale.log
grep -i traceback test_small_scale.log

# View recent entries
tail -100 test_small_scale.log
```

### Monitor GPU
```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Or use monitor script
python -m rl_implementation.training.monitor_test
```

### Check Ray
```bash
ray status
ray list nodes
ray list actors
```

## Next Steps After Successful Test

1. **Review Logs**
   - Check for warnings
   - Verify metrics look reasonable
   - Ensure GPU utilization was good

2. **Inspect Checkpoints**
   ```bash
   ls -lh outputs/checkpoints/test_small_scale/
   ```

3. **Scale Up Gradually**
   - 10 questions, 2 epochs
   - 50 questions, 5 epochs
   - 512 questions, 45 epochs (full scale)

4. **Tune Hyperparameters**
   - Learning rates
   - Batch sizes
   - Reward weights
   - Curriculum stages

5. **Enable Validation**
   - Add validation questions
   - Set `skip_validation: false`
   - Monitor validation metrics

## File Locations

All test files are in:
```
/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/
├── rl_implementation/
│   ├── config/
│   │   └── training_config_test.yaml
│   └── training/
│       ├── test_small_scale.py
│       ├── monitor_test.py
│       ├── TEST_README.md
│       ├── PRE_TEST_CHECKLIST.md
│       └── TEST_FILES_SUMMARY.md (this file)
└── scripts/
    ├── run_test_small_scale.sh
    └── check_prerequisites.sh
```

## Support

For issues:
1. Check `TEST_README.md` for detailed troubleshooting
2. Review `PRE_TEST_CHECKLIST.md` for prerequisites
3. Check `CUSTOM_TRAINING_LOOP_STATUS.md` for known issues
4. Review error logs carefully
5. Consult rllm documentation

## Summary

The test infrastructure is complete and ready to use. Start with:

```bash
./scripts/check_prerequisites.sh  # Verify prerequisites
./scripts/run_test_small_scale.sh  # Run test
```

Good luck! 🚀

