# Testing Guide - Collaborative Multi-Agent Training

## Quick Start

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

# 1. Check prerequisites
./scripts/check_prerequisites.sh

# 2. Run small-scale test
./scripts/run_test_small_scale.sh

# 3. Check results
tail -100 test_small_scale.log
```

## Documentation Index

### Getting Started
1. **[SMALL_SCALE_TEST_COMPLETE.md](SMALL_SCALE_TEST_COMPLETE.md)** ⭐ START HERE
   - Overview of test infrastructure
   - Quick start guide
   - What to expect

2. **[PRE_TEST_CHECKLIST.md](PRE_TEST_CHECKLIST.md)**
   - Prerequisite checklist
   - System requirements
   - Verification steps

### Running Tests
3. **[TEST_README.md](TEST_README.md)**
   - Comprehensive testing guide
   - Detailed instructions
   - Troubleshooting

4. **[TEST_FILES_SUMMARY.md](TEST_FILES_SUMMARY.md)**
   - Overview of all test files
   - File locations
   - Quick reference

### Implementation Status
5. **[CUSTOM_TRAINING_LOOP_STATUS.md](CUSTOM_TRAINING_LOOP_STATUS.md)**
   - What's implemented
   - What's pending
   - Known issues
   - Technical details

## Test Files

### Scripts
- **`test_small_scale.py`** - Main test script
- **`monitor_test.py`** - Real-time monitoring
- **`../scripts/run_test_small_scale.sh`** - Automated launcher
- **`../scripts/check_prerequisites.sh`** - Prerequisite checker

### Configuration
- **`../config/training_config_test.yaml`** - Test configuration (3 questions, 1 epoch)
- **`../config/ppo_collaborative.yaml`** - Cypher Generator PPO config
- **`../config/orchestrator_ppo.yaml`** - Orchestrator PPO config

## Test Workflow

```
┌─────────────────────────────────────┐
│  1. Check Prerequisites             │
│     ./scripts/check_prerequisites.sh│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Run Component Tests             │
│     - Imports                       │
│     - Configuration                 │
│     - Agents                        │
│     - Environment                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Initialize Training System      │
│     - Load models                   │
│     - Create worker groups          │
│     - Create execution engines      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Run Training Loop               │
│     - Generate questions (3)        │
│     - Collect rollouts (3)          │
│     - Compute rewards               │
│     - Train agents (PPO)            │
│     - Save checkpoint               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Review Results                  │
│     - Check logs                    │
│     - Verify checkpoints            │
│     - Analyze metrics               │
└─────────────────────────────────────┘
```

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Prerequisites | 5-10s | System checks |
| Component Tests | 10-30s | Validate components |
| Initialization | 2-5min | Load models, setup Ray |
| Training Loop | 5-10min | 3 questions, 1 epoch |
| **Total** | **~10-15min** | Full test run |

## Success Indicators

✅ **Test Passes If:**
- All component tests pass
- Trainer initializes successfully
- 3 questions generated
- 3 rollouts collected
- Rewards computed
- PPO updates complete
- Checkpoint saved
- No CUDA errors
- GPU utilization > 0%

❌ **Test Fails If:**
- Import errors
- Model loading fails
- CUDA out of memory
- Ray initialization fails
- Network timeout (Neo4j)
- Worker group errors

## Common Commands

```bash
# Check prerequisites
./scripts/check_prerequisites.sh

# Run test (basic)
./scripts/run_test_small_scale.sh

# Run with monitoring
./scripts/run_test_small_scale.sh --monitor

# Skip component tests
./scripts/run_test_small_scale.sh --skip-component-tests

# Monitor separately
python -m rl_implementation.training.monitor_test

# View logs
tail -100 test_small_scale.log
tail -f test_small_scale.log  # Follow

# Search logs
grep -i error test_small_scale.log
grep -i reward test_small_scale.log

# Check checkpoints
ls -lh outputs/checkpoints/test_small_scale/

# GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Ray status
ray status
ray list nodes
```

## Troubleshooting Quick Reference

| Error | Solution |
|-------|----------|
| Import errors | `cd /nfs/turbo/.../rllm && pip install -e .` |
| Model not found | Check paths in PPO configs |
| CUDA OOM | Reduce batch sizes or free GPU memory |
| Ray socket path | Already handled (uses /tmp) |
| Neo4j timeout | Check network, increase timeout |
| Worker group error | Check Ray logs, restart Ray |

For detailed troubleshooting, see [TEST_README.md](TEST_README.md).

## After Successful Test

### 1. Review Results
```bash
# View metrics
tail -100 test_small_scale.log | grep -i reward

# Check checkpoints
ls -lh outputs/checkpoints/test_small_scale/

# Verify GPU usage (should be > 0%)
grep -i "gpu" test_small_scale.log
```

### 2. Scale Up Gradually

Create configs for larger tests:

**Small (10 questions, 2 epochs)**
```yaml
# training_config_small.yaml
num_epochs: 2
questions_per_epoch: 10
```

**Medium (50 questions, 5 epochs)**
```yaml
# training_config_medium.yaml
num_epochs: 5
questions_per_epoch: 50
```

**Full (512 questions, 45 epochs)**
```yaml
# training_config.yaml
num_epochs: 45
questions_per_epoch: 512
```

### 3. Tune Hyperparameters

Adjust in PPO configs:
- Learning rates (`actor_rollout_ref.actor.lr`)
- Batch sizes (`ppo_micro_batch_size_per_gpu`)
- PPO parameters (clip_ratio, entropy_coef, etc.)

Adjust in reward config:
- Reward weights
- Penalty factors
- Thresholds

### 4. Enable Advanced Features

- Curriculum learning (set `curriculum_enabled: true`)
- Validation (add `val_questions_path`)
- EMA evaluation (already enabled)
- Checkpoint frequency (adjust `save_frequency`)

## Known Limitations

Current implementation has some placeholder components:

1. **Question Generation** - Uses prompt builder but generates placeholders
2. **Orchestrator Evaluations** - Returns stub values
3. **Orchestrator PPO Updates** - Basic implementation
4. **Trajectory Parsing** - Simplified parsing

See [CUSTOM_TRAINING_LOOP_STATUS.md](CUSTOM_TRAINING_LOOP_STATUS.md) for details.

## Support Resources

- **Test Guide**: [TEST_README.md](TEST_README.md)
- **Prerequisites**: [PRE_TEST_CHECKLIST.md](PRE_TEST_CHECKLIST.md)
- **Implementation Status**: [CUSTOM_TRAINING_LOOP_STATUS.md](CUSTOM_TRAINING_LOOP_STATUS.md)
- **File Overview**: [TEST_FILES_SUMMARY.md](TEST_FILES_SUMMARY.md)
- **Completion Summary**: [SMALL_SCALE_TEST_COMPLETE.md](SMALL_SCALE_TEST_COMPLETE.md)

## Project Structure

```
rl_implementation/
├── training/
│   ├── train_collaborative_system.py  # Main training script
│   ├── test_small_scale.py           # Test script
│   ├── monitor_test.py               # Monitoring script
│   ├── README_TESTING.md             # This file
│   ├── TEST_README.md                # Detailed test guide
│   ├── PRE_TEST_CHECKLIST.md         # Prerequisites
│   ├── TEST_FILES_SUMMARY.md         # File overview
│   ├── CUSTOM_TRAINING_LOOP_STATUS.md # Implementation status
│   └── SMALL_SCALE_TEST_COMPLETE.md  # Completion summary
│   └── utils/
│       ├── rllm_components.py        # rllm helpers
│       ├── training_stability.py     # EMA, normalization
│       ├── curriculum_utils.py       # Curriculum learning
│       ├── checkpoint_manager.py     # Checkpointing
│       └── validation.py             # Validation
├── config/
│   ├── training_config_test.yaml     # Test config
│   ├── ppo_collaborative.yaml        # Cypher PPO config
│   └── orchestrator_ppo.yaml         # Orchestrator PPO config
└── ...

scripts/
├── run_test_small_scale.sh           # Test launcher
└── check_prerequisites.sh            # Prerequisite checker
```

## Getting Help

1. **Read Documentation** - Start with [SMALL_SCALE_TEST_COMPLETE.md](SMALL_SCALE_TEST_COMPLETE.md)
2. **Check Prerequisites** - Run `./scripts/check_prerequisites.sh`
3. **Review Logs** - Check `test_small_scale.log` for errors
4. **Consult Troubleshooting** - See [TEST_README.md](TEST_README.md)
5. **Check Implementation Status** - See [CUSTOM_TRAINING_LOOP_STATUS.md](CUSTOM_TRAINING_LOOP_STATUS.md)

## Ready to Test?

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
./scripts/check_prerequisites.sh && ./scripts/run_test_small_scale.sh
```

Good luck! 🚀

---

**Last Updated**: 2025-11-27  
**Status**: Ready for testing  
**Next Step**: Run `./scripts/run_test_small_scale.sh`

