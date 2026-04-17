# Custom Multi-Agent Training Loop - Implementation Status

## Overview

Successfully refactored `train_collaborative_system.py` to use **Strategy 2: Sequential Within-Epoch Updates with Model Swapping** for maximum GPU utilization and flexible multi-agent training.

## Training Strategy: Sequential Within-Epoch Updates

### Epoch Structure

Each epoch consists of 3 phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EPOCH N                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: ROLLOUT COLLECTION                                                 │
│ ┌───────────────────────────────────┬───────────────────────────────────┐  │
│ │ Cypher Generator (4 GPUs)         │ Orchestrator (4 GPUs)             │  │
│ │ - Inference mode                  │ - EMA for evaluation (Roles 2,4) │  │
│ │ - Generate Cypher queries         │ - Trainable for gen/synth (1,3)  │  │
│ └───────────────────────────────────┴───────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: CYPHER GENERATOR PPO UPDATE                                        │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │ Cypher Generator Training (8 GPUs - ALL)                              │  │
│ │ - PPO update from collected trajectories                              │  │
│ │ - Orchestrator unloaded from GPU memory                               │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: ORCHESTRATOR PPO UPDATE (Conditional)                              │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │ Orchestrator Training (8 GPUs - ALL)                                  │  │
│ │ - PPO update for Roles 1 & 3 (Question Gen, Answer Synth)            │  │
│ │ - Cypher Generator unloaded from GPU memory                           │  │
│ │ - EMA model updated after training                                    │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Orchestrator Update Schedule

```
Phase       | Epochs  | Orch Update Frequency
------------|---------|----------------------
Warmup      | 1-5     | Never (freeze Orch)
Alternating | 6-20    | Every 3 epochs
Joint       | 21+     | Every 2 epochs (or 1 for medium/hard)
```

### Orchestrator 4 Roles

| Role | Name              | Training | GPU Usage              |
|------|-------------------|----------|------------------------|
| 1    | Question Gen      | ✅ Yes   | Training pool (8 GPU)  |
| 2    | Data Quality Eval | ❌ No    | EMA model (inference)  |
| 3    | Answer Synthesis  | ✅ Yes   | Training pool (8 GPU)  |
| 4    | Answer Quality    | ❌ No    | EMA model (inference)  |

## Completed Tasks ✅

### 1. Sequential Training Configuration
**Files**: 
- `config/training_config.yaml`
- `config/training_config_test.yaml`

Implemented:
- `gpu_allocation.mode: "sequential"` setting
- `rollout.cypher_gpus` and `rollout.orch_gpus` for inference phase
- `training.gpus` for training phase (all GPUs)
- `orchestrator_schedule` with warmup, alternating, joint phases
- `ema` configuration for stable evaluation

### 2. Sequential Resource Manager
**File**: `training/utils/rllm_components.py`

New components:
- `SequentialResourceManager` - Manages GPU pools for different phases
- `create_sequential_resource_manager()` - Factory function
- `should_update_orchestrator()` - Schedule-based update decision
- `get_training_phase()` - Current phase detection
- `EMAOrchestrator` - EMA model management for stable evaluation
- `create_ema_orchestrator()` - Factory function

### 3. Refactored CollaborativeTrainer
**File**: `train_collaborative_system.py`

Changes:
- Added `_init_sequential_mode()` for Strategy 2 initialization
- Updated `__init__` to support `mode: "sequential"` or `mode: "parallel"`
- Added `orch_schedule_config` for update scheduling
- Added `ema_orchestrator` for EMA model tracking
- Updated imports for new components

### 4. 3-Phase Epoch Implementation
**File**: `train_collaborative_system.py` (`_train_epoch` method)

Implemented:
- Phase 1: Rollout collection with both models
- Phase 2: Cypher Generator PPO update
- Phase 3: Orchestrator PPO update (conditional based on schedule)
- EMA update after Orchestrator training
- Detailed logging for each phase

### 5. Updated PPO Configs
**Files**:
- `config/ppo_collaborative.yaml`
- `config/orchestrator_ppo.yaml`

Changes:
- `n_gpus_per_node: 4` - Default for rollout phase
- `n_training_gpus_per_node: 8` - For training phase
- Documentation of GPU allocation strategy

## Configuration Example

```yaml
# training_config.yaml
gpu_allocation:
  total_gpus: 8
  nnodes: 1
  mode: "sequential"  # Key setting for Strategy 2
  
  rollout:
    cypher_gpus: 4
    orch_gpus: 4
  
  training:
    gpus: 8  # All GPUs for training

orchestrator_schedule:
  warmup_epochs: 5
  alternating_end_epoch: 20
  warmup_frequency: 0
  alternating_frequency: 3
  joint_frequency: 2
  stage_overrides:
    medium: { joint_frequency: 1 }
    hard: { joint_frequency: 1 }

ema:
  enabled: true
  decay: 0.99
  decay_on_drift: 0.995
```

## Benefits of Strategy 2

1. **Maximum GPU Utilization**: Training uses all 8 GPUs for one model at a time
2. **Memory Efficient**: Only one model in training mode at a time
3. **Shared Experience**: Both agents learn from same rollout trajectories
4. **Flexible Scheduling**: Orchestrator updates controlled by phase
5. **Stable Evaluation**: EMA model provides consistent evaluation signals
6. **No Circular Dependency**: EMA breaks feedback loop between agents

## What Still Needs to Be Done

### High Priority

1. **Implement Actual Model Swapping**
   - Currently initializes both models for rollout
   - Need to implement loading/unloading for training phases
   - Save/restore model weights between phases

2. **Complete EMA Weight Updates**
   - Currently tracks update count
   - Need to implement actual weight copying between models
   - Integrate with worker group state management

3. **Test Sequential Training**
   - Run with 2-3 questions per epoch
   - Verify phase transitions work correctly
   - Monitor GPU memory during swaps

### Medium Priority

4. **Optimize Model Swapping**
   - Minimize checkpoint I/O
   - Use GPU memory efficiently
   - Consider weight sharing where possible

5. **Add Phase Monitoring**
   - Log GPU utilization per phase
   - Track model loading times
   - Monitor memory usage

### Low Priority

6. **Support Parallel Mode (Legacy)**
   - Maintain backward compatibility
   - Test with `mode: "parallel"` setting

## Testing Strategy

### Phase 1: Configuration Testing
- [x] Verify config loading with sequential mode
- [x] Verify schedule calculations
- [x] Verify EMA configuration

### Phase 2: Initialization Testing
- [ ] Test `_init_sequential_mode()` with 8 GPUs
- [ ] Verify both worker groups initialize
- [ ] Check resource pool creation

### Phase 3: Epoch Testing
- [ ] Run single epoch with 3 questions
- [ ] Verify all 3 phases execute
- [ ] Check Orchestrator update schedule

### Phase 4: Full Training
- [ ] Run 10 epochs
- [ ] Verify curriculum progression
- [ ] Monitor training metrics

## Files Modified

### New/Updated for Strategy 2
- `config/training_config.yaml` - Sequential mode config
- `config/training_config_test.yaml` - Test config
- `training/utils/rllm_components.py` - Sequential components
- `training/utils/__init__.py` - Exports
- `training/train_collaborative_system.py` - Main trainer
- `config/ppo_collaborative.yaml` - GPU settings
- `config/orchestrator_ppo.yaml` - GPU settings

## Success Criteria

- [x] Sequential mode configuration implemented
- [x] SequentialResourceManager implemented
- [x] Orchestrator schedule implemented
- [x] EMA Orchestrator tracking implemented
- [x] 3-phase epoch structure implemented
- [ ] Model swapping between phases working
- [ ] EMA weight updates working
- [ ] Small scale test passes
- [ ] Full training runs without errors
- [ ] All 8 GPUs utilized during training phases

## References

- Plan document: `cypher.plan.md`
- rllm execution engine: `rllm/engine/agent_execution_engine.py`
- rllm PPO trainer: `rllm/trainer/verl/agent_ppo_trainer.py`
- Worker groups: `verl/single_controller/ray.py`
- DataProto: `verl/__init__.py`
