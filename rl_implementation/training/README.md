# Training Module

This module provides the complete training infrastructure for the collaborative multi-agent RL system, including stability mechanisms, curriculum learning, and checkpoint management.

## Quick Start

### Launch Training

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
bash scripts/launch_training.sh
```

### Custom Configuration

```bash
# Set environment variables
export NUM_EPOCHS=30
export QUESTIONS_PER_EPOCH=256
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/launch_training.sh
```

## Module Structure

```
training/
├── __init__.py                      # Module exports
├── train_collaborative_system.py   # Main training orchestration
├── test_training_utils.py          # Test suite
├── README.md                        # This file
├── TRAINING_STATUS.md               # Implementation status
└── utils/
    ├── training_stability.py        # EMA, reward normalization, drift detection
    ├── curriculum_utils.py          # Curriculum learning utilities
    ├── checkpoint_manager.py        # Checkpoint save/load
    └── validation.py                # Validation utilities
```

## Key Components

### 1. CollaborativeTrainer

Main training orchestration class that manages the entire training loop.

**Usage**:

```python
from rl_implementation.training import CollaborativeTrainer

config = {
    'schema_path': '/path/to/kg_schema.json',
    'checkpoint_dir': '/path/to/checkpoints',
    'log_dir': '/path/to/logs',
    'val_questions_path': '/path/to/val_questions.json',
    'questions_per_epoch': 512,
    'neo4j_url': 'https://...'
}

trainer = CollaborativeTrainer(config)
trainer.train(num_epochs=45)
```

### 2. Training Stability Utilities

**EMA Model Updates**:

```python
from rl_implementation.training import update_ema_model

update_ema_model(ema_model, train_model, decay=0.99)
```

**Reward Normalization**:

```python
from rl_implementation.training import RunningStats

reward_tracker = RunningStats(window_size=100)
reward_tracker.update(rewards, 'cypher')
normalized_rewards = reward_tracker.normalize(rewards, 'cypher')
```

**Training Phase Logic**:

```python
from rl_implementation.training import should_update_orchestrator, get_training_phase

phase = get_training_phase(epoch)  # 'warmup', 'alternating', or 'joint'
should_update = should_update_orchestrator(epoch, current_stage)
```

**Drift Detection**:

```python
from rl_implementation.training import detect_reward_drift, adjust_ema_decay

drift_detected = detect_reward_drift(train_metrics, val_metrics, threshold=0.2)
new_decay = adjust_ema_decay(current_decay, drift_detected)
```

### 3. Curriculum Learning

**CurriculumTracker**:

```python
from rl_implementation.training import CurriculumTracker

tracker = CurriculumTracker(initial_stage='easy', window_size=100)

# Update with success rate
stage_changed, new_stage = tracker.update(success_rate=0.8, epoch=10)

# Get current configuration
config = tracker.get_current_config()
# {'max_hops': 2, 'max_turns': 3, 'target_success': 0.7, ...}
```

**Manual Stage Control**:

```python
from rl_implementation.training import advance_stage, regress_stage, get_stage_config

next_stage = advance_stage('easy')  # Returns 'medium'
prev_stage = regress_stage('hard')  # Returns 'medium'
config = get_stage_config('easy')   # Get stage configuration
```

### 4. Checkpoint Management

**CheckpointManager**:

```python
from rl_implementation.training import CheckpointManager

manager = CheckpointManager(checkpoint_dir='/path/to/checkpoints')

# Save checkpoint
manager.save_checkpoint(
    epoch=10,
    cypher_gen=cypher_generator,
    orch_train=orchestrator_train,
    orch_eval=orchestrator_eval,
    exp_buffer=experience_buffer,
    reward_tracker=reward_tracker,
    current_stage='easy',
    metrics=epoch_metrics,
    is_best=True
)

# Load checkpoint
exp_buffer, reward_tracker, stage, metrics = manager.load_checkpoint(
    checkpoint_path='/path/to/checkpoint',
    cypher_gen=cypher_generator,
    orch_train=orchestrator_train,
    orch_eval=orchestrator_eval
)

# List checkpoints
checkpoints = manager.list_checkpoints()

# Cleanup old checkpoints
manager.cleanup_old_checkpoints(keep_last_n=3)
```

### 5. Validation

**Fixed Validation Set**:

```python
from rl_implementation.training import load_validation_set, validate_on_fixed_set

# Load validation questions
val_questions = load_validation_set('/path/to/val_questions.json', num_questions=100)

# Run validation
val_metrics = validate_on_fixed_set(
    cypher_gen=cypher_generator,
    orchestrator_eval=orchestrator_eval,
    questions=val_questions,
    env=environment
)
```

**Drift Detection**:

```python
from rl_implementation.training import compare_train_val_metrics

comparison = compare_train_val_metrics(
    train_metrics={'avg_answer_quality': 0.8, ...},
    val_metrics={'avg_answer_quality': 0.75, ...},
    threshold=0.2
)

if comparison['drift_detected']:
    print(f"Drift warnings: {comparison['warnings']}")
```

## Training Loop Overview

The main training loop (`CollaborativeTrainer.train()`) performs the following steps each epoch:

1. **Question Generation**: Orchestrator generates 512 training questions
2. **Rollout Collection**: 
   - Cypher Generator explores (up to 5 steps)
   - EMA Orchestrator evaluates data quality
   - Trainable Orchestrator synthesizes answer
   - EMA Orchestrator evaluates answer quality
3. **Reward Computation**: Compute rewards for both agents
4. **Reward Normalization**: Normalize using running statistics
5. **Cypher Training**: Train with PPO (every epoch)
6. **Orchestrator Training**: Train with PPO (conditional, phased)
7. **EMA Update**: Update EMA evaluator if Orchestrator trained
8. **Experience Buffer Update**: Extract patterns from episodes
9. **Validation**: Every 5 epochs, check for drift
10. **Curriculum Progression**: Check if should advance/regress
11. **Checkpointing**: Save every 5 epochs

## Phased Training Schedule

### Phase 1: Warmup (Epochs 0-4)
- **Cypher Generator**: Train every epoch
- **Orchestrator**: Frozen (not updated)
- **Purpose**: Let Cypher Generator learn basic patterns

### Phase 2: Alternating (Epochs 5-19)
- **Cypher Generator**: Train every epoch
- **Orchestrator**: Train every 3 epochs
- **Purpose**: Controlled co-adaptation

### Phase 3: Joint (Epochs 20+)
- **Cypher Generator**: Train every epoch
- **Orchestrator**: Train every 1-2 epochs (stage-dependent)
- **Purpose**: Full co-adaptation with stable evaluation

## Curriculum Stages

### Easy Stage
- Max hops: 2
- Max turns: 3
- Target success: 0.7
- Trajectory length: 1-2 steps
- Subgraph size: 10-50 nodes

### Medium Stage
- Max hops: 3
- Max turns: 4
- Target success: 0.5
- Trajectory length: 2-3 steps
- Subgraph size: 50-200 nodes

### Hard Stage
- Max hops: 5
- Max turns: 5
- Target success: 0.4
- Trajectory length: 3-5 steps
- Subgraph size: 200+ nodes

**Progression Rules**:
- Advance: success_rate > target + 0.1 for 100 episodes
- Regress: success_rate < target - 0.2 for 100 episodes

## Configuration Files

### Cypher Generator PPO Config (`config/ppo_collaborative.yaml`)

Key settings:
- Model: `qwen2.5-coder-14b`
- Batch size: 512
- Learning rate: 1e-6
- Max prompt length: 2450 tokens
- Multi-turn: Enabled (max 5 turns)
- Tensor parallelism: 2 GPUs per model

### Orchestrator PPO Config (`config/orchestrator_ppo.yaml`)

Key settings:
- Model: `qwen2.5-14b`
- Batch size: 512
- Learning rate: 1e-6
- Max prompt length: 2200 tokens
- Multi-turn: Disabled (single-turn per role)
- Tensor parallelism: 2 GPUs per model

## Testing

Run the test suite:

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation/training
python test_training_utils.py
```

Tests cover:
- RunningStats normalization
- EMA model updates
- Training phase determination
- Curriculum progression
- Checkpoint save/load
- Validation metrics

## Monitoring

### Metrics Logged

**Per Epoch**:
- Success rate
- Average answer quality
- Average data quality
- Average trajectory length
- Cypher reward
- Orchestrator generation reward
- Orchestrator synthesis reward
- Experience buffer size

**Validation (Every 5 Epochs)**:
- Validation answer quality
- Validation success rate
- Train/val gap (drift detection)

### Log Files

- Training log: `outputs/logs/{experiment_name}.log`
- Metrics history: `outputs/logs/metrics_history.json`
- Checkpoints: `outputs/checkpoints/checkpoint_epoch_XXX/`

## Troubleshooting

### Reward Drift Detected

If validation performance decreases while training rewards increase:

1. **Automatic**: EMA decay is automatically increased (0.99 → 0.995)
2. **Manual**: Reduce learning rate or increase EMA decay in config

### Curriculum Not Progressing

If stuck in a stage for too long:

1. Check success rate trends in logs
2. Adjust target success rates in `curriculum_utils.py`
3. Manually advance stage if needed

### Out of Memory

If GPU memory errors occur:

1. Reduce `train_batch_size` in config (512 → 256)
2. Reduce `tensor_model_parallel_size` (2 → 1)
3. Enable gradient checkpointing (already enabled by default)

### Training Too Slow

If training is slower than expected:

1. Increase `train_batch_size` if GPU memory allows
2. Reduce validation frequency (every 10 epochs instead of 5)
3. Use fewer GPUs for rollout generation

## Advanced Usage

### Custom Reward Functions

To use custom reward functions, update the config:

```yaml
custom_reward_function:
  path: my_module.my_rewards
  name: my_custom_reward_fn
```

### Resume from Specific Checkpoint

```python
trainer = CollaborativeTrainer(config)

# Load specific checkpoint
manager = CheckpointManager(config['checkpoint_dir'])
exp_buffer, reward_tracker, stage, metrics = manager.load_checkpoint(
    '/path/to/checkpoint_epoch_020',
    trainer.cypher_generator,
    trainer.orchestrator_train,
    trainer.orchestrator_eval
)

# Update trainer state
trainer.experience_buffer = exp_buffer
trainer.reward_tracker = reward_tracker
trainer.curriculum_tracker.current_stage = stage
trainer.current_epoch = metrics['epoch']

# Resume training
trainer.train(num_epochs=45)
```

### Custom Curriculum

```python
from rl_implementation.training.utils.curriculum_utils import CURRICULUM_STAGES

# Modify stage definitions
CURRICULUM_STAGES['easy']['target_success'] = 0.8
CURRICULUM_STAGES['medium']['max_hops'] = 4

# Create custom tracker
tracker = CurriculumTracker(initial_stage='medium', window_size=50)
```

## References

- **Implementation Plan**: `docs/planning/IMPLEMENTATION_PLAN.md`
- **Implementation Overview**: `docs/planning/IMPLEMENTATION_OVERVIEW.md`
- **Training Status**: `training/TRAINING_STATUS.md`
- **rllm Framework**: `rllm/trainer/agent_trainer.py`

## Support

For issues or questions:
1. Check `TRAINING_STATUS.md` for implementation details
2. Review test suite for usage examples
3. Consult `IMPLEMENTATION_PLAN.md` for design rationale

