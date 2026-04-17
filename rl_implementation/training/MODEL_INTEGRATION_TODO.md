# Model Integration TODO

## Current Status

The training infrastructure (Phase 4) is complete, but currently uses **stub implementations** of the agents. The actual model loading and inference needs to be integrated with rllm's `AgentTrainer`.

## Available Models

- **Cypher Generator**: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b`
- **Orchestrator**: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b`

## What's Working (Stub Implementation)

✅ Training loop orchestration
✅ EMA evaluator logic (gracefully skips when no model weights)
✅ Reward computation
✅ Curriculum learning
✅ Checkpoint management (saves buffers and trackers)
✅ Validation (basic structure)

## What Needs Integration

### 1. Replace Stub Training with rllm AgentTrainer

Current stub code in `train_collaborative_system.py`:

```python
def _train_cypher_generator(self, trajectories, rewards):
    # Stub implementation
    logger.debug("Cypher Generator training (stub)")
    return {'avg_reward': float(np.mean(rewards))}
```

**Needs to be replaced with**:

```python
def _train_cypher_generator(self, trajectories, rewards):
    from rllm.trainer.agent_trainer import AgentTrainer
    from rllm.data import Dataset
    
    # Create dataset from trajectories
    dataset = Dataset(
        data=trajectories,
        name='cypher_generator',
        split='train'
    )
    dataset.register()
    
    # Create trainer (or use existing one)
    if not hasattr(self, 'cypher_trainer'):
        self.cypher_trainer = AgentTrainer(
            agent_class=CypherGeneratorAgent,
            env_class=GraphReasoningEnvironment,
            agent_args={
                'schema_path': self.schema_path,
                'experience_buffer': self.experience_buffer,
                'max_steps': 5
            },
            env_args={
                'api_url': self.config.get('neo4j_url'),
                'max_turns': 5
            },
            config=self.cypher_ppo_config,
            train_dataset=dataset
        )
    
    # Train for one epoch
    metrics = self.cypher_trainer.train(num_epochs=1)
    return metrics
```

### 2. Load PPO Configurations

```python
def __init__(self, config):
    # ... existing code ...
    
    # Load PPO configs
    import yaml
    with open('rl_implementation/config/ppo_collaborative.yaml', 'r') as f:
        self.cypher_ppo_config = yaml.safe_load(f)
    
    with open('rl_implementation/config/orchestrator_ppo.yaml', 'r') as f:
        self.orch_ppo_config = yaml.safe_load(f)
```

### 3. Model Path Configuration

Update the config files to use the correct model paths:

**`config/ppo_collaborative.yaml`**:
```yaml
actor_rollout_ref:
  model:
    path: /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b
```

**`config/orchestrator_ppo.yaml`**:
```yaml
actor_rollout_ref:
  model:
    path: /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b
```

### 4. Rollout Collection with Actual Models

Current stub code:

```python
def _collect_rollouts(self, questions):
    # Stub implementation
    trajectories = []
    for question in questions:
        trajectory = {
            'question': question,
            'trajectory': [],
            'data_quality': {'data_quality_score': 0.5},
            'answer': "Placeholder",
            'answer_quality': {'score': 0.5}
        }
        trajectories.append(trajectory)
    return trajectories
```

**Needs to be replaced with actual rollout collection using the models**.

### 5. EMA Model Updates

The EMA update function now gracefully handles stub agents, but with actual models:

```python
# After Orchestrator training
update_ema_model(
    self.orchestrator_eval.model,  # Access the actual PyTorch model
    self.orchestrator_train.model,
    self.ema_decay
)
```

## Integration Steps

1. **Load Models in AgentTrainer**: rllm's `AgentTrainer` will load the models from the paths specified in the config files

2. **Create Trainers Once**: Initialize `cypher_trainer` and `orch_trainer` in `__init__()` instead of creating them each epoch

3. **Use Trainers for Rollouts**: rllm's `AgentTrainer` handles both rollout collection and training

4. **Update Checkpoint Manager**: Save/load the actual model checkpoints from rllm's trainer

## Example Full Integration

```python
class CollaborativeTrainer:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Load PPO configs
        self.cypher_ppo_config = self._load_config('config/ppo_collaborative.yaml')
        self.orch_ppo_config = self._load_config('config/orchestrator_ppo.yaml')
        
        # Create trainers
        self.cypher_trainer = self._create_cypher_trainer()
        self.orch_trainer = self._create_orch_trainer()
        
        # Get model references for EMA
        self.orchestrator_train_model = self.orch_trainer.actor_model
        self.orchestrator_eval_model = copy.deepcopy(self.orchestrator_train_model)
    
    def _train_epoch(self):
        # Use trainer's rollout collection
        rollouts = self.cypher_trainer.collect_rollouts(
            num_episodes=self.config['questions_per_epoch']
        )
        
        # Compute rewards
        rewards = self._compute_rewards(rollouts)
        
        # Train with PPO
        self.cypher_trainer.train_on_batch(rollouts, rewards)
        
        # Conditional Orchestrator training
        if should_update_orchestrator(self.current_epoch):
            self.orch_trainer.train_on_batch(orch_rollouts, orch_rewards)
            
            # Update EMA
            update_ema_model(
                self.orchestrator_eval_model,
                self.orchestrator_train_model,
                self.ema_decay
            )
```

## Current Workaround

The current implementation runs without errors by:
- Skipping EMA updates when models don't have `parameters()`
- Skipping model checkpoint saves when agents don't have `state_dict()`
- Using placeholder trajectories and rewards

This allows testing the training loop structure without actual model training.

## Next Steps

To enable actual training with the downloaded models:

1. Study rllm's `AgentTrainer` API and examples
2. Implement `_create_cypher_trainer()` and `_create_orch_trainer()`
3. Replace stub rollout collection with actual model inference
4. Replace stub training with actual PPO updates
5. Test with a small number of questions first (e.g., 10 questions, 2 epochs)

## References

- rllm AgentTrainer: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/rllm/trainer/agent_trainer.py`
- rllm examples: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/`
- PPO configs: `rl_implementation/config/ppo_*.yaml`

