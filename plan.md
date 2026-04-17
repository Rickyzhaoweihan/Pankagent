# Multi-Agent Text2Cypher RL Training Implementation Plan

## Phase 0: Data Generation & Setup (Days 1-2)

### 0.1 Generate Synthetic SFT Data for Schema Learning

**Location**: `examples/text2cypher/data_generation/`

Create `generate_schema_sft_data.py`:

- Load Neo4j schema from `examples/PanKLLM_RL_post-training/PankBaseAgent/text_to_cypher/data/input/neo4j_schema.json`
- Generate 1000-1500 synthetic examples covering:
  - Node property queries (200 examples)
  - Relationship type queries (200 examples)
  - Simple 1-hop traversals (300 examples)
  - 2-hop traversals with filtering (300 examples)
  - Aggregation queries (200 examples)
- Format: `{"question": str, "cypher": str, "schema_focus": str}`
- Save to `examples/text2cypher/data/schema_sft_train.jsonl`

### 0.2 Download Models

**Location**: `examples/text2cypher/models/`

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct --local-dir models/qwen2.5-coder-14b
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/qwen2.5-14b
```

### 0.3 Setup vLLM Inference Server

**Location**: `examples/text2cypher/scripts/`

Create `start_judge_server.sh`:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/qwen2.5-coder-14b \
  --served-model-name cypher-writer \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --tensor-parallel-size 1
```

---

## Phase 1: SFT Pre-Training (Days 3-4)

### 1.1 Create SFT Dataset Loader

**Location**: `examples/text2cypher/data/sft_dataset.py`

Implement dataset class that:

- Loads JSONL data
- Formats with Qwen chat template
- Converts to rLLM Dataset format
- Registers with `DatasetRegistry`

### 1.2 Create SFT Training Script for Text2Cypher

**Location**: `examples/text2cypher/train_text2cypher_sft.py`

Based on `examples/deepcoder/train_deepcoder.py` structure:

```python
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="sft_trainer")
def main(config):
    # Load schema SFT dataset
    train_dataset = DatasetRegistry.load_dataset("text2cypher_schema_sft", "train")
    
    # Override config
    config.model.partial_pretrain = "models/qwen2.5-coder-14b"
    config.model.lora_rank = 32
    config.trainer.total_epochs = 3
    config.data.train_batch_size = 128
    
    # Use AgentSFTTrainer from rllm.trainer.agent_sft_trainer
    trainer = AgentSFTTrainer(config)
    trainer.train()
```

Run: `python train_text2cypher_sft.py`

Expected output: `checkpoints/text2cypher_sft/epoch_3/`

---

## Phase 2: Core Agent Implementation (Days 5-6)

### 2.1 Multi-Step Text2Cypher Agent

**Location**: `examples/text2cypher/agents/text2cypher_agent.py`

Implement `MultiStepText2CypherAgent(BaseAgent)`:

- `__init__`: Load schema, set max_reasoning_steps=5
- `update_from_env`: Format observations for multi-turn reasoning
  - Turn 1: Schema + question
  - Turn 2-5: Previous results + cumulative context
- `update_from_model`: Extract Cypher from response, track reasoning history
- `reset`: Clear trajectory and reasoning history
- Properties: `chat_completions`, `trajectory`

Key: Each step generates ONE Cypher query, executes it, receives results as next observation.

### 2.2 Query Generator Agent

**Location**: `examples/text2cypher/agents/query_generator_agent.py`

Implement `QueryGeneratorAgent(BaseAgent)`:

- `__init__`: Load schema, set difficulty_level
- `update_from_env`: 
  - Initial: Schema elements + difficulty constraints
  - Feedback: Judge evaluation + suggestions
- `update_from_model`: Extract question from response
- Include curriculum stage constraints in prompt (max_hops, node_types)

### 2.3 Multi-Turn Graph Reasoning Environment

**Location**: `examples/text2cypher/environments/graph_reasoning_env.py`

Implement `GraphReasoningEnvironment(MultiTurnEnvironment)`:

- `__init__`: Set max_turns=5, load Neo4j connection details
- `get_reward_and_next_obs`: 
  - Execute Cypher against Pankbase API (reuse from `utils.py` line 170-175)
  - Compute turn reward (info_gain + diversity + relevance)
  - Return results as next observation
  - Track query_history and result_history
- `compute_turn_reward`: Reward individual steps
- `reset`: Clear history

---

## Phase 3: Reward Functions (Days 7-8)

### 3.1 LLM Judge Client

**Location**: `examples/text2cypher/rewards/llm_judge.py`

Implement `LLMJudge`:

- `__init__`: Connect to vLLM server at localhost:8001
- `evaluate(question, cypher, execution_results)`:
  - Build evaluation prompt with schema
  - Request: question_validity (0-100), cypher_correctness (0-100), difficulty (0-100)
  - Parse JSON response
  - Return dict with scores and feedback

### 3.2 Query Generator Reward Function

**Location**: `examples/text2cypher/rewards/query_gen_reward.py`

Implement `query_generator_reward_fn(task_info, action) -> RewardOutput`:

- Extract question from action
- Get Text2Cypher attempts (N=4 samples) from task_info
- Compute group-based entropy:
  ```python
  rewards = [attempt['reward'] for attempt in attempts]
  mean_reward = np.mean(rewards)
  std_reward = np.std(rewards)
  
  # Penalize low variance (too easy/hard)
  if std_reward < 0.1:
      entropy_score = 0.3 if mean_reward > 0.8 else 0.2
  else:
      entropy_score = 1.0 - abs(mean_reward - 0.5) * 2
      entropy_score = 0.6 * entropy_score + 0.4 * min(std_reward/0.3, 1.0)
  ```

- Get validity from LLM judge
- Compute diversity vs recent questions
- Return RewardOutput with weighted combination

### 3.3 Text2Cypher Multi-Step Reward Function

**Location**: `examples/text2cypher/rewards/text2cypher_reward.py`

Implement `text2cypher_multistep_reward_fn(task_info, action) -> RewardOutput`:

- Extract Cypher from action
- Validate syntax using existing validator
- Execute against Pankbase API
- Get correctness from LLM judge
- Compute step-level reward (syntax + execution + correctness)
- For final answer: Add outcome-based reward discounted by step distance
- Return RewardOutput with metadata

---

## Phase 4: Training Orchestration (Days 9-10)

### 4.1 Curriculum Manager

**Location**: `examples/text2cypher/training/curriculum.py`

Implement `CurriculumManager`:

```python
STAGES = {
    'easy': {'max_hops': 2, 'max_turns': 3, 'target_success': 0.7, 'epochs': 10},
    'medium': {'max_hops': 3, 'max_turns': 4, 'target_success': 0.5, 'epochs': 15},
    'hard': {'max_hops': 5, 'max_turns': 5, 'target_success': 0.4, 'epochs': 20}
}

def should_advance(recent_performance, current_stage):
    avg_success = np.mean(recent_performance[-100:])
    return avg_success > current_stage['target_success'] + 0.1

def should_regress(recent_performance, current_stage):
    avg_success = np.mean(recent_performance[-100:])
    return avg_success < current_stage['target_success'] - 0.2
```

### 4.2 Main Orchestration Script

**Location**: `examples/text2cypher/train_multi_agent.py`

Implement orchestration loop:

```python
def train_one_epoch(epoch, stage, query_gen_ckpt, text2cypher_ckpt):
    # 1. Generate questions (Query Generator inference)
    questions = generate_questions_batch(
        checkpoint=query_gen_ckpt,
        n=512,
        stage=stage,
        gpu=0
    )
    
    # 2. Generate Cypher responses (Text2Cypher inference, N=4 samples)
    text2cypher_attempts = []
    for question in questions:
        attempts = generate_cypher_samples(
            checkpoint=text2cypher_ckpt,
            question=question,
            n_samples=4,
            max_turns=stage['max_turns'],
            gpu=1
        )
        text2cypher_attempts.append(attempts)
    
    # 3. Judge all pairs (LLM Judge via vLLM server)
    evaluations = judge_batch(questions, text2cypher_attempts)
    
    # 4. Compute rewards
    query_gen_rewards = [
        query_generator_reward_fn(q, attempts, eval)
        for q, attempts, eval in zip(questions, text2cypher_attempts, evaluations)
    ]
    
    text2cypher_rewards = [
        text2cypher_multistep_reward_fn(q, attempt, eval)
        for q, attempts, eval in zip(questions, text2cypher_attempts, evaluations)
        for attempt in attempts
    ]
    
    # 5. Create datasets
    query_gen_dataset = create_query_gen_dataset(questions, query_gen_rewards)
    text2cypher_dataset = create_text2cypher_dataset(
        flatten_attempts(text2cypher_attempts),
        text2cypher_rewards
    )
    
    # 6. Register datasets
    DatasetRegistry.register_dataset(f"query_gen_epoch_{epoch}", query_gen_dataset)
    DatasetRegistry.register_dataset(f"text2cypher_epoch_{epoch}", text2cypher_dataset)
    
    # 7. Train Query Generator (subprocess)
    subprocess.run([
        "python", "train_query_generator_rl.py",
        f"data.train_files=query_gen_epoch_{epoch}",
        f"actor_rollout_ref.model.path={query_gen_ckpt}",
        "trainer.total_epochs=1"
    ])
    
    # 8. Train Text2Cypher (subprocess)
    subprocess.run([
        "python", "train_text2cypher_rl.py",
        f"data.train_files=text2cypher_epoch_{epoch}",
        f"actor_rollout_ref.model.path={text2cypher_ckpt}",
        "trainer.total_epochs=1"
    ])
    
    # 9. Return performance metrics
    return {
        'avg_query_gen_reward': np.mean(query_gen_rewards),
        'avg_text2cypher_reward': np.mean(text2cypher_rewards),
        'success_rate': np.mean([r > 0.8 for r in text2cypher_rewards])
    }

def main():
    curriculum = CurriculumManager()
    current_stage = 'easy'
    
    query_gen_ckpt = "checkpoints/query_gen_sft/epoch_3"
    text2cypher_ckpt = "checkpoints/text2cypher_sft/epoch_3"
    
    performance_history = []
    
    for epoch in range(100):  # Max epochs
        stage_config = curriculum.STAGES[current_stage]
        
        print(f"Epoch {epoch}: Stage={current_stage}")
        
        metrics = train_one_epoch(epoch, stage_config, query_gen_ckpt, text2cypher_ckpt)
        performance_history.append(metrics['success_rate'])
        
        # Update checkpoints
        query_gen_ckpt = f"checkpoints/query_gen/epoch_{epoch}"
        text2cypher_ckpt = f"checkpoints/text2cypher/epoch_{epoch}"
        
        # Check curriculum progression
        if curriculum.should_advance(performance_history, stage_config):
            if current_stage == 'easy':
                current_stage = 'medium'
            elif current_stage == 'medium':
                current_stage = 'hard'
        elif curriculum.should_regress(performance_history, stage_config):
            if current_stage == 'hard':
                current_stage = 'medium'
            elif current_stage == 'medium':
                current_stage = 'easy'
        
        # Save metrics
        save_metrics(epoch, current_stage, metrics)
        
        # Check completion
        if current_stage == 'hard' and len(performance_history) > 10:
            recent_success = np.mean(performance_history[-10:])
            if recent_success > 0.8:
                print("Training complete!")
                break
```

### 4.3 Individual RL Training Scripts

**Location**: `examples/text2cypher/`

Create `train_query_generator_rl.py`:

```python
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer")
def main(config):
    dataset = DatasetRegistry.load_dataset(config.data.train_files, "train")
    
    trainer = AgentTrainer(
        agent_class=QueryGeneratorAgent,
        agent_args={"schema": load_schema()},
        env_args={"reward_fn": query_generator_reward_fn},
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=dataset,
        val_dataset=None
    )
    trainer.train()
```

Create `train_text2cypher_rl.py`:

```python
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer")
def main(config):
    dataset = DatasetRegistry.load_dataset(config.data.train_files, "train")
    
    # Override for multi-turn
    config.agent.max_steps = 5
    config.agent.use_stepwise_advantage = True
    
    trainer = AgentTrainer(
        agent_class=MultiStepText2CypherAgent,
        agent_args={"max_reasoning_steps": 5},
        env_args={"reward_fn": text2cypher_multistep_reward_fn, "max_turns": 5},
        env_class=GraphReasoningEnvironment,
        config=config,
        train_dataset=dataset,
        val_dataset=None
    )
    trainer.train()
```

---

## Phase 5: Configuration & Utilities (Days 11-12)

### 5.1 PPO Training Config

**Location**: `examples/text2cypher/config/ppo_text2cypher.yaml`

Override defaults:

```yaml
data:
  max_prompt_length: 3072
  max_response_length: 512
  train_batch_size: 512

actor_rollout_ref:
  model:
    lora_rank: 32
    lora_alpha: 16
    enable_gradient_checkpointing: True
  actor:
    ppo_mini_batch_size: 128
    ppo_micro_batch_size_per_gpu: 4
    optim:
      lr: 5e-7
  rollout:
    name: vllm
    temperature: 0.7
    gpu_memory_utilization: 0.6

agent:
  max_steps: 5
  use_stepwise_advantage: True

trainer:
  n_gpus_per_node: 2
  total_epochs: 1
```

### 5.2 Helper Utilities

**Location**: `examples/text2cypher/utils/`

Create `inference_utils.py`:

- `generate_questions_batch()`: Batch inference for Query Generator
- `generate_cypher_samples()`: Multi-sample inference for Text2Cypher
- `judge_batch()`: Batch evaluation via vLLM server

Create `dataset_utils.py`:

- `create_query_gen_dataset()`: Format for Query Generator training
- `create_text2cypher_dataset()`: Format for Text2Cypher training
- `flatten_attempts()`: Convert multi-sample structure to flat list

Create `metrics_utils.py`:

- `save_metrics()`: Log to JSON/CSV
- `plot_training_curves()`: Visualize progress
- `compute_diversity()`: Track question diversity

---

## Phase 6: Testing & Validation (Days 13-14)

### 6.1 Unit Tests

**Location**: `examples/text2cypher/tests/`

Create tests for:

- Agent observation/action formatting
- Reward function calculations
- Curriculum progression logic
- Dataset creation

### 6.2 Integration Test

**Location**: `examples/text2cypher/test_pipeline.py`

Run mini version:

- 10 questions per epoch
- 2 epochs per stage
- Verify all components work together

### 6.3 Launch Full Training

```bash
# Terminal 1: Start vLLM judge server
bash scripts/start_judge_server.sh

# Terminal 2: Start training
python train_multi_agent.py
```

Monitor:

- `checkpoints/` for model saves
- `logs/metrics.json` for performance
- GPU memory usage (should be ~37-40GB per GPU)

---

## Expected Timeline

- **Days 1-2**: Data generation, model download, setup
- **Days 3-4**: SFT training (Text2Cypher only)
- **Days 5-6**: Agent implementations
- **Days 7-8**: Reward functions
- **Days 9-10**: Orchestration scripts
- **Days 11-12**: Config and utilities
- **Days 13-14**: Testing
- **Days 15-28**: Full RL training (3 stages)

**Total**: ~4 weeks from start to trained models

---

## Key Files Structure

```
examples/text2cypher/
├── agents/
│   ├── text2cypher_agent.py
│   └── query_generator_agent.py
├── environments/
│   └── graph_reasoning_env.py
├── rewards/
│   ├── llm_judge.py
│   ├── query_gen_reward.py
│   └── text2cypher_reward.py
├── training/
│   └── curriculum.py
├── data/
│   ├── sft_dataset.py
│   └── schema_sft_train.jsonl
├── data_generation/
│   └── generate_schema_sft_data.py
├── utils/
│   ├── inference_utils.py
│   ├── dataset_utils.py
│   └── metrics_utils.py
├── config/
│   └── ppo_text2cypher.yaml
├── scripts/
│   └── start_judge_server.sh
├── train_text2cypher_sft.py
├── train_query_generator_rl.py
├── train_text2cypher_rl.py
└── train_multi_agent.py
```