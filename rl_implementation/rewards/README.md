# Reward Functions Module

This module implements reward functions for the PanKLLM RL post-training system, providing multi-component rewards for both Cypher Generator and Orchestrator agents.

## Overview

The reward functions follow the `rllm` framework interface, returning `RewardOutput` objects with:
- `reward`: Float value in [0.0, 1.0]
- `metadata`: Dictionary with component scores and penalties
- `is_correct`: Boolean indicating success

## Implemented Reward Functions

### 1. Cypher Generator Reward (`cypher_reward.py`)

**Function**: `cypher_generator_reward_fn(task_info: dict, action: str) -> RewardOutput`

**Components** (6 total):
1. **Answer Quality (25%)**: From Orchestrator's answer evaluation
2. **Data Quality (25%)**: From Orchestrator's data evaluation
3. **Trajectory Quality (20%)**: From Orchestrator's trajectory evaluation
4. **Cypher Correctness (15%)**: Syntax validation + execution success
5. **Retrieval Efficiency (10%)**: Data retrieved per step
6. **Execution Time (10%)**: Query speed (exponential decay)

**Penalties**:
- Step penalty: 0.05 per step above 3
- Early stop penalty: 0.2 (< 2 steps, < 5 results)
- Late stop penalty: 0.1 (5 steps, low score)
- Data quality penalty: 0.1 (data_quality < 0.4)
- Semantic ambiguity penalty: 0.05 × doubt_level (if doubt > 0.6)

**Input Format**:
```python
task_info = {
    'question': str,
    'cypher_trajectory': list[dict],  # query, result, success, execution_time_ms, num_results, has_data
    'answer_quality_score': float,    # 0-1 from Orchestrator
    'data_quality_score': float,      # 0-1 from Orchestrator
    'trajectory_quality_score': float, # 0-1 from Orchestrator
    'doubt_level': float,             # 0-1 from Orchestrator
    'num_steps': int
}
```

### 2. Orchestrator Generation Reward (`orchestrator_reward.py`)

**Function**: `orchestrator_generation_reward_fn(task_info: dict, action: str) -> RewardOutput`

**Components** (4 total):
1. **Answerability (40%)**: Did Cypher Generator succeed?
2. **Difficulty Appropriateness (30%)**: Success rate near target?
3. **Diversity (20%)**: Different from recent questions?
4. **Scope Adherence (10%)**: Within allowed types?

**Input Format**:
```python
task_info = {
    'question': str,
    'answerability': bool,
    'success_rate': float,
    'target_success_rate': float,
    'recent_questions': list[str],
    'scope_constraints': dict,
    'question_used_types': list[str]
}
```

### 3. Orchestrator Synthesis Reward (`orchestrator_reward.py`)

**Function**: `orchestrator_synthesis_reward_fn(task_info: dict, action: str) -> RewardOutput`

**Components** (2 total):
1. **Answer Quality (70%)**: From self-evaluation (Role 4)
2. **Data Utilization (30%)**: How well data is used

**Input Format**:
```python
task_info = {
    'question': str,
    'answer': str,
    'answer_quality_score': float,
    'trajectory': list[dict],
    'data_utilization': float  # Optional, computed if not provided
}
```

## Utility Functions (`reward_utils.py`)

### `validate_cypher(query: str) -> dict`
Validates Cypher query syntax (stub implementation).

**Returns**: `{'score': int (0-100), 'errors': list[str]}`

**Checks**:
- Basic keywords (MATCH, RETURN)
- Relationship variable syntax `[r:type]`
- WHERE clause presence
- LIMIT clause presence

### `compute_diversity_score(question: str, recent_questions: list[str]) -> float`
Computes diversity using Jaccard similarity on tokenized questions.

**Returns**: Float in [0.0, 1.0] where 1.0 is most diverse

### `compute_data_utilization(answer: str, trajectory: list[dict]) -> float`
Measures entity overlap between answer and retrieved data.

**Returns**: Float in [0.0, 1.0] indicating utilization ratio

### `normalize_reward(reward: float, mean: float, std: float) -> float`
Normalizes reward using running statistics.

### `clip_reward(reward: float, min_val: float, max_val: float) -> float`
Clips reward to prevent extreme values.

## Configuration

See `config/reward_config.yaml` for:
- Component weights
- Penalty rates
- Thresholds
- Normalization settings

## Testing

Run the test suite:
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation
python test_rewards.py
```

**Tests include**:
1. Cypher reward with good performance
2. Cypher reward with poor performance
3. Orchestrator generation reward
4. Orchestrator synthesis reward
5. Cypher validation utility
6. Diversity scoring utility
7. Data utilization utility
8. Edge cases (empty trajectory, perfect scores)

## Usage Example

```python
from rl_implementation.rewards import cypher_generator_reward_fn

# Prepare task info
task_info = {
    'question': 'What genes interact with PTPN22?',
    'cypher_trajectory': [
        {
            'query': 'MATCH (g:gene)-[r]-(g2:gene {name: "PTPN22"}) RETURN g, r',
            'success': True,
            'has_data': True,
            'num_results': 10,
            'execution_time_ms': 150.0
        }
    ],
    'answer_quality_score': 0.85,
    'data_quality_score': 0.80,
    'trajectory_quality_score': 0.75,
    'doubt_level': 0.2,
    'num_steps': 1
}

# Compute reward
result = cypher_generator_reward_fn(task_info, "")

print(f"Reward: {result.reward:.3f}")
print(f"Is correct: {result.is_correct}")
print(f"Metadata: {result.metadata}")
```

## Integration with rllm Framework

All reward functions follow the `rllm.rewards.reward_fn.RewardFunction` protocol:

```python
def reward_fn(task_info: dict, action: str) -> RewardOutput:
    ...
```

This ensures compatibility with `rllm.trainer.agent_trainer.AgentTrainer` for PPO training.

## Design Principles

1. **Multi-Component Rewards**: Balance multiple objectives (quality, efficiency, speed)
2. **Explicit Penalties**: Clear penalties for undesirable behavior
3. **Normalized Outputs**: All rewards in [0.0, 1.0] range
4. **Rich Metadata**: Detailed component scores for analysis
5. **Configurable**: Weights and thresholds in YAML config

## Future Enhancements

1. **Full Cypher Validation**: Replace stub with proper Cypher parser
2. **Semantic Similarity**: Use embeddings for diversity scoring
3. **Entity Recognition**: Improve data utilization with NER
4. **Adaptive Weights**: Learn component weights during training
5. **Curriculum-Aware**: Adjust thresholds based on difficulty stage

## References

- Implementation Plan: `docs/planning/IMPLEMENTATION_PLAN.md` (lines 162-287)
- File Structure: `docs/planning/FILE_STRUCTURE.md` (lines 449-459)
- Configuration: `config/reward_config.yaml`

