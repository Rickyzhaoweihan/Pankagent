# Graph Reasoning Environment

This module implements the environment for multi-step Cypher query generation and execution against the Pankbase Neo4j knowledge graph.

## Components

### 1. Neo4jExecutor (`neo4j_executor.py`)

Wrapper for executing Cypher queries against the Pankbase Neo4j database via AWS Lambda endpoint.

**Features:**
- Executes Cypher queries via HTTP POST requests
- Handles query cleaning and JSON formatting
- Tracks execution time and result counts
- Robust error handling (timeouts, invalid queries, empty results)
- Detects data presence vs empty results

**Usage:**

```python
from rl_implementation.environments import Neo4jExecutor

executor = Neo4jExecutor(
    api_url="https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j",
    timeout=60
)

result = executor.execute_query("MATCH (g:gene {name: 'INS'}) RETURN g")

print(f"Success: {result['success']}")
print(f"Has Data: {result['has_data']}")
print(f"Num Results: {result['num_results']}")
print(f"Execution Time: {result['execution_time_ms']:.0f}ms")

executor.close()
```

### 2. GraphReasoningEnvironment (`graph_reasoning_env.py`)

Multi-turn RL environment for Cypher query generation. Inherits from `rllm.environments.base.multi_turn_env.MultiTurnEnvironment`.

**Features:**
- Multi-step query execution (up to 5 queries per episode)
- Structured observations with query history
- Execution metrics tracking
- DONE signal handling for early stopping
- Integration with rllm training framework

**Usage:**

```python
from rl_implementation.environments import GraphReasoningEnvironment

# Create environment
task = {'question': 'What genes interact with INS?'}
env = GraphReasoningEnvironment(task=task, max_turns=5)

# Reset for new episode
obs, info = env.reset()
print(f"Question: {obs['question']}")

# Execute queries
query1 = "MATCH (g:gene {name: 'INS'})-[r:physical_interaction]-(g2:gene) RETURN g2.name LIMIT 10"
next_obs, reward, done, info = env.step(query1)

print(f"Success: {next_obs['previous_result']['success']}")
print(f"Has Data: {next_obs['previous_result']['has_data']}")
print(f"Summary: {next_obs['previous_result']['data_summary']}")

# Continue with more queries or send DONE signal
if not done:
    env.step("DONE")

# Get trajectory data for reward computation
trajectory = env.get_trajectory_data()
print(f"Executed {trajectory['num_steps']} queries")

env.close()
```

## Observation Format

The environment returns structured observations with the following format:

```python
{
    'question': str,                    # Original NL question
    'previous_query': str,              # Last Cypher query executed
    'previous_result': {
        'success': bool,                # Query executed without errors
        'has_data': bool,               # Results returned (not empty)
        'result': dict,                 # Actual Neo4j result
        'num_results': int,             # Count of nodes/edges returned
        'execution_time_ms': float,     # Query execution time
        'data_summary': str,            # Brief summary for prompt
        'error': str | None             # Error message if any
    },
    'turn': int,                        # Current step (0-5)
    'history': {
        'queries': list[str],           # All previous queries
        'results': list[dict]           # All previous results
    }
}
```

## Integration with rllm Framework

The `GraphReasoningEnvironment` properly implements the `MultiTurnEnvironment` interface:

1. **`reset()`**: Initializes episode with question
2. **`get_reward_and_next_obs(task, action)`**: Executes query and returns observation
3. **`from_dict(env_args)`**: Factory method for rllm's AgentTrainer

**Example with rllm:**

```python
from rllm.trainer.agent_trainer import AgentTrainer
from rl_implementation.environments import GraphReasoningEnvironment
from rl_implementation.agents import CypherGeneratorAgent

# Environment args
env_args = {
    'api_url': 'https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j',
    'max_turns': 5,
    'schema': schema_dict
}

# Create trainer
trainer = AgentTrainer(
    agent_class=CypherGeneratorAgent,
    env_class=GraphReasoningEnvironment,
    agent_args={'schema_path': 'path/to/schema.json'},
    env_args=env_args,
    config=ppo_config
)

# Train
trainer.train()
```

## Testing

Run the test suite to verify functionality:

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation
python test_environment.py
```

**Tests include:**
1. Neo4j executor with sample queries
2. Environment reset
3. Environment step with query execution
4. Multi-turn trajectory
5. DONE signal handling
6. Error handling (invalid queries, timeouts)
7. Factory method (`from_dict`)

## Configuration

See `config/environment_config.yaml` for configuration options:

- **api_url**: Neo4j API endpoint
- **max_turns**: Maximum queries per episode (default: 5)
- **timeout**: Request timeout in seconds (default: 60)
- **schema_path**: Path to KG schema JSON file

## API Endpoints

**Production (default):**
```
https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j
```

**Development:**
```
https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/development/pank2-neo4j-api-development
```

## Error Handling

The environment handles various error conditions gracefully:

- **Timeout**: Returns error after 60s (configurable)
- **Invalid Cypher**: Returns error with message
- **Empty results**: Detects "No results" and empty arrays
- **API errors**: Captures and reports API error messages
- **Network errors**: Handles connection failures

All errors are logged and returned in the observation's `error` field.

## Execution Time Feedback

The environment provides visual indicators for query performance:

- ⚡ **Fast** (< 100ms)
- ○ **Good** (< 500ms)
- △ **Acceptable** (< 1000ms)
- ✗ **Slow** (> 1000ms)

These indicators are included in the `data_summary` field of observations.

## Next Steps

After implementing the environment, the next components to build are:

1. **Reward Functions** (`rewards/cypher_reward.py`)
   - Compute multi-component rewards
   - Integrate with Orchestrator evaluations

2. **Training Loop** (`training/train_collaborative_system.py`)
   - Orchestrate agent training
   - Manage experience buffer updates
   - Handle curriculum progression

3. **Experience Buffer** (full implementation in `agents/experience_buffer.py`)
   - Extract good patterns
   - Track bad data regions
   - Discover semantic ambiguities

