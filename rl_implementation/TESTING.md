# Testing the Cypher Generator Agent

## Quick Test (Recommended)

The fastest way to test if the agent can generate Cypher queries:

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation
python quick_test.py
```

This will:
1. Load the Qwen2.5-Coder-14B model
2. Initialize the Cypher Generator Agent
3. Generate a Cypher query for the question: "What genes have physical interactions with INS?"
4. Validate the generated query

**Expected output**: A valid Cypher query with MATCH, WHERE, and RETURN statements.

## Comprehensive Test Suite

For more thorough testing:

```bash
python test_cypher_agent.py
```

This runs three tests:
1. **Basic Functionality Test** - Tests agent without model (fast)
2. **Multi-Step Interaction Test** - Tests multiple query steps (fast)
3. **Model Inference Test** - Tests with actual model (slow, optional)

## What Gets Tested

### 1. Agent Initialization
- ✓ Loads schema from JSON
- ✓ Initializes experience buffer
- ✓ Sets up prompt builder

### 2. Prompt Construction
- ✓ Builds prompts within 2450 token budget
- ✓ Includes system rules, schema, and question
- ✓ Formats history with execution metrics
- ✓ Adds learned patterns (currently empty)

### 3. Model Interaction
- ✓ Generates chat completions for model
- ✓ Parses Cypher queries from model responses
- ✓ Detects DONE signals
- ✓ Updates trajectory correctly

### 4. Multi-Step Reasoning
- ✓ Handles multiple query steps
- ✓ Maintains history across steps
- ✓ Formats observations with execution results
- ✓ Compresses history for token budget

### 5. Cypher Validation
- ✓ Extracts queries from code blocks
- ✓ Validates basic Cypher syntax
- ✓ Checks for required keywords (MATCH, RETURN, WHERE)

## Example Output

```
Question: What genes have physical interactions with INS?

MODEL RESPONSE:
================================================================================
Let me query the knowledge graph to find genes with physical interactions to INS.

```cypher
MATCH (g1:gene)-[r:physical_interaction]->(g2:gene)
WHERE g2.name = 'INS'
WITH collect(DISTINCT g1) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```
================================================================================

EXTRACTED CYPHER QUERY:
================================================================================
MATCH (g1:gene)-[r:physical_interaction]->(g2:gene)
WHERE g2.name = 'INS'
WITH collect(DISTINCT g1) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
================================================================================

Validation:
  ✓ Contains MATCH: True
  ✓ Contains RETURN: True
  ✓ Contains relationship []: True

✅ SUCCESS: Generated valid Cypher query!
```

## Requirements

- Python 3.10+
- PyTorch with CUDA (recommended) or CPU
- transformers library
- Access to Qwen2.5-Coder-14B model at: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b`

## Performance Notes

- **GPU (H100)**: ~2-5 seconds per query
- **CPU**: ~30-60 seconds per query (not recommended)
- **Memory**: ~28GB GPU memory for float16, ~56GB for float32

## Troubleshooting

### Model not found
If you get a model loading error, check the path:
```python
model_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b"
```

### Out of memory
If you run out of GPU memory, try:
1. Using fewer GPUs: `CUDA_VISIBLE_DEVICES=0 python quick_test.py`
2. Reducing max_new_tokens in generation

### Schema not found
The schema should be at:
```
/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json
```

## Next Steps

After verifying the agent works:
1. Implement GraphReasoningEnvironment to execute queries against Neo4j
2. Implement reward function to score generated queries
3. Set up full training loop with rllm's AgentTrainer
4. Test end-to-end with environment integration

## Manual Testing

You can also test the agent interactively:

```python
from rl_implementation.agents import CypherGeneratorAgent

# Initialize
agent = CypherGeneratorAgent(
    schema_path="path/to/schema.json",
    max_steps=5
)

# Reset for new episode
agent.reset()

# Provide observation
observation = {'question': 'What genes interact with INS?'}
agent.update_from_env(observation, reward=0.0, done=False, info={})

# Get prompt for model
messages = agent.chat_completions
print(messages[0]['content'])

# Simulate model response
response = "```cypher\nMATCH (g:gene)...\n```"
action = agent.update_from_model(response)

print(f"Generated action: {action.action}")
```

