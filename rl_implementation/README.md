# RL Implementation for PanKLLM

Multi-agent reinforcement learning system for biomedical knowledge graph reasoning using Cypher queries.

## 🎯 Current Status

### ✅ Phase 1 Complete: Cypher Generator Agent

The Cypher Generator Agent is fully implemented and ready to test!

**Implemented Components:**
- `agents/cypher_generator_agent.py` - Multi-step Cypher query generator (inherits from rllm.BaseAgent)
- `agents/experience_buffer.py` - Pattern storage stub (ready for full implementation)
- `utils/prompt_builder.py` - Token-budgeted prompt construction
- Test scripts for validation

**Key Features:**
- ✅ Integrates with rllm framework (BaseAgent, Action, Step, Trajectory)
- ✅ Multi-step reasoning (up to 5 Cypher queries per question)
- ✅ Token budget management (2450 tokens with intelligent compression)
- ✅ Schema-aware query generation
- ✅ Experience buffer integration (patterns, semantic warnings)
- ✅ Robust Cypher parsing from model responses
- ✅ Works with Qwen2.5-Coder-14B model

## 🚀 Quick Start

### Test the Agent

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/rl_implementation

# Quick test (generates one Cypher query)
python quick_test.py

# Comprehensive test suite
python test_cypher_agent.py
```

### Use the Agent

```python
from rl_implementation.agents import CypherGeneratorAgent

# Initialize with schema
agent = CypherGeneratorAgent(
    schema_path="path/to/kg_schema.json",
    max_steps=5
)

# Reset for new episode
agent.reset()

# Process observation from environment
observation = {'question': 'What genes interact with INS?'}
agent.update_from_env(observation, reward=0.0, done=False, info={})

# Get prompt for model inference
messages = agent.chat_completions

# After model generates response, parse it
action = agent.update_from_model(model_response)
print(action.action)  # The Cypher query or "DONE"
```

## 📁 Project Structure

```
rl_implementation/
├── agents/
│   ├── __init__.py
│   ├── cypher_generator_agent.py      ✅ Complete
│   ├── experience_buffer.py           ✅ Complete (stub)
│   └── orchestrator_agent.py          ⏳ TODO
│
├── utils/
│   ├── __init__.py
│   ├── prompt_builder.py              ✅ Complete
│   ├── schema_utils.py                ⏳ TODO
│   ├── data_quality_evaluator.py      ⏳ TODO
│   └── training_stability.py          ⏳ TODO
│
├── environments/
│   ├── __init__.py
│   ├── graph_reasoning_env.py         ⏳ TODO (next priority)
│   └── neo4j_executor.py              ⏳ TODO (next priority)
│
├── rewards/
│   ├── __init__.py
│   ├── cypher_reward.py               ⏳ TODO
│   └── orchestrator_reward.py         ⏳ TODO
│
├── training/
│   ├── __init__.py
│   ├── train_collaborative_system.py  ⏳ TODO
│   ├── train_cypher_rl.py             ⏳ TODO
│   └── train_orchestrator_rl.py       ⏳ TODO
│
├── config/
│   └── (YAML config files)            ⏳ TODO
│
├── test_cypher_agent.py               ✅ Complete
├── quick_test.py                      ✅ Complete
├── TESTING.md                         ✅ Complete
├── IMPLEMENTATION_STATUS.md           ✅ Complete
└── README.md                          ✅ This file
```

## 🧪 Testing

See [TESTING.md](TESTING.md) for detailed testing instructions.

**Quick validation:**
```bash
python quick_test.py
```

**Expected output:** A valid Cypher query like:
```cypher
MATCH (g1:gene)-[r:physical_interaction]->(g2:gene)
WHERE g2.name = 'INS'
WITH collect(DISTINCT g1) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```

## 📋 Implementation Plan

### Phase 1: Cypher Generator Agent ✅ COMPLETE
- [x] ExperienceBuffer stub
- [x] PromptBuilder with token management
- [x] CypherGeneratorAgent (BaseAgent implementation)
- [x] Test scripts

### Phase 2: Environment (NEXT)
- [ ] Neo4jExecutor - Query execution wrapper
- [ ] GraphReasoningEnvironment - MultiTurnEnvironment implementation
- [ ] Integration with agent

### Phase 3: Rewards
- [ ] Cypher reward function (multi-component)
- [ ] Orchestrator reward functions
- [ ] Reward utilities

### Phase 4: Orchestrator Agent
- [ ] Four-role orchestrator (question gen, data eval, synthesis, answer eval)
- [ ] Full ExperienceBuffer implementation
- [ ] Data quality evaluation utilities

### Phase 5: Training
- [ ] Training orchestration script
- [ ] EMA evaluator for stability
- [ ] Curriculum learning
- [ ] Validation and checkpointing

## 🔧 Technical Details

### Agent Architecture

The `CypherGeneratorAgent` follows rllm's BaseAgent interface:

```python
class CypherGeneratorAgent(BaseAgent):
    # Required methods
    def reset(self) -> None
    def update_from_env(self, observation, reward, done, info) -> None
    def update_from_model(self, response: str) -> Action
    
    # Required properties
    @property
    def chat_completions(self) -> list[dict[str, str]]
    @property
    def trajectory(self) -> Trajectory
```

### Prompt Structure

Total budget: 2450 tokens
- System Rules: 200 tokens (Cypher syntax requirements)
- Learned Rules: 200 tokens (patterns from experience buffer)
- Schema: 600 tokens (filtered to relevant types)
- History: 1200 tokens (compressed, last 2 steps detailed)
- Question + Instructions: 250 tokens

### Token Management

- Uses character-based estimation (1 token ≈ 4 chars)
- Intelligent compression for history
- Schema filtering based on question relevance
- Execution time indicators: ⚡(<100ms) ○(<500ms) △(<1000ms) ✗(>1000ms)
- Data quality indicators: ✓(good) ○(ok) ✗(poor)

## 🔗 Integration with rllm

The agent is designed to work seamlessly with rllm's training infrastructure:

```python
from rllm.trainer.agent_trainer import AgentTrainer
from rl_implementation.agents import CypherGeneratorAgent
from rl_implementation.environments import GraphReasoningEnvironment  # TODO

trainer = AgentTrainer(
    agent_class=CypherGeneratorAgent,
    env_class=GraphReasoningEnvironment,
    agent_args={'schema_path': 'path/to/schema.json'},
    env_args={'neo4j_config': {...}},
    config=ppo_config,
    train_dataset=dataset
)

trainer.train()
```

## 📚 Documentation

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Detailed implementation status
- [TESTING.md](TESTING.md) - Testing guide and examples
- [../docs/planning/IMPLEMENTATION_PLAN.md](../docs/planning/IMPLEMENTATION_PLAN.md) - Complete implementation plan
- [../docs/planning/IMPLEMENTATION_OVERVIEW.md](../docs/planning/IMPLEMENTATION_OVERVIEW.md) - Conceptual overview
- [../docs/planning/FILE_STRUCTURE.md](../docs/planning/FILE_STRUCTURE.md) - File organization guide

## 🎓 Model Information

**Cypher Generator Model:** Qwen2.5-Coder-14B
- Location: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b`
- Size: ~28GB (float16)
- Specialization: Code generation (ideal for Cypher queries)

**Orchestrator Model:** Qwen2.5-14B (to be used)
- Location: `/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b`
- Size: ~28GB (float16)
- Specialization: General language understanding

## 🐛 Troubleshooting

### Import Errors
Make sure you're running from the correct directory:
```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
python -m rl_implementation.quick_test
```

### Model Loading Issues
Check that the model path is correct and you have read permissions.

### Out of Memory
- Use fewer GPUs: `CUDA_VISIBLE_DEVICES=0 python quick_test.py`
- Reduce `max_new_tokens` in generation

## 🤝 Contributing

When implementing new components:
1. Follow the structure in `FILE_STRUCTURE.md`
2. Inherit from rllm base classes where appropriate
3. Add tests to verify functionality
4. Update this README with status

## 📝 License

[Add license information]

---

**Last Updated:** 2024-11-27  
**Status:** Phase 1 Complete - Cypher Generator Agent Ready for Testing  
**Next:** Implement GraphReasoningEnvironment and Neo4jExecutor
