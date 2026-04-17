# DDP Stage 1 Training

Clean, simple DDP-based training for the Cypher Generator agent with LoRA.
Uses external vLLM OpenAI-compatible servers for inference.

## Architecture

### GPU Allocation (8 H100s)

| Component | GPUs | Mode | Ports |
|-----------|------|------|-------|
| Orchestrator - Question | 0 | vLLM Server | 8001 |
| Orchestrator - DataEval | 1 | vLLM Server | 8002 |
| Orchestrator - Synthesis | 2 | vLLM Server | 8003 |
| Orchestrator - AnswerEval | 3 | vLLM Server | 8004 |
| Cypher Generator (inference) | 4 | vLLM Server | 8005 |
| Cypher Generator (training) | 5-7 | DDP + LoRA | - |

### Key Design Decisions

1. **External vLLM Servers** - Each model runs as an OpenAI-compatible API server
2. **HTTP API** - Training code connects to servers via HTTP requests
3. **PyTorch DDP** for multi-GPU training with stability
4. **PEFT LoRA** for parameter-efficient fine-tuning
5. **GRPO** for advantage estimation (no critic network)
6. **Reuses existing** agents, environments, and reward functions

## File Structure

```
ddp_training/
├── __init__.py              # Module exports
├── README.md                # This file
├── inference_engine.py      # HTTP client for vLLM servers
├── ddp_trainer.py           # PyTorch DDP + PEFT LoRA trainer
├── rollout_collector.py     # Trajectory collection
├── ppo_updater.py           # GRPO advantage + PPO loss
├── train_stage1.py          # Main training loop
├── config/
│   ├── stage1_config.yaml       # Full training config
│   └── stage1_test_config.yaml  # Test config (small scale)
└── scripts/
    ├── run_stage1.sh            # Full launch script (starts servers + training)
    ├── run_stage1_test.sh       # Test launch script
    ├── start_vllm_servers.sh    # Start vLLM servers only
    └── stop_vllm_servers.sh     # Stop vLLM servers
```

## Quick Start

### Option 1: All-in-One (Recommended)

The `run_stage1.sh` script handles everything: starting vLLM servers, waiting for them, running training, and cleanup.

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

# Set model paths
# Orchestrator: Qwen2.5-14B (general-purpose LLM)
export ORCHESTRATOR_MODEL=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b
# Cypher Generator: Qwen2.5-Coder-14B-SFT (code-specialized)
export CYPHER_MODEL=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b

# Run full training
bash rl_implementation/ddp_training/scripts/run_stage1.sh
```

### Option 2: Manual Server Management

For more control, start servers separately:

```bash
# Terminal 1: Start vLLM servers
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training

bash rl_implementation/ddp_training/scripts/start_vllm_servers.sh \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b \
    /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b

# Wait for "All vLLM servers are ready!" message

# Terminal 2: Run training
export ORCH_QUESTION_PORT=8001
export ORCH_DATA_EVAL_PORT=8002
export ORCH_SYNTHESIS_PORT=8003
export ORCH_ANSWER_EVAL_PORT=8004
export CYPHER_INFERENCE_PORT=8005
export CUDA_VISIBLE_DEVICES=5,6,7

python -m rl_implementation.ddp_training.train_stage1 \
    --config rl_implementation/ddp_training/config/stage1_config.yaml

# When done, stop servers
bash rl_implementation/ddp_training/scripts/stop_vllm_servers.sh
```

### Option 3: Minimal Manual Setup

Start servers directly with vLLM:

```bash
# Model paths
# Orchestrator: Qwen2.5-14B (general-purpose LLM)
ORCH_MODEL=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-14b
# Cypher Generator: Qwen2.5-Coder-14B-SFT (code-specialized)
CYPHER_MODEL=/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b

# Start 5 vLLM servers on GPUs 0-4
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $ORCH_MODEL --port 8001 &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $ORCH_MODEL --port 8002 &

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $ORCH_MODEL --port 8003 &

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $ORCH_MODEL --port 8004 &

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model $CYPHER_MODEL --port 8005 &

# Wait for servers to be ready (check health endpoints)
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health

# Run training on GPUs 5-7
CUDA_VISIBLE_DEVICES=5,6,7 python -m rl_implementation.ddp_training.train_stage1 \
    --config rl_implementation/ddp_training/config/stage1_config.yaml
```

## Training Loop

Each epoch follows this sequence:

1. **Question Generation** (Orchestrator server @ port 8001)
   - Generate training questions based on curriculum difficulty
   - Uses Orchestrator in `generation` mode

2. **Trajectory Collection** (Cypher Generator server @ port 8005 + Neo4j)
   - Multi-turn Cypher query generation
   - Execute queries against Neo4j database
   - Collect up to 5 queries per question

3. **Evaluation** (Orchestrator servers @ ports 8002-8004)
   - Data quality evaluation (port 8002)
   - Answer synthesis (port 8003)
   - Answer quality evaluation (port 8004)

4. **Reward Computation**
   - Uses existing `cypher_generator_reward_fn`
   - Multi-component reward with penalties

5. **PPO Update** (Cypher Generator training on GPUs 5-7)
   - GRPO advantage estimation
   - Clipped surrogate objective
   - Only LoRA parameters updated

## Configuration

### vLLM Server Settings

```yaml
# Server configuration
server_host: localhost
orchestrator_port_question: 8001
orchestrator_port_data_eval: 8002
orchestrator_port_synthesis: 8003
orchestrator_port_answer_eval: 8004
cypher_inference_port: 8005

# API settings
api_timeout: 120.0
max_retries: 3
```

### Training Settings

```yaml
# LoRA
lora_rank: 32
lora_alpha: 64

# Training
learning_rate: 1.0e-5
clip_ratio: 0.2
entropy_coeff: 0.01

# Curriculum
curriculum:
  easy:
    epochs: 10
    max_hops: 1
  medium:
    epochs: 15
    max_hops: 2
  hard:
    epochs: 20
    max_hops: 3
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_MODEL` | `models/qwen2.5-14b` | Path to orchestrator model (Qwen2.5-14B general-purpose) |
| `CYPHER_MODEL` | `models/qwen2.5-coder-14b` | Path to cypher generator model (Qwen2.5-Coder-14B base) |
| `ORCH_QUESTION_PORT` | 8001 | Orchestrator question server port |
| `ORCH_DATA_EVAL_PORT` | 8002 | Orchestrator data eval server port |
| `ORCH_SYNTHESIS_PORT` | 8003 | Orchestrator synthesis server port |
| `ORCH_ANSWER_EVAL_PORT` | 8004 | Orchestrator answer eval server port |
| `CYPHER_INFERENCE_PORT` | 8005 | Cypher generator server port |
| `VLLM_MAX_MODEL_LEN` | 8192 | Max model length for vLLM |
| `VLLM_GPU_MEMORY_UTIL` | 0.85 | GPU memory utilization |

## Components

### InferenceEngine

HTTP client that connects to external vLLM servers:

```python
from rl_implementation.ddp_training import InferenceEngine, InferenceConfig

config = InferenceConfig(
    cypher_model_path="...",
    orchestrator_model_path="...",
    orchestrator_port_question=8001,
    orchestrator_port_data_eval=8002,
    orchestrator_port_synthesis=8003,
    orchestrator_port_answer_eval=8004,
    cypher_inference_port=8005,
)
engine = InferenceEngine(config)
engine.initialize()  # Waits for servers to be ready

responses = engine.generate_cypher(prompts)
```

### DDPTrainer

PyTorch DDP wrapper with PEFT LoRA:

```python
from rl_implementation.ddp_training import DDPTrainer, DDPTrainerConfig

config = DDPTrainerConfig(
    model_path="...",
    lora_rank=32,
    lora_alpha=64,
)
trainer = DDPTrainer(config, local_rank=0, world_size=1)
trainer.setup()

metrics = trainer.train_step(input_ids, attention_mask, labels, advantages, old_log_probs)
```

### RolloutCollector

Collects trajectories using existing agents and environments:

```python
from rl_implementation.ddp_training import RolloutCollector, RolloutCollectorConfig

config = RolloutCollectorConfig(schema_path="...")
collector = RolloutCollector(config, inference_engine, tokenizer)

questions = collector.generate_questions(num_questions=64)
trajectories = collector.collect_trajectories(questions)
trajectories = collector.evaluate_trajectories(trajectories)
trajectories = collector.compute_rewards(trajectories)
```

### PPOUpdater

GRPO advantage estimation and PPO loss:

```python
from rl_implementation.ddp_training import PPOUpdater, PPOConfig

config = PPOConfig(clip_ratio=0.2, grpo_baseline="mean")
updater = PPOUpdater(config)

batch = updater.prepare_batch(input_ids, attention_mask, labels, response_mask, rewards, old_log_probs)
metrics = updater.update_step(trainer, batch)
```

## Outputs

### Checkpoints

Saved to `outputs/stage1_ddp/checkpoints/`:
- `epoch_XXXX/` - LoRA adapter weights
- `epoch_XXXX/trainer_state.json` - Training state

### Logs

Saved to `outputs/stage1_ddp/logs/`:
- `training_YYYYMMDD_HHMMSS.log` - Training log
- `metrics_history.json` - Epoch metrics
- `vllm_*.log` - vLLM server logs (when using scripts)

## Troubleshooting

### vLLM Server Not Starting

1. Check GPU availability: `nvidia-smi`
2. Check server logs: `cat vllm_logs/orch_question_gpu0.log`
3. Ensure port is not in use: `lsof -i :8001`
4. Kill existing servers: `pkill -f "vllm.entrypoints.openai.api_server"`

### Connection Refused

1. Wait longer for servers to start (model loading takes time)
2. Check health endpoint: `curl http://localhost:8001/health`
3. Verify correct port numbers in config

### CUDA Out of Memory

1. Reduce `VLLM_GPU_MEMORY_UTIL` (e.g., 0.7)
2. Reduce `batch_size` and `mini_batch_size`
3. Enable `gradient_checkpointing: true`
4. Reduce `VLLM_MAX_MODEL_LEN`

### Neo4j Connection Issues

Check the API endpoint and network connectivity:
```bash
curl -X POST "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j" \
    -H "Content-Type: application/json" \
    -d '{"query": "MATCH (n) RETURN count(n) LIMIT 1"}'
```

## Dependencies

- PyTorch >= 2.0
- vLLM >= 0.6.3
- PEFT >= 0.7.0
- transformers >= 4.36.0
- requests (for HTTP API calls)

## Advantages of External Server Architecture

1. **Isolation** - Model loading issues don't crash training
2. **Flexibility** - Start/stop servers independently
3. **Debugging** - Each server has its own logs
4. **Reusability** - Same servers can be used for multiple training runs
5. **Scalability** - Easy to add more servers or change GPU allocation
