# DDP Training Scripts

This directory contains all scripts for the PanKLLM reinforcement learning training pipeline.

## 📁 Directory Structure

```
scripts/
├── slurm/          # SLURM batch submission scripts (.sbatch)
├── tests/          # Test and debugging scripts
├── legacy/         # Old stage1 scripts (deprecated)
├── *.sh            # Shell scripts (runnable locally)
└── *.py            # Python scripts
```

## Quick Start

```bash
# Main entry point - runs everything (vLLM servers, collection, training)
sbatch slurm/submit_full_pipeline.sbatch

# With options
sbatch --export=CLEAR_HISTORY=1,ITERATIONS=20 slurm/submit_full_pipeline.sbatch
```

---

## 🎯 Main Pipeline (START HERE)

### SLURM Jobs (`slurm/`)
| Script | Description |
|--------|-------------|
| **`slurm/submit_full_pipeline.sbatch`** | **MAIN ENTRY POINT** - Handles everything |
| `slurm/submit_accelerated_pipeline.sbatch` | Faster version with overlapping collection+training |

### Shell Scripts
| Script | Description |
|--------|-------------|
| `run_full_training_pipeline.sh` | Dynamic training loop (called by sbatch) |
| `run_accelerated_pipeline.sh` | Accelerated version with overlapping |

### Python
| Script | Description |
|--------|-------------|
| `training_decision.py` | Analyzes metrics, decides which models to train |

---

## 📊 Rollout Collection

| Script | Description |
|--------|-------------|
| `collect_rollouts.py` | Main collection script - generates questions, runs Cypher |
| `run_collect_rollouts.sh` | Wrapper script for collection |
| `slurm/submit_collect_rollouts.sbatch` | SLURM job for standalone collection |
| `analyze_rollouts.py` | Analyze collected rollouts (rewards, errors, etc.) |

---

## 🏋️ Training Scripts

### Cypher Generator
| Script | Description |
|--------|-------------|
| `train_cypher_from_rollouts.py` | Train Cypher model from collected rollouts |
| `run_train_cypher.sh` | Wrapper script |
| `slurm/submit_train_cypher.sbatch` | SLURM job for standalone training |

### Orchestrator
| Script | Description |
|--------|-------------|
| `train_orchestrator_from_rollouts.py` | Train Orchestrator from rollouts |
| `run_train_orchestrator.sh` | Wrapper script |
| `slurm/submit_train_orchestrator.sbatch` | SLURM job for standalone training |

### Combined
| Script | Description |
|--------|-------------|
| `slurm/submit_train_both.sbatch` | Train both models (parallel or sequential) |
| `slurm/submit_training_only.sbatch` | Training without collection |
| `run_training_only.sh` | Training wrapper |

---

## 🖥️ vLLM Server Management

| Script | Description |
|--------|-------------|
| `start_vllm_servers.sh` | Start Orchestrator + Cypher vLLM servers |
| `stop_vllm_servers.sh` | Stop all vLLM servers |
| `restart_vllm_server.sh` | Restart a specific server with new model |
| `slurm/submit_vllm_servers.sbatch` | SLURM job to run vLLM servers |

---

## 🧪 Testing & Debugging (`tests/`)

| Script | Description |
|--------|-------------|
| `tests/test_ppo_only.py` | Test PPO training in isolation |
| `tests/test_ppo_simple.sh` | Simple PPO test wrapper |
| `tests/test_gpu_allocation.py` | Verify GPU allocation |
| `tests/test_gpu_isolation.py` | Test GPU isolation between processes |
| `tests/test_vllm_server.py` | Test vLLM server connectivity |
| `tests/diagnose_vllm.py` | Diagnose vLLM issues |

---

## 📦 Legacy (`legacy/`)

Old scripts from the integrated training approach. The new modular approach 
(rollout collection → training) is preferred.

| Script | Description |
|--------|-------------|
| `legacy/run_stage1.sh` | Original integrated training |
| `legacy/submit_stage1.sbatch` | SLURM job for stage1 |
| `legacy/run_stage1_test.sh` | Test stage1 |
| `legacy/run_fsdp_training.sh` | FSDP distributed training (experimental) |

---

## GPU Allocation (4 GPUs)

```
GPU 0: vLLM Orchestrator server (port 8001) - inference
GPU 1: vLLM Cypher Generator server (port 8002) - inference
GPU 2: Cypher Generator training
GPU 3: Orchestrator training
```

---

## Common Usage Patterns

### Fresh Start (Clear History)
```bash
sbatch --export=CLEAR_HISTORY=1 slurm/submit_full_pipeline.sbatch
```

### Use Existing Rollouts
```bash
sbatch --export=SKIP_FIRST_COLLECT=1 slurm/submit_full_pipeline.sbatch
```

### Custom Training
```bash
sbatch --export=ITERATIONS=20,QUESTIONS=128,DIFFICULTY=medium,EPOCHS=5 slurm/submit_full_pipeline.sbatch
```

### Run Until Convergence
```bash
sbatch --export=AUTO_MODE=1 slurm/submit_full_pipeline.sbatch
```

### Manual Steps (for debugging)
```bash
# 1. Start vLLM servers
./start_vllm_servers.sh

# 2. Collect rollouts
./run_collect_rollouts.sh

# 3. Analyze
python analyze_rollouts.py --rollouts /path/to/rollouts.jsonl

# 4. Train individually
./run_train_cypher.sh --rollouts /path/to/rollouts.jsonl
./run_train_orchestrator.sh --rollouts /path/to/rollouts.jsonl
```

---

## File Naming Convention

- `submit_*.sbatch` - SLURM batch submission scripts
- `run_*.sh` - Standalone shell scripts (can run locally)
- `*_from_rollouts.py` - Training scripts using pre-collected data
- `test_*.py/sh` - Test and debugging scripts

