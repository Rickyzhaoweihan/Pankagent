"""
DDP-based Stage 1 Training for Cypher Generator.

This module provides a clean, simple DDP implementation for training the
Cypher Generator agent with LoRA, while using the Orchestrator for inference only.

Key Components:
- inference_engine: vLLM-based inference for both models
- ddp_trainer: PyTorch DDP wrapper with PEFT LoRA
- rollout_collector: Trajectory collection using existing agents/envs
- rollout_loader: Load rollouts from JSONL for offline training
- ppo_updater: GRPO-based PPO updates
- train_stage1: Main training loop

Architecture:
- Cypher Generator: 4 GPUs (DDP training with LoRA)
- Orchestrator: 4 GPUs (inference only, frozen)

Usage:
    # Collect rollouts
    bash scripts/run_collect_rollouts.sh
    
    # Train Cypher Generator from rollouts
    python scripts/train_cypher_from_rollouts.py --config config/train_cypher_config.yaml
    
    # Train Orchestrator from rollouts
    python scripts/train_orchestrator_from_rollouts.py --config config/train_orchestrator_config.yaml
    
    # Update vLLM server with trained model
    bash scripts/restart_vllm_server.sh --server cypher --model /path/to/trained/model
"""

from .inference_engine import InferenceEngine, InferenceConfig
from .ddp_trainer import DDPTrainer, DDPTrainerConfig
from .fsdp_trainer import FSDPTrainer, FSDPTrainerConfig
from .rollout_collector import RolloutCollector, RolloutCollectorConfig, Trajectory
from .rollout_loader import RolloutLoader, RolloutEntry, CypherBatchPreparer, OrchestratorBatchPreparer
from .ppo_updater import PPOUpdater, PPOConfig, GRPOAdvantageEstimator

__all__ = [
    # Inference
    "InferenceEngine",
    "InferenceConfig",
    # Training (Single GPU)
    "DDPTrainer",
    "DDPTrainerConfig",
    # Training (Multi-GPU FSDP)
    "FSDPTrainer",
    "FSDPTrainerConfig",
    # Rollout Collection
    "RolloutCollector",
    "RolloutCollectorConfig",
    "Trajectory",
    # Rollout Loading (for offline training)
    "RolloutLoader",
    "RolloutEntry",
    "CypherBatchPreparer",
    "OrchestratorBatchPreparer",
    # PPO
    "PPOUpdater",
    "PPOConfig",
    "GRPOAdvantageEstimator",
]

