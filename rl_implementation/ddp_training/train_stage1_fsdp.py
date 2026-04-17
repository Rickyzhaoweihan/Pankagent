"""
Stage 1 FSDP Training: Cypher Generator with LoRA

Uses FSDP (Fully Sharded Data Parallel) to train the Cypher Generator
across 2 GPUs. This enables:
- Larger LoRA rank (128+)
- Full sequence lengths
- Entropy computation (optional)

Launch with:
    torchrun --nproc_per_node=2 -m rl_implementation.ddp_training.train_stage1_fsdp ...
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size():
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


def log_main(msg: str, level: str = "info"):
    """Log only on main process."""
    if is_main_process():
        getattr(logger, level)(msg)


class FSDPStage1Trainer:
    """
    Stage 1 FSDP Trainer for Cypher Generator.
    
    Coordinates:
    - Rollout collection (via external vLLM servers)
    - Reward computation
    - FSDP-based PPO updates
    """
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = get_world_size()
        
        self.fsdp_trainer = None
        self.rollout_collector = None
        self.ppo_updater = None
        self.reward_computer = None
        self.rollout_store = None
        
        self.current_epoch = 0
        
        log_main(f"FSDPStage1Trainer initialized (rank {self.local_rank}/{self.world_size})")
    
    def setup(self):
        """Initialize all components."""
        log_main("Setting up FSDP training...")
        
        # Initialize FSDP trainer
        self._init_fsdp_trainer()
        
        # Only main process handles rollout collection
        if is_main_process():
            self._init_inference_engine()
            self._init_rollout_collector()
            self._init_reward_computer()
            self._init_rollout_store()
        
        # All processes need PPO updater
        self._init_ppo_updater()
        
        # Barrier to sync all processes
        if dist.is_initialized():
            dist.barrier()
        
        log_main("FSDP setup complete!")
    
    def _init_fsdp_trainer(self):
        """Initialize FSDP trainer."""
        from rl_implementation.ddp_training.fsdp_trainer import FSDPTrainer, FSDPTrainerConfig
        
        trainer_config = FSDPTrainerConfig(
            model_path=self.config['cypher_model_path'],
            lora_rank=self.config.get('lora_rank', 128),
            lora_alpha=self.config.get('lora_alpha', 256),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            target_modules=self.config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            learning_rate=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            warmup_steps=self.config.get('warmup_steps', 100),
            total_steps=max(100, self.args.num_epochs * self.args.questions_per_epoch),
            sharding_strategy="FULL_SHARD",
            cpu_offload=self.config.get('cpu_offload', False),
            use_bf16=self.config.get('use_bf16', True),
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            checkpoint_dir=self.args.checkpoint_dir,
        )
        
        self.fsdp_trainer = FSDPTrainer(
            config=trainer_config,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        self.fsdp_trainer.setup()
    
    def _init_inference_engine(self):
        """Initialize inference engine for vLLM servers."""
        from rl_implementation.ddp_training.inference_engine import InferenceEngine, InferenceConfig
        
        inference_config = InferenceConfig(
            server_host=self.config.get('server_host', 'localhost'),
            orchestrator_port_question=int(os.environ.get('ORCH_QUESTION_PORT', 8001)),
            orchestrator_port_data_eval=int(os.environ.get('ORCH_DATA_EVAL_PORT', 8001)),
            orchestrator_port_synthesis=int(os.environ.get('ORCH_SYNTHESIS_PORT', 8001)),
            orchestrator_port_answer_eval=int(os.environ.get('ORCH_ANSWER_EVAL_PORT', 8001)),
            cypher_inference_port=int(os.environ.get('CYPHER_INFERENCE_PORT', 8002)),
            orchestrator_model_path=self.config['orchestrator_model_path'],
            cypher_model_path=self.config['cypher_model_path'],
            api_timeout=self.config.get('api_timeout', 180.0),
            max_retries=self.config.get('max_retries', 3),
        )
        
        self.inference_engine = InferenceEngine(inference_config)
        logger.info("Inference engine initialized")
    
    def _init_rollout_collector(self):
        """Initialize rollout collector."""
        from rl_implementation.ddp_training.rollout_collector import RolloutCollector
        
        self.rollout_collector = RolloutCollector(
            inference_engine=self.inference_engine,
            schema_path=self.config['schema_path'],
            neo4j_url=self.config['neo4j_url'],
            max_steps=self.config.get('max_steps', 5),
        )
        logger.info("Rollout collector initialized")
    
    def _init_reward_computer(self):
        """Initialize reward computer."""
        from rl_implementation.rewards.reward_aggregator import RewardAggregator
        
        self.reward_computer = RewardAggregator()
        logger.info("Reward computer initialized")
    
    def _init_rollout_store(self):
        """Initialize rollout store for persistence."""
        from rl_implementation.ddp_training.rollout_store import RolloutStore
        
        rollout_path = self.config.get(
            'rollout_store_path',
            os.path.join(self.args.log_dir, 'rollouts.jsonl')
        )
        self.rollout_store = RolloutStore(rollout_path)
        logger.info(f"Rollout store initialized: {rollout_path}")
    
    def _init_ppo_updater(self):
        """Initialize PPO updater."""
        from rl_implementation.ddp_training.ppo_updater import PPOUpdater, PPOConfig
        
        ppo_config = PPOConfig(
            clip_ratio=self.config.get('clip_ratio', 0.2),
            entropy_coeff=self.config.get('entropy_coeff', 0.0),
            use_kl_penalty=self.config.get('use_kl_penalty', False),
            kl_coeff=self.config.get('kl_coeff', 0.1),
            ppo_epochs=self.config.get('ppo_epochs', 1),
            mini_batch_size=self.config.get('mini_batch_size', 1),
            grpo_baseline=self.config.get('grpo_baseline', 'mean'),
            normalize_advantages=self.config.get('normalize_advantages', True),
            max_prompt_length=self.config.get('max_prompt_length', 2048),
            max_response_length=self.config.get('max_response_length', 512),
        )
        
        self.ppo_updater = PPOUpdater(ppo_config)
        log_main("PPO updater initialized")
    
    def train(self):
        """Main training loop."""
        log_main("=" * 60)
        log_main("Starting FSDP Training")
        log_main("=" * 60)
        
        num_epochs = self.args.num_epochs
        questions_per_epoch = self.args.questions_per_epoch
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            log_main(f"\n{'='*60}")
            log_main(f"Epoch {epoch}/{num_epochs}")
            log_main(f"{'='*60}")
            
            # Determine difficulty based on curriculum
            difficulty = self._get_difficulty(epoch)
            log_main(f"Difficulty: {difficulty}")
            
            try:
                epoch_metrics = self.train_epoch(difficulty)
                self._log_epoch_metrics(epoch, epoch_metrics)
                
                # Save checkpoint periodically
                if epoch % self.config.get('save_frequency', 5) == 0:
                    self._save_checkpoint(epoch)
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                raise
        
        # Final save
        self._save_checkpoint(num_epochs)
        log_main("Training complete!")
    
    def train_epoch(self, difficulty: str) -> Dict[str, float]:
        """Train for one epoch."""
        questions_per_epoch = self.args.questions_per_epoch
        
        # Phase 1: Collect rollouts (main process only)
        if is_main_process():
            log_main(f"Collecting {questions_per_epoch} rollouts...")
            trajectories = self.rollout_collector.collect_trajectories(
                num_questions=questions_per_epoch,
                difficulty=difficulty,
            )
            log_main(f"Collected {len(trajectories)} trajectories")
            
            # Phase 2: Compute rewards
            log_main("Computing rewards...")
            trajectories = self._compute_rewards(trajectories)
            
            # Save rollouts
            self.rollout_store.save_rollouts(
                [{"question": t.get("question", ""), "trajectory": t} for t in trajectories],
                epoch=self.current_epoch,
                difficulty=difficulty,
            )
        else:
            trajectories = None
        
        # Barrier before PPO update
        if dist.is_initialized():
            dist.barrier()
        
        # Broadcast trajectories to all ranks
        if dist.is_initialized():
            if is_main_process():
                # Serialize trajectories
                import pickle
                traj_bytes = pickle.dumps(trajectories)
                traj_tensor = torch.ByteTensor(list(traj_bytes)).cuda()
                size_tensor = torch.tensor([len(traj_bytes)], dtype=torch.long).cuda()
            else:
                size_tensor = torch.tensor([0], dtype=torch.long).cuda()
            
            # Broadcast size first
            dist.broadcast(size_tensor, src=0)
            
            if not is_main_process():
                traj_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8).cuda()
            
            # Broadcast data
            dist.broadcast(traj_tensor, src=0)
            
            if not is_main_process():
                import pickle
                trajectories = pickle.loads(bytes(traj_tensor.cpu().numpy()))
        
        # Phase 3: PPO Update (all ranks)
        log_main("Running PPO update...")
        update_metrics = self._ppo_update(trajectories)
        
        return update_metrics
    
    def _compute_rewards(self, trajectories: List[Dict]) -> List[Dict]:
        """Compute rewards for trajectories."""
        for traj in trajectories:
            reward_info = self.reward_computer.compute_reward(traj)
            traj['reward'] = reward_info['total_reward']
            traj['reward_breakdown'] = reward_info
        return trajectories
    
    def _ppo_update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Run PPO update on trajectories."""
        if not trajectories:
            return {"ppo_loss": 0.0}
        
        # Prepare batches
        batches = self.ppo_updater.prepare_batches(
            trajectories,
            self.fsdp_trainer.tokenizer,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in batches:
            # Compute old log probs
            old_log_probs = self.fsdp_trainer.compute_log_probs_for_rollout(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            batch["old_log_probs"] = old_log_probs
            
            # PPO step
            metrics = self.fsdp_trainer.train_step(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                advantages=batch["advantages"],
                old_log_probs=batch["old_log_probs"],
                clip_ratio=self.ppo_updater.config.clip_ratio,
                entropy_coeff=self.ppo_updater.config.entropy_coeff,
            )
            
            total_loss += metrics.get("loss", 0.0)
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"ppo_loss": avg_loss, "num_batches": num_batches}
    
    def _get_difficulty(self, epoch: int) -> str:
        """Get difficulty level based on curriculum."""
        curriculum = self.config.get('curriculum', {})
        easy_epochs = curriculum.get('easy', {}).get('epochs', 10)
        medium_epochs = curriculum.get('medium', {}).get('epochs', 15)
        
        if epoch <= easy_epochs:
            return 'easy'
        elif epoch <= easy_epochs + medium_epochs:
            return 'medium'
        else:
            return 'hard'
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        log_main(f"Epoch {epoch} metrics:")
        for key, value in metrics.items():
            log_main(f"  {key}: {value:.4f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir,
            f"checkpoint_epoch_{epoch}"
        )
        self.fsdp_trainer.save_checkpoint(checkpoint_path)
        log_main(f"Checkpoint saved: {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.fsdp_trainer:
            self.fsdp_trainer.cleanup()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 1 FSDP Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--questions_per_epoch", type=int, default=8, help="Questions per epoch")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = FSDPStage1Trainer(config, args)
    
    try:
        trainer.setup()
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

