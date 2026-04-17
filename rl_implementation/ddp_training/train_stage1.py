"""
Stage 1 Training: Cypher Generator Only.

Main training script for Stage 1 of the collaborative multi-agent system.
In Stage 1, only the Cypher Generator is trained while the Orchestrator
is used for inference only.

Architecture:
- Cypher Generator: 4 GPUs with DDP + LoRA training
- Orchestrator: 4 GPUs with vLLM inference (frozen)

Training Loop:
1. Generate questions using Orchestrator
2. Collect trajectories using Cypher Generator + Neo4j
3. Evaluate trajectories using Orchestrator
4. Compute rewards
5. PPO update on Cypher Generator
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# CRITICAL: Respect CUDA_VISIBLE_DEVICES if already set (e.g., by SLURM srun)
# Only set default if not already configured
_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cuda_visible is None or _cuda_visible == '':
    # Not in SLURM job, set default for 3-GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print(f"[train_stage1] Setting CUDA_VISIBLE_DEVICES=2 (default for training)")
else:
    # SLURM/srun already set it - don't override
    print(f"[train_stage1] Using SLURM-allocated GPU: CUDA_VISIBLE_DEVICES={_cuda_visible}")

# Verify GPU setting by importing torch and checking
def _verify_gpu():
    """Verify GPU allocation is correct."""
    import torch
    print(f"[train_stage1] torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
            print(f"[train_stage1] cuda:{i} = {props.name}, free: {free_mem:.1f} GB")
        # Verify we have at least 1 GPU visible
        if torch.cuda.device_count() >= 1:
            free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
            if free_mem > 60:  # Should have plenty of free memory
                print(f"[train_stage1] ✓ GPU allocation correct (free: {free_mem:.1f} GB)")
            else:
                print(f"[train_stage1] ⚠ Warning: GPU has only {free_mem:.1f} GB free")
        else:
            print(f"[train_stage1] ⚠ Warning: Expected 1 GPU, found {torch.cuda.device_count()}")

# Run verification
_verify_gpu()

# NOTE: Do NOT import torch at module level!
# This would initialize CUDA before vLLM worker processes start,
# causing "CUDA device busy" errors.
# torch is imported inside methods that need it.

import yaml

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import inference engine first (no torch dependency at module level)
from rl_implementation.ddp_training.inference_engine import InferenceConfig, InferenceEngine
from rl_implementation.ddp_training.rollout_store import RolloutStore

# NOTE: DDPTrainer, RolloutCollector, PPOUpdater are imported LAZILY
# inside methods to avoid importing torch before vLLM workers start.
# from rl_implementation.ddp_training.ddp_trainer import DDPTrainer, DDPTrainerConfig
# from rl_implementation.ddp_training.rollout_collector import RolloutCollector, RolloutCollectorConfig
# from rl_implementation.ddp_training.ppo_updater import PPOUpdater, PPOConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1Trainer:
    """
    Stage 1 Trainer for Cypher Generator.
    
    Optimized for single GPU training with LoRA.
    
    Training Loop:
    1. Question generation (Orchestrator inference via vLLM)
    2. Trajectory collection (Cypher Generator inference + Neo4j)
    3. Evaluation (Orchestrator inference via vLLM)
    4. Reward computation
    5. PPO update (Cypher Generator training with LoRA)
    6. Save rollouts to persistent storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Stage 1 trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.epoch = 0
        self.global_step = 0
        
        # Single GPU mode (default)
        self.local_rank = config.get('local_rank', 0)
        self.world_size = config.get('world_size', 1)
        self.is_main_process = True  # Always main process in single GPU mode
        
        # Extract paths
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 45)
        self.questions_per_epoch = config.get('questions_per_epoch', 64)
        self.save_frequency = config.get('save_frequency', 5)
        self.log_frequency = config.get('log_frequency', 1)
        
        # Curriculum settings
        self.curriculum_stages = config.get('curriculum', {
            'easy': {'epochs': 10, 'max_hops': 1},
            'medium': {'epochs': 15, 'max_hops': 2},
            'hard': {'epochs': 20, 'max_hops': 3},
        })
        
        # Initialize components in order
        logger.info("=" * 60)
        logger.info("INITIALIZING STAGE 1 TRAINER")
        logger.info("=" * 60)
        
        self._init_inference_engine()  # GPU 0-1 for vLLM servers
        self._init_ddp_trainer()        # GPU 2 for training
        self._init_rollout_collector()
        self._init_ppo_updater()
        self._init_rollout_store()      # Persistent storage for rollouts
        
        # Metrics tracking
        self.metrics_history = []
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_epoch = 0
        
        logger.info("=" * 60)
        logger.info("Stage1Trainer initialized successfully!")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Rollout store: {self.rollout_store.store_path}")
        logger.info("=" * 60)
    
    def _init_inference_engine(self):
        """Initialize connection to external vLLM OpenAI servers."""
        logger.info("=" * 60)
        logger.info("STEP 1: Connecting to external vLLM servers")
        logger.info("=" * 60)
        
        # Get ports from environment variables or config
        # Environment variables take precedence (set by run_stage1.sh)
        inference_config = InferenceConfig(
            cypher_model_path=self.config['cypher_model_path'],
            orchestrator_model_path=self.config['orchestrator_model_path'],
            # Server ports (from env vars or config)
            orchestrator_port_question=int(os.environ.get(
                'ORCH_QUESTION_PORT', 
                self.config.get('orchestrator_port_question', 8001)
            )),
            orchestrator_port_data_eval=int(os.environ.get(
                'ORCH_DATA_EVAL_PORT',
                self.config.get('orchestrator_port_data_eval', 8002)
            )),
            orchestrator_port_synthesis=int(os.environ.get(
                'ORCH_SYNTHESIS_PORT',
                self.config.get('orchestrator_port_synthesis', 8003)
            )),
            orchestrator_port_answer_eval=int(os.environ.get(
                'ORCH_ANSWER_EVAL_PORT',
                self.config.get('orchestrator_port_answer_eval', 8004)
            )),
            cypher_inference_port=int(os.environ.get(
                'CYPHER_INFERENCE_PORT',
                self.config.get('cypher_inference_port', 8005)
            )),
            # Server host
            server_host=self.config.get('server_host', 'localhost'),
            # API settings
            api_timeout=self.config.get('api_timeout', 180.0),
            max_retries=self.config.get('max_retries', 3),
            # Batch settings
            batch_size=self.config.get('batch_size', 64),
            # Generation settings
            max_new_tokens=self.config.get('max_new_tokens', 1024),
            temperature=self.config.get('temperature', 0.7),
        )
        
        self.inference_engine = InferenceEngine(inference_config)
        self.inference_engine.initialize()
        
        logger.info("Inference engine connected to external vLLM servers")
    
    def _init_ddp_trainer(self):
        """Initialize DDP trainer for Cypher Generator."""
        # Import here to avoid CUDA initialization before vLLM workers start
        import torch
        from rl_implementation.ddp_training.ddp_trainer import DDPTrainer, DDPTrainerConfig
        
        logger.info("=" * 60)
        logger.info("STEP 2: Initializing DDP trainer for Cypher Generator")
        logger.info("=" * 60)
        
        # Verify GPU allocation
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                logger.info(f"  cuda:{i} = {props.name}, free: {free_mem/1024**3:.1f}/{total_mem/1024**3:.1f} GB")
        
        # With CUDA_VISIBLE_DEVICES=2, we should see only 1 GPU
        # training_gpus should be [0] which maps to the visible GPU
        training_gpus = self.config.get('cypher_training_gpus', [0])
        logger.info(f"Training GPUs (logical, after remapping): {training_gpus}")
        
        # Get distributed training info
        local_rank = self.config.get('local_rank', 0)
        world_size = self.config.get('world_size', 1)
        
        # Calculate total steps for scheduler
        # Each epoch has questions_per_epoch / mini_batch_size PPO updates
        mini_batch_size = self.config.get('mini_batch_size', 1)
        steps_per_epoch = max(1, self.questions_per_epoch // mini_batch_size)
        total_steps = max(100, self.num_epochs * steps_per_epoch)  # At least 100 steps
        logger.info(f"Scheduler total_steps: {total_steps} ({self.num_epochs} epochs × {steps_per_epoch} steps/epoch)")
        
        trainer_config = DDPTrainerConfig(
            model_path=self.config['cypher_model_path'],
            lora_rank=self.config.get('lora_rank', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            learning_rate=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            warmup_steps=self.config.get('warmup_steps', 100),
            total_steps=total_steps,
            gpus=training_gpus,
            use_bf16=self.config.get('use_bf16', True),
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            checkpoint_dir=str(self.checkpoint_dir),
            mini_batch_size=mini_batch_size,  # For gradient accumulation
        )
        
        # DDP training with multiple GPUs
        self.ddp_trainer = DDPTrainer(
            trainer_config,
            local_rank=local_rank,
            world_size=world_size,
        )
        self.ddp_trainer.setup()
        
        if world_size > 1:
            logger.info(f"DDP trainer initialized: rank {local_rank}/{world_size}")
        else:
            logger.info(f"Single GPU trainer initialized")
    
    def _init_rollout_collector(self):
        """Initialize rollout collector."""
        # Import here to avoid CUDA initialization before vLLM workers start
        from rl_implementation.ddp_training.rollout_collector import RolloutCollector, RolloutCollectorConfig
        
        logger.info("Initializing rollout collector...")
        
        collector_config = RolloutCollectorConfig(
            schema_path=self.config['schema_path'],
            neo4j_url=self.config.get('neo4j_url', 
                "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"),
            experience_buffer_path=self.config.get('experience_buffer_path', None),
            entity_samples_path=self.config.get('entity_samples_path', None),  # For entity seeding
            max_steps=self.config.get('max_steps', 5),
            max_prompt_length=self.config.get('max_prompt_length', 4096),
            max_response_length=self.config.get('max_response_length', 1024),
            batch_size=self.config.get('batch_size', 8),
            max_patterns=self.config.get('max_patterns', 100),
            use_entity_seeding=self.config.get('use_entity_seeding', True),  # Enable entity seeding
            use_experience_keywords=self.config.get('use_experience_keywords', True),  # Enable experience keywords
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 0.9),
        )
        
        self.rollout_collector = RolloutCollector(
            collector_config,
            self.inference_engine,
            self.ddp_trainer.tokenizer,
        )
        
        logger.info("Rollout collector initialized")
    
    def _init_ppo_updater(self):
        """Initialize PPO updater."""
        # Import here to avoid CUDA initialization before vLLM workers start
        from rl_implementation.ddp_training.ppo_updater import PPOUpdater, PPOConfig
        
        logger.info("Initializing PPO updater...")
        
        ppo_config = PPOConfig(
            clip_ratio=self.config.get('clip_ratio', 0.2),
            entropy_coeff=self.config.get('entropy_coeff', 0.01),
            use_kl_penalty=self.config.get('use_kl_penalty', False),
            kl_coeff=self.config.get('kl_coeff', 0.1),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            grpo_baseline=self.config.get('grpo_baseline', 'mean'),
            normalize_advantages=self.config.get('normalize_advantages', True),
            ppo_epochs=self.config.get('ppo_epochs', 1),
            mini_batch_size=self.config.get('mini_batch_size', 4),
        )
        
        self.ppo_updater = PPOUpdater(ppo_config)
        
        logger.info("PPO updater initialized")
    
    def _init_rollout_store(self):
        """Initialize rollout store for data persistence."""
        # Default store path in log directory
        store_path = self.config.get(
            'rollout_store_path',
            str(self.log_dir / 'rollouts.jsonl')
        )
        
        self.rollout_store = RolloutStore(store_path)
        
        # Log existing data stats
        stats = self.rollout_store.get_stats()
        if stats['total_entries'] > 0:
            logger.info(f"Loaded existing rollout store with {stats['total_entries']} entries")
            logger.info(f"  Epochs: {stats.get('num_epochs', 0)}, Avg reward: {stats.get('avg_reward', 0):.3f}")
    
    def get_current_difficulty(self) -> str:
        """Get current curriculum difficulty based on epoch."""
        cumulative = 0
        for stage, settings in self.curriculum_stages.items():
            cumulative += settings['epochs']
            if self.epoch < cumulative:
                return stage
        return 'hard'  # Default to hard after all stages
    
    def train(self):
        """Run the full training loop."""
        logger.info(f"Starting Stage 1 training for {self.num_epochs} epochs")
        logger.info(f"Questions per epoch: {self.questions_per_epoch}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            difficulty = self.get_current_difficulty()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Difficulty: {difficulty}")
            logger.info(f"{'='*60}")
            
            # Train epoch
            epoch_metrics = self.train_epoch(difficulty)
            
            # Log metrics
            self.metrics_history.append({
                'epoch': epoch + 1,
                'difficulty': difficulty,
                **epoch_metrics,
            })
            
            if (epoch + 1) % self.log_frequency == 0:
                self._log_metrics(epoch_metrics)
            
            # Check if this is the best model by reward
            current_reward = epoch_metrics.get('avg_reward', 0.0)
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.best_epoch = epoch + 1
                logger.info(f"🏆 New best model! Reward: {current_reward:.4f} at epoch {epoch + 1}")
                self.save_best_checkpoint()
            
            # Save regular checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint()
        logger.info(f"Best model was at epoch {self.best_epoch} with reward {self.best_reward:.4f}")
        self._save_metrics_history()
        logger.info("Training complete!")
    
    def train_epoch(self, difficulty: str) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Phases:
        1. Generate questions (Orchestrator via vLLM)
        2. Collect trajectories (Cypher Generator + Neo4j)
        3. Evaluate trajectories (Orchestrator via vLLM)
        4. Compute rewards
        5. Save rollouts to persistent storage
        6. PPO update
        
        Args:
            difficulty: Current curriculum difficulty
            
        Returns:
            Dictionary with epoch metrics
        """
        import torch as _torch
        
        # Phase 1: Generate questions
        logger.info("Phase 1: Generating questions...")
        curriculum_constraints = self.curriculum_stages.get(difficulty, {})
        questions = self.rollout_collector.generate_questions(
            num_questions=self.questions_per_epoch,
            difficulty=difficulty,
            curriculum_constraints=curriculum_constraints,
        )
        
        # Phase 2: Collect trajectories
        logger.info("Phase 2: Collecting trajectories...")
        trajectories = self.rollout_collector.collect_trajectories(questions)
        
        # Phase 3: Evaluate trajectories
        logger.info("Phase 3: Evaluating trajectories...")
        trajectories = self.rollout_collector.evaluate_trajectories(trajectories)
        
        # Phase 4: Compute rewards
        logger.info("Phase 4: Computing rewards...")
        trajectories = self.rollout_collector.compute_rewards(trajectories)
        
        # Phase 4b: Update experience buffer with learned patterns
        logger.info("Phase 4b: Updating experience buffer...")
        self.rollout_collector.update_experience_buffer(trajectories)
        
        # Phase 5: Save rollouts to persistent storage (append mode)
        # This ensures data is not lost even if training crashes
        logger.info("Phase 5: Saving rollouts...")
        try:
            num_saved = self.rollout_store.save_epoch(
                epoch=self.epoch + 1,  # 1-indexed
                difficulty=difficulty,
                questions=questions,
                trajectories=trajectories,
                config={'questions_per_epoch': self.questions_per_epoch} if self.epoch == 0 else None,
            )
            logger.info(f"  Saved {num_saved} rollouts to {self.rollout_store.store_path}")
        except Exception as e:
            logger.error(f"Failed to save rollouts: {e}")
            # Continue training even if save fails
        
        # Phase 6: PPO update
        logger.info("Phase 6: PPO update...")
        update_metrics = {}
        
        try:
            batch = self.rollout_collector.prepare_training_batch(trajectories)
            update_metrics = self._ppo_update_from_batch(batch)
        except ValueError as e:
            logger.warning(f"Failed to prepare batch: {e}")
            update_metrics = {'ppo_update_skipped': 1.0}
        
        # Compute epoch metrics
        rewards = [t.reward for t in trajectories]
        success_rates = [t.success_rate for t in trajectories]
        data_quality_scores = [t.data_quality_score for t in trajectories]
        answer_quality_scores = [t.answer_quality_score for t in trajectories]
        
        epoch_metrics = {
            'reward_mean': sum(rewards) / len(rewards) if rewards else 0.0,
            'reward_std': _torch.tensor(rewards).std().item() if rewards else 0.0,
            'success_rate_mean': sum(success_rates) / len(success_rates) if success_rates else 0.0,
            'data_quality_mean': sum(data_quality_scores) / len(data_quality_scores) if data_quality_scores else 0.0,
            'answer_quality_mean': sum(answer_quality_scores) / len(answer_quality_scores) if answer_quality_scores else 0.0,
            'num_trajectories': len(trajectories),
            'rollouts_saved': self.rollout_store.entry_count,
            **update_metrics,
        }
        
        # Log sample trajectories
        self._log_sample_trajectories(trajectories[:3])
        
        return epoch_metrics
    
    def _ppo_update_from_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform PPO update from prepared batch.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, labels, response_mask, rewards
            
        Returns:
            Dictionary with update metrics
        """
        # Compute old log probs
        old_log_probs = self.ddp_trainer.compute_log_probs_for_rollout(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Prepare batch with advantages
        ppo_batch = self.ppo_updater.prepare_batch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            response_mask=batch['response_mask'],
            rewards=batch['rewards'],
            old_log_probs=old_log_probs,
        )
        
        # PPO update
        update_metrics = self.ppo_updater.update_step(self.ddp_trainer, ppo_batch)
        
        self.global_step += 1
        
        return update_metrics
    
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        logger.info(f"Epoch {self.epoch + 1} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _log_sample_trajectories(self, trajectories, num_samples: int = 3):
        """Log sample trajectories for debugging."""
        logger.info(f"\nSample Trajectories (first {min(num_samples, len(trajectories))}):")
        for i, traj in enumerate(trajectories[:num_samples]):
            logger.info(f"\n--- Trajectory {i + 1} ---")
            logger.info(f"Question: {traj.question[:100]}...")
            logger.info(f"Steps: {traj.num_steps}")
            logger.info(f"Reward: {traj.reward:.3f}")
            logger.info(f"Data Quality: {traj.data_quality_score:.3f}")
            logger.info(f"Answer Quality: {traj.answer_quality_score:.3f}")
            
            if traj.steps:
                logger.info(f"First Cypher: {traj.steps[0].cypher_query[:100]}...")
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.epoch + 1:04d}"
        self.ddp_trainer.save_checkpoint(str(checkpoint_path))
        
        # Save training state
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config,
        }
        with open(checkpoint_path / 'trainer_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_best_checkpoint(self):
        """Save the best model checkpoint (overwrites previous best)."""
        best_path = self.checkpoint_dir / "best_model"
        self.ddp_trainer.save_checkpoint(str(best_path))
        
        # Save training state with best info
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'best_epoch': self.best_epoch,
            'config': self.config,
        }
        with open(best_path / 'trainer_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Best model saved to {best_path} (reward: {self.best_reward:.4f})")
    
    def _save_metrics_history(self):
        """Save metrics history to JSON."""
        metrics_path = self.log_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Metrics history saved to {metrics_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")
        
        if hasattr(self, 'inference_engine'):
            self.inference_engine.shutdown()
        
        if hasattr(self, 'ddp_trainer'):
            self.ddp_trainer.cleanup()
        
        if hasattr(self, 'rollout_collector'):
            self.rollout_collector.close()
        
        logger.info("Cleanup complete")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stage 1 Training: Cypher Generator')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint directory')
    parser.add_argument('--log_dir', type=str, help='Override log directory')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--questions_per_epoch', type=int, help='Override questions per epoch')
    # DDP arguments (set by torchrun)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Get distributed training info from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"Starting training: local_rank={local_rank}, world_size={world_size}")
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['log_dir'] = args.log_dir
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.questions_per_epoch:
        config['questions_per_epoch'] = args.questions_per_epoch
    
    # Add distributed training info to config
    config['local_rank'] = local_rank
    config['world_size'] = world_size
    
    # Create trainer and run
    trainer = Stage1Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()

