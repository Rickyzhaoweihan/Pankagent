#!/usr/bin/env python3
"""
Train Orchestrator from Rollouts.

Standalone training script that reads from a rollout JSONL file
and trains the Orchestrator model using PPO/GRPO.

Trains both Orchestrator roles:
1. Question Generation - generates training questions
2. Answer Synthesis - synthesizes answers from retrieved data

Usage:
    python train_orchestrator_from_rollouts.py --config config/train_orchestrator_config.yaml
    python train_orchestrator_from_rollouts.py --rollouts /path/to/rollouts.jsonl --orchestrator-model /path/to/model

Requirements:
    - Rollouts JSONL file (from collect_rollouts.py)
    - Orchestrator model
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DDP_DIR = SCRIPT_DIR.parent
RL_IMPL_DIR = DDP_DIR.parent
PROJECT_DIR = RL_IMPL_DIR.parent

sys.path.insert(0, str(RL_IMPL_DIR))
sys.path.insert(0, str(DDP_DIR))

import yaml
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# Import training components
from ddp_trainer import DDPTrainer, DDPTrainerConfig
from ppo_updater import PPOUpdater, PPOConfig
from rollout_loader import RolloutLoader, OrchestratorBatchPreparer

# FSDP support - import conditionally to avoid errors if not using FSDP
try:
    from fsdp_trainer import FSDPTrainer, FSDPTrainerConfig
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Orchestrator from rollouts")
    
    # Config file (primary)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    
    # Rollouts
    parser.add_argument("--rollouts", type=str, default=None,
                        help="Path to rollouts JSONL file")
    
    # Model
    parser.add_argument("--orchestrator-model", type=str, default=None,
                        help="Path to Orchestrator model")
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for logs")
    
    # Training parameters (use None as default so YAML config takes precedence)
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    
    # Role selection
    parser.add_argument("--train-question-gen", action="store_true", default=True,
                        help="Train question generation role")
    parser.add_argument("--no-train-question-gen", action="store_false", dest="train_question_gen",
                        help="Skip question generation training")
    parser.add_argument("--train-answer-synthesis", action="store_true", default=True,
                        help="Train answer synthesis role")
    parser.add_argument("--no-train-answer-synthesis", action="store_false", dest="train_answer_synthesis",
                        help="Skip answer synthesis training")
    
    # Filtering
    parser.add_argument("--min-reward", type=float, default=None,
                        help="Minimum reward threshold for filtering rollouts")
    parser.add_argument("--min-epoch", type=int, default=None,
                        help="Minimum epoch to load from rollouts")
    parser.add_argument("--max-epoch", type=int, default=None,
                        help="Maximum epoch to load from rollouts")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty level")
    
    # LoRA (use None as default so YAML config takes precedence)
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank (overrides config)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (overrides config)")
    
    # Misc
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for logging")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge command line arguments into config.
    
    Command line args with non-None values override config file.
    Use 'is not None' checks to distinguish between 'not provided' and 'provided as 0/False'.
    """
    # Paths (override if provided)
    if args.rollouts:
        config['rollouts_path'] = args.rollouts
    if args.orchestrator_model:
        config['orchestrator_model_path'] = args.orchestrator_model
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['log_dir'] = args.log_dir
    
    # Training parameters (use 'is not None' to allow YAML to take precedence)
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # Filtering
    if args.min_reward is not None:
        config['min_reward'] = args.min_reward
    if args.min_epoch is not None:
        config['min_epoch'] = args.min_epoch
    if args.max_epoch is not None:
        config['max_epoch'] = args.max_epoch
    if args.difficulty:
        config['difficulty'] = args.difficulty
    
    # LoRA (use 'is not None' to allow YAML to take precedence)
    if args.lora_rank is not None:
        config['lora_rank'] = args.lora_rank
    if args.lora_alpha is not None:
        config['lora_alpha'] = args.lora_alpha
    
    # Role flags
    config['train_question_gen'] = args.train_question_gen
    config['train_answer_synthesis'] = args.train_answer_synthesis
    
    return config


class OrchestratorTrainer:
    """Trainer for Orchestrator from rollouts."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.epoch = 0
        self.global_step = 0
        
        # Role flags
        self.train_question_gen = config.get('train_question_gen', True)
        self.train_answer_synthesis = config.get('train_answer_synthesis', True)
        
        if not self.train_question_gen and not self.train_answer_synthesis:
            raise ValueError("At least one role must be trained!")
        
        # Paths
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config.get('log_dir', self.checkpoint_dir / 'logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 4)
        self.save_frequency = config.get('save_frequency', 1)
        
        # Initialize components
        logger.info("=" * 60)
        logger.info("ORCHESTRATOR TRAINING FROM ROLLOUTS")
        logger.info("=" * 60)
        logger.info(f"Training roles:")
        logger.info(f"  Question Generation: {'Yes' if self.train_question_gen else 'No'}")
        logger.info(f"  Answer Synthesis: {'Yes' if self.train_answer_synthesis else 'No'}")
        
        self._init_rollout_loader()
        self._init_ddp_trainer()
        self._init_batch_preparer()
        self._init_ppo_updater()
        
        # Metrics tracking
        self.metrics_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        logger.info("=" * 60)
        logger.info("Trainer initialized!")
        logger.info(f"  Rollouts: {len(self.rollouts)} entries")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info("=" * 60)
    
    def _init_rollout_loader(self):
        """Load rollouts from JSONL file."""
        logger.info("Loading rollouts...")
        
        rollouts_path = self.config['rollouts_path']
        loader = RolloutLoader(rollouts_path)
        
        # Load with filters
        self.rollouts = loader.load_rollouts(
            min_epoch=self.config.get('min_epoch'),
            max_epoch=self.config.get('max_epoch'),
            difficulty=self.config.get('difficulty'),
            min_reward=self.config.get('min_reward'),
        )
        
        if not self.rollouts:
            raise ValueError("No rollouts loaded! Check filters and file path.")
        
        # Log stats (including filtering info)
        stats = loader.get_stats()
        min_reward_filter = self.config.get('min_reward', 0.0)
        total_before_filter = stats['total_entries']
        total_after_filter = len(self.rollouts)
        filtered_out = total_before_filter - total_after_filter
        
        logger.info(f"  Total in file: {total_before_filter}")
        logger.info(f"  After filtering (min_reward={min_reward_filter}): {total_after_filter}")
        if filtered_out > 0:
            logger.info(f"  Filtered out: {filtered_out} ({100*filtered_out/total_before_filter:.1f}% removed)")
        
        # Compute filtered reward stats for both qgen and synth
        qgen_rewards = [r.orch_qgen_reward for r in self.rollouts]
        synth_rewards = [r.orch_synth_reward for r in self.rollouts]
        
        if qgen_rewards:
            low_qgen = sum(1 for r in qgen_rewards if r < 0.3)
            med_qgen = sum(1 for r in qgen_rewards if 0.3 <= r < 0.7)
            high_qgen = sum(1 for r in qgen_rewards if r >= 0.7)
            logger.info(f"  QGen reward distribution: low={low_qgen}, med={med_qgen}, high={high_qgen}")
            logger.info(f"  Avg qgen reward (filtered): {sum(qgen_rewards)/len(qgen_rewards):.3f}")
        
        if synth_rewards:
            low_synth = sum(1 for r in synth_rewards if r < 0.3)
            med_synth = sum(1 for r in synth_rewards if 0.3 <= r < 0.7)
            high_synth = sum(1 for r in synth_rewards if r >= 0.7)
            logger.info(f"  Synth reward distribution: low={low_synth}, med={med_synth}, high={high_synth}")
            logger.info(f"  Avg synth reward (filtered): {sum(synth_rewards)/len(synth_rewards):.3f}")
    
    def _init_ddp_trainer(self):
        """Initialize DDP or FSDP trainer for Orchestrator."""
        # Check if FSDP is requested
        use_fsdp = self.config.get('use_fsdp', False)
        
        if use_fsdp:
            if not FSDP_AVAILABLE:
                raise ImportError("FSDP requested but fsdp_trainer not available!")
            logger.info("Initializing FSDP trainer...")
        else:
            logger.info("Initializing DDP trainer...")
        
        # Calculate total steps (2 roles * samples)
        samples_per_epoch = len(self.rollouts)
        if self.train_question_gen:
            samples_per_epoch += len(self.rollouts)
        steps_per_epoch = max(1, samples_per_epoch // self.batch_size)
        total_steps = max(100, self.num_epochs * steps_per_epoch)
        
        logger.info(f"  Samples per epoch: {samples_per_epoch}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        
        # Check for existing best model (cumulative training)
        # If a previous best model exists, resume from it instead of starting fresh
        resume_from_adapter = ""
        best_model_path = self.checkpoint_dir / "best_model"
        if best_model_path.exists() and (best_model_path / "adapter_config.json").exists():
            resume_from_adapter = str(best_model_path)
            logger.info(f"  🔄 Found existing best model, will resume from: {resume_from_adapter}")
            logger.info(f"     (Cumulative training enabled - building on previous training)")
        else:
            logger.info(f"  Starting fresh (no previous best model found)")
        
        # Get distributed training info from torchrun environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if use_fsdp:
            # Use FSDP trainer
            # IMPORTANT: mini_batch_size controls memory usage during forward pass
            # It should be read from config, NOT use batch_size (which is larger)
            fsdp_mini_batch_size = self.config.get('mini_batch_size', 1)  # Default 1 for max memory safety
            
            logger.info(f"  FSDP Config: mini_batch_size={fsdp_mini_batch_size}, batch_size={self.batch_size}")
            logger.info(f"  FSDP Config: use_bf16={self.config.get('use_bf16', True)}, gradient_checkpointing={self.config.get('gradient_checkpointing', True)}")
            logger.info(f"  FSDP Config: sharding_strategy={self.config.get('sharding_strategy', 'FULL_SHARD')}")
            
            trainer_config = FSDPTrainerConfig(
                model_path=self.config['orchestrator_model_path'],
                resume_from_adapter=resume_from_adapter,
                lora_rank=self.config.get('lora_rank', 128),
                lora_alpha=self.config.get('lora_alpha', 256),
                lora_dropout=self.config.get('lora_dropout', 0.05),
                learning_rate=self.config.get('learning_rate', 5e-6),
                weight_decay=self.config.get('weight_decay', 0.01),
                max_grad_norm=self.config.get('max_grad_norm', 1.0),
                warmup_steps=self.config.get('warmup_steps', 50),
                total_steps=total_steps,
                sharding_strategy=self.config.get('sharding_strategy', 'FULL_SHARD'),
                backward_prefetch=self.config.get('backward_prefetch', 'BACKWARD_PRE'),
                cpu_offload=self.config.get('cpu_offload', False),
                use_bf16=self.config.get('use_bf16', True),
                gradient_checkpointing=self.config.get('gradient_checkpointing', True),
                checkpoint_dir=str(self.checkpoint_dir),
                mini_batch_size=fsdp_mini_batch_size,  # Fixed: use config value, not batch_size
            )
            
            self.ddp_trainer = FSDPTrainer(trainer_config, local_rank=local_rank, world_size=world_size)
            self.ddp_trainer.setup()
            logger.info(f"  FSDP trainer initialized (rank {local_rank}/{world_size}, sharding={self.config.get('sharding_strategy', 'FULL_SHARD')})")
        else:
            # Use DDP trainer
            # IMPORTANT: mini_batch_size controls memory usage during forward pass
            # It should be read from config, NOT use batch_size (which is larger)
            ddp_mini_batch_size = self.config.get('mini_batch_size', 1)  # Default 1 for max memory safety
            
            logger.info(f"  DDP Config: mini_batch_size={ddp_mini_batch_size}, batch_size={self.batch_size}")
            logger.info(f"  DDP Config: use_bf16={self.config.get('use_bf16', True)}, gradient_checkpointing={self.config.get('gradient_checkpointing', True)}")
            
            trainer_config = DDPTrainerConfig(
                model_path=self.config['orchestrator_model_path'],
                resume_from_adapter=resume_from_adapter,
                lora_rank=self.config.get('lora_rank', 16),
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=self.config.get('lora_dropout', 0.05),
                learning_rate=self.config.get('learning_rate', 1e-5),
                weight_decay=self.config.get('weight_decay', 0.01),
                max_grad_norm=self.config.get('max_grad_norm', 1.0),
                warmup_steps=self.config.get('warmup_steps', 50),
                total_steps=total_steps,
                gpus=[0],  # Single GPU
                use_bf16=self.config.get('use_bf16', True),
                gradient_checkpointing=self.config.get('gradient_checkpointing', True),
                activation_offloading=self.config.get('activation_offloading', False),
                checkpoint_dir=str(self.checkpoint_dir),
                mini_batch_size=ddp_mini_batch_size,  # Fixed: use config value, not batch_size
            )
            
            self.ddp_trainer = DDPTrainer(trainer_config, local_rank=local_rank, world_size=world_size)
            self.ddp_trainer.setup()
            logger.info(f"  DDP trainer initialized (rank {local_rank}/{world_size})")
    
    def _init_batch_preparer(self):
        """Initialize batch preparer."""
        logger.info("Initializing batch preparer...")
        
        self.batch_preparer = OrchestratorBatchPreparer(
            tokenizer=self.ddp_trainer.tokenizer,
            max_length=self.config.get('max_length', 4096),
        )
        
        # Config for prompt handling (default: use stored prompts)
        self.use_stored_prompts = self.config.get('use_stored_prompts', True)
        
        if self.use_stored_prompts:
            logger.info("  Batch preparer initialized (using stored prompts)")
        else:
            logger.info("  Batch preparer initialized (rebuilding prompts)")
    
    def _init_ppo_updater(self):
        """Initialize PPO updater."""
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
            mini_batch_size=self.batch_size,
        )
        
        self.ppo_updater = PPOUpdater(ppo_config)
        
        logger.info("  PPO updater initialized")
    
    def train(self):
        """Run training loop."""
        logger.info(f"\nStarting training for {self.num_epochs} epochs")
        logger.info(f"Total rollouts: {len(self.rollouts)}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Synchronize all ranks at start of each epoch
            # This ensures all ranks are ready before starting training
            if dist.is_initialized():
                dist.barrier()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            epoch_metrics = {}
            
            # Train question generation role
            if self.train_question_gen:
                logger.info("\n--- Question Generation Role ---")
                qgen_metrics = self.train_question_generation()
                epoch_metrics.update({f"qgen_{k}": v for k, v in qgen_metrics.items()})
            
            # Train answer synthesis role
            if self.train_answer_synthesis:
                logger.info("\n--- Answer Synthesis Role ---")
                synth_metrics = self.train_answer_synthesis_role()
                epoch_metrics.update({f"synth_{k}": v for k, v in synth_metrics.items()})
            
            # Log metrics
            self.metrics_history.append({
                'epoch': epoch + 1,
                **epoch_metrics,
            })
            self._log_metrics(epoch_metrics)
            
            # Track best (using average policy loss) - only save when we improve
            # CRITICAL: Synchronize checkpoint save decision across all ranks!
            # Without this, floating-point differences can cause one rank to save
            # while another continues training, causing NCCL deadlock.
            avg_loss = 0.0
            count = 0
            if 'qgen_policy_loss' in epoch_metrics:
                avg_loss += epoch_metrics['qgen_policy_loss']
                count += 1
            if 'synth_policy_loss' in epoch_metrics:
                avg_loss += epoch_metrics['synth_policy_loss']
                count += 1
            
            should_save = False
            if count > 0:
                avg_loss /= count
                should_save = avg_loss < self.best_loss
            
            if dist.is_initialized():
                # Broadcast rank 0's decision to all ranks
                should_save_tensor = torch.tensor([1 if should_save else 0], dtype=torch.int64, device=self.ddp_trainer.device)
                dist.broadcast(should_save_tensor, src=0)
                should_save = should_save_tensor.item() == 1
            
            if should_save:
                self.best_loss = avg_loss
                self.best_epoch = epoch + 1
                self.save_best_checkpoint()
                logger.info(f"🏆 New best model! Avg loss: {avg_loss:.4f}")
        
        # Final save (overwrites previous final if exists)
        self.save_final_checkpoint()
        self._save_metrics_history()
        logger.info(f"\nTraining complete! Best model at epoch {self.best_epoch} with loss {self.best_loss:.4f}")
    
    def _get_synchronized_seed(self) -> int:
        """
        Generate a random seed synchronized across all ranks.
        
        CRITICAL for FSDP: all ranks must sample the same data to avoid NCCL desync!
        """
        if dist.is_initialized():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_rank == 0:
                seed = torch.randint(0, 2**31, (1,), device=self.ddp_trainer.device)
            else:
                seed = torch.zeros(1, dtype=torch.int64, device=self.ddp_trainer.device)
            dist.broadcast(seed, src=0)
            return seed.item()
        return None
    
    def train_question_generation(self) -> Dict[str, float]:
        """Train question generation role."""
        logger.info("Preparing question generation batch...")
        
        # Synchronize random seed across ranks for balanced sampling
        use_balanced = self.config.get('use_balanced_sampling', False)
        seed = self._get_synchronized_seed() if use_balanced else None
        if seed is not None:
            logger.info(f"  Synchronized sampling seed: {seed}")
        
        try:
            # Use balanced sampling if configured (CRITICAL for bimodal qgen rewards!)
            batch = self.batch_preparer.prepare_question_gen_batch(
                self.rollouts,
                schema_summary=self.config.get('schema_summary', ''),
                use_stored_prompts=self.use_stored_prompts,
                use_balanced_sampling=use_balanced,
                batch_size=self.batch_size if use_balanced else None,
                seed=seed,  # Pass synchronized seed
            )
            batch_error = False
        except ValueError as e:
            logger.error(f"Failed to prepare batch: {e}")
            batch_error = True
        
        # CRITICAL: Synchronize error status across ranks to prevent NCCL deadlock
        # If one rank errors, all ranks must know and return together
        if dist.is_initialized():
            error_tensor = torch.tensor([1 if batch_error else 0], dtype=torch.int64, device=self.ddp_trainer.device)
            dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
            if error_tensor.item() > 0:
                logger.error("Batch preparation failed on one or more ranks, skipping qgen training")
                return {'error': 1.0}
        elif batch_error:
            return {'error': 1.0}
        
        logger.info(f"  Batch size: {batch['input_ids'].shape[0]}")
        
        return self._ppo_update(batch)
    
    def train_answer_synthesis_role(self) -> Dict[str, float]:
        """Train answer synthesis role."""
        logger.info("Preparing answer synthesis batch...")
        
        # Synchronize random seed across ranks for balanced sampling
        use_balanced = self.config.get('use_balanced_sampling', False)
        seed = self._get_synchronized_seed() if use_balanced else None
        if seed is not None:
            logger.info(f"  Synchronized sampling seed: {seed}")
        
        try:
            # Use balanced sampling if configured
            batch = self.batch_preparer.prepare_synthesis_batch(
                self.rollouts,
                use_stored_prompts=self.use_stored_prompts,
                use_balanced_sampling=use_balanced,
                batch_size=self.batch_size if use_balanced else None,
                seed=seed,  # Pass synchronized seed
            )
            batch_error = False
        except ValueError as e:
            logger.error(f"Failed to prepare batch: {e}")
            batch_error = True
        
        # CRITICAL: Synchronize error status across ranks to prevent NCCL deadlock
        # If one rank errors, all ranks must know and return together
        if dist.is_initialized():
            error_tensor = torch.tensor([1 if batch_error else 0], dtype=torch.int64, device=self.ddp_trainer.device)
            dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
            if error_tensor.item() > 0:
                logger.error("Batch preparation failed on one or more ranks, skipping synth training")
                return {'error': 1.0}
        elif batch_error:
            return {'error': 1.0}
        
        logger.info(f"  Batch size: {batch['input_ids'].shape[0]}")
        
        return self._ppo_update(batch)
    
    def _ppo_update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform PPO update on a batch.
        
        Architecture matches train_stage1.py:
        1. Compute old_log_probs ONCE for entire batch
        2. Compute advantages using GRPO
        3. ONE ppo_updater.update_step() - mini-batching handled inside ddp_trainer.train_step()
        """
        # Move batch to device
        device = self.ddp_trainer.device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        total_samples = batch['input_ids'].shape[0]
        logger.info(f"  Total samples: {total_samples}")
        
        # Compute old log probs ONCE for entire batch (same as train_stage1.py)
        logger.info("Computing old log probs...")
        old_log_probs = self.ddp_trainer.compute_log_probs_for_rollout(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Prepare PPO batch with advantages (uses GRPO internally)
        ppo_batch = self.ppo_updater.prepare_batch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            response_mask=batch['response_mask'],
            rewards=batch['rewards'],
            old_log_probs=old_log_probs,
        )
        
        # Log reward/advantage stats
        rewards = batch['rewards']
        advantages = ppo_batch['advantages']
        valid_adv = advantages[batch['response_mask'] > 0]
        
        logger.info(f"  Reward stats: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        logger.info(f"  Advantage stats: mean={valid_adv.mean():.4f}, std={valid_adv.std():.4f}")
        
        # ONE PPO update - mini-batching handled inside ddp_trainer.train_step()
        logger.info("Performing PPO update...")
        update_metrics = self.ppo_updater.update_step(self.ddp_trainer, ppo_batch)
        
        self.global_step += 1
        
        # Build metrics
        return {
            'policy_loss': update_metrics.get('ppo_loss', 0.0),
            'entropy': update_metrics.get('entropy', 0.0),
            'kl_divergence': update_metrics.get('approx_kl', 0.0),
            'clip_fraction': update_metrics.get('clip_fraction', 0.0),
            'grad_norm': update_metrics.get('grad_norm', 0.0),
            'samples_processed': total_samples,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        logger.info(f"Epoch {self.epoch + 1} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def save_best_checkpoint(self):
        """Save best model checkpoint (overwrites previous best)."""
        best_path = self.checkpoint_dir / "best_model"
        
        # Ensure directory exists (for FSDP, only rank 0 saves but all ranks call this)
        best_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter only (fast, keeps model trainable)
        # Merged model will be created at the end of training
        self.ddp_trainer.save_checkpoint(str(best_path))
        
        # Only rank 0 saves trainer state (avoid race condition in FSDP)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            state = {
                'epoch': self.epoch + 1,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'config': self.config,
                'roles_trained': {
                    'question_gen': self.train_question_gen,
                    'answer_synthesis': self.train_answer_synthesis,
                }
            }
            with open(best_path / 'trainer_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        
        logger.info(f"  Best model saved to {best_path}")
    
    def save_final_checkpoint(self):
        """Save final model checkpoint and create merged models for vLLM."""
        final_path = self.checkpoint_dir / "final_model"
        
        # Ensure directory exists
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter (all ranks participate in FSDP gather, but only rank 0 saves)
        self.ddp_trainer.save_checkpoint(str(final_path))
        
        # Only rank 0 handles the rest (trainer state + merged models)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            # Save trainer state
            state = {
                'epoch': self.epoch + 1,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch,
                'config': self.config,
                'roles_trained': {
                    'question_gen': self.train_question_gen,
                    'answer_synthesis': self.train_answer_synthesis,
                }
            }
            with open(final_path / 'trainer_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            # Now create merged models for vLLM (done at end to preserve training)
            # Only rank 0 creates merged models to avoid race conditions
            logger.info("Creating merged models for vLLM inference...")
            
            # Merge best model (from saved adapter)
            best_adapter_path = self.checkpoint_dir / "best_model"
            if best_adapter_path.exists():
                merged_best_path = self.checkpoint_dir / "best_model_merged"
                logger.info(f"Merging best model: {merged_best_path}")
                self._create_merged_model(str(best_adapter_path), str(merged_best_path))
            
            # Merge final model
            merged_final_path = self.checkpoint_dir / "final_model_merged"
            logger.info(f"Merging final model: {merged_final_path}")
            self._create_merged_model(str(final_path), str(merged_final_path))
        
        # CRITICAL: All ranks must wait here until model merging is complete!
        # Without this barrier, non-rank-0 processes would proceed to cleanup()
        # and hit its barrier while rank 0 is still merging, causing NCCL timeout.
        if dist.is_initialized():
            logger.info(f"[Rank {local_rank}] Waiting at barrier after final checkpoint...")
            dist.barrier()
            logger.info(f"[Rank {local_rank}] Passed final checkpoint barrier")
    
    def _create_merged_model(self, adapter_path: str, output_path: str):
        """Create a merged model from a LoRA adapter checkpoint."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_model_path = self.config['orchestrator_model_path']
        
        # Load fresh base model on CPU
        logger.info(f"  Loading base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        # Load adapter
        merged_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge and save
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        
        # Save tokenizer
        self.ddp_trainer.tokenizer.save_pretrained(output_path)
        
        # Cleanup
        del merged_model
        del base_model
        torch.cuda.empty_cache()
        
        logger.info(f"  Merged model saved to {output_path}")
    
    def _save_metrics_history(self):
        """Save metrics history."""
        metrics_path = self.log_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'ddp_trainer'):
            self.ddp_trainer.cleanup()
        logger.info("Cleanup complete")


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Merge command line args
    config = merge_config(config, args)
    
    # Validate required fields
    required = ['rollouts_path', 'orchestrator_model_path', 'checkpoint_dir']
    for field in required:
        if field not in config or not config[field]:
            logger.error(f"Missing required config: {field}")
            logger.error("Provide via --config file or command line arguments")
            sys.exit(1)
    
    # Create trainer and run
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")
    
    trainer = OrchestratorTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

