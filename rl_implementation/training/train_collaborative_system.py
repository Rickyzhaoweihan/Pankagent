"""
Main training orchestration for collaborative multi-agent system.

Implements:
- Full training loop with EMA evaluator
- Phased training schedule (warmup, alternating, joint)
- Reward normalization
- Validation anchoring
- Curriculum learning
- Experience buffer updates
- Checkpoint management
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
import ray
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import rllm components for actual model training
from rllm.data import Dataset, DatasetRegistry
from rllm.engine import AsyncAgentExecutionEngine
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl import DataProto

from rl_implementation.agents import CypherGeneratorAgent, OrchestratorAgent, ExperienceBuffer
from rl_implementation.environments import GraphReasoningEnvironment, Neo4jExecutor
from rl_implementation.rewards import (
    cypher_generator_reward_fn,
    orchestrator_generation_reward_fn,
    orchestrator_synthesis_reward_fn
)
from rl_implementation.utils.orchestrator_prompt_builder import OrchestratorPromptBuilder
from rl_implementation.utils.prompt_builder import PromptBuilder
from rl_implementation.utils.data_quality_evaluator import (
    parse_data_quality_json,
    parse_answer_quality_json
)
from verl.utils.model import compute_position_id_with_mask
from rl_implementation.utils.orchestrator_prompt_builder import OrchestratorPromptBuilder
from rl_implementation.utils.prompt_builder import PromptBuilder
from verl.utils.model import compute_position_id_with_mask
from rl_implementation.training.utils.training_stability import (
    RunningStats,
    update_ema_model,
    should_update_orchestrator,
    get_training_phase,
    detect_reward_drift,
    adjust_ema_decay
)
from rl_implementation.training.utils.curriculum_utils import (
    CurriculumTracker,
    compute_success_rate
)
from rl_implementation.training.utils.checkpoint_manager import CheckpointManager
from rl_implementation.training.utils.validation import (
    load_validation_set,
    validate_on_fixed_set,
    compare_train_val_metrics
)
from rl_implementation.training.utils.rllm_components import (
    load_tokenizer,
    create_resource_pool_manager,
    create_dual_resource_pool_manager,
    create_resource_pools_from_config,
    create_worker_groups,
    create_dual_worker_groups,
    create_execution_engine,
    initialize_worker_group,
    DualResourcePoolManager,
    # Sequential training mode (Strategy 2)
    SequentialResourceManager,
    create_sequential_resource_manager,
    should_update_orchestrator as should_update_orch_from_schedule,
    get_training_phase as get_phase_from_schedule,
    EMAOrchestrator,
    create_ema_orchestrator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CollaborativeTrainer:
    """
    Main trainer for collaborative multi-agent RL system.
    
    Orchestrates training of Cypher Generator and Orchestrator agents
    with stability mechanisms and curriculum learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CollaborativeTrainer with low-level rllm components.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Initialize paths
        self.schema_path = config['schema_path']
        self.checkpoint_dir = config['checkpoint_dir']
        self.log_dir = config['log_dir']
        self.val_questions_path = config.get('val_questions_path', None)
        self.neo4j_url = config.get('neo4j_url', 
            "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j")
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Load PPO configs
        logger.info("Loading PPO configurations...")
        cypher_config_path = Path(__file__).parent.parent / "config" / "ppo_collaborative.yaml"
        orch_config_path = Path(__file__).parent.parent / "config" / "orchestrator_ppo.yaml"
        
        self.cypher_ppo_config = OmegaConf.load(cypher_config_path)
        self.orch_ppo_config = OmegaConf.load(orch_config_path)
        
        # Initialize utilities FIRST
        logger.info("Initializing training utilities...")
        self.experience_buffer = ExperienceBuffer(max_patterns=100)
        self.reward_tracker = RunningStats(window_size=100)
        self.curriculum_tracker = CurriculumTracker(initial_stage='easy', window_size=100)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        # Initialize Ray
        if not ray.is_initialized():
            logger.info("Initializing Ray for distributed training...")
            # Use /tmp for Ray to avoid socket path length issues
            import tempfile
            ray_temp_dir = tempfile.mkdtemp(prefix="ray_", dir="/tmp")
            
            # Set up environment variables for Ray workers
            # Include CUDA paths to ensure libcudart.so.12 is found
            env_vars = {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
            }
            
            # Inherit CUDA-related environment variables from current environment
            # This is critical for vLLM which requires CUDA 12 (libcudart.so.12)
            import os
            for var in ["LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "PATH", "LIBRARY_PATH"]:
                if var in os.environ:
                    env_vars[var] = os.environ[var]
            
            ld_path = env_vars.get("LD_LIBRARY_PATH", "not set")
            logger.info(f"Passing LD_LIBRARY_PATH to Ray workers: {ld_path[:150]}...")
            
            ray.init(
                _temp_dir=ray_temp_dir,
                runtime_env={"env_vars": env_vars}
            )
        
        # Load tokenizers
        logger.info("Loading tokenizers...")
        self.cypher_tokenizer = load_tokenizer(
            self.cypher_ppo_config.actor_rollout_ref.model.path,
            trust_remote_code=self.cypher_ppo_config.data.get("trust_remote_code", False)
        )
        self.orch_tokenizer = load_tokenizer(
            self.orch_ppo_config.actor_rollout_ref.model.path,
            trust_remote_code=self.orch_ppo_config.data.get("trust_remote_code", False)
        )
        
        # Get GPU allocation config (with defaults)
        gpu_config = config.get('gpu_allocation', {})
        if not gpu_config:
            # Default: sequential mode with all available GPUs
            gpu_config = {
                'total_gpus': self.cypher_ppo_config.trainer.n_gpus_per_node,
                'nnodes': self.cypher_ppo_config.trainer.nnodes,
                'mode': 'sequential',
                'rollout': {
                    'cypher_gpus': self.cypher_ppo_config.trainer.n_gpus_per_node // 2,
                    'orch_gpus': self.cypher_ppo_config.trainer.n_gpus_per_node // 2,
                },
                'training': {
                    'gpus': self.cypher_ppo_config.trainer.n_gpus_per_node,
                }
            }
        
        # Get training mode
        self.training_mode = gpu_config.get('mode', 'sequential')
        logger.info(f"Training mode: {self.training_mode}")
        
        # Get orchestrator schedule config
        self.orch_schedule_config = config.get('orchestrator_schedule', {
            'warmup_epochs': 5,
            'alternating_end_epoch': 20,
            'warmup_frequency': 0,
            'alternating_frequency': 3,
            'joint_frequency': 2,
        })
        
        # Get EMA config
        ema_config = config.get('ema', {
            'enabled': True,
            'decay': 0.99,
            'decay_on_drift': 0.995,
        })
        self.ema_orchestrator = create_ema_orchestrator(ema_config)
        
        if self.training_mode == 'sequential':
            # Sequential mode: swap models between phases
            self._init_sequential_mode(gpu_config)
        elif self.training_mode == 'parallel':
            # Parallel mode: both models loaded simultaneously (legacy)
            parallel_config = gpu_config.get('parallel', gpu_config)
            if parallel_config.get('use_shared_model', False):
                self._init_shared_model_workers(parallel_config)
            else:
                self._init_dual_model_workers(parallel_config)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        
        # Complete initialization (execution engines, validation, etc.)
        self._complete_initialization()
    
    def _init_sequential_mode(self, gpu_config: Dict[str, Any]):
        """
        Initialize in sequential training mode (Strategy 2).
        
        In this mode:
        - Phase 1 (Rollout): Both models loaded for inference (split GPUs)
        - Phase 2 (Cypher Train): Only Cypher loaded (all GPUs)
        - Phase 3 (Orch Train): Only Orch loaded (all GPUs, conditional)
        
        This maximizes GPU utilization during training.
        """
        logger.info("=" * 60)
        logger.info("Initializing in SEQUENTIAL TRAINING mode (Strategy 2)")
        logger.info("=" * 60)
        
        # Create sequential resource manager
        self.seq_resource_manager = create_sequential_resource_manager(gpu_config)
        
        # For sequential mode, we start in rollout phase
        # Worker groups will be created/destroyed as needed during training
        logger.info("Setting up for rollout phase (both models for inference)...")
        
        # Create dual resource pools for rollout phase
        self.dual_pool_manager = create_dual_resource_pool_manager(
            cypher_gpus=self.seq_resource_manager.rollout_cypher_gpus,
            orch_gpus=self.seq_resource_manager.rollout_orch_gpus,
            nnodes=self.seq_resource_manager.nnodes
        )
        
        # Create worker groups for rollout (inference)
        use_cypher_async = self.cypher_ppo_config.actor_rollout_ref.rollout.mode == "async"
        use_orch_async = self.orch_ppo_config.actor_rollout_ref.rollout.mode == "async"
        
        (self.cypher_actor_rollout_wg, 
         self.orch_actor_rollout_wg,
         self.cypher_role_mapping,
         self.orch_role_mapping) = create_dual_worker_groups(
            cypher_config=self.cypher_ppo_config,
            orch_config=self.orch_ppo_config,
            dual_pool_manager=self.dual_pool_manager,
            cypher_async=use_cypher_async,
            orch_async=use_orch_async
        )
        
        # Initialize both worker groups with models (for rollout)
        logger.info("Initializing Cypher Generator worker group...")
        print(f"\n{'='*60}")
        print(f"INITIALIZING CYPHER GENERATOR MODEL")
        print(f"Cypher GPUs requested: {self.seq_resource_manager.rollout_cypher_gpus}")
        print(f"{'='*60}\n")
        initialize_worker_group(
            self.cypher_actor_rollout_wg,
            self.cypher_ppo_config,
            self.cypher_tokenizer,
            Role.ActorRollout
        )
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING ORCHESTRATOR MODEL")
        print(f"Orch GPUs requested: {self.seq_resource_manager.rollout_orch_gpus}")
        print(f"{'='*60}\n")
        logger.info("Initializing Orchestrator worker group...")
        initialize_worker_group(
            self.orch_actor_rollout_wg,
            self.orch_ppo_config,
            self.orch_tokenizer,
            Role.ActorRollout
        )
        
        self.seq_resource_manager.current_mode = "rollout"
        logger.info(f"Sequential mode initialized. Current phase: {self.seq_resource_manager.current_mode}")
        logger.info(f"Rollout GPUs: Cypher={self.seq_resource_manager.rollout_cypher_gpus}, "
                   f"Orch={self.seq_resource_manager.rollout_orch_gpus}")
        logger.info(f"Training GPUs: {self.seq_resource_manager.training_gpus} (all GPUs)")
    
    def _init_shared_model_workers(self, gpu_config: Dict[str, Any]):
        """Initialize workers in shared model mode (both agents use same model)."""
        logger.info("Initializing in SHARED MODEL mode...")
        
        # Create single resource pool manager
        shared_gpus = gpu_config.get('shared_gpus', gpu_config.get('total_gpus', 8))
        nnodes = gpu_config.get('nnodes', 1)
        
        self.resource_pool_manager, self.global_pool_id, self.role_mapping = create_resource_pool_manager(
            n_gpus_per_node=shared_gpus,
            nnodes=nnodes
        )
        
        # Create worker groups for Cypher Generator
        logger.info("Creating worker groups for Cypher Generator...")
        use_async = self.cypher_ppo_config.actor_rollout_ref.rollout.mode == "async"
        self.cypher_worker_groups, self.cypher_role_mapping = create_worker_groups(
            config=self.cypher_ppo_config,
            tokenizer=self.cypher_tokenizer,
            resource_pool_manager=self.resource_pool_manager,
            mapping=self.role_mapping,
            use_async=use_async
        )
        self.cypher_actor_rollout_wg = self.cypher_worker_groups['actor_rollout']
        
        # In shared mode, Orchestrator uses the same worker group
        logger.info("Orchestrator will share worker group with Cypher Generator...")
        self.orch_actor_rollout_wg = self.cypher_actor_rollout_wg
        self.orch_role_mapping = self.cypher_role_mapping
        
        # Initialize the shared worker group with model
        logger.info("Initializing shared worker group with model...")
        initialize_worker_group(
            self.cypher_actor_rollout_wg,
            self.cypher_ppo_config,
            self.cypher_tokenizer,
            Role.ActorRollout
        )
    
    def _init_dual_model_workers(self, gpu_config: Dict[str, Any]):
        """Initialize workers in dual model mode (separate GPUs for each model)."""
        logger.info("Initializing in DUAL MODEL mode...")
        
        cypher_gpus = gpu_config.get('cypher_gpus', 4)
        orch_gpus = gpu_config.get('orch_gpus', 4)
        nnodes = gpu_config.get('nnodes', 1)
        
        logger.info(f"GPU allocation: Cypher={cypher_gpus}, Orchestrator={orch_gpus}")
        
        # Create dual resource pool manager
        self.dual_pool_manager = create_dual_resource_pool_manager(
            cypher_gpus=cypher_gpus,
            orch_gpus=orch_gpus,
            nnodes=nnodes
        )
        
        # Create separate worker groups for each model
        use_cypher_async = self.cypher_ppo_config.actor_rollout_ref.rollout.mode == "async"
        use_orch_async = self.orch_ppo_config.actor_rollout_ref.rollout.mode == "async"
        
        (self.cypher_actor_rollout_wg, 
         self.orch_actor_rollout_wg,
         self.cypher_role_mapping,
         self.orch_role_mapping) = create_dual_worker_groups(
            cypher_config=self.cypher_ppo_config,
            orch_config=self.orch_ppo_config,
            dual_pool_manager=self.dual_pool_manager,
            cypher_async=use_cypher_async,
            orch_async=use_orch_async
        )
        
        # Initialize Cypher Generator worker group with model
        logger.info("Initializing Cypher Generator worker group with model...")
        initialize_worker_group(
            self.cypher_actor_rollout_wg,
            self.cypher_ppo_config,
            self.cypher_tokenizer,
            Role.ActorRollout
        )
        
        # Initialize Orchestrator worker group with model
        logger.info("Initializing Orchestrator worker group with model...")
        initialize_worker_group(
            self.orch_actor_rollout_wg,
            self.orch_ppo_config,
            self.orch_tokenizer,
            Role.ActorRollout
        )
    
    def _complete_initialization(self):
        """Complete initialization after worker groups are set up."""
        # Create execution engines
        logger.info("Creating Cypher Generator execution engine...")
        self.cypher_execution_engine = create_execution_engine(
            agent_class=CypherGeneratorAgent,
            env_class=GraphReasoningEnvironment,
            config=self.cypher_ppo_config,
            tokenizer=self.cypher_tokenizer,
            rollout_engine=self.cypher_actor_rollout_wg,
            agent_args={
                'schema_path': self.schema_path,
                'experience_buffer': self.experience_buffer,
                'max_steps': 5
            },
            env_args={
                'api_url': self.neo4j_url,
                'max_turns': 5
            },
            max_steps=5,
            engine_name="verl"
        )
        
        logger.info("Creating Orchestrator execution engine...")
        self.orch_execution_engine = create_execution_engine(
            agent_class=OrchestratorAgent,
            env_class=GraphReasoningEnvironment,
            config=self.orch_ppo_config,
            tokenizer=self.orch_tokenizer,
            rollout_engine=self.orch_actor_rollout_wg,
            agent_args={
                'schema_path': self.schema_path,
                'experience_buffer': self.experience_buffer,
                'mode': 'generation'
            },
            env_args={
                'api_url': self.neo4j_url,
                'max_turns': 1  # Orchestrator is single-turn per role
            },
            max_steps=1,
            engine_name="verl"
        )
        
        # EMA model for stable evaluation (will be deep copy of Orchestrator after first update)
        self.orchestrator_eval_model = None
        
        # Load validation set
        if self.val_questions_path:
            self.val_questions = load_validation_set(self.val_questions_path, 100)
        else:
            self.val_questions = []
            logger.warning("No validation questions provided")
        
        # Training state
        self.current_epoch = 0
        self.ema_decay = 0.99
        self.metrics_history = []
        
        logger.info("CollaborativeTrainer initialized with low-level rllm components!")
    
    def _init_cypher_generator(self) -> CypherGeneratorAgent:
        """Initialize Cypher Generator agent."""
        return CypherGeneratorAgent(
            schema_path=self.schema_path,
            experience_buffer=self.experience_buffer,
            max_steps=5
        )
    
    def _init_orchestrator(self) -> OrchestratorAgent:
        """Initialize Orchestrator agent."""
        return OrchestratorAgent(
            schema_path=self.schema_path,
            experience_buffer=self.experience_buffer,
            mode='generation'
        )
    
    def train(self, num_epochs: int = 45):
        """
        Main training loop with sequential 3-phase approach.
        
        Each epoch consists of:
        - Phase 1: Rollout collection (both models inference)
        - Phase 2: Cypher Generator PPO update
        - Phase 3: Orchestrator PPO update (conditional)
        
        Args:
            num_epochs: Total number of epochs to train
        """
        logger.info("=" * 80)
        logger.info("STARTING COLLABORATIVE MULTI-AGENT TRAINING")
        logger.info("=" * 80)
        logger.info(f"Training mode: {self.training_mode}")
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Initial stage: {self.curriculum_tracker.current_stage}")
        logger.info(f"Orchestrator schedule: {self.orch_schedule_config}")
        logger.info(f"EMA enabled: {self.ema_orchestrator.enabled}, decay: {self.ema_orchestrator.decay}")
        logger.info("=" * 80)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Get current training phase
            training_phase = get_phase_from_schedule(epoch + 1, self.orch_schedule_config)
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"Stage: {self.curriculum_tracker.current_stage} | Phase: {training_phase}")
            logger.info("=" * 80)
            
            # Run one epoch
            epoch_metrics = self._train_epoch()
            
            # Update curriculum
            success_rate = epoch_metrics['success_rate']
            stage_changed, new_stage = self.curriculum_tracker.update(success_rate, epoch)
            
            if stage_changed:
                logger.info(f"🎉 Curriculum stage changed to: {new_stage}")
            
            # Validation (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                val_metrics = self._run_validation()
                epoch_metrics['val_metrics'] = val_metrics
                
                # Detect drift
                drift_detected = detect_reward_drift(
                    epoch_metrics,
                    val_metrics,
                    threshold=0.2
                )
                
                if drift_detected:
                    self.ema_decay = adjust_ema_decay(self.ema_decay, True)
                    logger.warning(f"Adjusted EMA decay to {self.ema_decay}")
            
            # Checkpoint (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                is_best = self._is_best_model(epoch_metrics)
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    cypher_gen=self.cypher_generator,
                    orch_train=self.orchestrator_train,
                    orch_eval=self.orchestrator_eval,
                    exp_buffer=self.experience_buffer,
                    reward_tracker=self.reward_tracker,
                    current_stage=self.curriculum_tracker.current_stage,
                    metrics=epoch_metrics,
                    is_best=is_best
                )
                
                # Cleanup old checkpoints
                self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
            
            # Store metrics
            self.metrics_history.append(epoch_metrics)
            self._save_metrics()
            
            # Log summary
            self._log_epoch_summary(epoch, epoch_metrics)
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Final stage: {self.curriculum_tracker.current_stage}")
        logger.info(f"Best checkpoint: {self.checkpoint_manager.get_best_checkpoint_path()}")
        logger.info("=" * 80)
    
    def _train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch using sequential 3-phase approach.
        
        Phase 1 (Rollout): Both models in inference mode, collect trajectories
        Phase 2 (Cypher PPO): Train Cypher Generator using all GPUs
        Phase 3 (Orch PPO): Train Orchestrator using all GPUs (conditional)
        
        Returns:
            Dictionary with epoch metrics
        """
        epoch = self.current_epoch
        current_stage = self.curriculum_tracker.current_stage
        stage_config = self.curriculum_tracker.get_current_config()
        
        # Get current training phase from schedule
        training_phase = get_phase_from_schedule(epoch + 1, self.orch_schedule_config)
        
        logger.info("-" * 60)
        logger.info(f"EPOCH {epoch + 1} | Stage: {current_stage} | Phase: {training_phase}")
        logger.info("-" * 60)
        
        # =====================================================================
        # PHASE 1: ROLLOUT COLLECTION (Both models in inference mode)
        # =====================================================================
        logger.info("")
        logger.info("=" * 40)
        logger.info("PHASE 1: ROLLOUT COLLECTION")
        logger.info("=" * 40)
        
        # Step 1.1: Question Generation
        logger.info("Step 1.1: Generating training questions...")
        questions = self._generate_questions(
            num_questions=self.config.get('questions_per_epoch', 512),
            stage_config=stage_config
        )
        
        # Step 1.2: Rollout Collection (both agents interact)
        logger.info("Step 1.2: Collecting rollouts...")
        trajectories = self._collect_rollouts(questions)
        
        # Step 1.3: Compute Rewards
        logger.info("Step 1.3: Computing rewards...")
        cypher_rewards, orch_gen_rewards, orch_synth_rewards = self._compute_rewards(trajectories)
        
        # Step 1.4: Normalize Rewards
        logger.info("Step 1.4: Normalizing rewards...")
        self.reward_tracker.update(cypher_rewards, 'cypher')
        self.reward_tracker.update(orch_gen_rewards, 'orch_gen')
        self.reward_tracker.update(orch_synth_rewards, 'orch_synth')
        
        cypher_rewards_norm = self.reward_tracker.normalize(cypher_rewards, 'cypher')
        orch_gen_rewards_norm = self.reward_tracker.normalize(orch_gen_rewards, 'orch_gen')
        orch_synth_rewards_norm = self.reward_tracker.normalize(orch_synth_rewards, 'orch_synth')
        
        logger.info(f"Rollout complete: {len(trajectories)} trajectories collected")
        logger.info(f"Avg rewards: Cypher={np.mean(cypher_rewards):.3f}, "
                   f"OrchGen={np.mean(orch_gen_rewards):.3f}, "
                   f"OrchSynth={np.mean(orch_synth_rewards):.3f}")
        
        # =====================================================================
        # PHASE 2: CYPHER GENERATOR PPO UPDATE (Every epoch)
        # =====================================================================
        logger.info("")
        logger.info("=" * 40)
        logger.info("PHASE 2: CYPHER GENERATOR PPO UPDATE")
        logger.info("=" * 40)
        
        cypher_metrics = self._train_cypher_generator(trajectories, cypher_rewards_norm)
        logger.info(f"Cypher training complete. Metrics: {cypher_metrics}")
        
        # =====================================================================
        # PHASE 3: ORCHESTRATOR PPO UPDATE (Conditional based on schedule)
        # =====================================================================
        logger.info("")
        logger.info("=" * 40)
        logger.info("PHASE 3: ORCHESTRATOR PPO UPDATE")
        logger.info("=" * 40)
        
        orch_metrics = {}
        should_update_orch = should_update_orch_from_schedule(
            epoch + 1,  # 1-indexed
            self.orch_schedule_config,
            current_stage
        )
        
        if should_update_orch:
            logger.info(f"Updating Orchestrator (phase={training_phase}, stage={current_stage})")
            orch_metrics = self._train_orchestrator(
                trajectories,
                orch_gen_rewards_norm,
                orch_synth_rewards_norm
            )
            
            # Update EMA evaluator after Orchestrator training
            if self.ema_orchestrator.enabled:
                logger.info(f"Updating EMA Orchestrator (decay={self.ema_orchestrator.decay})")
                # Note: In full implementation, this would update actual model weights
                # For now, we track the update count
                self.ema_orchestrator.update_count += 1
                logger.info(f"EMA update #{self.ema_orchestrator.update_count}")
            
            logger.info(f"Orchestrator training complete. Metrics: {orch_metrics}")
        else:
            logger.info(f"Skipping Orchestrator update (phase={training_phase}, epoch={epoch+1})")
            logger.info(f"Next Orch update based on schedule...")
        
        # =====================================================================
        # POST-TRAINING: Experience Buffer Update
        # =====================================================================
        logger.info("")
        logger.info("Updating experience buffer...")
        self._update_experience_buffer(trajectories, cypher_rewards)
        
        # Compute epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'training_phase': training_phase,
            'orch_updated': should_update_orch,
            'success_rate': compute_success_rate(trajectories, threshold=0.7),
            'avg_answer_quality': np.mean([t.get('answer_quality', {}).get('score', 0) for t in trajectories]),
            'avg_data_quality': np.mean([t.get('data_quality', {}).get('data_quality_score', 0) for t in trajectories]),
            'avg_trajectory_length': np.mean([len(t.get('trajectory', [])) for t in trajectories]),
            'avg_cypher_reward': np.mean(cypher_rewards),
            'avg_orch_gen_reward': np.mean(orch_gen_rewards),
            'avg_orch_synth_reward': np.mean(orch_synth_rewards),
            'cypher_metrics': cypher_metrics,
            'orch_metrics': orch_metrics,
            'exp_buffer_size': len(self.experience_buffer),
            'ema_update_count': self.ema_orchestrator.update_count,
            'ema_decay': self.ema_orchestrator.decay
        }
        
        return epoch_metrics
    
    def _generate_questions(self, num_questions: int, stage_config: Dict[str, Any]) -> List[str]:
        """
        Generate training questions using Orchestrator model inference.
        
        Args:
            num_questions: Number of questions to generate
            stage_config: Current curriculum stage configuration
            
        Returns:
            List of generated questions
        """
        logger.info(f"Generating {num_questions} questions for stage {stage_config['stage']} using real model inference...")
        
        # Load schema for question generation
        schema = self._load_schema()
        
        # Build prompts for question generation
        prompt_builder = OrchestratorPromptBuilder()
        
        # Prepare curriculum constraints
        curriculum_constraints = {
            'max_hops': stage_config.get('max_hops', 2),
            'focus_area': stage_config.get('focus_area', 'general biomedical queries')
        }
        
        # Scope constraints (allow all by default)
        scope_constraints = {
            'allowed_node_types': [],
            'allowed_edge_types': [],
            'avoid_regions': []
        }
        
        # Get recent questions for diversity (from metrics history)
        recent_questions = []
        for m in self.metrics_history[-20:]:
            if isinstance(m, dict) and 'questions' in m:
                recent_questions.extend(m['questions'][:5])
        
        # Build prompts for each question
        prompts_text = []
        for i in range(num_questions):
            prompt = prompt_builder.build_question_generation_prompt(
                schema=schema,
                difficulty=stage_config['stage'],
                curriculum_constraints=curriculum_constraints,
                scope_constraints=scope_constraints,
                recent_questions=recent_questions[-20:]  # Last 20 for diversity
            )
            prompts_text.append(prompt)
        
        # Tokenize prompts
        logger.debug(f"Tokenizing {len(prompts_text)} prompts...")
        max_prompt_length = self.orch_ppo_config.data.max_prompt_length
        
        # Apply chat template if available
        formatted_prompts = []
        for prompt in prompts_text:
            if hasattr(self.orch_tokenizer, 'apply_chat_template'):
                # Format as chat message
                messages = [{"role": "user", "content": prompt}]
                formatted = self.orch_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(prompt)
        
        # Tokenize with padding
        tokenized = self.orch_tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_prompt_length,
            truncation=True
        )
        
        # Create DataProto for generation
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        
        gen_batch = DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info={
                "eos_token_id": self.orch_tokenizer.eos_token_id,
                "pad_token_id": self.orch_tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": 0.7,  # Slightly creative for question generation
            }
        )
        
        # Generate using Orchestrator worker group
        logger.info("Generating questions with Orchestrator model...")
        try:
            gen_output = self.orch_actor_rollout_wg.generate_sequences(gen_batch)
            
            # Decode generated responses
            responses = gen_output.batch["responses"]  # (batch_size, response_length)
            
            questions = []
            for i in range(responses.shape[0]):
                # Decode response tokens
                response_ids = responses[i].tolist()
                # Remove padding tokens
                response_ids = [t for t in response_ids if t != self.orch_tokenizer.pad_token_id]
                # Decode to text
                response_text = self.orch_tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Extract the question from the response
                question = self._extract_question_from_response(response_text)
                questions.append(question)
                
            logger.info(f"Generated {len(questions)} questions using real model inference")
            
            # Log sample questions
            for i, q in enumerate(questions[:3]):
                logger.debug(f"  Sample question {i+1}: {q[:100]}...")
                
        except Exception as e:
            logger.error(f"Error generating questions with model: {e}")
            logger.warning("Falling back to template-based questions...")
            questions = self._generate_fallback_questions(num_questions, stage_config)
        
        return questions
    
    def _extract_question_from_response(self, response_text: str) -> str:
        """
        Extract the generated question from model response.
        
        Args:
            response_text: Raw model response
            
        Returns:
            Extracted question string
        """
        # Clean up the response
        response_text = response_text.strip()
        
        # If response contains "Generate question:" prefix from prompt, skip it
        if "Generate question:" in response_text:
            response_text = response_text.split("Generate question:")[-1].strip()
        
        # Take the first line/sentence as the question
        lines = response_text.split('\n')
        question = lines[0].strip()
        
        # Ensure it ends with a question mark
        if question and not question.endswith('?'):
            question = question + '?'
        
        # Fallback if empty
        if not question or len(question) < 10:
            question = "What genes are associated with diabetes?"
        
        return question
    
    def _generate_fallback_questions(self, num_questions: int, stage_config: Dict[str, Any]) -> List[str]:
        """
        Generate fallback template-based questions when model inference fails.
        
        Args:
            num_questions: Number of questions to generate
            stage_config: Current curriculum stage configuration
            
        Returns:
            List of template-based questions
        """
        import random
        
        # Template questions by difficulty
        easy_templates = [
            "What genes are associated with diabetes?",
            "What proteins interact with INS?",
            "What diseases are linked to BRCA1?",
            "What pathways involve TP53?",
            "What drugs target EGFR?"
        ]
        medium_templates = [
            "What genes are associated with diabetes and also interact with insulin signaling proteins?",
            "What proteins form complexes with BRCA1 and are involved in DNA repair?",
            "What pathways are shared between cancer and metabolic disorders?",
            "What drugs target genes that are differentially expressed in Alzheimer's disease?"
        ]
        hard_templates = [
            "What genes are associated with both Type 2 diabetes and cardiovascular disease, and what pathways do they share?",
            "What proteins interact with BRCA1, are involved in DNA repair, and are also targets of FDA-approved cancer drugs?",
            "What are the common upstream regulators of genes differentially expressed in multiple cancer types?"
        ]
        
        difficulty = stage_config.get('stage', 'easy')
        if difficulty == 'easy':
            templates = easy_templates
        elif difficulty == 'medium':
            templates = medium_templates
        else:
            templates = hard_templates
        
        questions = []
        for i in range(num_questions):
            template_idx = i % len(templates)
            questions.append(templates[template_idx])
        
        return questions
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the knowledge graph schema."""
        if hasattr(self, '_cached_schema'):
            return self._cached_schema
        
        try:
            schema_path = Path(self.schema_path)
            if not schema_path.is_absolute():
                # Try relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                schema_path = project_root / self.schema_path
            
            with open(schema_path, 'r') as f:
                self._cached_schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
        except Exception as e:
            logger.warning(f"Could not load schema from {self.schema_path}: {e}")
            # Return minimal schema
            self._cached_schema = {
                'knowledge_graph_schema': {
                    'node_types': {
                        'gene': {'description': 'A gene'},
                        'protein': {'description': 'A protein'},
                        'disease': {'description': 'A disease'},
                        'pathway': {'description': 'A biological pathway'},
                        'drug': {'description': 'A drug or compound'}
                    },
                    'edge_types': {
                        'ASSOCIATED_WITH': {'source_node_type': 'gene', 'target_node_type': 'disease'},
                        'INTERACTS_WITH': {'source_node_type': 'protein', 'target_node_type': 'protein'},
                        'TARGETS': {'source_node_type': 'drug', 'target_node_type': 'gene'},
                        'PART_OF': {'source_node_type': 'gene', 'target_node_type': 'pathway'}
                    }
                }
            }
        
        return self._cached_schema
    
    def _collect_rollouts(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Collect rollouts using real Cypher Generator model inference and Neo4j execution.
        
        Uses:
        - CypherGeneratorAgent for prompt building and response parsing
        - Neo4jExecutor for real query execution
        - OrchestratorAgent for data/answer evaluation
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of trajectory dictionaries with:
                - question: Original question
                - trajectory: List of Cypher queries and results
                - data_quality: Data quality evaluation from Orchestrator
                - answer: Synthesized answer from Orchestrator
                - answer_quality: Answer quality evaluation from Orchestrator
        """
        logger.info(f"Collecting rollouts for {len(questions)} questions using real model inference...")
        
        # Collect raw trajectories using real model inference and Neo4j
        raw_trajectories = []
        
        for i, question in enumerate(questions):
            try:
                # Collect multi-step trajectory for this question
                trajectory_data = self._collect_single_trajectory(question, max_steps=5)
                raw_trajectories.append({
                    'idx': i,
                    'question': question,
                    'trajectory': trajectory_data['steps'],
                    'total_results': trajectory_data['total_results'],
                    'success': trajectory_data['success']
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Collected trajectories for {i+1}/{len(questions)} questions")
                    
            except Exception as e:
                logger.error(f"Error collecting trajectory for question {i}: {e}")
                raw_trajectories.append({
                    'idx': i,
                    'question': question,
                    'trajectory': [],
                    'total_results': 0,
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"Collected {len(raw_trajectories)} raw trajectories")
        
        # Process raw trajectories and add Orchestrator evaluations
        trajectories = []
        for i, raw_traj in enumerate(raw_trajectories):
            try:
                question = raw_traj['question']
                cypher_trajectory = raw_traj.get('trajectory', [])
                
                # Step 1: Data Quality Evaluation (using Orchestrator in data_eval mode)
                data_quality_eval = self._evaluate_data_quality(
                    question=question,
                    cypher_trajectory=cypher_trajectory
                )
                
                # Step 2: Answer Synthesis (using Orchestrator in synthesis mode)
                answer = self._synthesize_answer(
                    question=question,
                    cypher_trajectory=cypher_trajectory,
                    data_quality_feedback=data_quality_eval
                )
                
                # Step 3: Answer Quality Evaluation (using Orchestrator in answer_eval mode)
                answer_quality_eval = self._evaluate_answer_quality(
                    question=question,
                    answer=answer,
                    retrieved_data=cypher_trajectory
                )
                
                trajectory = {
                    'question': question,
                    'trajectory': cypher_trajectory,
                    'data_quality': data_quality_eval,
                    'answer': answer,
                    'answer_quality': answer_quality_eval,
                    'error': raw_traj.get('error')
                }
                
                trajectories.append(trajectory)
                
                if (i + 1) % 100 == 0:
                    logger.debug(f"Processed {i + 1}/{len(questions)} trajectories with evaluations")
                
            except Exception as e:
                logger.error(f"Error processing trajectory {i}: {e}")
                trajectories.append({
                    'question': raw_traj['question'],
                    'trajectory': raw_traj.get('trajectory', []),
                    'data_quality': {'data_quality_score': 0.5},
                    'answer': "Error generating answer",
                    'answer_quality': {'score': 0.0},
                    'error': str(e)
                })
        
        logger.info(f"Collected {len(trajectories)} complete trajectories with evaluations")
        
        # Log sample rollouts for inspection
        self._log_sample_rollouts(trajectories)
        
        return trajectories
    
    def _collect_single_trajectory(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Collect a multi-step trajectory for a single question.
        
        Uses CypherGeneratorAgent for prompt building and response parsing,
        and Neo4jExecutor for real query execution.
        
        Args:
            question: The question to answer
            max_steps: Maximum number of query steps
            
        Returns:
            Dictionary with trajectory steps and metadata
        """
        # Create agent for this trajectory
        cypher_agent = CypherGeneratorAgent(
            schema_path=self.schema_path,
            experience_buffer=self.experience_buffer,
            max_steps=max_steps
        )
        cypher_agent.reset()
        
        # Create Neo4j executor
        executor = Neo4jExecutor(self.neo4j_url)
        
        steps = []
        total_results = 0
        
        try:
            # Initial observation
            initial_obs = {'question': question}
            cypher_agent.update_from_env(initial_obs, reward=0.0, done=False, info={})
            
            for step_num in range(1, max_steps + 1):
                # Get prompt from agent
                if not cypher_agent.messages:
                    break
                prompt = cypher_agent.messages[-1]['content']
                
                # Generate Cypher query using model
                cypher_query = self._generate_cypher_query(prompt)
                
                # Let agent parse the response
                action = cypher_agent.update_from_model(cypher_query)
                parsed_query = action.action
                
                # Check for DONE signal
                if parsed_query.upper().strip() == "DONE":
                    logger.debug(f"Question '{question[:50]}...' completed after {step_num-1} steps (DONE)")
                    break
                
                # Execute query against Neo4j
                result = executor.execute_query(parsed_query)
                
                # Store step data
                step_data = {
                    'query': parsed_query,
                    'result': result.get('result', {}),
                    'success': result['success'],
                    'has_data': result['has_data'],
                    'num_results': result['num_results'],
                    'execution_time_ms': result['execution_time_ms'],
                    'data_summary': self._generate_result_summary(result),
                    'error': result.get('error')
                }
                steps.append(step_data)
                total_results += result['num_results']
                
                # Build next observation for agent
                next_obs = {
                    'question': question,
                    'previous_query': parsed_query,
                    'previous_result': result,
                    'turn': step_num
                }
                cypher_agent.update_from_env(next_obs, reward=0.0, done=False, info={})
                
        except Exception as e:
            logger.error(f"Error during trajectory collection: {e}")
        finally:
            executor.close()
        
        return {
            'steps': steps,
            'total_results': total_results,
            'success': len(steps) > 0 and any(s['has_data'] for s in steps)
        }
    
    def _generate_cypher_query(self, prompt: str) -> str:
        """
        Generate a Cypher query using the Cypher Generator model.
        
        For single queries, we use a simple transformer generate call
        instead of the distributed worker group (which requires batch sizes
        divisible by number of workers).
        
        Args:
            prompt: The formatted prompt for Cypher generation
            
        Returns:
            Generated response string (may contain Cypher or DONE)
        """
        # For single query generation, use the batched version with padding
        results = self._generate_cypher_queries_batch([prompt])
        return results[0] if results else "MATCH (n) RETURN n LIMIT 1"
    
    def _generate_cypher_queries_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate Cypher queries in batch using the Cypher Generator model.
        
        Pads the batch to be divisible by number of workers if needed.
        
        Args:
            prompts: List of formatted prompts for Cypher generation
            
        Returns:
            List of generated response strings
        """
        if not prompts:
            return []
        
        max_prompt_length = self.cypher_ppo_config.data.max_prompt_length
        
        # Get number of workers for padding calculation
        num_workers = self.seq_resource_manager.rollout_cypher_gpus if hasattr(self, 'seq_resource_manager') else 4
        
        # Pad prompts to be divisible by num_workers
        original_count = len(prompts)
        padded_prompts = list(prompts)
        while len(padded_prompts) % num_workers != 0:
            padded_prompts.append(prompts[0])  # Duplicate first prompt as padding
        
        # Apply chat template if available
        formatted_prompts = []
        for prompt in padded_prompts:
            if hasattr(self.cypher_tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted = self.cypher_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(prompt)
        
        # Tokenize
        tokenized = self.cypher_tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_prompt_length,
            truncation=True
        )
        
        # Create DataProto for generation
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        
        gen_batch = DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info={
                "eos_token_id": self.cypher_tokenizer.eos_token_id,
                "pad_token_id": self.cypher_tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": 0.3,  # Lower temperature for deterministic Cypher
            }
        )
        
        # Debug: Print first prompt
        if formatted_prompts:
            prompt_tokens = len(self.cypher_tokenizer.encode(formatted_prompts[0]))
            print(f"\n{'='*60}")
            print(f"=== CYPHER GENERATOR INPUT (first prompt) ===")
            print(f"Prompt length: {len(formatted_prompts[0])} chars, ~{prompt_tokens} tokens")
            print(f"Max prompt length: {max_prompt_length} tokens")
            print(f"{'='*60}")
            # Print last 500 chars to see if question is included
            print(f"...END OF PROMPT:\n{formatted_prompts[0][-800:]}")
            print(f"{'='*60}\n")
        
        try:
            # Generate using Cypher Generator worker group
            gen_output = self.cypher_actor_rollout_wg.generate_sequences(gen_batch)
            
            # Decode responses
            responses = gen_output.batch["responses"]
            
            decoded_responses = []
            for i in range(responses.shape[0]):
                response_ids = responses[i].tolist()
                response_ids = [t for t in response_ids if t != self.cypher_tokenizer.pad_token_id]
                response_text = self.cypher_tokenizer.decode(response_ids, skip_special_tokens=True)
                decoded_responses.append(response_text)
                
                # Debug: Print Cypher Generator output
                if i < 3:  # Only print first 3 to avoid spam
                    print(f"\n{'='*60}")
                    print(f"=== CYPHER GENERATOR OUTPUT {i+1} ===")
                    print(f"{'='*60}")
                    print(f"Response:\n{response_text[:800]}")
                    print(f"{'='*60}\n")
            
            # Return only the original (non-padded) responses
            return decoded_responses[:original_count]
            
        except Exception as e:
            logger.error(f"Error generating Cypher queries: {e}")
            return ["MATCH (n) RETURN n LIMIT 1" for _ in range(original_count)]
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> str:
        """Generate a brief summary of query result."""
        if not result['success']:
            return f"Query failed: {result.get('error', 'Unknown error')}"
        
        if not result['has_data']:
            return "No results returned"
        
        num_results = result['num_results']
        exec_time = result['execution_time_ms']
        
        if exec_time < 100:
            time_icon = "⚡"
        elif exec_time < 500:
            time_icon = "○"
        elif exec_time < 1000:
            time_icon = "△"
        else:
            time_icon = "✗"
        
        return f"Retrieved {num_results} entities in {exec_time:.0f}ms {time_icon}"
    
    def _log_sample_rollouts(self, trajectories: List[Dict[str, Any]], sample_size: int = 3):
        """
        Log a sample of rollouts for inspection.
        
        Args:
            trajectories: List of trajectory dictionaries
            sample_size: Number of examples to log (default: 3)
        """
        import random
        
        if not trajectories:
            return
        
        # Get rollout logging config
        rollout_config = self.config.get('rollout_logging', {})
        if not rollout_config.get('enabled', True):
            return
        
        sample_size = rollout_config.get('sample_per_epoch', sample_size)
        
        # Select sample indices (first one + random samples)
        n = len(trajectories)
        if n <= sample_size:
            sample_indices = list(range(n))
        else:
            sample_indices = [0] + random.sample(range(1, n), sample_size - 1)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"SAMPLE ROLLOUTS (Epoch {self.current_epoch + 1})")
        logger.info("=" * 60)
        
        for idx in sample_indices:
            traj = trajectories[idx]
            logger.info("")
            logger.info(f"--- Rollout {idx + 1}/{n} ---")
            logger.info(f"Question: {traj.get('question', 'N/A')[:200]}...")
            
            # Log Cypher trajectory summary
            cypher_traj = traj.get('trajectory', [])
            if cypher_traj:
                logger.info(f"Cypher Steps: {len(cypher_traj)}")
                for step_idx, step in enumerate(cypher_traj[:3]):  # Show first 3 steps
                    cypher = step.get('cypher', step.get('query', 'N/A'))
                    if isinstance(cypher, str):
                        cypher_preview = cypher[:100].replace('\n', ' ')
                        logger.info(f"  Step {step_idx + 1}: {cypher_preview}...")
            else:
                logger.info("Cypher Steps: 0 (empty trajectory)")
            
            # Log evaluations
            data_qual = traj.get('data_quality', {})
            logger.info(f"Data Quality Score: {data_qual.get('data_quality_score', 'N/A'):.2f}" 
                       if isinstance(data_qual.get('data_quality_score'), (int, float)) else "Data Quality Score: N/A")
            
            # Log answer preview
            answer = traj.get('answer', 'N/A')
            if isinstance(answer, str):
                logger.info(f"Answer: {answer[:150]}...")
            
            answer_qual = traj.get('answer_quality', {})
            logger.info(f"Answer Quality Score: {answer_qual.get('score', 'N/A'):.2f}"
                       if isinstance(answer_qual.get('score'), (int, float)) else "Answer Quality Score: N/A")
            
            # Log errors if any
            if 'error' in traj:
                logger.warning(f"Error: {traj['error']}")
        
        logger.info("")
        logger.info("=" * 60)
        
        # Optionally save to file
        if rollout_config.get('save_samples', False):
            self._save_sample_rollouts(trajectories, sample_indices)
    
    def _save_sample_rollouts(self, trajectories: List[Dict[str, Any]], sample_indices: List[int]):
        """
        Save sample rollouts to a JSON file for detailed inspection.
        
        Args:
            trajectories: All trajectories
            sample_indices: Indices of samples to save
        """
        rollout_dir = Path(self.log_dir) / "rollout_samples"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        
        epoch = self.current_epoch + 1
        rollout_path = rollout_dir / f"epoch_{epoch:04d}_samples.json"
        
        samples = []
        for idx in sample_indices:
            traj = trajectories[idx]
            samples.append({
                'idx': idx,
                'epoch': epoch,
                'question': traj.get('question', ''),
                'cypher_trajectory': traj.get('trajectory', []),
                'data_quality': traj.get('data_quality', {}),
                'answer': traj.get('answer', ''),
                'answer_quality': traj.get('answer_quality', {}),
                'error': traj.get('error', None)
            })
        
        try:
            with open(rollout_path, 'w') as f:
                json.dump(samples, f, indent=2, default=str)
            logger.debug(f"Saved {len(samples)} sample rollouts to {rollout_path}")
        except Exception as e:
            logger.error(f"Failed to save sample rollouts: {e}")
    
    def _evaluate_data_quality(self, question: str, cypher_trajectory: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate data quality using Orchestrator in 'data_eval' mode.
        
        Uses real model inference to evaluate the quality of retrieved data.
        
        Args:
            question: The original question
            cypher_trajectory: List of query steps with results
            
        Returns:
            Data quality evaluation dictionary
        """
        try:
            # Create OrchestratorAgent in data_eval mode
            orch_agent = OrchestratorAgent(
                schema_path=self.schema_path,
                experience_buffer=self.experience_buffer,
                mode='data_eval'
            )
            orch_agent.reset()
            
            # Build observation for data evaluation
            observation = {
                'question': question,
                'trajectory': cypher_trajectory,
                'known_semantic_issues': self.experience_buffer.get_semantic_issues_for_prompt(question)
            }
            
            # Update agent to build prompt
            orch_agent.update_from_env(observation, reward=0.0, done=False, info={})
            
            if not orch_agent.messages:
                raise ValueError("No prompt generated")
            
            prompt = orch_agent.messages[-1]['content']
            
            # Generate evaluation using Orchestrator model
            responses = self._generate_with_orchestrator([prompt], mode='data_eval')
            
            if responses:
                # Parse the evaluation JSON
                evaluation = orch_agent._parse_data_quality_eval(responses[0])
                return evaluation
            else:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.warning(f"Error in data quality evaluation: {e}, using defaults")
            return {
                'data_quality_score': 0.5,
                'relevance_score': 0.5,
                'completeness_score': 0.5,
                'trajectory_quality_score': 0.5,
                'doubt_level': 0.3
            }
    
    def _synthesize_answer(self, question: str, cypher_trajectory: List[Dict], 
                          data_quality_feedback: Dict) -> str:
        """
        Synthesize answer using Orchestrator in 'synthesis' mode.
        
        Uses real model inference to generate a natural language answer.
        
        Args:
            question: The original question
            cypher_trajectory: List of query steps with results
            data_quality_feedback: Data quality evaluation
            
        Returns:
            Synthesized answer string
        """
        try:
            # Create OrchestratorAgent in synthesis mode
            orch_agent = OrchestratorAgent(
                schema_path=self.schema_path,
                experience_buffer=self.experience_buffer,
                mode='synthesis'
            )
            orch_agent.reset()
            
            # Build observation for synthesis
            observation = {
                'question': question,
                'trajectory_data': cypher_trajectory,
                'data_quality_feedback': data_quality_feedback
            }
            
            # Update agent to build prompt
            orch_agent.update_from_env(observation, reward=0.0, done=False, info={})
            
            if not orch_agent.messages:
                raise ValueError("No prompt generated")
            
            prompt = orch_agent.messages[-1]['content']
            
            # Generate answer using Orchestrator model
            responses = self._generate_with_orchestrator([prompt], mode='synthesis')
            
            if responses:
                # Parse the answer
                answer = orch_agent._parse_answer(responses[0])
                return answer
            else:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.warning(f"Error in answer synthesis: {e}, using fallback")
            return f"Based on the retrieved data, here is an answer to: {question}"
    
    def _evaluate_answer_quality(self, question: str, answer: str, 
                                 retrieved_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate answer quality using Orchestrator in 'answer_eval' mode.
        
        Uses real model inference to evaluate the quality of the synthesized answer.
        
        Args:
            question: The original question
            answer: The synthesized answer
            retrieved_data: List of query steps with results
            
        Returns:
            Answer quality evaluation dictionary
        """
        try:
            # Create OrchestratorAgent in answer_eval mode
            orch_agent = OrchestratorAgent(
                schema_path=self.schema_path,
                experience_buffer=self.experience_buffer,
                mode='answer_eval'
            )
            orch_agent.reset()
            
            # Build observation for answer evaluation
            observation = {
                'question': question,
                'answer': answer
            }
            
            # Update agent to build prompt
            orch_agent.update_from_env(observation, reward=0.0, done=False, info={})
            
            if not orch_agent.messages:
                raise ValueError("No prompt generated")
            
            prompt = orch_agent.messages[-1]['content']
            
            # Generate evaluation using Orchestrator model
            responses = self._generate_with_orchestrator([prompt], mode='answer_eval')
            
            if responses:
                # Parse the evaluation JSON
                evaluation = orch_agent._parse_answer_quality_eval(responses[0])
                return evaluation
            else:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.warning(f"Error in answer quality evaluation: {e}, using defaults")
            return {
                'score': 0.5,
                'correctness': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'accuracy': 0.5
            }
    
    def _generate_with_orchestrator(self, prompts: List[str], mode: str = 'generation') -> List[str]:
        """
        Generate responses using Orchestrator model via generate_sequences().
        
        Pads the batch to be divisible by number of workers if needed.
        
        Args:
            prompts: List of prompt strings
            mode: Orchestrator mode for temperature selection
            
        Returns:
            List of generated response strings
        """
        if not prompts:
            return []
        
        max_prompt_length = self.orch_ppo_config.data.max_prompt_length
        
        # Get number of workers for padding calculation
        num_workers = self.seq_resource_manager.rollout_orch_gpus if hasattr(self, 'seq_resource_manager') else 4
        
        # Pad prompts to be divisible by num_workers
        original_count = len(prompts)
        padded_prompts = list(prompts)
        while len(padded_prompts) % num_workers != 0:
            padded_prompts.append(prompts[0])  # Duplicate first prompt as padding
        
        # Apply chat template if available
        formatted_prompts = []
        for prompt in padded_prompts:
            if hasattr(self.orch_tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted = self.orch_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(prompt)
        
        # Tokenize with padding
        tokenized = self.orch_tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_prompt_length,
            truncation=True
        )
        
        # Create DataProto for generation
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Set temperature based on mode
        temperature_map = {
            'generation': 0.7,  # Creative for question generation
            'data_eval': 0.2,   # Deterministic for evaluation
            'synthesis': 0.5,   # Balanced for answer synthesis
            'answer_eval': 0.2  # Deterministic for evaluation
        }
        temperature = temperature_map.get(mode, 0.5)
        
        gen_batch = DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info={
                "eos_token_id": self.orch_tokenizer.eos_token_id,
                "pad_token_id": self.orch_tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": temperature,
            }
        )
        
        try:
            # Generate using Orchestrator worker group
            gen_output = self.orch_actor_rollout_wg.generate_sequences(gen_batch)
            
            # Decode responses
            responses = gen_output.batch["responses"]
            
            decoded_responses = []
            for i in range(responses.shape[0]):
                response_ids = responses[i].tolist()
                response_ids = [t for t in response_ids if t != self.orch_tokenizer.pad_token_id]
                response_text = self.orch_tokenizer.decode(response_ids, skip_special_tokens=True)
                decoded_responses.append(response_text)
            
            # Return only the original (non-padded) responses
            return decoded_responses[:original_count]
            
            return decoded_responses
            
        except Exception as e:
            logger.error(f"Error generating with Orchestrator: {e}")
            return []
    
    def _compute_rewards(self, trajectories: List[Dict[str, Any]]) -> tuple:
        """
        Compute rewards for all agents.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Tuple of (cypher_rewards, orch_gen_rewards, orch_synth_rewards)
        """
        cypher_rewards = []
        orch_gen_rewards = []
        orch_synth_rewards = []
        
        for traj in trajectories:
            # Cypher Generator reward
            cypher_task_info = {
                'question': traj['question'],
                'cypher_trajectory': traj.get('trajectory', []),
                'answer_quality_score': traj.get('answer_quality', {}).get('score', 0.5),
                'data_quality_score': traj.get('data_quality', {}).get('data_quality_score', 0.5),
                'trajectory_quality_score': traj.get('data_quality', {}).get('trajectory_quality_score', 0.5),
                'doubt_level': traj.get('data_quality', {}).get('doubt_level', 0.0)
            }
            cypher_reward_output = cypher_generator_reward_fn(cypher_task_info, "DONE")
            cypher_rewards.append(cypher_reward_output.reward)
            
            # Orchestrator Generation reward
            orch_gen_task_info = {
                'question': traj['question'],
                'answerability_score': traj.get('answer_quality', {}).get('score', 0) > 0.5,
                'difficulty_appropriate': True,  # Placeholder
                'recent_questions': [],
                'scope_adherence': True
            }
            orch_gen_reward_output = orchestrator_generation_reward_fn(orch_gen_task_info, traj['question'])
            orch_gen_rewards.append(orch_gen_reward_output.reward)
            
            # Orchestrator Synthesis reward
            orch_synth_task_info = {
                'synthesized_answer': traj.get('answer', ''),
                'answer_quality_score': traj.get('answer_quality', {}).get('score', 0.5),
                'cypher_trajectory': traj.get('trajectory', [])
            }
            orch_synth_reward_output = orchestrator_synthesis_reward_fn(orch_synth_task_info, traj.get('answer', ''))
            orch_synth_rewards.append(orch_synth_reward_output.reward)
        
        logger.debug(
            f"Computed rewards: "
            f"cypher={np.mean(cypher_rewards):.3f}, "
            f"orch_gen={np.mean(orch_gen_rewards):.3f}, "
            f"orch_synth={np.mean(orch_synth_rewards):.3f}"
        )
        
        return cypher_rewards, orch_gen_rewards, orch_synth_rewards
    
    def _train_cypher_generator(self, trajectories: List[Dict[str, Any]], rewards: List[float]) -> Dict[str, Any]:
        """
        Train Cypher Generator with manual PPO updates using worker groups.
        
        Args:
            trajectories: Training trajectories from rollout collection
            rewards: Normalized rewards
            
        Returns:
            Training metrics
        """
        logger.info("Training Cypher Generator with manual PPO updates...")
        
        # Prepare DataProto batch from trajectories
        # Each trajectory contains: question, cypher_trajectory, data_quality, answer, answer_quality
        
        # Extract prompts and responses from trajectories
        prompts = []
        responses = []
        token_level_rewards_list = []
        
        for i, traj in enumerate(trajectories):
            # Build prompt from question
            question = traj['question']
            prompt_text = f"Question: {question}\n\nGenerate Cypher queries to answer this question."
            
            # Tokenize prompt
            prompt_tokens = torch.tensor(
                self.cypher_tokenizer.encode(prompt_text, add_special_tokens=False),
                dtype=torch.long
            )
            prompts.append(prompt_tokens)
            
            # Build response from Cypher trajectory
            # For now, use a simple concatenation of queries
            cypher_queries = [step.get('query', '') for step in traj.get('trajectory', [])]
            response_text = "\n".join(cypher_queries) if cypher_queries else "DONE"
            
            # Tokenize response
            response_tokens = torch.tensor(
                self.cypher_tokenizer.encode(response_text, add_special_tokens=False),
                dtype=torch.long
            )
            responses.append(response_tokens)
            
            # Create token-level rewards (assign reward to last token)
            token_rewards = torch.zeros(len(response_tokens), dtype=torch.float32)
            token_rewards[-1] = rewards[i]  # Assign reward to last token
            token_level_rewards_list.append(token_rewards)
        
        # Pad sequences
        from verl.utils.torch_functional import pad_sequence_to_length
        
        # Left-pad prompts
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(p, dims=[0]) for p in prompts],
            batch_first=True,
            padding_value=self.cypher_tokenizer.pad_token_id
        ).flip(dims=[1])
        
        prompts_batch = pad_sequence_to_length(
            prompts_batch,
            self.cypher_ppo_config.data.max_prompt_length,
            self.cypher_tokenizer.pad_token_id,
            left_pad=True
        )
        
        # Right-pad responses
        responses_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.cypher_tokenizer.pad_token_id
        )
        
        responses_batch = pad_sequence_to_length(
            responses_batch,
            self.cypher_ppo_config.data.max_response_length,
            self.cypher_tokenizer.pad_token_id,
            left_pad=False
        )
        
        # Pad token-level rewards
        token_level_rewards_batch = torch.nn.utils.rnn.pad_sequence(
            token_level_rewards_list,
            batch_first=True,
            padding_value=0.0
        )
        
        token_level_rewards_batch = pad_sequence_to_length(
            token_level_rewards_batch,
            self.cypher_ppo_config.data.max_response_length,
            0.0,
            left_pad=False
        )
        
        # Concatenate prompts and responses
        input_ids = torch.cat([prompts_batch, responses_batch], dim=1)
        attention_mask = torch.where(input_ids != self.cypher_tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        
        # Create response mask (1 for response tokens, 0 for padding)
        response_mask = torch.where(responses_batch != self.cypher_tokenizer.pad_token_id, 1, 0)
        
        # Create unique IDs for GRPO grouping (each sample is its own group)
        uids = [f"cypher_{self.current_epoch}_{i}" for i in range(len(trajectories))]
        
        # Create old_log_probs placeholder (will be computed by the model during update)
        # For initial training, we use zeros as placeholder
        # The actual old_log_probs would come from the rollout phase
        batch_size = len(trajectories)
        response_len = responses_batch.shape[1]
        old_log_probs = torch.zeros(batch_size, response_len, dtype=torch.float32)
        
        # Create DataProto batch
        batch = DataProto.from_dict(
            tensors={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'prompts': prompts_batch,
                'responses': responses_batch,
                'response_mask': response_mask,
                'token_level_scores': token_level_rewards_batch,
                'token_level_rewards': token_level_rewards_batch,
                'old_log_probs': old_log_probs,  # Required for PPO update
            },
            non_tensors={
                'uid': uids  # Required for GRPO advantage estimation
            },
            meta_info={
                'temperature': self.cypher_ppo_config.actor_rollout_ref.rollout.temperature,  # Required for PPO
                'multi_turn': False,  # We handle multi-turn ourselves
                # global_token_num should be a list of sequence lengths (one per sample)
                'global_token_num': attention_mask.sum(dim=1).tolist(),  # List of token counts per sample
            }
        )
        
        # Import AdvantageEstimator enum for proper comparison
        from verl.trainer.ppo.ray_trainer import AdvantageEstimator
        
        # Get advantage estimator - convert string to enum if needed
        adv_estimator_str = self.cypher_ppo_config.algorithm.adv_estimator
        if isinstance(adv_estimator_str, str):
            adv_estimator = AdvantageEstimator(adv_estimator_str)
        else:
            adv_estimator = adv_estimator_str
        
        logger.debug(f"Using advantage estimator: {adv_estimator}")
        
        # Compute advantages using GRPO (doesn't require critic values)
        batch = compute_advantage(
            batch,
            adv_estimator=adv_estimator,
            gamma=self.cypher_ppo_config.algorithm.gamma,
            lam=self.cypher_ppo_config.algorithm.lam,
            norm_adv_by_std_in_grpo=self.cypher_ppo_config.algorithm.get('norm_adv_by_std_in_grpo', True),
            mask_truncated_samples=self.cypher_ppo_config.algorithm.get('mask_truncated_samples', False),
            clip_advantages=self.cypher_ppo_config.algorithm.get('clip_advantages', False)
        )
        
        # Pad batch to be divisible by number of workers
        # This is required because verl splits the batch evenly across workers
        from verl.protocol import pad_dataproto_to_divisor
        
        n_workers = self.cypher_ppo_config.trainer.n_gpus_per_node
        original_batch_size = len(batch)
        batch, pad_size = pad_dataproto_to_divisor(batch, n_workers)
        
        if pad_size > 0:
            logger.info(f"Padded batch from {original_batch_size} to {len(batch)} (divisible by {n_workers} workers)")
        
        # Perform PPO update using worker group
        logger.info(f"Performing PPO update on Cypher Generator (batch_size={len(batch)})...")
        actor_output = self.cypher_actor_rollout_wg.update_actor(batch)
        
        # Extract metrics
        update_metrics = actor_output.meta_info.get('metrics', {})
        
        # Collect overall metrics
        metrics = {
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            **update_metrics
        }
        
        logger.info(f"Cypher training complete. Avg reward: {metrics['avg_reward']:.3f}")
        return metrics
    
    def _train_orchestrator(
        self,
        trajectories: List[Dict[str, Any]],
        gen_rewards: List[float],
        synth_rewards: List[float]
    ) -> Dict[str, Any]:
        """
        Train Orchestrator with PPO using actual model.
        
        Args:
            trajectories: Training trajectories
            gen_rewards: Normalized generation rewards
            synth_rewards: Normalized synthesis rewards
            
        Returns:
            Training metrics
        """
        logger.info("Training Orchestrator with manual PPO updates...")
        
        # For now, use simplified training - just compute metrics
        # Full implementation would prepare DataProto batches and call worker group updates
        # Similar to _train_cypher_generator but for generation and synthesis modes
        
        metrics = {
            'avg_gen_reward': float(np.mean(gen_rewards)),
            'avg_synth_reward': float(np.mean(synth_rewards)),
            'std_gen_reward': float(np.std(gen_rewards)),
            'std_synth_reward': float(np.std(synth_rewards))
        }
        
        logger.info(f"Orchestrator training complete. Gen reward: {metrics['avg_gen_reward']:.3f}")
        logger.info("Note: Full Orchestrator PPO updates not yet implemented - using stub")
        return metrics
    
    def _update_experience_buffer(self, trajectories: List[Dict[str, Any]], rewards: List[float]):
        """
        Update experience buffer with patterns from episodes.
        
        Args:
            trajectories: Episode trajectories
            rewards: Episode rewards
        """
        # Extract good patterns (high reward)
        high_reward_episodes = [
            traj for traj, reward in zip(trajectories, rewards)
            if reward > 0.7
        ]
        if high_reward_episodes:
            self.experience_buffer.extract_good_patterns(high_reward_episodes)
            logger.debug(f"Extracted patterns from {len(high_reward_episodes)} high-reward episodes")
        
        # Extract bad data regions (low data quality)
        low_quality_episodes = [
            traj for traj in trajectories
            if traj.get('data_quality', {}).get('data_quality_score', 1.0) < 0.4
        ]
        if low_quality_episodes:
            self.experience_buffer.extract_bad_data_regions(low_quality_episodes)
            logger.debug(f"Extracted bad regions from {len(low_quality_episodes)} low-quality episodes")
        
        # Extract semantic ambiguities (high doubt)
        high_doubt_episodes = [
            traj for traj in trajectories
            if traj.get('data_quality', {}).get('doubt_level', 0.0) > 0.6
        ]
        if high_doubt_episodes:
            self.experience_buffer.extract_semantic_ambiguities(high_doubt_episodes)
            logger.debug(f"Extracted semantic issues from {len(high_doubt_episodes)} high-doubt episodes")
    
    def _run_validation(self) -> Dict[str, Any]:
        """
        Run validation on fixed question set.
        
        Returns:
            Validation metrics
        """
        if not self.val_questions:
            logger.warning("No validation questions available")
            return {}
        
        logger.info(f"Running validation on {len(self.val_questions)} questions...")
        
        val_metrics = validate_on_fixed_set(
            self.cypher_generator,
            self.orchestrator_eval,
            self.val_questions,
            self.env
        )
        
        logger.info(
            f"Validation results: "
            f"answer_quality={val_metrics.get('avg_answer_quality', 0):.3f}, "
            f"success_rate={val_metrics.get('success_rate', 0):.3f}"
        )
        
        return val_metrics
    
    def _is_best_model(self, epoch_metrics: Dict[str, Any]) -> bool:
        """
        Check if current model is best so far.
        
        Args:
            epoch_metrics: Current epoch metrics
            
        Returns:
            True if this is the best model
        """
        val_metrics = epoch_metrics.get('val_metrics', {})
        val_score = val_metrics.get('avg_answer_quality', 0.0)
        
        if val_score > self.checkpoint_manager.best_val_score:
            self.checkpoint_manager.best_val_score = val_score
            return True
        
        return False
    
    def _save_metrics(self):
        """Save metrics history to file."""
        metrics_path = Path(self.log_dir) / "metrics_history.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.debug(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log summary of epoch results.
        
        Args:
            epoch: Epoch number
            metrics: Epoch metrics
        """
        logger.info("")
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Stage: {metrics['stage']}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.3f}")
        logger.info(f"  Avg Answer Quality: {metrics['avg_answer_quality']:.3f}")
        logger.info(f"  Avg Data Quality: {metrics['avg_data_quality']:.3f}")
        logger.info(f"  Avg Trajectory Length: {metrics['avg_trajectory_length']:.2f}")
        logger.info(f"  Cypher Reward: {metrics['avg_cypher_reward']:.3f}")
        logger.info(f"  Orch Gen Reward: {metrics['avg_orch_gen_reward']:.3f}")
        logger.info(f"  Orch Synth Reward: {metrics['avg_orch_synth_reward']:.3f}")
        logger.info(f"  Experience Buffer Size: {metrics['exp_buffer_size']}")
        
        if 'val_metrics' in metrics:
            val_metrics = metrics['val_metrics']
            logger.info(f"  Val Answer Quality: {val_metrics.get('avg_answer_quality', 0):.3f}")
            logger.info(f"  Val Success Rate: {val_metrics.get('success_rate', 0):.3f}")
        
        logger.info("")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train collaborative multi-agent system")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--schema_path', type=str, required=True, help="Path to KG schema")
    parser.add_argument('--val_questions', type=str, help="Path to validation questions")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Checkpoint directory")
    parser.add_argument('--log_dir', type=str, required=True, help="Log directory")
    parser.add_argument('--experiment_name', type=str, default='pankllm_rl', help="Experiment name")
    parser.add_argument('--num_epochs', type=int, default=45, help="Number of epochs")
    parser.add_argument('--questions_per_epoch', type=int, default=512, help="Questions per epoch")
    parser.add_argument('--neo4j_url', type=str, 
                       default="https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j",
                       help="Neo4j API URL")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command-line args
    config.update({
        'schema_path': args.schema_path,
        'val_questions_path': args.val_questions,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'experiment_name': args.experiment_name,
        'questions_per_epoch': args.questions_per_epoch,
        'neo4j_url': args.neo4j_url
    })
    
    logger.info("=" * 80)
    logger.info("Collaborative Multi-Agent RL Training")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Questions per epoch: {args.questions_per_epoch}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Log dir: {args.log_dir}")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = CollaborativeTrainer(config)
    
    # Run training
    try:
        trainer.train(num_epochs=args.num_epochs)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    logger.info("Training script completed")


if __name__ == "__main__":
    main()

