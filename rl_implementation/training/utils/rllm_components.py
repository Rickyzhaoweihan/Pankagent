"""
Helper functions for initializing rllm low-level components.

This module provides utilities for setting up:
- AsyncAgentExecutionEngine for rollout collection
- Ray worker groups for distributed PPO updates
- Resource pool managers for GPU allocation (single or dual pool)
- Tokenizers and model loading
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import ray
from omegaconf import DictConfig

from rllm.engine import AsyncAgentExecutionEngine
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Role definitions for multi-model setup
# =============================================================================

class MultiModelRole:
    """Extended roles for multi-model training."""
    CypherActorRollout = "cypher_actor_rollout"
    OrchActorRollout = "orch_actor_rollout"


@dataclass
class DualResourcePoolManager:
    """
    Resource pool manager for dual-model training.
    
    Manages separate GPU pools for Cypher Generator and Orchestrator models.
    """
    cypher_pool_spec: Dict[str, list]
    orch_pool_spec: Dict[str, list]
    cypher_pool_id: str = "cypher_pool"
    orch_pool_id: str = "orch_pool"
    cypher_pool: RayResourcePool = field(default=None, repr=False)
    orch_pool: RayResourcePool = field(default=None, repr=False)
    
    def create_resource_pools(self):
        """Create both resource pools."""
        logger.info(f"Creating Cypher pool with spec: {self.cypher_pool_spec}")
        self.cypher_pool = RayResourcePool(
            process_on_nodes=self.cypher_pool_spec[self.cypher_pool_id],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix=self.cypher_pool_id
        )
        
        logger.info(f"Creating Orchestrator pool with spec: {self.orch_pool_spec}")
        self.orch_pool = RayResourcePool(
            process_on_nodes=self.orch_pool_spec[self.orch_pool_id],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix=self.orch_pool_id
        )
        
        self._check_resource_available()
        logger.info("Both resource pools created successfully")
    
    def get_cypher_pool(self) -> RayResourcePool:
        """Get the Cypher Generator resource pool."""
        if self.cypher_pool is None:
            raise RuntimeError("Cypher pool not initialized. Call create_resource_pools() first.")
        return self.cypher_pool
    
    def get_orch_pool(self) -> RayResourcePool:
        """Get the Orchestrator resource pool."""
        if self.orch_pool is None:
            raise RuntimeError("Orchestrator pool not initialized. Call create_resource_pools() first.")
        return self.orch_pool
    
    def get_total_gpus(self) -> int:
        """Get total number of GPUs across both pools."""
        cypher_gpus = sum(self.cypher_pool_spec[self.cypher_pool_id])
        orch_gpus = sum(self.orch_pool_spec[self.orch_pool_id])
        return cypher_gpus + orch_gpus
    
    def _check_resource_available(self):
        """Check if the resource pools can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) 
            for node, node_info in node_available_resources.items()
        }
        
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = self.get_total_gpus()
        
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than "
                f"total required GPUs {total_required_gpus} "
                f"(Cypher: {sum(self.cypher_pool_spec[self.cypher_pool_id])}, "
                f"Orch: {sum(self.orch_pool_spec[self.orch_pool_id])})"
            )
        
        logger.info(f"Resource check passed: {total_available_gpus} available >= {total_required_gpus} required")


def load_tokenizer(model_path: str, trust_remote_code: bool = False):
    """
    Load tokenizer from model path.
    
    Args:
        model_path: Path to the model (local or HDFS)
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {model_path}")
    
    # Download from HDFS if needed
    local_path = copy_local_path_from_hdfs(model_path)
    
    # Load tokenizer
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    
    logger.info(f"Tokenizer loaded successfully")
    return tokenizer


def create_resource_pool_manager(
    n_gpus_per_node: int,
    nnodes: int = 1
) -> Tuple[ResourcePoolManager, str, Dict]:
    """
    Create a single shared resource pool manager for GPU allocation.
    
    Use this when both models share the same GPUs (use_shared_model=true).
    
    Args:
        n_gpus_per_node: Number of GPUs per node
        nnodes: Number of nodes
        
    Returns:
        Tuple of (ResourcePoolManager instance, global_pool_id, mapping)
    """
    logger.info(f"Creating shared resource pool: {n_gpus_per_node} GPUs x {nnodes} nodes")
    
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [n_gpus_per_node] * nnodes,
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    # Create resource pool manager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )
    
    # Initialize the resource pools
    logger.info(f"Creating resource pools with spec: {resource_pool_spec}")
    logger.info(f"Role mapping: {mapping}")
    resource_pool_manager.create_resource_pool()
    
    # Verify pools were created
    logger.info(f"Resource pool dict keys: {list(resource_pool_manager.resource_pool_dict.keys())}")
    if global_pool_id not in resource_pool_manager.resource_pool_dict:
        raise RuntimeError(f"Failed to create resource pool '{global_pool_id}'. "
                          f"Available pools: {list(resource_pool_manager.resource_pool_dict.keys())}")
    
    logger.info("Resource pool manager created and initialized successfully")
    return resource_pool_manager, global_pool_id, mapping


def create_dual_resource_pool_manager(
    cypher_gpus: int,
    orch_gpus: int,
    nnodes: int = 1
) -> DualResourcePoolManager:
    """
    Create dual resource pool manager for separate GPU allocation per model.
    
    Use this when each model has its own dedicated GPUs (use_shared_model=false).
    
    Args:
        cypher_gpus: Number of GPUs for Cypher Generator model
        orch_gpus: Number of GPUs for Orchestrator model
        nnodes: Number of nodes
        
    Returns:
        DualResourcePoolManager instance with separate pools for each model
    
    Example:
        For 8 GPUs split 4+4:
        - Cypher Generator: GPUs 0-3
        - Orchestrator: GPUs 4-7
    """
    logger.info(f"Creating dual resource pools: Cypher={cypher_gpus} GPUs, Orch={orch_gpus} GPUs, nnodes={nnodes}")
    
    cypher_pool_id = "cypher_pool"
    orch_pool_id = "orch_pool"
    
    cypher_pool_spec = {
        cypher_pool_id: [cypher_gpus] * nnodes,
    }
    
    orch_pool_spec = {
        orch_pool_id: [orch_gpus] * nnodes,
    }
    
    dual_manager = DualResourcePoolManager(
        cypher_pool_spec=cypher_pool_spec,
        orch_pool_spec=orch_pool_spec,
        cypher_pool_id=cypher_pool_id,
        orch_pool_id=orch_pool_id,
    )
    
    # Initialize both pools
    dual_manager.create_resource_pools()
    
    logger.info(f"Dual resource pool manager created: "
                f"Cypher={cypher_gpus} GPUs, Orch={orch_gpus} GPUs")
    return dual_manager


def create_resource_pools_from_config(gpu_config: Dict[str, Any]) -> Tuple[Any, bool]:
    """
    Create resource pool(s) based on configuration.
    
    Args:
        gpu_config: GPU allocation configuration dict with keys:
            - total_gpus: Total GPUs available
            - nnodes: Number of nodes
            - cypher_gpus: GPUs for Cypher Generator
            - orch_gpus: GPUs for Orchestrator
            - use_shared_model: Whether to use shared model
            - shared_gpus: GPUs for shared model (if use_shared_model=true)
    
    Returns:
        Tuple of (resource_manager, is_dual_mode)
        - If is_dual_mode=True: resource_manager is DualResourcePoolManager
        - If is_dual_mode=False: resource_manager is (ResourcePoolManager, pool_id, mapping)
    """
    use_shared = gpu_config.get('use_shared_model', False)
    nnodes = gpu_config.get('nnodes', 1)
    
    if use_shared:
        # Shared model mode: both agents use the same model
        shared_gpus = gpu_config.get('shared_gpus', gpu_config.get('total_gpus', 8))
        logger.info(f"Using shared model mode with {shared_gpus} GPUs")
        resource_manager = create_resource_pool_manager(shared_gpus, nnodes)
        return resource_manager, False
    else:
        # Dual model mode: each agent has its own model
        cypher_gpus = gpu_config.get('cypher_gpus', 4)
        orch_gpus = gpu_config.get('orch_gpus', 4)
        logger.info(f"Using dual model mode: Cypher={cypher_gpus} GPUs, Orch={orch_gpus} GPUs")
        resource_manager = create_dual_resource_pool_manager(cypher_gpus, orch_gpus, nnodes)
        return resource_manager, True


def create_worker_group_for_pool(
    config: DictConfig,
    resource_pool: RayResourcePool,
    num_gpus: int,
    use_async: bool = False,
    name_prefix: str = "actor_rollout",
    role: str = "actor"
) -> Tuple[RayWorkerGroup, Any]:
    """
    Create a Ray worker group for a specific resource pool.
    
    Args:
        config: PPO configuration (full config, not just actor_rollout_ref)
        resource_pool: The resource pool to use
        num_gpus: Number of GPUs/workers to create
        use_async: Whether to use async rollout worker
        name_prefix: Prefix for naming (e.g., "cypher" or "orch")
        role: Role string passed to worker ("actor", "rollout", etc.)
        
    Returns:
        Tuple of (RayWorkerGroup, role_worker_mapping)
    """
    logger.info(f"Creating worker group '{name_prefix}' with {num_gpus} GPUs, role='{role}'...")
    
    # Determine worker class based on async mode
    if use_async:
        actor_rollout_cls = AsyncActorRolloutRefWorker
    else:
        actor_rollout_cls = ActorRolloutRefWorker
    
    # Create ray remote class with max_concurrency for async operations
    remote_actor_rollout_cls = ray.remote(max_concurrency=2048)(actor_rollout_cls)
    
    role_worker_mapping = {
        Role.ActorRollout: remote_actor_rollout_cls,
    }
    
    # Create RayClassWithInitArgs wrapper for RayWorkerGroup
    # The kwargs are passed to ActorRolloutRefWorker.__init__(config, role)
    actor_rollout_cls_with_init = RayClassWithInitArgs(
        cls=remote_actor_rollout_cls,
        config=config.actor_rollout_ref,
        role=role,
    )
    
    # Create worker group
    actor_rollout_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=actor_rollout_cls_with_init,
        num_workers=num_gpus,
    )
    
    logger.info(f"Worker group '{name_prefix}' created with {num_gpus} workers")
    return actor_rollout_wg, role_worker_mapping


def create_worker_groups(
    config: DictConfig,
    tokenizer,
    resource_pool_manager: ResourcePoolManager,
    mapping: Dict[Role, str],
    use_async: bool = False
) -> Tuple[Dict[str, RayWorkerGroup], Dict]:
    """
    Create Ray worker groups for distributed training (shared pool mode).
    
    Args:
        config: PPO configuration
        tokenizer: Tokenizer instance
        resource_pool_manager: Resource pool manager
        mapping: Role to pool mapping
        use_async: Whether to use async rollout worker
        
    Returns:
        Tuple of (worker_groups dict, role_worker_mapping)
    """
    logger.info("Creating Ray worker groups (shared pool mode)...")
    
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    logger.info(f"Got resource pool: {resource_pool}")
    
    actor_rollout_wg, role_worker_mapping = create_worker_group_for_pool(
        config=config,
        resource_pool=resource_pool,
        num_gpus=config.trainer.n_training_gpus_per_node,
        use_async=use_async,
        name_prefix="actor_rollout",
        role="actor"
    )
    
    worker_groups = {'actor_rollout': actor_rollout_wg}
    
    logger.info(f"Worker groups created: {list(worker_groups.keys())}")
    return worker_groups, role_worker_mapping


def create_dual_worker_groups(
    cypher_config: DictConfig,
    orch_config: DictConfig,
    dual_pool_manager: DualResourcePoolManager,
    cypher_async: bool = False,
    orch_async: bool = False
) -> Tuple[RayWorkerGroup, RayWorkerGroup, Dict, Dict]:
    """
    Create separate worker groups for Cypher Generator and Orchestrator.
    
    Args:
        cypher_config: PPO configuration for Cypher Generator
        orch_config: PPO configuration for Orchestrator
        dual_pool_manager: Dual resource pool manager
        cypher_async: Whether to use async for Cypher Generator
        orch_async: Whether to use async for Orchestrator
        
    Returns:
        Tuple of (cypher_worker_group, orch_worker_group, cypher_role_mapping, orch_role_mapping)
    """
    logger.info("Creating dual worker groups for separate models...")
    
    # Get pools
    cypher_pool = dual_pool_manager.get_cypher_pool()
    orch_pool = dual_pool_manager.get_orch_pool()
    
    # Get GPU counts from pool specs
    cypher_gpus = sum(dual_pool_manager.cypher_pool_spec[dual_pool_manager.cypher_pool_id])
    orch_gpus = sum(dual_pool_manager.orch_pool_spec[dual_pool_manager.orch_pool_id])
    
    # Create Cypher Generator worker group
    logger.info(f"Creating Cypher Generator worker group ({cypher_gpus} GPUs)...")
    cypher_wg, cypher_role_mapping = create_worker_group_for_pool(
        config=cypher_config,
        resource_pool=cypher_pool,
        num_gpus=cypher_gpus,
        use_async=cypher_async,
        name_prefix="cypher_actor_rollout",
        role="actor_rollout"  # Need rollout capability for generate_sequences
    )
    
    # Create Orchestrator worker group
    logger.info(f"Creating Orchestrator worker group ({orch_gpus} GPUs)...")
    orch_wg, orch_role_mapping = create_worker_group_for_pool(
        config=orch_config,
        resource_pool=orch_pool,
        num_gpus=orch_gpus,
        use_async=orch_async,
        name_prefix="orch_actor_rollout",
        role="actor_rollout"  # Need rollout capability for generate_sequences
    )
    
    logger.info(f"Dual worker groups created: Cypher={cypher_gpus} GPUs, Orch={orch_gpus} GPUs")
    return cypher_wg, orch_wg, cypher_role_mapping, orch_role_mapping


def create_execution_engine(
    agent_class: type,
    env_class: type,
    config: DictConfig,
    tokenizer,
    rollout_engine: RayWorkerGroup,
    agent_args: Optional[Dict[str, Any]] = None,
    env_args: Optional[Dict[str, Any]] = None,
    max_steps: int = 5,
    engine_name: str = "verl"
) -> AsyncAgentExecutionEngine:
    """
    Create agent execution engine for rollout collection.
    
    Args:
        agent_class: Agent class to use
        env_class: Environment class to use
        config: PPO configuration
        tokenizer: Tokenizer instance
        rollout_engine: Ray worker group for rollout
        agent_args: Arguments for agent initialization
        env_args: Arguments for environment initialization
        max_steps: Maximum steps per trajectory
        engine_name: Engine name ("verl" or "openai")
        
    Returns:
        AsyncAgentExecutionEngine instance
    """
    if agent_args is None:
        agent_args = {}
    if env_args is None:
        env_args = {}
    
    logger.info(f"Creating execution engine for {agent_class.__name__}")
    
    execution_engine = AsyncAgentExecutionEngine(
        rollout_engine=rollout_engine,
        config=config,
        engine_name=engine_name,
        tokenizer=tokenizer,
        model_path=config.actor_rollout_ref.model.path,
        max_steps=max_steps,
        max_response_length=config.data.max_response_length,
        max_prompt_length=config.data.max_prompt_length,
        agent_class=agent_class,
        agent_args=agent_args,
        env_class=env_class,
        env_args=env_args,
        enforce_max_prompt_length=config.agent.use_stepwise_advantage,
        trajectory_timeout=config.agent.trajectory_timeout,
        overlong_filter=config.agent.overlong_filter,
    )
    
    logger.info("Execution engine created successfully")
    return execution_engine


def initialize_worker_group(
    worker_group: RayWorkerGroup,
    config: DictConfig,
    tokenizer,
    role: Role
):
    """
    Initialize a worker group with model.
    
    Note: The config is passed via RayClassWithInitArgs during worker group creation,
    so init_model() takes no arguments.
    
    Args:
        worker_group: Worker group to initialize
        config: PPO configuration (for reference/logging only)
        tokenizer: Tokenizer instance (for reference/logging only)
        role: Role of the worker group
    """
    logger.info(f"Initializing worker group for role: {role}")
    logger.info(f"Model path: {config.actor_rollout_ref.model.path}")
    
    # Initialize workers - config was passed via RayClassWithInitArgs
    worker_group.init_model()
    
    # Set tensor parallel size for rollout
    worker_group.tp_size = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
    
    logger.info(f"Worker group initialized for role: {role}")


# =============================================================================
# Sequential Training Mode - Model Swapping Utilities
# =============================================================================

@dataclass
class SequentialResourceManager:
    """
    Resource manager for sequential training mode.
    
    In sequential mode:
    - During rollout: Both models loaded for inference (split GPUs)
    - During training: Only one model loaded (all GPUs)
    
    This maximizes GPU utilization during training phases.
    """
    total_gpus: int
    nnodes: int = 1
    rollout_cypher_gpus: int = 4
    rollout_orch_gpus: int = 4
    training_gpus: int = 8
    
    # Current state
    current_mode: str = field(default="uninitialized")  # "rollout", "cypher_train", "orch_train"
    
    # Resource pools (created on demand)
    _rollout_cypher_pool: RayResourcePool = field(default=None, repr=False)
    _rollout_orch_pool: RayResourcePool = field(default=None, repr=False)
    _training_pool: RayResourcePool = field(default=None, repr=False)
    
    def create_rollout_pools(self):
        """Create resource pools for rollout phase (split GPUs)."""
        logger.info(f"Creating rollout pools: Cypher={self.rollout_cypher_gpus}, Orch={self.rollout_orch_gpus}")
        
        self._rollout_cypher_pool = RayResourcePool(
            process_on_nodes=[self.rollout_cypher_gpus] * self.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="rollout_cypher"
        )
        
        self._rollout_orch_pool = RayResourcePool(
            process_on_nodes=[self.rollout_orch_gpus] * self.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="rollout_orch"
        )
        
        self.current_mode = "rollout"
        logger.info("Rollout pools created")
    
    def create_training_pool(self):
        """Create resource pool for training phase (all GPUs)."""
        logger.info(f"Creating training pool: {self.training_gpus} GPUs")
        
        self._training_pool = RayResourcePool(
            process_on_nodes=[self.training_gpus] * self.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="training"
        )
        
        logger.info("Training pool created")
    
    def get_rollout_cypher_pool(self) -> RayResourcePool:
        """Get Cypher pool for rollout phase."""
        if self._rollout_cypher_pool is None:
            self.create_rollout_pools()
        return self._rollout_cypher_pool
    
    def get_rollout_orch_pool(self) -> RayResourcePool:
        """Get Orchestrator pool for rollout phase."""
        if self._rollout_orch_pool is None:
            self.create_rollout_pools()
        return self._rollout_orch_pool
    
    def get_training_pool(self) -> RayResourcePool:
        """Get pool for training phase (all GPUs)."""
        if self._training_pool is None:
            self.create_training_pool()
        return self._training_pool


def create_sequential_resource_manager(gpu_config: Dict[str, Any]) -> SequentialResourceManager:
    """
    Create a sequential resource manager from config.
    
    Args:
        gpu_config: GPU allocation configuration with:
            - total_gpus: Total GPUs available
            - nnodes: Number of nodes
            - rollout.cypher_gpus: GPUs for Cypher during rollout
            - rollout.orch_gpus: GPUs for Orch during rollout
            - training.gpus: GPUs for training (should be total_gpus)
    
    Returns:
        SequentialResourceManager instance
    """
    total_gpus = gpu_config.get('total_gpus', 8)
    nnodes = gpu_config.get('nnodes', 1)
    
    rollout_config = gpu_config.get('rollout', {})
    training_config = gpu_config.get('training', {})
    
    manager = SequentialResourceManager(
        total_gpus=total_gpus,
        nnodes=nnodes,
        rollout_cypher_gpus=rollout_config.get('cypher_gpus', total_gpus // 2),
        rollout_orch_gpus=rollout_config.get('orch_gpus', total_gpus // 2),
        training_gpus=training_config.get('gpus', total_gpus),
    )
    
    logger.info(f"Created SequentialResourceManager: "
                f"rollout={manager.rollout_cypher_gpus}+{manager.rollout_orch_gpus}, "
                f"training={manager.training_gpus}")
    
    return manager


def should_update_orchestrator(
    epoch: int,
    schedule_config: Dict[str, Any],
    current_stage: str = "easy"
) -> bool:
    """
    Determine if Orchestrator should be updated this epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        schedule_config: Orchestrator schedule configuration with:
            - warmup_epochs: Number of warmup epochs (freeze Orch)
            - alternating_end_epoch: End of alternating phase
            - warmup_frequency: Update frequency during warmup (0 = never)
            - alternating_frequency: Update frequency during alternating phase
            - joint_frequency: Update frequency during joint phase
            - stage_overrides: Optional per-stage frequency overrides
        current_stage: Current curriculum stage ("easy", "medium", "hard")
    
    Returns:
        True if Orchestrator should be updated this epoch
    """
    warmup_epochs = schedule_config.get('warmup_epochs', 5)
    alternating_end = schedule_config.get('alternating_end_epoch', 20)
    
    warmup_freq = schedule_config.get('warmup_frequency', 0)
    alternating_freq = schedule_config.get('alternating_frequency', 3)
    joint_freq = schedule_config.get('joint_frequency', 2)
    
    # Check for stage-specific overrides
    stage_overrides = schedule_config.get('stage_overrides', {})
    if current_stage in stage_overrides:
        stage_config = stage_overrides[current_stage]
        joint_freq = stage_config.get('joint_frequency', joint_freq)
    
    # Determine current phase
    if epoch <= warmup_epochs:
        phase = "warmup"
        frequency = warmup_freq
    elif epoch <= alternating_end:
        phase = "alternating"
        frequency = alternating_freq
    else:
        phase = "joint"
        frequency = joint_freq
    
    # Check if should update
    if frequency == 0:
        should_update = False
    else:
        should_update = (epoch % frequency == 0)
    
    logger.debug(f"Epoch {epoch}: phase={phase}, stage={current_stage}, "
                 f"frequency={frequency}, should_update={should_update}")
    
    return should_update


def get_training_phase(epoch: int, schedule_config: Dict[str, Any]) -> str:
    """
    Get the current training phase based on epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        schedule_config: Orchestrator schedule configuration
    
    Returns:
        Phase name: "warmup", "alternating", or "joint"
    """
    warmup_epochs = schedule_config.get('warmup_epochs', 5)
    alternating_end = schedule_config.get('alternating_end_epoch', 20)
    
    if epoch <= warmup_epochs:
        return "warmup"
    elif epoch <= alternating_end:
        return "alternating"
    else:
        return "joint"


@dataclass
class EMAOrchestrator:
    """
    Exponential Moving Average version of Orchestrator for stable evaluation.
    
    The EMA Orchestrator is used for:
    - Role 2: Data Quality Evaluation
    - Role 4: Answer Quality Evaluation
    
    It's updated slowly after each Orchestrator training step to provide
    stable evaluation signals and prevent circular dependency issues.
    """
    decay: float = 0.99
    decay_on_drift: float = 0.995
    enabled: bool = True
    
    # State
    initialized: bool = field(default=False)
    update_count: int = field(default=0)
    
    def update_from_trained(self, trained_state_dict: Dict, ema_state_dict: Dict) -> Dict:
        """
        Update EMA weights from trained model weights.
        
        Formula: ema = decay * ema + (1 - decay) * trained
        
        Args:
            trained_state_dict: State dict of trained Orchestrator
            ema_state_dict: Current EMA state dict
        
        Returns:
            Updated EMA state dict
        """
        if not self.enabled:
            return trained_state_dict
        
        if not self.initialized:
            # First update: just copy
            self.initialized = True
            self.update_count = 1
            logger.info("EMA Orchestrator initialized from trained model")
            return trained_state_dict.copy()
        
        # EMA update
        updated_state = {}
        for key in trained_state_dict:
            if key in ema_state_dict:
                updated_state[key] = (
                    self.decay * ema_state_dict[key] + 
                    (1 - self.decay) * trained_state_dict[key]
                )
            else:
                updated_state[key] = trained_state_dict[key]
        
        self.update_count += 1
        logger.info(f"EMA Orchestrator updated (count={self.update_count}, decay={self.decay})")
        
        return updated_state
    
    def increase_decay_on_drift(self):
        """Increase decay rate when drift is detected."""
        old_decay = self.decay
        self.decay = self.decay_on_drift
        logger.warning(f"EMA decay increased due to drift: {old_decay} -> {self.decay}")


def create_ema_orchestrator(ema_config: Dict[str, Any]) -> EMAOrchestrator:
    """
    Create EMA Orchestrator from config.
    
    Args:
        ema_config: EMA configuration with:
            - enabled: Whether EMA is enabled
            - decay: EMA decay rate
            - decay_on_drift: Decay rate to use when drift detected
    
    Returns:
        EMAOrchestrator instance
    """
    return EMAOrchestrator(
        decay=ema_config.get('decay', 0.99),
        decay_on_drift=ema_config.get('decay_on_drift', 0.995),
        enabled=ema_config.get('enabled', True),
    )

