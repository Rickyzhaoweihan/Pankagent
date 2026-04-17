"""
Training utilities for PanKLLM RL post-training.

Exports utilities for:
- Training stability (EMA, reward normalization)
- Curriculum learning
- Checkpoint management
- Validation
"""

from .training_stability import (
    RunningStats,
    update_ema_model,
    should_update_orchestrator,
    get_training_phase,
    detect_reward_drift,
    adjust_ema_decay,
    compute_evaluation_consistency,
    check_score_inflation
)
from .curriculum_utils import (
    CurriculumTracker,
    check_curriculum_progression,
    advance_stage,
    regress_stage,
    get_stage_config,
    compute_success_rate,
    CURRICULUM_STAGES,
    STAGE_ORDER
)
from .checkpoint_manager import CheckpointManager
from .validation import (
    load_validation_set,
    validate_on_fixed_set,
    compute_validation_metrics,
    compare_train_val_metrics,
    save_validation_results,
    load_validation_results
)
from .rllm_components import (
    # Tokenizer
    load_tokenizer,
    
    # Single pool mode
    create_resource_pool_manager,
    create_worker_groups,
    create_execution_engine,
    initialize_worker_group,
    
    # Dual pool mode (parallel multi-model)
    create_dual_resource_pool_manager,
    create_resource_pools_from_config,
    create_dual_worker_groups,
    create_worker_group_for_pool,
    DualResourcePoolManager,
    MultiModelRole,
    
    # Sequential training mode (Strategy 2)
    SequentialResourceManager,
    create_sequential_resource_manager,
    should_update_orchestrator,
    get_training_phase,
    EMAOrchestrator,
    create_ema_orchestrator,
)

__all__ = [
    # Training stability
    'RunningStats',
    'update_ema_model',
    'detect_reward_drift',
    'adjust_ema_decay',
    'compute_evaluation_consistency',
    'check_score_inflation',
    
    # Curriculum learning
    'CurriculumTracker',
    'check_curriculum_progression',
    'advance_stage',
    'regress_stage',
    'get_stage_config',
    'compute_success_rate',
    'CURRICULUM_STAGES',
    'STAGE_ORDER',
    
    # Checkpoint management
    'CheckpointManager',
    
    # Validation
    'load_validation_set',
    'validate_on_fixed_set',
    'compute_validation_metrics',
    'compare_train_val_metrics',
    'save_validation_results',
    'load_validation_results',
    
    # rllm components - single pool mode
    'load_tokenizer',
    'create_resource_pool_manager',
    'create_worker_groups',
    'create_execution_engine',
    'initialize_worker_group',
    
    # rllm components - dual pool mode (multi-model)
    'create_dual_resource_pool_manager',
    'create_resource_pools_from_config',
    'create_dual_worker_groups',
    'create_worker_group_for_pool',
    'DualResourcePoolManager',
    'MultiModelRole',
    
    # rllm components - sequential training mode (Strategy 2)
    'SequentialResourceManager',
    'create_sequential_resource_manager',
    'should_update_orchestrator',
    'get_training_phase',
    'EMAOrchestrator',
    'create_ema_orchestrator',
]

