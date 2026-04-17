"""
Training module for PanKLLM RL post-training.

Exports:
- Main training orchestration
- Training stability utilities
- Curriculum learning utilities
- Checkpoint management
- Validation utilities
"""

from .train_collaborative_system import CollaborativeTrainer
from .utils.training_stability import (
    RunningStats,
    update_ema_model,
    should_update_orchestrator,
    get_training_phase,
    detect_reward_drift,
    adjust_ema_decay
)
from .utils.curriculum_utils import (
    CurriculumTracker,
    check_curriculum_progression,
    advance_stage,
    regress_stage,
    get_stage_config,
    compute_success_rate
)
from .utils.checkpoint_manager import CheckpointManager
from .utils.validation import (
    validate_on_fixed_set,
    load_validation_set,
    compute_validation_metrics,
    compare_train_val_metrics
)

__all__ = [
    # Main trainer
    'CollaborativeTrainer',
    
    # Training stability
    'RunningStats',
    'update_ema_model',
    'should_update_orchestrator',
    'get_training_phase',
    'detect_reward_drift',
    'adjust_ema_decay',
    
    # Curriculum learning
    'CurriculumTracker',
    'check_curriculum_progression',
    'advance_stage',
    'regress_stage',
    'get_stage_config',
    'compute_success_rate',
    
    # Checkpoint management
    'CheckpointManager',
    
    # Validation
    'validate_on_fixed_set',
    'load_validation_set',
    'compute_validation_metrics',
    'compare_train_val_metrics'
]

