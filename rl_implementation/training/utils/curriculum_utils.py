"""
Curriculum learning utilities for progressive difficulty training.

Provides:
- Stage definitions (Easy, Medium, Hard)
- Progression logic (advance/regress based on success rate)
- Success rate computation
- Stage configuration management
"""

import logging
from collections import deque
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# Curriculum stage definitions (from IMPLEMENTATION_PLAN.md lines 1015-1019)
CURRICULUM_STAGES = {
    'easy': {
        'max_hops': 2,
        'max_turns': 3,
        'target_success': 0.7,
        'trajectory_length': (1, 2),
        'subgraph_size': (10, 50)
    },
    'medium': {
        'max_hops': 3,
        'max_turns': 4,
        'target_success': 0.5,
        'trajectory_length': (2, 3),
        'subgraph_size': (50, 200)
    },
    'hard': {
        'max_hops': 5,
        'max_turns': 5,
        'target_success': 0.4,
        'trajectory_length': (3, 5),
        'subgraph_size': (200, 1000)
    }
}

# Stage progression order
STAGE_ORDER = ['easy', 'medium', 'hard']


def get_stage_config(stage_name: str) -> Dict[str, Any]:
    """
    Get configuration for a curriculum stage.
    
    Args:
        stage_name: Name of the stage ('easy', 'medium', 'hard')
        
    Returns:
        Dictionary with stage configuration including 'stage' key
        
    Raises:
        ValueError: If stage_name is invalid
    """
    if stage_name not in CURRICULUM_STAGES:
        raise ValueError(
            f"Invalid stage name: {stage_name}. "
            f"Must be one of {list(CURRICULUM_STAGES.keys())}"
        )
    
    config = CURRICULUM_STAGES[stage_name].copy()
    config['stage'] = stage_name  # Include the stage name in the config
    return config


def compute_success_rate(trajectories: List[Dict[str, Any]], threshold: float = 0.7) -> float:
    """
    Compute success rate from trajectories.
    
    Success is defined as answer_quality > threshold.
    
    Args:
        trajectories: List of trajectory dictionaries with 'answer_quality' key
        threshold: Minimum answer quality to consider successful
        
    Returns:
        Success rate (0.0 to 1.0)
    """
    if not trajectories:
        return 0.0
    
    successes = sum(
        1 for traj in trajectories
        if traj.get('answer_quality', {}).get('score', 0.0) > threshold
    )
    
    success_rate = successes / len(trajectories)
    
    logger.debug(
        f"Success rate: {success_rate:.3f} "
        f"({successes}/{len(trajectories)} above {threshold})"
    )
    
    return success_rate


def check_curriculum_progression(
    success_rates: List[float],
    current_stage: str,
    window: int = 100
) -> Tuple[bool, bool]:
    """
    Check if curriculum should advance or regress.
    
    Progression rules:
    - Advance: success_rate > target + 0.1 for last 100 episodes
    - Regress: success_rate < target - 0.2 for last 100 episodes
    
    Args:
        success_rates: Recent success rates (one per episode)
        current_stage: Current curriculum stage
        window: Number of recent episodes to consider
        
    Returns:
        Tuple of (should_advance, should_regress)
    """
    if not success_rates:
        return False, False
    
    # Get current stage config
    stage_config = get_stage_config(current_stage)
    target = stage_config['target_success']
    
    # Use last 'window' episodes
    recent_rates = success_rates[-window:] if len(success_rates) > window else success_rates
    
    if len(recent_rates) < window:
        logger.debug(
            f"Not enough episodes for progression check "
            f"({len(recent_rates)}/{window})"
        )
        return False, False
    
    # Compute average success rate
    avg_success_rate = sum(recent_rates) / len(recent_rates)
    
    # Check for advancement
    should_advance = False
    if current_stage != STAGE_ORDER[-1]:  # Not already at hardest stage
        if avg_success_rate > target + 0.1:
            should_advance = True
            logger.info(
                f"Curriculum advancement triggered: "
                f"success_rate={avg_success_rate:.3f} > target+0.1={target+0.1:.3f}"
            )
    
    # Check for regression
    should_regress = False
    if current_stage != STAGE_ORDER[0]:  # Not already at easiest stage
        if avg_success_rate < target - 0.2:
            should_regress = True
            logger.warning(
                f"Curriculum regression triggered: "
                f"success_rate={avg_success_rate:.3f} < target-0.2={target-0.2:.3f}"
            )
    
    return should_advance, should_regress


def advance_stage(current_stage: str) -> str:
    """
    Move to next difficulty stage.
    
    Args:
        current_stage: Current curriculum stage
        
    Returns:
        Next stage name
    """
    if current_stage not in STAGE_ORDER:
        raise ValueError(f"Invalid current stage: {current_stage}")
    
    current_idx = STAGE_ORDER.index(current_stage)
    
    if current_idx == len(STAGE_ORDER) - 1:
        logger.warning(f"Already at hardest stage ({current_stage}), cannot advance")
        return current_stage
    
    next_stage = STAGE_ORDER[current_idx + 1]
    logger.info(f"Advancing curriculum: {current_stage} -> {next_stage}")
    
    return next_stage


def regress_stage(current_stage: str) -> str:
    """
    Move to previous difficulty stage.
    
    Args:
        current_stage: Current curriculum stage
        
    Returns:
        Previous stage name
    """
    if current_stage not in STAGE_ORDER:
        raise ValueError(f"Invalid current stage: {current_stage}")
    
    current_idx = STAGE_ORDER.index(current_stage)
    
    if current_idx == 0:
        logger.warning(f"Already at easiest stage ({current_stage}), cannot regress")
        return current_stage
    
    prev_stage = STAGE_ORDER[current_idx - 1]
    logger.warning(f"Regressing curriculum: {current_stage} -> {prev_stage}")
    
    return prev_stage


class CurriculumTracker:
    """
    Track curriculum progression over training.
    
    Maintains a history of success rates and automatically determines
    when to advance or regress stages.
    """
    
    def __init__(self, initial_stage: str = 'easy', window_size: int = 100):
        """
        Initialize CurriculumTracker.
        
        Args:
            initial_stage: Starting curriculum stage
            window_size: Number of episodes to track for progression decisions
        """
        self.current_stage = initial_stage
        self.window_size = window_size
        self.success_rates = deque(maxlen=window_size * 3)  # Keep extra history
        self.stage_history = [(0, initial_stage)]  # (epoch, stage) tuples
        
        logger.info(
            f"CurriculumTracker initialized: stage={initial_stage}, "
            f"window={window_size}"
        )
    
    def update(self, success_rate: float, epoch: int) -> Tuple[bool, str]:
        """
        Update tracker with new success rate and check for progression.
        
        Args:
            success_rate: Success rate for current epoch
            epoch: Current epoch number
            
        Returns:
            Tuple of (stage_changed, new_stage)
        """
        self.success_rates.append(success_rate)
        
        # Check for progression
        should_advance, should_regress = check_curriculum_progression(
            list(self.success_rates),
            self.current_stage,
            self.window_size
        )
        
        old_stage = self.current_stage
        
        if should_advance:
            self.current_stage = advance_stage(self.current_stage)
        elif should_regress:
            self.current_stage = regress_stage(self.current_stage)
        
        # Record stage change
        if self.current_stage != old_stage:
            self.stage_history.append((epoch, self.current_stage))
            logger.info(
                f"Curriculum stage changed at epoch {epoch}: "
                f"{old_stage} -> {self.current_stage}"
            )
            return True, self.current_stage
        
        return False, self.current_stage
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get configuration for current stage.
        
        Returns:
            Stage configuration dictionary
        """
        return get_stage_config(self.current_stage)
    
    def get_stage_history(self) -> List[Tuple[int, str]]:
        """
        Get history of stage changes.
        
        Returns:
            List of (epoch, stage) tuples
        """
        return list(self.stage_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get curriculum statistics.
        
        Returns:
            Dictionary with current stage, success rate, and history
        """
        recent_rates = list(self.success_rates)[-self.window_size:]
        avg_success_rate = sum(recent_rates) / len(recent_rates) if recent_rates else 0.0
        
        return {
            'current_stage': self.current_stage,
            'avg_success_rate': avg_success_rate,
            'n_episodes': len(self.success_rates),
            'stage_changes': len(self.stage_history) - 1,
            'stage_history': self.stage_history
        }

