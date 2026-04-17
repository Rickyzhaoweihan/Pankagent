"""
Reward function for Cypher Generator Agent.

Implements multi-component reward with 6 main components:
1. Answer Quality (25%) - From Orchestrator evaluation
2. Data Quality (25%) - From Orchestrator evaluation
3. Trajectory Quality (20%) - From Orchestrator evaluation
4. Cypher Correctness (15%) - Syntax and execution success
5. Retrieval Efficiency (10%) - Data per step
6. Execution Time (10%) - Query speed

Plus penalties for poor stopping decisions and low data quality.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rllm.rewards.reward_types import RewardOutput

# Add parent directory to path for imports when loaded as standalone file
_reward_dir = Path(__file__).parent
if str(_reward_dir) not in sys.path:
    sys.path.insert(0, str(_reward_dir))

from reward_utils import validate_cypher

logger = logging.getLogger(__name__)


def cypher_generator_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Compute reward for Cypher Generator Agent.
    
    Args:
        task_info: Dictionary containing:
            - question: str
            - cypher_trajectory: list[dict] with query, result, success, execution_time_ms, num_results, has_data
            - answer_quality_score: float (0-1) from Orchestrator
            - data_quality_score: float (0-1) from Orchestrator
            - trajectory_quality_score: float (0-1) from Orchestrator
            - doubt_level: float (0-1) from Orchestrator
            - num_steps: int
        action: Final action (not used, info in task_info)
        
    Returns:
        RewardOutput with reward value, metadata, and is_correct flag
    """
    # Extract components from task_info
    answer_quality = task_info.get('answer_quality_score', 0.5)
    data_quality = task_info.get('data_quality_score', 0.5)
    trajectory_quality = task_info.get('trajectory_quality_score', 0.5)
    doubt_level = task_info.get('doubt_level', 0.0)
    cypher_trajectory = task_info.get('cypher_trajectory', [])
    num_steps = len(cypher_trajectory)
    
    logger.debug(
        f"Computing Cypher reward: answer_quality={answer_quality:.3f}, "
        f"data_quality={data_quality:.3f}, trajectory_quality={trajectory_quality:.3f}, "
        f"num_steps={num_steps}"
    )
    
    # Component 1: Answer Quality (25%)
    answer_quality_component = answer_quality
    
    # Component 2: Data Quality (25%)
    data_quality_component = data_quality
    
    # Component 3: Trajectory Quality (20%)
    trajectory_quality_component = trajectory_quality
    
    # Component 4: Cypher Correctness (15%)
    cypher_correctness = _compute_cypher_correctness(cypher_trajectory)
    
    # Component 5: Retrieval Efficiency (10%)
    efficiency_score = _compute_efficiency(cypher_trajectory, num_steps)
    
    # Component 6: Execution Time (10%)
    time_score, avg_time = _compute_time_score(cypher_trajectory, num_steps)
    
    # Compute penalties
    step_penalty = _compute_step_penalty(num_steps)
    stopping_penalty = _compute_stopping_penalty(cypher_trajectory, num_steps)
    data_quality_penalty = _compute_data_quality_penalty(data_quality)
    semantic_ambiguity_penalty = _compute_semantic_ambiguity_penalty(doubt_level)
    
    # Combine components with weights
    reward = (
        0.25 * answer_quality_component +
        0.25 * data_quality_component +
        0.20 * trajectory_quality_component +
        0.15 * cypher_correctness +
        0.10 * efficiency_score +
        0.10 * time_score -
        step_penalty -
        stopping_penalty -
        data_quality_penalty -
        semantic_ambiguity_penalty
    )
    
    # Clamp reward to [0.0, 1.0]
    reward = max(0.0, min(1.0, reward))
    
    # Determine if answer is correct
    is_correct = (answer_quality > 0.7 and data_quality > 0.6)
    
    # Build metadata
    metadata = {
        'answer_quality': answer_quality,
        'data_quality': data_quality,
        'trajectory_quality': trajectory_quality,
        'cypher_correctness': cypher_correctness,
        'efficiency': efficiency_score,
        'time_score': time_score,
        'avg_execution_time_ms': avg_time,
        'num_steps': num_steps,
        'step_penalty': step_penalty,
        'stopping_penalty': stopping_penalty,
        'data_quality_penalty': data_quality_penalty,
        'semantic_ambiguity_penalty': semantic_ambiguity_penalty,
        'doubt_level': doubt_level
    }
    
    logger.info(
        f"Cypher reward: {reward:.3f} (answer={answer_quality:.2f}, "
        f"data={data_quality:.2f}, traj={trajectory_quality:.2f}, "
        f"cypher={cypher_correctness:.2f}, eff={efficiency_score:.2f}, "
        f"time={time_score:.2f}, penalties={step_penalty+stopping_penalty+data_quality_penalty+semantic_ambiguity_penalty:.3f})"
    )
    
    return RewardOutput(
        reward=reward,
        metadata=metadata,
        is_correct=is_correct
    )


def _compute_cypher_correctness(trajectory: List[Dict[str, Any]]) -> float:
    """
    Compute Cypher correctness score based on syntax and execution.
    
    Args:
        trajectory: List of trajectory steps
        
    Returns:
        Average correctness score (0-1)
    """
    if not trajectory:
        return 0.0
    
    scores = []
    for step in trajectory:
        query = step.get('query', '')
        success = step.get('success', False)
        has_data = step.get('has_data', False)
        
        if not success:
            # Query failed to execute
            scores.append(0.0)
        elif not has_data:
            # Query executed but returned no data
            scores.append(0.3)
        else:
            # Query executed successfully with data
            # Validate syntax
            validation = validate_cypher(query)
            syntax_score = validation['score'] / 100.0
            
            # Combine syntax (60%) and execution success (40%)
            step_score = 0.6 * syntax_score + 0.4 * 1.0
            scores.append(step_score)
    
    avg_score = np.mean(scores) if scores else 0.0
    return float(avg_score)


def _compute_efficiency(trajectory: List[Dict[str, Any]], num_steps: int) -> float:
    """
    Compute retrieval efficiency (data per step).
    
    Args:
        trajectory: List of trajectory steps
        num_steps: Number of steps taken
        
    Returns:
        Efficiency score (0-1)
    """
    if num_steps == 0:
        return 0.0
    
    total_results = sum(step.get('num_results', 0) for step in trajectory)
    
    # Efficiency: more data with fewer steps is better
    # Target: 10 results per step
    efficiency = min(1.0, total_results / (num_steps * 10))
    
    return float(efficiency)


def _compute_time_score(trajectory: List[Dict[str, Any]], num_steps: int) -> tuple[float, float]:
    """
    Compute execution time score.
    
    Rewards faster queries using exponential decay.
    
    Args:
        trajectory: List of trajectory steps
        num_steps: Number of steps taken
        
    Returns:
        Tuple of (time_score, avg_execution_time_ms)
    """
    if num_steps == 0:
        return 1.0, 0.0
    
    total_time = sum(step.get('execution_time_ms', 0) for step in trajectory)
    avg_time = total_time / num_steps
    
    # Exponential decay: < 100ms: 1.0, 500ms: 0.6, 1000ms: 0.4, > 2000ms: 0.2
    time_score = np.exp(-avg_time / 1000.0)
    
    # Clamp to [0.2, 1.0]
    time_score = max(0.2, min(1.0, time_score))
    
    return float(time_score), float(avg_time)


def _compute_step_penalty(num_steps: int) -> float:
    """
    Compute penalty for taking too many steps.
    
    Args:
        num_steps: Number of steps taken
        
    Returns:
        Penalty value (0 or positive)
    """
    # Penalize steps beyond 3
    penalty = max(0.0, (num_steps - 3) * 0.05)
    return float(penalty)


def _compute_stopping_penalty(trajectory: List[Dict[str, Any]], num_steps: int) -> float:
    """
    Compute penalty for poor stopping decisions.
    
    Args:
        trajectory: List of trajectory steps
        num_steps: Number of steps taken
        
    Returns:
        Penalty value (0 or positive)
    """
    if num_steps == 0:
        return 0.2  # No steps taken is bad
    
    total_results = sum(step.get('num_results', 0) for step in trajectory)
    
    # Early stopping: < 2 steps and < 5 results
    if num_steps < 2 and total_results < 5:
        return 0.2
    
    # Late stopping: 5 steps and last query had low score
    if num_steps >= 5:
        last_step = trajectory[-1]
        if not last_step.get('has_data', False):
            return 0.1  # Should have stopped earlier
    
    return 0.0


def _compute_data_quality_penalty(data_quality: float) -> float:
    """
    Compute penalty for retrieving low-quality data.
    
    Args:
        data_quality: Data quality score from Orchestrator
        
    Returns:
        Penalty value (0 or positive)
    """
    if data_quality < 0.4:
        return 0.1
    return 0.0


def _compute_semantic_ambiguity_penalty(doubt_level: float) -> float:
    """
    Compute penalty for using semantically ambiguous edges.
    
    Args:
        doubt_level: Doubt level from Orchestrator
        
    Returns:
        Penalty value (0 or positive)
    """
    if doubt_level > 0.6:
        # Penalize proportional to doubt level
        penalty = 0.05 * doubt_level
        return float(penalty)
    return 0.0

