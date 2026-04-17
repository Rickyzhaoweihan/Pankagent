"""
Reward functions for Orchestrator Agent.

Implements two reward functions for the Orchestrator's trainable roles:
1. Question Generation Reward - For generating training questions
2. Answer Synthesis Reward - For synthesizing answers from retrieved data
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from rllm.rewards.reward_types import RewardOutput

# Add parent directory to path for imports when loaded as standalone file
_reward_dir = Path(__file__).parent
if str(_reward_dir) not in sys.path:
    sys.path.insert(0, str(_reward_dir))

from reward_utils import compute_diversity_score, compute_data_utilization

logger = logging.getLogger(__name__)


def orchestrator_generation_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Compute reward for Orchestrator's question generation role.
    
    REDESIGNED: The reward is now STRONGLY COUPLED to Cypher success.
    
    Key principle: A question is only valuable if it can be answered!
    - If Cypher fails (reward=0), Orchestrator gets near-zero reward
    - Diversity/scope only provide small bonuses on top of answerability
    
    Args:
        task_info: Dictionary containing:
            - question: str (generated question)
            - answerability: bool (did Cypher Generator succeed?)
            - cypher_reward: float (actual Cypher reward, 0-1)
            - data_richness: float (how much data was retrieved, 0-1)
            - success_rate: float (recent success rate)
            - target_success_rate: float (target for current stage)
            - recent_questions: list[str] (for diversity check)
            - scope_constraints: dict (allowed types)
            - question_used_types: list[str] (types used in question)
        action: Generated question (same as task_info['question'])
        
    Returns:
        RewardOutput with reward value and metadata
    """
    question = task_info.get('question', action)
    answerability = task_info.get('answerability', False)
    cypher_reward = task_info.get('cypher_reward', 0.0)  # Direct Cypher reward signal
    data_richness = task_info.get('data_richness', 0.0)  # How much data was retrieved
    success_rate = task_info.get('success_rate', 0.5)
    target_success_rate = task_info.get('target_success_rate', 0.6)
    recent_questions = task_info.get('recent_questions', [])
    scope_constraints = task_info.get('scope_constraints', {})
    question_used_types = task_info.get('question_used_types', [])
    
    logger.debug(
        f"Computing Orchestrator generation reward: answerability={answerability}, "
        f"cypher_reward={cypher_reward:.3f}, success_rate={success_rate:.3f}"
    )
    
    # ==========================================================================
    # NEW REWARD DESIGN: Answerability is the PRIMARY signal
    # ==========================================================================
    
    # Component 1: Cypher Success (PRIMARY - 70%)
    # Directly use the Cypher reward as the main signal
    # If Cypher fails, this is 0. If Cypher succeeds well, this is high.
    cypher_success_score = cypher_reward
    
    # Component 2: Data Richness Bonus (15%)
    # Reward questions that lead to rich data retrieval
    # Normalized: 0 results = 0, 10+ results = 1.0
    data_richness_score = min(1.0, data_richness)
    
    # Component 3: Diversity Bonus (10%)
    # Small bonus for asking different questions (only matters if answerable)
    diversity_score = compute_diversity_score(question, recent_questions)
    
    # Component 4: Scope Adherence (5%)
    # Minor bonus for staying within allowed types
    scope_score = _compute_scope_adherence(question_used_types, scope_constraints)
    
    # ==========================================================================
    # CRITICAL: Gate reward by answerability
    # ==========================================================================
    if answerability and cypher_reward > 0:
        # Question was answerable - full reward calculation
        reward = (
            0.70 * cypher_success_score +
            0.15 * data_richness_score +
            0.10 * diversity_score +
            0.05 * scope_score
        )
    else:
        # Question was NOT answerable - heavy penalty
        # Give tiny reward for diversity to encourage exploration, but mostly 0
        reward = 0.05 * diversity_score
    
    # Clamp reward to [0.0, 1.0]
    reward = max(0.0, min(1.0, reward))
    
    # Build metadata
    metadata = {
        'cypher_success': cypher_success_score,
        'data_richness': data_richness_score,
        'diversity': diversity_score,
        'scope': scope_score,
        'answerability': 1.0 if answerability else 0.0,
        'cypher_reward': cypher_reward,
        'success_rate': success_rate,
        'target_success_rate': target_success_rate,
        'reward_gated': not (answerability and cypher_reward > 0),
    }
    
    logger.info(
        f"Orchestrator generation reward: {reward:.3f} "
        f"(cypher={cypher_success_score:.2f}, data={data_richness_score:.2f}, "
        f"diversity={diversity_score:.2f}, gated={not answerability})"
    )
    
    return RewardOutput(
        reward=reward,
        metadata=metadata,
        is_correct=answerability
    )


def orchestrator_synthesis_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Compute reward for Orchestrator's answer synthesis role.
    
    Rewards answers that are:
    - High quality (from self-evaluation)
    - Utilize retrieved data effectively
    
    Args:
        task_info: Dictionary containing:
            - question: str
            - answer: str (synthesized answer)
            - answer_quality_score: float (from self-evaluation Role 4)
            - trajectory: list[dict] (retrieved data)
            - data_utilization: float (optional, computed if not provided)
        action: Synthesized answer (same as task_info['answer'])
        
    Returns:
        RewardOutput with reward value and metadata
    """
    answer = task_info.get('answer', action)
    answer_quality = task_info.get('answer_quality_score', 0.5)
    trajectory = task_info.get('trajectory', [])
    
    # Compute data utilization if not provided
    if 'data_utilization' in task_info:
        data_utilization = task_info['data_utilization']
    else:
        data_utilization = compute_data_utilization(answer, trajectory)
    
    logger.debug(
        f"Computing Orchestrator synthesis reward: answer_quality={answer_quality:.3f}, "
        f"data_utilization={data_utilization:.3f}"
    )
    
    # Component 1: Answer Quality (70%)
    # From self-evaluation (Role 4)
    answer_quality_component = answer_quality
    
    # Component 2: Data Utilization (30%)
    # How well the answer uses retrieved data
    data_utilization_component = data_utilization
    
    # Combine components with weights
    reward = (
        0.70 * answer_quality_component +
        0.30 * data_utilization_component
    )
    
    # Clamp reward to [0.0, 1.0]
    reward = max(0.0, min(1.0, reward))
    
    # Build metadata
    metadata = {
        'answer_quality': answer_quality,
        'data_utilization': data_utilization
    }
    
    logger.info(
        f"Orchestrator synthesis reward: {reward:.3f} "
        f"(answer_quality={answer_quality:.2f}, data_utilization={data_utilization:.2f})"
    )
    
    return RewardOutput(
        reward=reward,
        metadata=metadata,
        is_correct=(answer_quality > 0.7)
    )


def _compute_difficulty_score(success_rate: float, target_success_rate: float) -> float:
    """
    Compute difficulty appropriateness score.
    
    Rewards questions that lead to success rate near the target.
    
    Args:
        success_rate: Recent success rate
        target_success_rate: Target success rate for current stage
        
    Returns:
        Difficulty score (0-1)
    """
    if target_success_rate <= 0:
        return 0.5  # Neutral if no target
    
    # Compute normalized distance from target
    distance = abs(success_rate - target_success_rate) / target_success_rate
    
    # Score decreases with distance
    # Perfect match: 1.0, 50% off target: 0.5, 100% off: 0.0
    score = max(0.0, 1.0 - distance)
    
    return float(score)


def _compute_scope_adherence(used_types: List[str], scope_constraints: Dict[str, Any]) -> float:
    """
    Compute scope adherence score.
    
    Checks if all types used in the question are within allowed types.
    
    Args:
        used_types: Types used in the generated question
        scope_constraints: Dictionary with 'allowed_node_types' and 'allowed_edge_types'
        
    Returns:
        Scope score (0 or 1)
    """
    if not used_types:
        return 1.0  # No types used, no violation
    
    allowed_node_types = scope_constraints.get('allowed_node_types', [])
    allowed_edge_types = scope_constraints.get('allowed_edge_types', [])
    
    # If no constraints specified, everything is allowed
    if not allowed_node_types and not allowed_edge_types:
        return 1.0
    
    # Combine all allowed types
    allowed_types = set(allowed_node_types + allowed_edge_types)
    
    # Check if all used types are allowed
    for used_type in used_types:
        if used_type not in allowed_types:
            return 0.0  # Violation found
    
    return 1.0  # All types are allowed

