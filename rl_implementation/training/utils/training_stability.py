"""
Training stability utilities for circular dependency mitigation.

Provides:
- RunningStats: Reward normalization with running statistics
- EMA model updates: Exponential moving average for stable evaluation
- Training phase logic: Warmup, alternating, joint update phases
- Drift detection: Monitor reward inflation and evaluation consistency
"""

import logging
from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RunningStats:
    """
    Track running statistics for reward normalization.
    
    Maintains separate statistics per agent to prevent cross-agent interference.
    Uses a sliding window to adapt to changing reward distributions.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize RunningStats.
        
        Args:
            window_size: Number of recent episodes to track
        """
        self.window_size = window_size
        self.stats: Dict[str, Dict[str, Any]] = {}
        logger.info(f"RunningStats initialized with window_size={window_size}")
    
    def update(self, rewards: List[float], agent_name: str):
        """
        Update running statistics for an agent.
        
        Args:
            rewards: List of reward values from recent episodes
            agent_name: Name of the agent (e.g., 'cypher', 'orch_gen', 'orch_synth')
        """
        if agent_name not in self.stats:
            self.stats[agent_name] = {
                'rewards': deque(maxlen=self.window_size),
                'mean': 0.0,
                'std': 1.0
            }
        
        # Add new rewards to window
        self.stats[agent_name]['rewards'].extend(rewards)
        
        # Compute statistics
        if len(self.stats[agent_name]['rewards']) > 0:
            rewards_array = np.array(list(self.stats[agent_name]['rewards']))
            self.stats[agent_name]['mean'] = float(np.mean(rewards_array))
            self.stats[agent_name]['std'] = float(np.std(rewards_array) + 1e-8)
        
        logger.debug(
            f"Updated stats for {agent_name}: "
            f"mean={self.stats[agent_name]['mean']:.3f}, "
            f"std={self.stats[agent_name]['std']:.3f}, "
            f"n={len(self.stats[agent_name]['rewards'])}"
        )
    
    def normalize(self, rewards: List[float], agent_name: str) -> List[float]:
        """
        Normalize rewards using running statistics.
        
        Args:
            rewards: Raw reward values
            agent_name: Name of the agent
            
        Returns:
            Normalized and clipped rewards
        """
        if agent_name not in self.stats:
            logger.warning(f"No stats for {agent_name}, returning raw rewards")
            return rewards
        
        mean = self.stats[agent_name]['mean']
        std = self.stats[agent_name]['std']
        
        # Normalize
        normalized = [(r - mean) / std for r in rewards]
        
        # Clip to prevent extreme values
        clipped = [float(np.clip(r, -10.0, 10.0)) for r in normalized]
        
        logger.debug(
            f"Normalized {len(rewards)} rewards for {agent_name}: "
            f"mean={mean:.3f}, std={std:.3f}"
        )
        
        return clipped
    
    def get_stats(self, agent_name: str) -> Dict[str, float]:
        """
        Get current statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary with 'mean', 'std', and 'n_samples'
        """
        if agent_name not in self.stats:
            return {'mean': 0.0, 'std': 1.0, 'n_samples': 0}
        
        return {
            'mean': self.stats[agent_name]['mean'],
            'std': self.stats[agent_name]['std'],
            'n_samples': len(self.stats[agent_name]['rewards'])
        }


def update_ema_model(ema_model: torch.nn.Module, train_model: torch.nn.Module, decay: float = 0.99):
    """
    Update EMA (Exponential Moving Average) model parameters.
    
    This creates a slow-moving version of the model for stable evaluation.
    Similar to target networks in TD3/SAC algorithms.
    
    Formula: ema_param = decay * ema_param + (1 - decay) * train_param
    
    Args:
        ema_model: EMA model to update (evaluation version)
        train_model: Trained model (trainable version)
        decay: EMA decay factor (0.99 = slow updates, 0.9 = fast updates)
        
    Note:
        In stub implementation (agents without loaded models), this is a no-op.
        In full implementation with rllm AgentTrainer, the actual model weights are updated.
    """
    # Check if models have parameters method (actual PyTorch models)
    if not hasattr(ema_model, 'parameters') or not hasattr(train_model, 'parameters'):
        logger.debug(f"Models don't have parameters() method, skipping EMA update (stub implementation)")
        return
    
    try:
        with torch.no_grad():
            for ema_param, train_param in zip(ema_model.parameters(), train_model.parameters()):
                ema_param.data.mul_(decay).add_(train_param.data, alpha=1.0 - decay)
        
        logger.debug(f"Updated EMA model with decay={decay}")
    except Exception as e:
        logger.warning(f"Failed to update EMA model: {e}. This is expected in stub implementation.")


def should_update_orchestrator(epoch: int, current_stage: str = 'easy') -> bool:
    """
    Determine if Orchestrator should be updated this epoch.
    
    Implements phased training schedule:
    - Phase 1 (Warmup, epochs 0-4): Freeze Orchestrator
    - Phase 2 (Alternating, epochs 5-19): Update every 3 epochs
    - Phase 3 (Joint, epochs 20+): Update every epoch (easy) or every 2 epochs (medium/hard)
    
    Args:
        epoch: Current epoch number (0-indexed)
        current_stage: Current curriculum stage ('easy', 'medium', 'hard')
        
    Returns:
        True if Orchestrator should be updated
    """
    # Phase 1: Warmup (first 5 epochs)
    if epoch < 5:
        return False
    
    # Phase 2: Alternating (epochs 5-19)
    elif epoch < 20:
        return epoch % 3 == 0  # Every 3rd epoch
    
    # Phase 3: Joint updates (epoch 20+)
    else:
        if current_stage == 'easy':
            return True  # Every epoch
        else:
            return epoch % 2 == 0  # Every 2nd epoch


def get_training_phase(epoch: int) -> str:
    """
    Get current training phase name.
    
    Args:
        epoch: Current epoch number (0-indexed)
        
    Returns:
        Phase name: 'warmup', 'alternating', or 'joint'
    """
    if epoch < 5:
        return 'warmup'
    elif epoch < 20:
        return 'alternating'
    else:
        return 'joint'


def detect_reward_drift(train_metrics: Dict[str, float], val_metrics: Dict[str, float], 
                       threshold: float = 0.2) -> bool:
    """
    Detect reward inflation by comparing training and validation metrics.
    
    Reward inflation occurs when training rewards increase but validation
    performance doesn't improve (or decreases), indicating the agents are
    gaming the evaluation system.
    
    Args:
        train_metrics: Training metrics (must include 'answer_quality')
        val_metrics: Validation metrics (must include 'answer_quality')
        threshold: Maximum acceptable gap between train and val
        
    Returns:
        True if drift detected (train >> val)
    """
    train_quality = train_metrics.get('answer_quality', 0.0)
    val_quality = val_metrics.get('answer_quality', 0.0)
    
    gap = train_quality - val_quality
    
    if gap > threshold:
        logger.warning(
            f"Reward drift detected! Train quality: {train_quality:.3f}, "
            f"Val quality: {val_quality:.3f}, Gap: {gap:.3f} > {threshold}"
        )
        return True
    
    logger.debug(
        f"No drift detected. Train: {train_quality:.3f}, "
        f"Val: {val_quality:.3f}, Gap: {gap:.3f}"
    )
    return False


def adjust_ema_decay(current_decay: float, drift_detected: bool) -> float:
    """
    Adjust EMA decay rate if drift is detected.
    
    Increases decay (slows down updates) when drift is detected to
    stabilize evaluation.
    
    Args:
        current_decay: Current EMA decay factor
        drift_detected: Whether drift was detected
        
    Returns:
        Adjusted decay factor
    """
    if drift_detected:
        # Increase decay to slow down EMA updates
        new_decay = min(0.995, current_decay + 0.005)
        logger.info(f"Adjusting EMA decay: {current_decay} -> {new_decay} (drift detected)")
        return new_decay
    
    return current_decay


def compute_evaluation_consistency(current_scores: List[float], previous_scores: List[float]) -> float:
    """
    Compute consistency of evaluation scores over time.
    
    High variance in scores for similar inputs indicates unstable evaluation.
    
    Args:
        current_scores: Evaluation scores from current epoch
        previous_scores: Evaluation scores from previous epoch
        
    Returns:
        Consistency score (0-1, higher is more consistent)
    """
    if not current_scores or not previous_scores:
        return 1.0  # Assume consistent if no data
    
    # Compute correlation between current and previous scores
    if len(current_scores) != len(previous_scores):
        logger.warning("Score lists have different lengths, using min length")
        min_len = min(len(current_scores), len(previous_scores))
        current_scores = current_scores[:min_len]
        previous_scores = previous_scores[:min_len]
    
    # Compute Pearson correlation
    if len(current_scores) < 2:
        return 1.0
    
    correlation = np.corrcoef(current_scores, previous_scores)[0, 1]
    
    # Handle NaN (can occur if all scores are identical)
    if np.isnan(correlation):
        correlation = 1.0
    
    # Convert to 0-1 range (correlation is in [-1, 1])
    consistency = (correlation + 1.0) / 2.0
    
    logger.debug(f"Evaluation consistency: {consistency:.3f}")
    
    return float(consistency)


def check_score_inflation(current_avg: float, previous_avg: float, 
                         performance_improved: bool, threshold: float = 0.1) -> bool:
    """
    Check if scores are inflating without real performance improvement.
    
    Args:
        current_avg: Average score in current epoch
        previous_avg: Average score in previous epoch
        performance_improved: Whether actual performance improved
        threshold: Minimum score increase to consider inflation
        
    Returns:
        True if inflation detected
    """
    score_increase = current_avg - previous_avg
    
    if score_increase > threshold and not performance_improved:
        logger.warning(
            f"Score inflation detected! Scores increased by {score_increase:.3f} "
            f"but performance did not improve"
        )
        return True
    
    return False

