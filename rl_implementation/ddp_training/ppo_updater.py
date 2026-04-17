"""
PPO Updater with GRPO Advantage Estimation.

Implements Group Relative Policy Optimization (GRPO) for advantage estimation,
which doesn't require a critic network. Uses standard PPO loss with clipping.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO updates."""
    # PPO hyperparameters
    clip_ratio: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_low: float = 0.2
    
    # Entropy
    entropy_coeff: float = 0.01
    
    # KL penalty (optional)
    use_kl_penalty: bool = False
    kl_coeff: float = 0.1
    target_kl: float = 0.01
    
    # Value loss (not used with GRPO, but kept for compatibility)
    value_coeff: float = 0.5
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # GRPO settings
    grpo_baseline: str = "mean"  # 'mean', 'median', 'min', 'max'
    normalize_advantages: bool = True
    
    # Mini-batch settings
    ppo_epochs: int = 1
    mini_batch_size: int = 4


class GRPOAdvantageEstimator:
    """
    Group Relative Policy Optimization (GRPO) advantage estimator.
    
    GRPO computes advantages by comparing rewards within a group (batch),
    eliminating the need for a learned value function (critic).
    
    Advantage = (reward - baseline) / std
    
    Where baseline can be mean, median, min, or max of the group.
    """
    
    def __init__(self, baseline: str = "mean", normalize: bool = True):
        """
        Initialize GRPO advantage estimator.
        
        Args:
            baseline: Baseline method ('mean', 'median', 'min', 'max')
            normalize: Whether to normalize advantages
        """
        self.baseline = baseline
        self.normalize = normalize
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        response_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using GRPO.
        
        Args:
            rewards: Reward tensor [batch] or [batch, seq_len]
            response_masks: Optional mask for valid tokens [batch, seq_len]
            
        Returns:
            Advantages tensor (same shape as rewards)
        """
        # Ensure rewards are float
        rewards = rewards.float()
        
        # Compute baseline
        if self.baseline == "mean":
            baseline = rewards.mean()
        elif self.baseline == "median":
            baseline = rewards.median()
        elif self.baseline == "min":
            baseline = rewards.min()
        elif self.baseline == "max":
            baseline = rewards.max()
        else:
            raise ValueError(f"Unknown baseline method: {self.baseline}")
        
        # Compute advantages
        advantages = rewards - baseline
        
        # Normalize if requested
        if self.normalize:
            std = advantages.std() + 1e-8
            advantages = advantages / std
        
        return advantages
    
    def compute_token_level_advantages(
        self,
        rewards: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level advantages for response tokens.
        
        Distributes the trajectory-level reward across response tokens.
        
        Args:
            rewards: Trajectory rewards [batch]
            response_masks: Mask for response tokens [batch, seq_len]
            
        Returns:
            Token-level advantages [batch, seq_len]
        """
        batch_size, seq_len = response_masks.shape
        
        # Compute trajectory-level advantages
        traj_advantages = self.compute_advantages(rewards)
        
        # Expand to token level
        # Each token in a trajectory gets the same advantage
        token_advantages = traj_advantages.unsqueeze(-1).expand(-1, seq_len)
        
        # Mask non-response tokens
        token_advantages = token_advantages * response_masks
        
        return token_advantages


class PPOUpdater:
    """
    PPO updater for training with GRPO advantages.
    
    Handles:
    - GRPO advantage estimation
    - PPO loss computation with clipping
    - Entropy bonus
    - Optional KL penalty
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPO updater.
        
        Args:
            config: PPOConfig with hyperparameters
        """
        self.config = config
        self.advantage_estimator = GRPOAdvantageEstimator(
            baseline=config.grpo_baseline,
            normalize=config.normalize_advantages,
        )
        
        # Running statistics for logging
        self.stats = {
            "policy_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "clip_fraction": [],
            "advantage_mean": [],
            "advantage_std": [],
        }
    
    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with clipping.
        
        Args:
            log_probs: Current log probabilities [batch, seq_len]
            old_log_probs: Log probabilities from rollout [batch, seq_len]
            advantages: Advantage estimates [batch, seq_len]
            response_mask: Mask for response tokens [batch, seq_len]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio_low,
            1.0 + self.config.clip_ratio_high,
        ) * advantages
        
        # Take minimum (pessimistic bound)
        surr = torch.min(surr1, surr2)
        
        # Masked mean
        policy_loss = -(surr * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        # Compute metrics
        with torch.no_grad():
            # KL divergence approximation
            kl_div = 0.5 * ((log_probs - old_log_probs) ** 2 * response_mask).sum() / (response_mask.sum() + 1e-8)
            
            # Clip fraction
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float()
            clip_fraction = (clip_fraction * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_div.item(),
            "clip_fraction": clip_fraction.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }
        
        return policy_loss, metrics
    
    def compute_entropy_loss(
        self,
        logits: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute entropy bonus.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            response_mask: Mask for response tokens [batch, seq_len]
            
        Returns:
            Tuple of (entropy_loss, entropy_value)
        """
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Token-level entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Masked mean
        entropy_mean = (entropy * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -self.config.entropy_coeff * entropy_mean
        
        return entropy_loss, entropy_mean.item()
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty.
        
        Args:
            log_probs: Current log probabilities [batch, seq_len]
            old_log_probs: Log probabilities from rollout [batch, seq_len]
            response_mask: Mask for response tokens [batch, seq_len]
            
        Returns:
            KL penalty loss
        """
        if not self.config.use_kl_penalty:
            return torch.tensor(0.0, device=log_probs.device)
        
        # KL divergence approximation
        kl_div = 0.5 * (log_probs - old_log_probs) ** 2
        kl_mean = (kl_div * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        return self.config.kl_coeff * kl_mean
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a training batch with computed advantages.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            response_mask: Response token mask [batch, seq_len]
            rewards: Trajectory rewards [batch]
            old_log_probs: Log probs from rollout [batch, seq_len]
            
        Returns:
            Dictionary with all batch data including advantages
        """
        # Compute token-level advantages
        advantages = self.advantage_estimator.compute_token_level_advantages(
            rewards=rewards,
            response_masks=response_mask,
        )
        
        # Log advantage statistics
        valid_advantages = advantages[response_mask > 0]
        self.stats["advantage_mean"].append(valid_advantages.mean().item())
        self.stats["advantage_std"].append(valid_advantages.std().item())
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_mask": response_mask,
            "rewards": rewards,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        }
    
    def update_step(
        self,
        trainer,  # DDPTrainer instance
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single PPO update step.
        
        Args:
            trainer: DDPTrainer instance for model and optimizer
            batch: Prepared batch with advantages
            
        Returns:
            Dictionary with training metrics
        """
        metrics = trainer.train_step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            advantages=batch["advantages"],
            old_log_probs=batch["old_log_probs"],
            clip_ratio=self.config.clip_ratio,
            entropy_coeff=self.config.entropy_coeff,
        )
        
        # Track statistics
        self.stats["policy_loss"].append(metrics.get("ppo_loss", 0.0))
        self.stats["entropy"].append(metrics.get("entropy", 0.0))
        self.stats["kl_divergence"].append(metrics.get("approx_kl", 0.0))
        self.stats["clip_fraction"].append(metrics.get("clip_fraction", 0.0))
        
        return metrics
    
    def get_stats_summary(self) -> Dict[str, float]:
        """
        Get summary of training statistics.
        
        Returns:
            Dictionary with mean statistics
        """
        summary = {}
        for key, values in self.stats.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_last"] = values[-1]
        
        return summary
    
    def reset_stats(self):
        """Reset training statistics."""
        for key in self.stats:
            self.stats[key] = []


def compute_returns_and_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_gae: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute returns and advantages (GAE or simple).
    
    This is an alternative to GRPO that uses temporal difference.
    Not used in GRPO mode, but provided for compatibility.
    
    Args:
        rewards: Reward tensor [batch, seq_len] or [batch]
        values: Value estimates [batch, seq_len] (required for GAE)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        use_gae: Whether to use GAE
        
    Returns:
        Tuple of (returns, advantages)
    """
    if not use_gae:
        # Simple returns (no discounting for single-step)
        returns = rewards.clone()
        advantages = rewards.clone()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    if values is None:
        raise ValueError("Values required for GAE")
    
    # GAE computation
    batch_size = rewards.shape[0]
    
    if rewards.dim() == 1:
        # Trajectory-level rewards
        returns = rewards.clone()
        advantages = rewards - values.mean(dim=-1)
    else:
        # Token-level rewards (rare case)
        seq_len = rewards.shape[1]
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae = torch.zeros(batch_size, device=rewards.device)
        last_value = torch.zeros(batch_size, device=rewards.device)
        
        for t in reversed(range(seq_len)):
            delta = rewards[:, t] + gamma * last_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[:, t] = advantages[:, t] + values[:, t]
            last_value = values[:, t]
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns, advantages

