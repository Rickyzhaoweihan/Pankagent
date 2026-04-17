"""
Adaptive Entity Sampler for Question Generation.

Uses Thompson Sampling to balance:
1. Prior knowledge (entity degree - more connected = more answerable)
2. Learned knowledge (reward history - what worked before)

Key insight: Sample (entity, relationship) pairs together, not just entities.
This ensures questions are about relationships the entity actually HAS.

The sampler updates based on ANSWERABILITY only (cypher success + results > 0),
with slack for failures to avoid over-penalizing Cypher generator errors.

Usage:
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    
    # Initialize from degree data
    sampler = AdaptiveEntitySampler('outputs/entity_degrees.json')
    
    # Sample an entity-relationship pair
    entity, relationship, entity_type = sampler.sample(entity_type='gene')
    
    # Update based on answerability
    sampler.update(entity, relationship, answerable=True)
    
    # Persist state
    sampler.save('outputs/adaptive_sampler_state.json')
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntityRelationshipStats:
    """Statistics for an (entity, relationship) pair."""
    
    entity: str
    entity_type: str
    relationship: str
    degree: int  # Number of this relationship from graph
    
    # Thompson Sampling Beta distribution parameters
    alpha: float = 1.0      # Pseudo-successes (higher = more likely to succeed)
    beta_param: float = 1.0  # Pseudo-failures (higher = less likely to succeed)
    
    # Tracking
    times_sampled: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0  # For slack mechanism
    
    @property
    def success_rate(self) -> float:
        """Thompson Sampling success probability estimate (mean of Beta)."""
        return self.alpha / (self.alpha + self.beta_param)
    
    @property
    def uncertainty(self) -> float:
        """Uncertainty in the estimate (variance of Beta)."""
        total = self.alpha + self.beta_param
        return (self.alpha * self.beta_param) / (total * total * (total + 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entity': self.entity,
            'entity_type': self.entity_type,
            'relationship': self.relationship,
            'degree': self.degree,
            'alpha': self.alpha,
            'beta_param': self.beta_param,
            'times_sampled': self.times_sampled,
            'total_successes': self.total_successes,
            'consecutive_failures': self.consecutive_failures,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRelationshipStats":
        """Create from dictionary."""
        return cls(**data)


class AdaptiveEntitySampler:
    """
    Samples entities with adaptive prior based on:
    1. Graph degree (how connected the entity is)
    2. Answerability history (what worked before)
    
    Uses Thompson Sampling for explore/exploit balance with slack for failures.
    """
    
    def __init__(
        self,
        entity_degrees_path: Optional[str] = None,
        degree_weight: float = 0.1,
        exploration_bonus: float = 0.3,
        slack_threshold: int = 3,
        soft_penalty: float = 0.3,
    ):
        """
        Initialize the adaptive entity sampler.
        
        Args:
            entity_degrees_path: Path to entity_degrees.json file
            degree_weight: How much to weight degree in prior (0.1 = modest boost)
            exploration_bonus: Bonus for rarely-sampled pairs (0-1)
            slack_threshold: Number of consecutive failures before full penalty
            soft_penalty: Beta increment for soft failures (before threshold)
        """
        self.degree_weight = degree_weight
        self.exploration_bonus = exploration_bonus
        self.slack_threshold = slack_threshold
        self.soft_penalty = soft_penalty
        
        # (entity, relationship) -> stats mapping
        self.stats: Dict[Tuple[str, str], EntityRelationshipStats] = {}
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.total_samples = 0
        self.total_updates = 0
        
        # Initialize from degree data if provided
        if entity_degrees_path:
            self._init_from_degrees(entity_degrees_path)
    
    def _init_from_degrees(self, path: str):
        """
        Initialize stats from degree JSON file.
        
        Prior: alpha = 1 + degree * degree_weight
        Higher degree = higher initial P(success)
        """
        logger.info(f"Initializing adaptive sampler from {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        degrees = data.get('degrees', {})
        
        for entity_type, entities in degrees.items():
            for entity_name, rel_degrees in entities.items():
                for rel, degree in rel_degrees.items():
                    # Clean up relationship name (remove _in suffix for key)
                    # but keep it in the stats for reference
                    key = (entity_name, rel)
                    
                    # Initialize with degree-based prior
                    initial_alpha = 1.0 + degree * self.degree_weight
                    
                    self.stats[key] = EntityRelationshipStats(
                        entity=entity_name,
                        entity_type=entity_type,
                        relationship=rel,
                        degree=degree,
                        alpha=initial_alpha,
                        beta_param=1.0,
                    )
        
        logger.info(f"Initialized {len(self.stats)} entity-relationship pairs")
        
        # Log distribution by entity type
        by_type = {}
        for stats in self.stats.values():
            by_type[stats.entity_type] = by_type.get(stats.entity_type, 0) + 1
        logger.info(f"Distribution by type: {by_type}")
    
    def sample(
        self,
        entity_type: Optional[str] = None,
        relationship: Optional[str] = None,
        n_samples: int = 1,
    ) -> List[Tuple[str, str, str]]:
        """
        Sample (entity, relationship, entity_type) tuples using Thompson Sampling.
        
        Args:
            entity_type: Optional filter for entity type (e.g., 'gene', 'cell_type')
            relationship: Optional filter for relationship type
            n_samples: Number of samples to return
            
        Returns:
            List of (entity_name, relationship_type, entity_type) tuples
        """
        # Get candidates matching filters
        candidates = []
        for key, stats in self.stats.items():
            if entity_type and stats.entity_type != entity_type:
                continue
            if relationship and stats.relationship != relationship:
                continue
            candidates.append((key, stats))
        
        if not candidates:
            logger.warning(
                f"No candidates for entity_type={entity_type}, relationship={relationship}"
            )
            return [(None, None, None)] * n_samples
        
        # Thompson Sampling: sample from Beta distribution for each candidate
        samples = []
        for key, stats in candidates:
            # Sample from Beta(alpha, beta)
            theta = np.random.beta(stats.alpha, stats.beta_param)
            
            # Add exploration bonus for rarely-sampled pairs
            if stats.times_sampled < 3:
                bonus = self.exploration_bonus * (3 - stats.times_sampled) / 3
                theta += bonus
            
            samples.append((theta, key, stats))
        
        # Sort by sampled value (descending) and pick top n
        samples.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for i in range(min(n_samples, len(samples))):
            _, key, stats = samples[i]
            results.append((stats.entity, stats.relationship, stats.entity_type))
            
            # Track that we sampled this
            stats.times_sampled += 1
            self.total_samples += 1
        
        return results
    
    def sample_one(
        self,
        entity_type: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Sample a single (entity, relationship, entity_type) tuple.
        
        Convenience method for single samples.
        """
        results = self.sample(entity_type, relationship, n_samples=1)
        return results[0]
    
    def update(
        self,
        entity: str,
        relationship: str,
        answerable: bool,
    ):
        """
        Update Thompson Sampling parameters based on answerability.
        
        Updates with SLACK for failures:
        - Success (answerable=True): alpha += 1.0 (full reward)
        - Failure (answerable=False):
          - If consecutive_failures < slack_threshold: beta += soft_penalty
          - If consecutive_failures >= slack_threshold: beta += 1.0 (full penalty)
        
        Slack rationale: Early failures may be Cypher generator errors,
        not entity unanswerability. Only penalize strongly after repeated failures.
        
        Args:
            entity: Entity name
            relationship: Relationship type
            answerable: Whether the question was answerable (cypher success + results > 0)
        """
        key = (entity, relationship)
        stats = self.stats.get(key)
        
        if not stats:
            logger.debug(f"Unknown entity-relationship pair: {key}")
            return
        
        if answerable:
            # Success: full reward
            stats.alpha += 1.0
            stats.total_successes += 1
            stats.consecutive_failures = 0  # Reset failure streak
            
            logger.debug(
                f"Success: {entity}+{relationship} α={stats.alpha:.1f}, "
                f"P(success)={stats.success_rate:.3f}"
            )
        else:
            # Failure: soft or full penalty based on consecutive failures
            if stats.consecutive_failures < self.slack_threshold:
                # Soft penalty - might be Cypher generator error
                stats.beta_param += self.soft_penalty
                logger.debug(
                    f"Soft failure ({stats.consecutive_failures + 1}/{self.slack_threshold}): "
                    f"{entity}+{relationship} β+={self.soft_penalty}"
                )
            else:
                # Full penalty - repeated failures suggest entity is unanswerable
                stats.beta_param += 1.0
                logger.debug(
                    f"Full failure: {entity}+{relationship} β+=1.0"
                )
            
            stats.consecutive_failures += 1
        
        self.total_updates += 1
        self.last_updated = datetime.now().isoformat()
    
    def batch_update(
        self,
        results: List[Tuple[str, str, bool]],
    ):
        """
        Update from batch of (entity, relationship, answerable) tuples.
        
        Args:
            results: List of (entity, relationship, answerable) tuples
        """
        for entity, relationship, answerable in results:
            self.update(entity, relationship, answerable)
    
    def get_top_entities(
        self,
        entity_type: Optional[str] = None,
        n: int = 10,
    ) -> List[Tuple[str, str, float, int]]:
        """
        Get top entities by success rate.
        
        Args:
            entity_type: Optional filter for entity type
            n: Number of entities to return
            
        Returns:
            List of (entity, relationship, success_rate, times_sampled) tuples
        """
        candidates = []
        for key, stats in self.stats.items():
            if entity_type and stats.entity_type != entity_type:
                continue
            candidates.append((
                stats.entity,
                stats.relationship,
                stats.success_rate,
                stats.times_sampled,
            ))
        
        # Sort by success rate, with tie-breaker on times_sampled
        candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        return candidates[:n]
    
    def get_exploration_priorities(
        self,
        n: int = 10,
    ) -> List[Tuple[str, str, int, int]]:
        """
        Get entities that need more exploration (high potential, rarely sampled).
        
        Returns:
            List of (entity, relationship, times_sampled, degree) tuples
        """
        # High degree but few samples = high exploration priority
        candidates = []
        for key, stats in self.stats.items():
            if stats.times_sampled < 5 and stats.degree > 3:
                priority = stats.degree / (stats.times_sampled + 1)
                candidates.append((
                    stats.entity,
                    stats.relationship,
                    stats.times_sampled,
                    stats.degree,
                    priority,
                ))
        
        candidates.sort(key=lambda x: x[4], reverse=True)
        return [(e, r, t, d) for e, r, t, d, _ in candidates[:n]]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the sampler state."""
        by_type = {}
        total_sampled = 0
        total_successes = 0
        
        for stats in self.stats.values():
            etype = stats.entity_type
            if etype not in by_type:
                by_type[etype] = {
                    'count': 0,
                    'sampled': 0,
                    'successes': 0,
                    'avg_success_rate': 0,
                }
            
            by_type[etype]['count'] += 1
            by_type[etype]['sampled'] += stats.times_sampled
            by_type[etype]['successes'] += stats.total_successes
            total_sampled += stats.times_sampled
            total_successes += stats.total_successes
        
        # Compute average success rates
        for etype, info in by_type.items():
            type_stats = [s for s in self.stats.values() if s.entity_type == etype]
            if type_stats:
                info['avg_success_rate'] = sum(s.success_rate for s in type_stats) / len(type_stats)
        
        return {
            'total_pairs': len(self.stats),
            'total_samples': total_sampled,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / max(1, total_sampled),
            'by_entity_type': by_type,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }
    
    def save(self, path: str):
        """Save sampler state to JSON file."""
        state = {
            'metadata': {
                'degree_weight': self.degree_weight,
                'exploration_bonus': self.exploration_bonus,
                'slack_threshold': self.slack_threshold,
                'soft_penalty': self.soft_penalty,
                'created_at': self.created_at,
                'last_updated': self.last_updated,
                'total_samples': self.total_samples,
                'total_updates': self.total_updates,
            },
            'stats': {
                f"{e}|{r}": s.to_dict()
                for (e, r), s in self.stats.items()
            }
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved sampler state to {path} ({len(self.stats)} pairs)")
    
    @classmethod
    def load(cls, path: str) -> "AdaptiveEntitySampler":
        """Load sampler state from JSON file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        metadata = state['metadata']
        
        # Create sampler with saved parameters (don't load from degrees)
        sampler = cls(
            entity_degrees_path=None,
            degree_weight=metadata['degree_weight'],
            exploration_bonus=metadata['exploration_bonus'],
            slack_threshold=metadata['slack_threshold'],
            soft_penalty=metadata['soft_penalty'],
        )
        
        # Restore metadata
        sampler.created_at = metadata['created_at']
        sampler.last_updated = metadata['last_updated']
        sampler.total_samples = metadata['total_samples']
        sampler.total_updates = metadata['total_updates']
        
        # Restore stats
        for key_str, data in state['stats'].items():
            e, r = key_str.split('|', 1)
            sampler.stats[(e, r)] = EntityRelationshipStats.from_dict(data)
        
        logger.info(f"Loaded sampler state from {path} ({len(sampler.stats)} pairs)")
        
        return sampler


def main():
    """CLI for testing the adaptive sampler."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test adaptive entity sampler')
    parser.add_argument(
        '--degrees', '-d',
        default='outputs/entity_degrees.json',
        help='Path to entity degrees JSON'
    )
    parser.add_argument(
        '--state', '-s',
        default=None,
        help='Path to saved sampler state (to load instead of degrees)'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=10,
        help='Number of samples to draw'
    )
    parser.add_argument(
        '--entity-type', '-t',
        default=None,
        help='Filter by entity type'
    )
    
    args = parser.parse_args()
    
    # Load or create sampler
    if args.state and Path(args.state).exists():
        sampler = AdaptiveEntitySampler.load(args.state)
        print(f"Loaded sampler from {args.state}")
    else:
        sampler = AdaptiveEntitySampler(args.degrees)
        print(f"Created sampler from {args.degrees}")
    
    # Print summary
    summary = sampler.get_stats_summary()
    print(f"\nSampler Summary:")
    print(f"  Total pairs: {summary['total_pairs']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  By type: {summary['by_entity_type']}")
    
    # Draw samples
    print(f"\nSampling {args.samples} entities (type={args.entity_type}):")
    for i in range(args.samples):
        entity, rel, etype = sampler.sample_one(entity_type=args.entity_type)
        stats = sampler.stats.get((entity, rel))
        if stats:
            print(f"  {i+1}. {entity} + {rel} (type={etype}, degree={stats.degree}, P={stats.success_rate:.3f})")
        else:
            print(f"  {i+1}. {entity} + {rel} (type={etype})")
    
    # Show top entities
    print(f"\nTop 5 by success rate:")
    for entity, rel, rate, sampled in sampler.get_top_entities(n=5):
        print(f"  {entity} + {rel}: P={rate:.3f}, sampled={sampled}")
    
    # Show exploration priorities
    print(f"\nTop 5 for exploration:")
    for entity, rel, sampled, degree in sampler.get_exploration_priorities(n=5):
        print(f"  {entity} + {rel}: degree={degree}, sampled={sampled}")


if __name__ == '__main__':
    main()

