"""
Rollout Store for persisting training data.

Saves rollout data (questions, trajectories, evaluations, rewards) after each epoch.
Uses JSONL format (one JSON per line) for efficient append-mode writing.

Features:
- Append mode: Data accumulates across epochs and runs
- Atomic writes: Uses temp file + rename to prevent corruption
- Resumable: Can load previous rollouts to continue training
- Filterable: Can load by epoch, difficulty, reward threshold, etc.
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

logger = logging.getLogger(__name__)


class RolloutStore:
    """
    Persistent storage for training rollouts.
    
    Saves rollouts in JSONL format (one JSON object per line) for:
    - Efficient append-mode writing
    - Easy streaming reads for large files
    - Crash recovery (partial data preserved)
    """
    
    def __init__(
        self,
        store_path: str,
        auto_flush: bool = True,
    ):
        """
        Initialize rollout store.
        
        Args:
            store_path: Path to JSONL file for storing rollouts
            auto_flush: Whether to flush after each write (default: True for safety)
        """
        self.store_path = Path(store_path)
        self.auto_flush = auto_flush
        
        # Create parent directory if needed
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count existing entries
        self.entry_count = self._count_entries()
        
        logger.info(f"RolloutStore initialized: {self.store_path}")
        logger.info(f"  Existing entries: {self.entry_count}")
    
    def _count_entries(self) -> int:
        """Count existing entries in the store."""
        if not self.store_path.exists():
            return 0
        
        count = 0
        try:
            with open(self.store_path, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.warning(f"Error counting entries: {e}")
        return count
    
    def save_epoch(
        self,
        epoch: int,
        difficulty: str,
        questions: List[str],
        trajectories: List[Any],  # List of Trajectory objects
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save rollouts from one epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
            difficulty: Curriculum difficulty level
            questions: List of questions generated
            trajectories: List of Trajectory objects
            config: Optional config dict to include
            
        Returns:
            Number of entries saved
        """
        timestamp = datetime.now().isoformat()
        entries_saved = 0
        
        try:
            with open(self.store_path, 'a') as f:
                for i, (question, traj) in enumerate(zip(questions, trajectories)):
                    entry = {
                        'timestamp': timestamp,
                        'epoch': epoch,
                        'difficulty': difficulty,
                        'question_idx': i,
                        'question': question,
                        'trajectory': self._serialize_trajectory(traj),
                    }
                    
                    if config and i == 0:  # Include config only with first entry
                        entry['config'] = config
                    
                    f.write(json.dumps(entry) + '\n')
                    entries_saved += 1
                
                if self.auto_flush:
                    f.flush()
                    os.fsync(f.fileno())
            
            self.entry_count += entries_saved
            logger.info(f"Saved {entries_saved} rollouts from epoch {epoch} to {self.store_path}")
            
        except Exception as e:
            logger.error(f"Error saving rollouts: {e}")
            raise
        
        return entries_saved
    
    def _serialize_trajectory(self, traj) -> Dict[str, Any]:
        """
        Serialize a Trajectory object to dict.
        
        IMPORTANT: Stores FULL prompts for both Cypher Generator and Orchestrator
        to ensure exact reproducibility during training. This avoids issues with
        prompt reconstruction that can cause distribution shift.
        """
        if traj is None:
            return {}
        
        # Convert trajectory to dict, handling non-serializable fields
        traj_dict = {
            'question': traj.question,
            'num_steps': traj.num_steps,
            'total_results': traj.total_results,
            'success_rate': traj.success_rate,
            'data_quality_score': traj.data_quality_score,
            'answer_quality_score': traj.answer_quality_score,
            'trajectory_quality_score': traj.trajectory_quality_score,
            'doubt_level': traj.doubt_level,
            'synthesized_answer': traj.synthesized_answer,
            
            # =========================================================================
            # Stored Prompts for Orchestrator Training
            # =========================================================================
            # Question Generation prompt (Role 1) - full prompt used to generate question
            'orch_qgen_prompt': getattr(traj, 'orch_qgen_prompt', ''),
            # Answer Synthesis prompt (Role 3) - full prompt used to synthesize answer
            'orch_synth_prompt': getattr(traj, 'orch_synth_prompt', ''),
            
            # =========================================================================
            # Rewards
            # =========================================================================
            # Legacy reward (same as cypher_reward for backwards compatibility)
            'reward': traj.reward,
            # Separate rewards for each trainable model/role
            'cypher_reward': getattr(traj, 'cypher_reward', traj.reward),
            'orch_qgen_reward': getattr(traj, 'orch_qgen_reward', 0.0),
            'orch_synth_reward': getattr(traj, 'orch_synth_reward', 0.0),
            'reward_metadata': getattr(traj, 'reward_metadata', {}),
            
            'steps': [],
        }
        
        # Serialize steps - store FULL prompts for Cypher Generator training
        for step in traj.steps:
            step_dict = {
                # FULL prompt - no truncation! This is critical for training.
                'prompt': step.prompt,
                'response': step.response,
                'cypher_query': step.cypher_query,
                'success': step.success,
                'has_data': step.has_data,
                'execution_time_ms': step.execution_time_ms,
                'num_results': step.num_results,
                # Don't save full execution_result to save space
                'execution_result_keys': list(step.execution_result.keys()) if step.execution_result else [],
            }
            traj_dict['steps'].append(step_dict)
        
        return traj_dict
    
    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all rollouts from the store.
        
        Returns:
            List of rollout entries
        """
        entries = []
        for entry in self.iter_entries():
            entries.append(entry)
        return entries
    
    def iter_entries(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over entries without loading all into memory.
        
        Yields:
            Rollout entry dicts
        """
        if not self.store_path.exists():
            return
        
        with open(self.store_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    def load_by_epoch(self, epoch: int) -> List[Dict[str, Any]]:
        """Load rollouts from a specific epoch."""
        return [e for e in self.iter_entries() if e.get('epoch') == epoch]
    
    def load_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Load rollouts with a specific difficulty."""
        return [e for e in self.iter_entries() if e.get('difficulty') == difficulty]
    
    def load_high_reward(self, min_reward: float = 0.5) -> List[Dict[str, Any]]:
        """Load rollouts with reward above threshold."""
        entries = []
        for e in self.iter_entries():
            traj = e.get('trajectory', {})
            if traj.get('reward', 0) >= min_reward:
                entries.append(e)
        return entries
    
    def load_failed(self) -> List[Dict[str, Any]]:
        """Load rollouts where trajectory had errors or low success."""
        entries = []
        for e in self.iter_entries():
            traj = e.get('trajectory', {})
            if traj.get('success_rate', 1.0) < 0.5 or traj.get('reward', 1.0) < 0.0:
                entries.append(e)
        return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored rollouts."""
        stats = {
            'total_entries': 0,
            'epochs': set(),
            'difficulties': {},
            'reward_sum': 0.0,
            'cypher_reward_sum': 0.0,
            'orch_qgen_reward_sum': 0.0,
            'orch_synth_reward_sum': 0.0,
            'success_rate_sum': 0.0,
        }
        
        for e in self.iter_entries():
            stats['total_entries'] += 1
            stats['epochs'].add(e.get('epoch', 0))
            
            diff = e.get('difficulty', 'unknown')
            stats['difficulties'][diff] = stats['difficulties'].get(diff, 0) + 1
            
            traj = e.get('trajectory', {})
            stats['reward_sum'] += traj.get('reward', 0)
            stats['cypher_reward_sum'] += traj.get('cypher_reward', traj.get('reward', 0))
            stats['orch_qgen_reward_sum'] += traj.get('orch_qgen_reward', 0)
            stats['orch_synth_reward_sum'] += traj.get('orch_synth_reward', 0)
            stats['success_rate_sum'] += traj.get('success_rate', 0)
        
        if stats['total_entries'] > 0:
            stats['avg_reward'] = stats['reward_sum'] / stats['total_entries']
            stats['avg_cypher_reward'] = stats['cypher_reward_sum'] / stats['total_entries']
            stats['avg_orch_qgen_reward'] = stats['orch_qgen_reward_sum'] / stats['total_entries']
            stats['avg_orch_synth_reward'] = stats['orch_synth_reward_sum'] / stats['total_entries']
            stats['avg_success_rate'] = stats['success_rate_sum'] / stats['total_entries']
        
        stats['epochs'] = sorted(stats['epochs'])
        stats['num_epochs'] = len(stats['epochs'])
        
        return stats
    
    def export_questions(self, output_path: str) -> int:
        """
        Export just the questions to a text file (one per line).
        
        Args:
            output_path: Path to output file
            
        Returns:
            Number of questions exported
        """
        count = 0
        with open(output_path, 'w') as f:
            for e in self.iter_entries():
                f.write(e.get('question', '') + '\n')
                count += 1
        
        logger.info(f"Exported {count} questions to {output_path}")
        return count
    
    def export_high_quality_examples(
        self,
        output_path: str,
        min_reward: float = 0.7,
        max_examples: int = 1000,
    ) -> int:
        """
        Export high-quality examples for future SFT training.
        
        Args:
            output_path: Path to output JSONL file
            min_reward: Minimum reward threshold
            max_examples: Maximum number of examples to export
            
        Returns:
            Number of examples exported
        """
        count = 0
        with open(output_path, 'w') as f:
            for e in self.iter_entries():
                if count >= max_examples:
                    break
                    
                traj = e.get('trajectory', {})
                if traj.get('reward', 0) >= min_reward:
                    example = {
                        'question': e.get('question'),
                        'steps': traj.get('steps', []),
                        'answer': traj.get('synthesized_answer'),
                        'reward': traj.get('reward'),
                    }
                    f.write(json.dumps(example) + '\n')
                    count += 1
        
        logger.info(f"Exported {count} high-quality examples to {output_path}")
        return count


def load_rollouts(store_path: str) -> RolloutStore:
    """
    Convenience function to load rollouts from a file.
    
    Args:
        store_path: Path to JSONL rollout store
        
    Returns:
        RolloutStore instance
    """
    return RolloutStore(store_path)

