"""
Rollout Loader for Training Scripts.

Loads and parses rollout JSONL files for both Cypher Generator and Orchestrator training.
Provides filtering and batch preparation utilities.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Parsed step from a trajectory."""
    prompt: str
    response: str
    cypher_query: str
    success: bool
    has_data: bool
    execution_time_ms: float
    num_results: int


@dataclass
class RolloutEntry:
    """Parsed rollout entry from JSONL."""
    timestamp: str
    epoch: int
    difficulty: str
    question_idx: int
    question: str
    
    # Trajectory data
    num_steps: int
    total_results: int
    success_rate: float
    data_quality_score: float
    answer_quality_score: float
    trajectory_quality_score: float
    doubt_level: float
    synthesized_answer: str
    
    # Rewards (separate for each trainable model/role)
    reward: float  # Legacy: same as cypher_reward for backwards compatibility
    cypher_reward: float = 0.0  # Reward for Cypher Generator
    orch_qgen_reward: float = 0.0  # Reward for Orchestrator Question Generation
    orch_synth_reward: float = 0.0  # Reward for Orchestrator Answer Synthesis
    reward_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Stored prompts for training (full prompts, no truncation)
    orch_qgen_prompt: str = ""  # Question generation prompt (Orchestrator Role 1)
    orch_synth_prompt: str = ""  # Answer synthesis prompt (Orchestrator Role 3)
    
    steps: List[TrajectoryStep] = field(default_factory=list)
    
    # Optional config (first entry of epoch may have this)
    config: Optional[Dict[str, Any]] = None


class RolloutLoader:
    """
    Loads and filters rollouts from JSONL files.
    
    Provides utilities for preparing training batches for both
    Cypher Generator and Orchestrator training.
    """
    
    def __init__(self, rollouts_path: str):
        """
        Initialize rollout loader.
        
        Args:
            rollouts_path: Path to rollouts JSONL file
        """
        self.rollouts_path = Path(rollouts_path)
        
        if not self.rollouts_path.exists():
            raise FileNotFoundError(f"Rollouts file not found: {self.rollouts_path}")
        
        logger.info(f"RolloutLoader initialized: {self.rollouts_path}")
    
    def load_rollouts(
        self,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        difficulty: Optional[str] = None,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[RolloutEntry]:
        """
        Load rollouts with optional filtering.
        
        Args:
            min_epoch: Minimum epoch number (inclusive)
            max_epoch: Maximum epoch number (inclusive)
            difficulty: Filter by difficulty level ('easy', 'medium', 'hard')
            min_reward: Minimum reward threshold
            max_reward: Maximum reward threshold
            limit: Maximum number of entries to return
            
        Returns:
            List of RolloutEntry objects
        """
        entries = []
        
        with open(self.rollouts_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
                
                # Apply filters
                if min_epoch is not None and data.get('epoch', 0) < min_epoch:
                    continue
                if max_epoch is not None and data.get('epoch', 0) > max_epoch:
                    continue
                if difficulty is not None and data.get('difficulty') != difficulty:
                    continue
                
                traj = data.get('trajectory', {})
                reward = traj.get('reward', 0.0)
                
                if min_reward is not None and reward < min_reward:
                    continue
                if max_reward is not None and reward > max_reward:
                    continue
                
                # Parse entry
                entry = self._parse_entry(data)
                if entry is not None:
                    entries.append(entry)
                
                # Check limit
                if limit is not None and len(entries) >= limit:
                    break
        
        logger.info(f"Loaded {len(entries)} rollouts from {self.rollouts_path}")
        return entries
    
    def _parse_entry(self, data: Dict[str, Any]) -> Optional[RolloutEntry]:
        """Parse a raw JSON entry into a RolloutEntry object."""
        try:
            traj = data.get('trajectory', {})
            
            # Parse steps
            steps = []
            for step_data in traj.get('steps', []):
                step = TrajectoryStep(
                    prompt=step_data.get('prompt', ''),
                    response=step_data.get('response', ''),
                    cypher_query=step_data.get('cypher_query', ''),
                    success=step_data.get('success', False),
                    has_data=step_data.get('has_data', False),
                    execution_time_ms=step_data.get('execution_time_ms', 0.0),
                    num_results=step_data.get('num_results', 0),
                )
                steps.append(step)
            
            # Parse rewards (handle backwards compatibility)
            legacy_reward = traj.get('reward', 0.0)
            cypher_reward = traj.get('cypher_reward', legacy_reward)  # Fall back to legacy
            orch_qgen_reward = traj.get('orch_qgen_reward', 0.0)
            orch_synth_reward = traj.get('orch_synth_reward', 0.0)
            reward_metadata = traj.get('reward_metadata', {})
            
            # Parse stored prompts (for training without reconstruction)
            orch_qgen_prompt = traj.get('orch_qgen_prompt', '')
            orch_synth_prompt = traj.get('orch_synth_prompt', '')
            
            return RolloutEntry(
                timestamp=data.get('timestamp', ''),
                epoch=data.get('epoch', 0),
                difficulty=data.get('difficulty', 'easy'),
                question_idx=data.get('question_idx', 0),
                question=data.get('question', ''),
                num_steps=traj.get('num_steps', 0),
                total_results=traj.get('total_results', 0),
                success_rate=traj.get('success_rate', 0.0),
                data_quality_score=traj.get('data_quality_score', 0.0),
                answer_quality_score=traj.get('answer_quality_score', 0.0),
                trajectory_quality_score=traj.get('trajectory_quality_score', 0.0),
                doubt_level=traj.get('doubt_level', 0.0),
                synthesized_answer=traj.get('synthesized_answer', ''),
                reward=legacy_reward,
                cypher_reward=cypher_reward,
                orch_qgen_reward=orch_qgen_reward,
                orch_synth_reward=orch_synth_reward,
                reward_metadata=reward_metadata,
                orch_qgen_prompt=orch_qgen_prompt,
                orch_synth_prompt=orch_synth_prompt,
                steps=steps,
                config=data.get('config'),
            )
        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None
    
    def iterate_rollouts(
        self,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        difficulty: Optional[str] = None,
        min_reward: Optional[float] = None,
    ) -> Iterator[RolloutEntry]:
        """
        Iterate over rollouts without loading all into memory.
        
        Useful for very large rollout files.
        """
        with open(self.rollouts_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Apply filters
                if min_epoch is not None and data.get('epoch', 0) < min_epoch:
                    continue
                if max_epoch is not None and data.get('epoch', 0) > max_epoch:
                    continue
                if difficulty is not None and data.get('difficulty') != difficulty:
                    continue
                
                traj = data.get('trajectory', {})
                reward = traj.get('reward', 0.0)
                
                if min_reward is not None and reward < min_reward:
                    continue
                
                entry = self._parse_entry(data)
                if entry is not None:
                    yield entry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the rollouts file."""
        stats = {
            'total_entries': 0,
            'epochs': set(),
            'difficulties': {},
            'rewards': [],
            'success_rates': [],
        }
        
        for entry in self.iterate_rollouts():
            stats['total_entries'] += 1
            stats['epochs'].add(entry.epoch)
            stats['difficulties'][entry.difficulty] = stats['difficulties'].get(entry.difficulty, 0) + 1
            stats['rewards'].append(entry.reward)
            stats['success_rates'].append(entry.success_rate)
        
        # Compute aggregates
        stats['num_epochs'] = len(stats['epochs'])
        stats['epochs'] = sorted(stats['epochs'])
        
        if stats['rewards']:
            stats['avg_reward'] = sum(stats['rewards']) / len(stats['rewards'])
            stats['min_reward'] = min(stats['rewards'])
            stats['max_reward'] = max(stats['rewards'])
        
        if stats['success_rates']:
            stats['avg_success_rate'] = sum(stats['success_rates']) / len(stats['success_rates'])
        
        # Remove lists from final stats
        del stats['rewards']
        del stats['success_rates']
        
        return stats


class BalancedBatchSampler:
    """
    Sample batches with balanced reward distribution for better GRPO training.
    
    With bimodal reward distributions (many near-zero, many high rewards),
    GRPO's baseline estimate becomes unreliable. This sampler ensures each
    batch contains a mix of reward levels for more stable advantage estimation.
    
    Strategy: Stratify rollouts by reward into 3 groups, then sample
    proportionally from each group to create balanced batches.
    """
    
    def __init__(
        self,
        rollouts: List[RolloutEntry],
        reward_attr: str = 'cypher_reward',
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            rollouts: List of rollout entries
            reward_attr: Which reward attribute to use for stratification
            low_threshold: Threshold for low reward (< this value)
            high_threshold: Threshold for high reward (>= this value)
        """
        self.rollouts = rollouts
        self.reward_attr = reward_attr
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Stratify rollouts by reward
        self.low_reward = []   # reward < 0.3
        self.med_reward = []   # 0.3 <= reward < 0.7
        self.high_reward = []  # reward >= 0.7
        
        for entry in rollouts:
            reward = getattr(entry, reward_attr, entry.reward)
            if reward < low_threshold:
                self.low_reward.append(entry)
            elif reward < high_threshold:
                self.med_reward.append(entry)
            else:
                self.high_reward.append(entry)
        
        logger.info(
            f"BalancedBatchSampler stratified {len(rollouts)} rollouts: "
            f"low={len(self.low_reward)}, med={len(self.med_reward)}, high={len(self.high_reward)}"
        )
    
    def sample_balanced_batch(self, batch_size: int, seed: Optional[int] = None) -> List[RolloutEntry]:
        """
        Sample a batch with balanced reward distribution.
        
        Each batch contains approximately:
        - 33% low reward samples
        - 33% medium reward samples  
        - 33% high reward samples
        
        Args:
            batch_size: Target batch size
            seed: Optional random seed for reproducibility across distributed ranks.
                  CRITICAL for FSDP: all ranks must use the same seed to get the same batch!
            
        Returns:
            List of sampled rollout entries
        """
        import random
        
        # Set seed for reproducibility across ranks
        if seed is not None:
            random.seed(seed)
        
        # Calculate samples per stratum (round to ensure we hit batch_size)
        samples_per_stratum = batch_size // 3
        remainder = batch_size % 3
        
        n_low = samples_per_stratum + (1 if remainder > 0 else 0)
        n_med = samples_per_stratum + (1 if remainder > 1 else 0)
        n_high = samples_per_stratum
        
        # Sample from each stratum (with replacement if needed)
        batch = []
        
        if self.low_reward:
            batch.extend(random.choices(self.low_reward, k=n_low))
        else:
            # No low reward samples, take from medium instead
            n_med += n_low
        
        if self.med_reward:
            batch.extend(random.choices(self.med_reward, k=n_med))
        else:
            # No medium reward samples, distribute to low and high
            if self.low_reward:
                batch.extend(random.choices(self.low_reward, k=n_med // 2))
            if self.high_reward:
                batch.extend(random.choices(self.high_reward, k=n_med - n_med // 2))
        
        if self.high_reward:
            batch.extend(random.choices(self.high_reward, k=n_high))
        else:
            # No high reward samples, take from medium instead
            if self.med_reward:
                batch.extend(random.choices(self.med_reward, k=n_high))
            elif self.low_reward:
                batch.extend(random.choices(self.low_reward, k=n_high))
        
        # Shuffle to avoid ordering effects
        random.shuffle(batch)
        
        return batch
    
    def create_balanced_batches(self, batch_size: int, seed: Optional[int] = None) -> Iterator[List[RolloutEntry]]:
        """
        Create multiple balanced batches iteratively.
        
        Yields balanced batches until all strata are exhausted.
        
        Args:
            batch_size: Size of each batch
            seed: Optional random seed for reproducibility across distributed ranks.
                  CRITICAL for FSDP: all ranks must use the same seed to get the same batches!
            
        Yields:
            Lists of rollout entries (balanced batches)
        """
        import random
        
        # Set seed for reproducibility across ranks
        if seed is not None:
            random.seed(seed)
        
        # Create working copies to avoid modifying originals
        low_pool = list(self.low_reward)
        med_pool = list(self.med_reward)
        high_pool = list(self.high_reward)
        
        # Shuffle pools
        random.shuffle(low_pool)
        random.shuffle(med_pool)
        random.shuffle(high_pool)
        
        # Calculate samples per stratum per batch
        samples_per_stratum = batch_size // 3
        remainder = batch_size % 3
        
        n_low = samples_per_stratum + (1 if remainder > 0 else 0)
        n_med = samples_per_stratum + (1 if remainder > 1 else 0)
        n_high = samples_per_stratum
        
        # Yield batches until we run out of data
        while low_pool or med_pool or high_pool:
            batch = []
            
            # Sample from each pool
            if len(low_pool) >= n_low:
                batch.extend(low_pool[:n_low])
                low_pool = low_pool[n_low:]
            elif low_pool:
                batch.extend(low_pool)
                low_pool = []
            
            if len(med_pool) >= n_med:
                batch.extend(med_pool[:n_med])
                med_pool = med_pool[n_med:]
            elif med_pool:
                batch.extend(med_pool)
                med_pool = []
            
            if len(high_pool) >= n_high:
                batch.extend(high_pool[:n_high])
                high_pool = high_pool[n_high:]
            elif high_pool:
                batch.extend(high_pool)
                high_pool = []
            
            if not batch:
                break
            
            # Shuffle to avoid ordering effects
            random.shuffle(batch)
            
            yield batch


class CypherBatchPreparer:
    """
    Prepares training batches for Cypher Generator from rollouts.
    
    IMPORTANT: This preparer REBUILDS prompts from stored data rather than
    using stored prompts (which are truncated to 500 chars). This ensures
    the full prompt context is available for training.
    
    Rebuilding requires:
    - schema: KG schema loaded from file
    - experience_buffer: Optional, for learned rules
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
        schema_path: Optional[str] = None,
        experience_buffer: Optional[Any] = None,
    ):
        """
        Initialize batch preparer with prompt rebuilding support.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            schema: KG schema dict (if not provided, must provide schema_path)
            schema_path: Path to schema JSON file
            experience_buffer: Optional ExperienceBuffer for learned rules
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.experience_buffer = experience_buffer
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load schema
        if schema is not None:
            self.schema = schema
        elif schema_path is not None:
            import json
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
        else:
            self.schema = None
            logger.warning("No schema provided - will use truncated prompts from rollouts")
        
        # Initialize prompt builder (lazy import to avoid circular deps)
        self._prompt_builder = None
    
    @property
    def prompt_builder(self):
        """Lazy-load prompt builder."""
        if self._prompt_builder is None:
            import sys
            from pathlib import Path
            # Add utils to path if needed
            utils_path = Path(__file__).parent.parent / "utils"
            if str(utils_path) not in sys.path:
                sys.path.insert(0, str(utils_path))
            from prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder(self.tokenizer)
        return self._prompt_builder
    
    def _build_history_from_steps(
        self,
        steps: List[TrajectoryStep],
        current_step_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct history from previous trajectory steps.
        
        Args:
            steps: All steps in the trajectory
            current_step_idx: Index of current step (0-based)
            
        Returns:
            List of history dicts for prompt builder
        """
        history = []
        for i in range(current_step_idx):
            step = steps[i]
            history.append({
                'query': step.cypher_query,
                'result': {
                    'execution_time_ms': step.execution_time_ms,
                    'num_results': step.num_results,
                    'success': step.success,
                    'has_data': step.has_data,
                }
            })
        return history
    
    def _get_learned_rules(self, question: str) -> List[str]:
        """Get learned rules from experience buffer."""
        if self.experience_buffer is None:
            return []
        
        try:
            # Get relevant patterns
            patterns = self.experience_buffer.get_relevant_patterns(question, top_k=3)
            rules = []
            for p in patterns:
                if isinstance(p, dict):
                    rules.append(p.get('description', str(p)))
                else:
                    rules.append(str(p))
            
            # Get semantic warnings
            warnings = self.experience_buffer.get_semantic_issues_for_prompt(question, top_k=2)
            rules.extend(warnings)
            
            return rules
        except Exception as e:
            logger.warning(f"Error getting learned rules: {e}")
            return []
    
    def _rebuild_prompt(
        self,
        question: str,
        steps: List[TrajectoryStep],
        step_idx: int,
    ) -> str:
        """
        Rebuild the full Cypher prompt for a given step.
        
        Args:
            question: The question being answered
            steps: All trajectory steps
            step_idx: Index of the step to build prompt for (0-based)
            
        Returns:
            Full prompt string
        """
        # Build history from previous steps
        history = self._build_history_from_steps(steps, step_idx)
        
        # Get learned rules
        learned_rules = self._get_learned_rules(question)
        
        # Build prompt (step is 1-indexed)
        prompt = self.prompt_builder.build_cypher_prompt(
            question=question,
            schema=self.schema,
            history=history,
            learned_rules=learned_rules,
            step=step_idx + 1,  # Convert to 1-indexed
        )
        
        return prompt
    
    def prepare_batch(
        self,
        rollouts: List[RolloutEntry],
        use_all_steps: bool = True,
        rebuild_prompts: bool = False,
        use_balanced_sampling: bool = False,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a training batch from rollouts for Cypher Generator.
        
        IMPORTANT: Now defaults to using stored prompts (rebuild_prompts=False)
        since we store full prompts. This ensures exact reproducibility.
        Rebuilding is only needed for old rollouts with truncated prompts.
        
        Args:
            rollouts: List of RolloutEntry objects
            use_all_steps: If True, create samples for all steps; if False, only first step
            rebuild_prompts: If True, rebuild prompts (for old data with truncated prompts).
                            If False (default), use stored prompts.
            use_balanced_sampling: If True, use BalancedBatchSampler to ensure diverse
                                  rewards in batch for better GRPO baseline estimation
            batch_size: Batch size for balanced sampling (required if use_balanced_sampling=True)
            seed: Random seed for balanced sampling. CRITICAL for FSDP: all ranks must
                  use the same seed to get identical batches and avoid NCCL desync!
            
        Returns:
            Dictionary with input_ids, attention_mask, labels, response_mask, rewards
        """
        # Apply balanced sampling if requested
        if use_balanced_sampling:
            if batch_size is None:
                raise ValueError("batch_size must be specified when use_balanced_sampling=True")
            sampler = BalancedBatchSampler(rollouts, reward_attr='cypher_reward')
            rollouts = sampler.sample_balanced_batch(batch_size, seed=seed)
            logger.info(f"Using balanced sampling with batch_size={batch_size}, seed={seed}")
        prompts = []
        responses = []
        rewards = []
        stored_count = 0
        rebuilt_count = 0
        
        # Decide whether to rebuild prompts
        should_rebuild = rebuild_prompts and self.schema is not None
        
        if rebuild_prompts and self.schema is None:
            logger.warning(
                "rebuild_prompts=True but no schema provided. "
                "Using stored prompts."
            )
            should_rebuild = False
        
        for entry in rollouts:
            # Each step in the trajectory is a training sample
            steps_to_use = entry.steps if use_all_steps else entry.steps[:1]
            
            for step_idx, step in enumerate(steps_to_use):
                if not step.response:
                    continue
                
                # Get prompt - prefer stored full prompts, rebuild only if needed
                if should_rebuild:
                    try:
                        prompt = self._rebuild_prompt(
                            question=entry.question,
                            steps=entry.steps,
                            step_idx=step_idx,
                        )
                        rebuilt_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to rebuild prompt: {e}. Using stored prompt.")
                        prompt = step.prompt
                        stored_count += 1
                else:
                    # Use stored prompt (preferred for new rollouts with full prompts)
                    prompt = step.prompt
                    stored_count += 1
                
                if not prompt:
                    continue
                
                prompts.append(prompt)
                responses.append(step.response)
                # Use cypher_reward (falls back to legacy reward for old data)
                rewards.append(entry.cypher_reward)
        
        if not prompts:
            raise ValueError("No valid samples found in rollouts")
        
        logger.info(f"Cypher batch: {stored_count} stored prompts, {rebuilt_count} rebuilt")
        
        return self._tokenize_batch(prompts, responses, rewards)
    
    def _tokenize_batch(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and responses into training batch."""
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_response_mask = []
        
        for prompt, response in zip(prompts, responses):
            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Combine: prompt + response + eos
            full_tokens = prompt_tokens + response_tokens + [self.tokenizer.eos_token_id]
            
            # Truncate if needed
            if len(full_tokens) > self.max_length:
                # Keep as much of prompt as possible, truncate response
                max_response = self.max_length - len(prompt_tokens) - 1
                if max_response < 10:
                    # Prompt too long, truncate it
                    prompt_tokens = prompt_tokens[:self.max_length - 100]
                    max_response = 99
                response_tokens = response_tokens[:max_response]
                full_tokens = prompt_tokens + response_tokens + [self.tokenizer.eos_token_id]
            
            # Create labels (mask prompt with -100)
            labels = [-100] * len(prompt_tokens) + response_tokens + [self.tokenizer.eos_token_id]
            
            # Create response mask (1 for response tokens, 0 for prompt)
            response_mask = [0] * len(prompt_tokens) + [1] * (len(response_tokens) + 1)
            
            # Pad to max_length
            padding_length = self.max_length - len(full_tokens)
            if padding_length > 0:
                full_tokens = full_tokens + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                response_mask = response_mask + [0] * padding_length
            
            attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in full_tokens]
            
            batch_input_ids.append(full_tokens)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_response_mask.append(response_mask)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'response_mask': torch.tensor(batch_response_mask, dtype=torch.float),
            'rewards': torch.tensor(rewards, dtype=torch.float),
        }


class OrchestratorBatchPreparer:
    """Prepares training batches for Orchestrator from rollouts."""
    
    def __init__(self, tokenizer, max_length: int = 4096):
        """
        Initialize batch preparer.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def prepare_question_gen_batch(
        self,
        rollouts: List[RolloutEntry],
        schema_summary: str = "",
        use_stored_prompts: bool = True,
        use_balanced_sampling: bool = False,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare training batch for question generation role.
        
        Uses the pre-computed orch_qgen_reward from rollouts.
        
        IMPORTANT: Prefers stored prompts (orch_qgen_prompt) when available
        to ensure exact reproducibility. Falls back to rebuilding only for
        old rollouts that don't have stored prompts.
        
        Args:
            rollouts: List of RolloutEntry objects
            schema_summary: Schema summary to include in prompt
            use_stored_prompts: If True, prefer stored prompts over rebuilding
            use_balanced_sampling: If True, use BalancedBatchSampler for diverse rewards
            batch_size: Batch size for balanced sampling (required if use_balanced_sampling=True)
            seed: Random seed for balanced sampling. CRITICAL for FSDP: all ranks must
                  use the same seed to get identical batches and avoid NCCL desync!
            
        Returns:
            Training batch dictionary
        """
        # Apply balanced sampling if requested (CRITICAL for bimodal qgen rewards!)
        if use_balanced_sampling:
            if batch_size is None:
                raise ValueError("batch_size must be specified when use_balanced_sampling=True")
            sampler = BalancedBatchSampler(rollouts, reward_attr='orch_qgen_reward')
            rollouts = sampler.sample_balanced_batch(batch_size, seed=seed)
            logger.info(f"Using balanced sampling for qgen with batch_size={batch_size}, seed={seed}")
        
        prompts = []
        responses = []
        rewards = []
        stored_count = 0
        rebuilt_count = 0
        
        for entry in rollouts:
            # Prefer stored prompt (exact prompt used during generation)
            if use_stored_prompts and entry.orch_qgen_prompt:
                prompt = entry.orch_qgen_prompt
                stored_count += 1
            else:
                # Fallback: rebuild prompt (for backwards compatibility)
                prompt = self._build_question_gen_prompt(
                    difficulty=entry.difficulty,
                    schema_summary=schema_summary,
                )
                rebuilt_count += 1
            
            # Response is the generated question
            response = entry.question
            
            # Use pre-computed orchestrator question generation reward
            # Falls back to computed reward for backwards compatibility with old data
            if entry.orch_qgen_reward > 0:
                reward = entry.orch_qgen_reward
            else:
                # Fallback for old data: NEW DESIGN - strongly couple to cypher success
                # If cypher failed, orchestrator gets near-zero (question was unanswerable)
                cypher_reward = entry.cypher_reward if entry.cypher_reward > 0 else entry.reward
                answerability = entry.success_rate > 0.5 and entry.total_results > 0
                
                if answerability and cypher_reward > 0:
                    # Question was answerable - reward based on cypher success
                    data_richness = min(1.0, entry.total_results / 50.0)
                    reward = 0.70 * cypher_reward + 0.15 * data_richness + 0.10 * entry.data_quality_score
                else:
                    # Question was NOT answerable - heavy penalty
                    reward = 0.05 * entry.data_quality_score  # Near-zero
            
            prompts.append(prompt)
            responses.append(response)
            rewards.append(reward)
        
        if not prompts:
            raise ValueError("No valid samples found for question generation")
        
        logger.info(f"Question gen batch: {stored_count} stored prompts, {rebuilt_count} rebuilt")
        
        return self._tokenize_batch(prompts, responses, rewards)
    
    def prepare_synthesis_batch(
        self,
        rollouts: List[RolloutEntry],
        use_stored_prompts: bool = True,
        use_balanced_sampling: bool = False,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare training batch for answer synthesis role.
        
        Uses the pre-computed orch_synth_reward from rollouts.
        
        IMPORTANT: Prefers stored prompts (orch_synth_prompt) when available
        to ensure exact reproducibility. Falls back to rebuilding only for
        old rollouts that don't have stored prompts.
        
        Args:
            rollouts: List of RolloutEntry objects
            use_stored_prompts: If True, prefer stored prompts over rebuilding
            use_balanced_sampling: If True, use BalancedBatchSampler for diverse rewards
            batch_size: Batch size for balanced sampling (required if use_balanced_sampling=True)
            seed: Random seed for balanced sampling. CRITICAL for FSDP: all ranks must
                  use the same seed to get identical batches and avoid NCCL desync!
            
        Returns:
            Training batch dictionary
        """
        # Apply balanced sampling if requested
        if use_balanced_sampling:
            if batch_size is None:
                raise ValueError("batch_size must be specified when use_balanced_sampling=True")
            # Filter rollouts with synthesized answers first
            valid_rollouts = [r for r in rollouts if r.synthesized_answer]
            if not valid_rollouts:
                raise ValueError("No rollouts with synthesized answers available")
            sampler = BalancedBatchSampler(valid_rollouts, reward_attr='orch_synth_reward')
            rollouts = sampler.sample_balanced_batch(batch_size, seed=seed)
            logger.info(f"Using balanced sampling for synthesis with batch_size={batch_size}, seed={seed}")
        
        prompts = []
        responses = []
        rewards = []
        stored_count = 0
        rebuilt_count = 0
        
        for entry in rollouts:
            # Skip entries without synthesized answers
            if not entry.synthesized_answer:
                continue
            
            # Prefer stored prompt (exact prompt used during generation)
            if use_stored_prompts and entry.orch_synth_prompt:
                prompt = entry.orch_synth_prompt
                stored_count += 1
            else:
                # Fallback: rebuild prompt (for backwards compatibility)
                prompt = self._build_synthesis_prompt(entry)
                rebuilt_count += 1
            
            # Response is the synthesized answer
            response = entry.synthesized_answer
            
            # Use pre-computed orchestrator synthesis reward
            # Falls back to computed reward for backwards compatibility with old data
            if entry.orch_synth_reward > 0:
                reward = entry.orch_synth_reward
            else:
                # Fallback for old data without orch_synth_reward
                reward = 0.7 * entry.answer_quality_score + 0.3 * (1.0 - entry.doubt_level)
            
            prompts.append(prompt)
            responses.append(response)
            rewards.append(reward)
        
        if not prompts:
            raise ValueError("No valid samples found for answer synthesis")
        
        logger.info(f"Synthesis batch: {stored_count} stored prompts, {rebuilt_count} rebuilt")
        
        return self._tokenize_batch(prompts, responses, rewards)
    
    def _build_question_gen_prompt(
        self,
        difficulty: str,
        schema_summary: str,
    ) -> str:
        """Build prompt for question generation training."""
        prompt = f"""You are a question generator for a biomedical knowledge graph (PankBase).

Generate a {difficulty} difficulty question that can be answered by querying the knowledge graph.

{f"Schema summary: {schema_summary}" if schema_summary else ""}

Requirements:
- Question should be clear and specific
- Question should be answerable using the knowledge graph
- Difficulty: {difficulty}

Generate a biomedical question:"""
        return prompt
    
    def _build_synthesis_prompt(self, entry: RolloutEntry) -> str:
        """Build prompt for answer synthesis training."""
        # Summarize trajectory data
        trajectory_summary = []
        for i, step in enumerate(entry.steps[:5]):  # Limit to first 5 steps
            if step.cypher_query:
                result_info = f"returned {step.num_results} results" if step.success else "failed"
                trajectory_summary.append(f"Step {i+1}: Query executed, {result_info}")
        
        trajectory_text = "\n".join(trajectory_summary) if trajectory_summary else "No data retrieved"
        
        prompt = f"""You are an answer synthesizer for a biomedical knowledge graph.

Question: {entry.question}

Retrieved Data Summary:
{trajectory_text}

Total results: {entry.total_results}
Data quality: {entry.data_quality_score:.2f}

Based on the retrieved data, synthesize a comprehensive answer to the question:"""
        return prompt
    
    def _tokenize_batch(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and responses into training batch."""
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_response_mask = []
        
        for prompt, response in zip(prompts, responses):
            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Combine: prompt + response + eos
            full_tokens = prompt_tokens + response_tokens + [self.tokenizer.eos_token_id]
            
            # Truncate if needed
            if len(full_tokens) > self.max_length:
                max_response = self.max_length - len(prompt_tokens) - 1
                if max_response < 10:
                    prompt_tokens = prompt_tokens[:self.max_length - 100]
                    max_response = 99
                response_tokens = response_tokens[:max_response]
                full_tokens = prompt_tokens + response_tokens + [self.tokenizer.eos_token_id]
            
            # Create labels (mask prompt with -100)
            labels = [-100] * len(prompt_tokens) + response_tokens + [self.tokenizer.eos_token_id]
            
            # Create response mask
            response_mask = [0] * len(prompt_tokens) + [1] * (len(response_tokens) + 1)
            
            # Pad to max_length
            padding_length = self.max_length - len(full_tokens)
            if padding_length > 0:
                full_tokens = full_tokens + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                response_mask = response_mask + [0] * padding_length
            
            attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in full_tokens]
            
            batch_input_ids.append(full_tokens)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_response_mask.append(response_mask)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'response_mask': torch.tensor(batch_response_mask, dtype=torch.float),
            'rewards': torch.tensor(rewards, dtype=torch.float),
        }

