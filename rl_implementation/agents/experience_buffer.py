"""
Experience Buffer for storing and retrieving learned patterns.

Stores patterns discovered during training:
- Fast Query Patterns: Queries that execute quickly (<200ms)
- High-Yield Patterns: Queries that return substantial data
- Optimal Stopping Patterns: Good heuristics for when to stop
- Bad Data Regions: Node/edge combinations that return confusing data
- Semantic Ambiguities: Edge definitions that differ from common understanding

Based on IMPLEMENTATION_OVERVIEW.md design.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a learned pattern from training."""
    
    # Pattern identification
    pattern_type: str  # 'fast_query', 'high_yield', 'optimal_stopping', 'bad_region', 'semantic_ambiguity'
    pattern_id: str  # Unique ID
    
    # Pattern content
    description: str  # Human-readable description
    structure: str  # Generalized query/pattern structure
    context_keywords: List[str]  # Keywords for relevance matching
    
    # For semantic ambiguities
    edge_type: Optional[str] = None  # The problematic edge
    node_types: Optional[List[str]] = None  # Associated node types
    evidence: Optional[str] = None  # Supporting evidence
    recommendation: Optional[str] = None  # How to handle
    
    # Statistics
    frequency: int = 1  # How often this pattern was observed
    total_reward: float = 0.0  # Sum of rewards for episodes with this pattern
    total_data_quality: float = 0.0  # Sum of data quality scores
    confidence: float = 0.5  # Confidence in this pattern (0-1)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def avg_reward(self) -> float:
        """Average reward for episodes with this pattern."""
        return self.total_reward / max(1, self.frequency)
    
    @property
    def avg_data_quality(self) -> float:
        """Average data quality for episodes with this pattern."""
        return self.total_data_quality / max(1, self.frequency)
    
    @property
    def usefulness_score(self) -> float:
        """
        Compute usefulness score for eviction decisions.
        Higher score = more useful = less likely to evict.
        """
        # Combine frequency, reward, and confidence
        return self.frequency * self.avg_reward * self.confidence
    
    def update(self, reward: float = 0.0, data_quality: float = 0.0):
        """Update pattern statistics with new observation."""
        self.frequency += 1
        self.total_reward += reward
        self.total_data_quality += data_quality
        self.last_updated = datetime.now().isoformat()
        
        # Increase confidence with more observations (asymptotic to 1.0)
        self.confidence = min(0.99, self.confidence + (1 - self.confidence) * 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create Pattern from dictionary."""
        return cls(**data)


class ExperienceBuffer:
    """
    Stores and retrieves learned patterns from training episodes.
    
    Pattern Types (max 25 each, 100 total):
    1. Fast Query Patterns: execution_time < 200ms
    2. High-Yield Patterns: high result count
    3. Optimal Stopping Patterns: good stopping decisions
    4. Bad Data Regions: data_quality < 0.4
    5. Semantic Ambiguities: doubt_level > 0.6
    
    Features:
    - Context-aware pattern retrieval based on question keywords
    - Pattern eviction when capacity reached (lowest usefulness score)
    - Persistence to JSON file
    - Integration with prompt builders
    """
    
    # Pattern type constants
    FAST_QUERY = "fast_query"
    HIGH_YIELD = "high_yield"
    OPTIMAL_STOPPING = "optimal_stopping"
    BAD_REGION = "bad_region"
    SEMANTIC_AMBIGUITY = "semantic_ambiguity"
    
    # Thresholds from IMPLEMENTATION_OVERVIEW.md
    FAST_QUERY_THRESHOLD_MS = 200
    HIGH_YIELD_THRESHOLD = 50  # min results
    HIGH_REWARD_THRESHOLD = 0.7
    LOW_QUALITY_THRESHOLD = 0.4
    HIGH_DOUBT_THRESHOLD = 0.6
    SEMANTIC_CONFIDENCE_THRESHOLD = 0.6
    
    # Capacity limits
    MAX_PATTERNS_PER_TYPE = 25
    MAX_TOTAL_PATTERNS = 100
    
    def __init__(
        self,
        max_patterns: int = 100,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize the experience buffer.
        
        Args:
            max_patterns: Maximum total patterns to store (default: 100)
            persist_path: Optional path to persist patterns (JSON file)
        """
        self.max_patterns = max_patterns
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Storage organized by pattern type
        self.patterns: Dict[str, List[Pattern]] = {
            self.FAST_QUERY: [],
            self.HIGH_YIELD: [],
            self.OPTIMAL_STOPPING: [],
            self.BAD_REGION: [],
            self.SEMANTIC_AMBIGUITY: [],
        }
        
        # Index for fast lookup by pattern_id
        self.pattern_index: Dict[str, Pattern] = {}
        
        # Track recent questions for diversity (rolling window)
        self.recent_questions: List[str] = []
        self.max_recent_questions = 100
        
        # Load from persistence if exists
        if self.persist_path and self.persist_path.exists():
            self.load()
        
        logger.info(f"ExperienceBuffer initialized (max_patterns={max_patterns})")
    
    def __len__(self) -> int:
        """Return total number of patterns."""
        return sum(len(patterns) for patterns in self.patterns.values())
    
    def __repr__(self) -> str:
        """String representation."""
        counts = {k: len(v) for k, v in self.patterns.items()}
        return f"ExperienceBuffer(total={len(self)}, {counts})"
    
    # =========================================================================
    # Pattern Addition
    # =========================================================================
    
    def add_pattern(self, pattern: Pattern) -> bool:
        """
        Add a pattern to the buffer.
        
        If pattern with same ID exists, updates it instead.
        If at capacity, evicts least useful pattern of same type.
        
        Args:
            pattern: Pattern to add
            
        Returns:
            True if added/updated, False if rejected
        """
        pattern_type = pattern.pattern_type
        
        # Check if pattern already exists (by ID)
        if pattern.pattern_id in self.pattern_index:
            existing = self.pattern_index[pattern.pattern_id]
            existing.update(pattern.total_reward, pattern.total_data_quality)
            logger.debug(f"Updated existing pattern: {pattern.pattern_id}")
            return True
        
        # Check capacity for this type
        type_patterns = self.patterns.get(pattern_type, [])
        if len(type_patterns) >= self.MAX_PATTERNS_PER_TYPE:
            # Evict least useful pattern
            evicted = self._evict_pattern(pattern_type)
            if evicted:
                logger.debug(f"Evicted pattern {evicted.pattern_id} to make room")
        
        # Add pattern
        if pattern_type not in self.patterns:
            self.patterns[pattern_type] = []
        
        self.patterns[pattern_type].append(pattern)
        self.pattern_index[pattern.pattern_id] = pattern
        
        logger.debug(f"Added pattern: {pattern.pattern_type}/{pattern.pattern_id}")
        return True
    
    def _evict_pattern(self, pattern_type: str) -> Optional[Pattern]:
        """
        Evict the least useful pattern of a given type.
        
        Args:
            pattern_type: Type of pattern to evict from
            
        Returns:
            Evicted pattern or None
        """
        type_patterns = self.patterns.get(pattern_type, [])
        if not type_patterns:
            return None
        
        # Find pattern with lowest usefulness score
        least_useful = min(type_patterns, key=lambda p: p.usefulness_score)
        
        # Remove from storage
        type_patterns.remove(least_useful)
        del self.pattern_index[least_useful.pattern_id]
        
        return least_useful
    
    def add_from_episode(
        self,
        question: str,
        trajectory: List[Dict[str, Any]],
        reward: float,
        data_quality: float,
        doubt_level: float,
        semantic_issues: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Extract and add patterns from a completed episode.
        
        This is the main entry point for learning from experience.
        
        Args:
            question: The question answered
            trajectory: List of query steps with results
            reward: Final reward for the episode
            data_quality: Data quality score (0-1)
            doubt_level: Doubt level from orchestrator (0-1)
            semantic_issues: Optional list of semantic issues identified
        """
        # Track question for diversity
        self._add_recent_question(question)
        
        # Extract context keywords from question
        context_keywords = self._extract_keywords(question)
        
        # 1. Extract fast query patterns (high reward, fast execution)
        if reward >= self.HIGH_REWARD_THRESHOLD:
            for i, step in enumerate(trajectory):
                exec_time = step.get('execution_time_ms', 1000)
                if exec_time < self.FAST_QUERY_THRESHOLD_MS:
                    pattern = self._create_fast_query_pattern(
                        step, context_keywords, reward, data_quality
                    )
                    self.add_pattern(pattern)
        
        # 2. Extract high-yield patterns (high reward, many results)
        if reward >= self.HIGH_REWARD_THRESHOLD:
            for i, step in enumerate(trajectory):
                num_results = step.get('num_results', 0)
                if num_results >= self.HIGH_YIELD_THRESHOLD:
                    pattern = self._create_high_yield_pattern(
                        step, context_keywords, reward, data_quality
                    )
                    self.add_pattern(pattern)
        
        # 3. Extract optimal stopping patterns (high reward, good trajectory)
        if reward >= self.HIGH_REWARD_THRESHOLD and len(trajectory) > 0:
            pattern = self._create_stopping_pattern(
                trajectory, context_keywords, reward, data_quality
            )
            self.add_pattern(pattern)
        
        # 4. Extract bad data region patterns (low data quality)
        if data_quality < self.LOW_QUALITY_THRESHOLD:
            for step in trajectory:
                pattern = self._create_bad_region_pattern(
                    step, context_keywords, reward, data_quality
                )
                if pattern is not None:  # Skip empty/garbage patterns
                    self.add_pattern(pattern)
        
        # 5. Extract semantic ambiguity patterns (high doubt)
        if doubt_level > self.HIGH_DOUBT_THRESHOLD and semantic_issues:
            for issue in semantic_issues:
                confidence = issue.get('confidence', 0.5)
                if confidence >= self.SEMANTIC_CONFIDENCE_THRESHOLD:
                    pattern = self._create_semantic_pattern(
                        issue, context_keywords, reward, data_quality
                    )
                    self.add_pattern(pattern)
        
        # Persist after adding
        if self.persist_path:
            self.save()
    
    def _add_recent_question(self, question: str):
        """
        Track recent questions for diversity.
        Filters out conversational/garbage responses that aren't real questions.
        """
        if not question or len(question) < 15:
            return
        
        # Filter out conversational/meta responses
        question_lower = question.lower()
        garbage_patterns = [
            'would you like',
            'do you want',
            'shall i',
            'let me know',
            'adjust anything',
            'another question',
            'proceed with',
            'confirm this',
            'further refinement',
            'here\'s a question',
            'here is a question',
            '🚀',
            '🤔',
        ]
        
        for pattern in garbage_patterns:
            if pattern in question_lower:
                logger.debug(f"Skipping garbage question: {question[:50]}...")
                return
        
        # Only add if it looks like a real question
        self.recent_questions.append(question)
        if len(self.recent_questions) > self.max_recent_questions:
            self.recent_questions.pop(0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract context keywords from text for relevance matching.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of lowercase keywords
        """
        # Remove punctuation and split
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter stop words and short words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'which', 'who', 'how', 'when', 'where', 'why',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'as', 'that', 'this', 'it', 'its',
            'can', 'do', 'does', 'did', 'have', 'has', 'had',
            'all', 'any', 'some', 'each', 'every', 'both', 'more', 'most',
        }
        
        keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
        
        # Return unique keywords (preserve order)
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        
        return unique[:20]  # Limit to 20 keywords
    
    def _generalize_query(self, query: str) -> str:
        """
        Generalize a Cypher query by replacing specific values with placeholders.
        
        Args:
            query: Cypher query
            
        Returns:
            Generalized query structure
        """
        # Replace string values with <STRING>
        generalized = re.sub(r"'[^']*'", "'<STRING>'", query)
        generalized = re.sub(r'"[^"]*"', '"<STRING>"', generalized)
        
        # Replace numbers with <NUM>
        generalized = re.sub(r'\b\d+\b', '<NUM>', generalized)
        
        # Normalize whitespace
        generalized = ' '.join(generalized.split())
        
        return generalized
    
    def _create_fast_query_pattern(
        self,
        step: Dict[str, Any],
        context_keywords: List[str],
        reward: float,
        data_quality: float,
    ) -> Pattern:
        """Create a fast query pattern from a step."""
        query = step.get('query', '')
        exec_time = step.get('execution_time_ms', 0)
        
        structure = self._generalize_query(query)
        pattern_id = f"fast_{hash(structure) % 100000:05d}"
        
        # Extract node/edge types from query
        node_types = re.findall(r':(\w+)', query)
        
        return Pattern(
            pattern_type=self.FAST_QUERY,
            pattern_id=pattern_id,
            description=f"Fast query ({exec_time:.0f}ms): {structure[:100]}...",
            structure=structure,
            context_keywords=context_keywords,
            node_types=list(set(node_types)),
            total_reward=reward,
            total_data_quality=data_quality,
            confidence=0.6,
        )
    
    def _create_high_yield_pattern(
        self,
        step: Dict[str, Any],
        context_keywords: List[str],
        reward: float,
        data_quality: float,
    ) -> Pattern:
        """Create a high-yield pattern from a step."""
        query = step.get('query', '')
        num_results = step.get('num_results', 0)
        
        structure = self._generalize_query(query)
        pattern_id = f"yield_{hash(structure) % 100000:05d}"
        
        # Extract node/edge types from query
        node_types = re.findall(r':(\w+)', query)
        
        return Pattern(
            pattern_type=self.HIGH_YIELD,
            pattern_id=pattern_id,
            description=f"High-yield query ({num_results} results): {structure[:100]}...",
            structure=structure,
            context_keywords=context_keywords,
            node_types=list(set(node_types)),
            total_reward=reward,
            total_data_quality=data_quality,
            confidence=0.6,
        )
    
    def _create_stopping_pattern(
        self,
        trajectory: List[Dict[str, Any]],
        context_keywords: List[str],
        reward: float,
        data_quality: float,
    ) -> Pattern:
        """Create an optimal stopping pattern from a trajectory."""
        num_steps = len(trajectory)
        total_results = sum(s.get('num_results', 0) for s in trajectory)
        
        # Heuristic: what triggered good stopping?
        last_step = trajectory[-1]
        last_results = last_step.get('num_results', 0)
        
        description = (
            f"Stopped after {num_steps} steps with {total_results} total results. "
            f"Last step: {last_results} results."
        )
        
        pattern_id = f"stop_{num_steps}_{total_results % 1000:03d}"
        
        return Pattern(
            pattern_type=self.OPTIMAL_STOPPING,
            pattern_id=pattern_id,
            description=description,
            structure=f"steps={num_steps}, total_results={total_results}",
            context_keywords=context_keywords,
            total_reward=reward,
            total_data_quality=data_quality,
            confidence=0.6,
        )
    
    def _create_bad_region_pattern(
        self,
        step: Dict[str, Any],
        context_keywords: List[str],
        reward: float,
        data_quality: float,
    ) -> Optional[Pattern]:
        """Create a bad data region pattern from a step. Returns None if step has garbage data."""
        query = step.get('query', '')
        
        # Skip if query is empty or garbage
        if not query or len(query) < 10:
            return None
        
        # Extract node and edge types
        node_types = re.findall(r':(\w+)', query)
        edge_matches = re.findall(r'\[(\w*):([\w]+)\]', query)
        edge_types = [m[1] for m in edge_matches] if edge_matches else []
        
        if not edge_types:
            edge_types = re.findall(r'\[:(\w+)\]', query)
        
        # Skip if we couldn't extract meaningful types
        if not node_types and not edge_types:
            return None
        
        structure = self._generalize_query(query)
        pattern_id = f"bad_{hash(structure) % 100000:05d}"
        
        # Build a meaningful description
        node_str = ', '.join(node_types[:3]) if node_types else 'unknown nodes'
        edge_str = ', '.join(edge_types[:2]) if edge_types else 'unknown edges'
        description = f"Low quality region: [{edge_str}] between ({node_str})"
        
        return Pattern(
            pattern_type=self.BAD_REGION,
            pattern_id=pattern_id,
            description=description,
            structure=structure,
            context_keywords=context_keywords,
            node_types=list(set(node_types)) if node_types else [],
            edge_type=edge_types[0] if edge_types else None,
            total_reward=reward,
            total_data_quality=data_quality,
            confidence=0.5,
        )
    
    def _create_semantic_pattern(
        self,
        issue: Dict[str, Any],
        context_keywords: List[str],
        reward: float,
        data_quality: float,
    ) -> Pattern:
        """Create a semantic ambiguity pattern from an identified issue."""
        edge_type = issue.get('edge_type', 'unknown')
        node_types = issue.get('node_types', [])
        description = issue.get('description', 'Semantic ambiguity detected')
        evidence = issue.get('evidence', '')
        recommendation = issue.get('recommendation', '')
        confidence = issue.get('confidence', 0.6)
        
        pattern_id = f"sem_{edge_type}_{hash(description) % 10000:04d}"
        
        return Pattern(
            pattern_type=self.SEMANTIC_AMBIGUITY,
            pattern_id=pattern_id,
            description=description,
            structure=f"[:{edge_type}] ambiguity",
            context_keywords=context_keywords,
            edge_type=edge_type,
            node_types=node_types,
            evidence=evidence,
            recommendation=recommendation,
            total_reward=reward,
            total_data_quality=data_quality,
            confidence=confidence,
        )
    
    # =========================================================================
    # Pattern Retrieval
    # =========================================================================
    
    def get_relevant_patterns(
        self,
        question: str,
        top_k: int = 5,
        pattern_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant patterns for a given question.
        
        Uses keyword overlap for relevance scoring.
        
        Args:
            question: The natural language question
            top_k: Number of top patterns to return (default: 5)
            pattern_types: Optional list of pattern types to include
                           (default: all types)
            
        Returns:
            List of relevant patterns as dictionaries
        """
        if pattern_types is None:
            pattern_types = [
                self.FAST_QUERY,
                self.HIGH_YIELD,
                self.OPTIMAL_STOPPING,
            ]
        
        question_keywords = set(self._extract_keywords(question))
        
        if not question_keywords:
            return []
        
        # Score all patterns
        scored_patterns = []
        for ptype in pattern_types:
            for pattern in self.patterns.get(ptype, []):
                relevance = self._compute_relevance(question_keywords, pattern)
                if relevance > 0:
                    scored_patterns.append((relevance, pattern))
        
        # Sort by relevance and return top_k
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        return [p.to_dict() for _, p in scored_patterns[:top_k]]
    
    def _compute_relevance(
        self,
        question_keywords: Set[str],
        pattern: Pattern,
    ) -> float:
        """
        Compute relevance score for a pattern given question keywords.
        
        Combines:
        - Keyword overlap (Jaccard similarity)
        - Pattern frequency
        - Average reward
        - Confidence
        
        Args:
            question_keywords: Set of keywords from question
            pattern: Pattern to score
            
        Returns:
            Relevance score (higher = more relevant)
        """
        pattern_keywords = set(pattern.context_keywords)
        
        if not pattern_keywords:
            return 0.0
        
        # Jaccard similarity for keyword overlap
        intersection = len(question_keywords & pattern_keywords)
        union = len(question_keywords | pattern_keywords)
        keyword_score = intersection / union if union > 0 else 0
        
        # Combine with pattern quality metrics
        # Normalize frequency (log scale)
        freq_score = min(1.0, pattern.frequency / 10.0)
        
        # Combine scores
        relevance = (
            0.5 * keyword_score +
            0.2 * freq_score +
            0.2 * pattern.avg_reward +
            0.1 * pattern.confidence
        )
        
        return relevance
    
    def get_semantic_issues_for_prompt(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        Retrieve semantic ambiguity warnings formatted for prompt.
        
        Args:
            question: The natural language question
            top_k: Number of top semantic issues to return (default: 3)
            
        Returns:
            List of formatted warning strings for prompt
        """
        question_keywords = set(self._extract_keywords(question))
        
        # Score semantic ambiguity patterns
        scored_issues = []
        for pattern in self.patterns.get(self.SEMANTIC_AMBIGUITY, []):
            if pattern.confidence < self.SEMANTIC_CONFIDENCE_THRESHOLD:
                continue
            
            relevance = self._compute_relevance(question_keywords, pattern)
            scored_issues.append((relevance, pattern))
        
        # Sort by relevance
        scored_issues.sort(key=lambda x: x[0], reverse=True)
        
        # Format as warning strings
        warnings = []
        for _, pattern in scored_issues[:top_k]:
            edge = pattern.edge_type or "unknown"
            desc = pattern.description
            rec = pattern.recommendation or "Use with caution"
            
            warning = f"⚠️ [:{edge}]: {desc}. {rec}"
            warnings.append(warning)
        
        return warnings
    
    def get_bad_region_warnings(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        Retrieve bad data region warnings for prompt.
        
        Args:
            question: The natural language question
            top_k: Number of warnings to return
            
        Returns:
            List of formatted warning strings
        """
        question_keywords = set(self._extract_keywords(question))
        
        scored_warnings = []
        for pattern in self.patterns.get(self.BAD_REGION, []):
            relevance = self._compute_relevance(question_keywords, pattern)
            if relevance > 0:
                scored_warnings.append((relevance, pattern))
        
        scored_warnings.sort(key=lambda x: x[0], reverse=True)
        
        warnings = []
        for _, pattern in scored_warnings[:top_k]:
            warnings.append(f"⚠️ Avoid: {pattern.description}")
        
        return warnings
    
    def get_scope_constraints(self) -> Dict[str, Any]:
        """
        Get scope constraints for question generation.
        
        Builds lists of:
        - Allowed node/edge types (from good patterns)
        - Prohibited regions (from bad patterns)
        
        Returns:
            Dictionary with allowed/prohibited node and edge types
        """
        # Collect node types from good patterns
        good_node_types = set()
        good_edge_types = set()
        
        for ptype in [self.FAST_QUERY, self.HIGH_YIELD]:
            for pattern in self.patterns.get(ptype, []):
                if pattern.avg_reward >= self.HIGH_REWARD_THRESHOLD:
                    if pattern.node_types:
                        good_node_types.update(pattern.node_types)
                    if pattern.edge_type:
                        good_edge_types.add(pattern.edge_type)
        
        # Collect bad regions to avoid
        avoid_regions = []
        for pattern in self.patterns.get(self.BAD_REGION, []):
            if pattern.confidence >= 0.5:
                region = {
                    'node_types': pattern.node_types or [],
                    'edge_type': pattern.edge_type,
                    'description': pattern.description,
                    'severity': 'high' if pattern.avg_data_quality < 0.3 else 'medium',
                }
                avoid_regions.append(region)
        
        # Collect semantic ambiguity warnings
        semantic_warnings = []
        for pattern in self.patterns.get(self.SEMANTIC_AMBIGUITY, []):
            if pattern.confidence >= self.SEMANTIC_CONFIDENCE_THRESHOLD:
                semantic_warnings.append({
                    'edge_type': pattern.edge_type,
                    'description': pattern.description,
                    'recommendation': pattern.recommendation,
                })
        
        return {
            'allowed_node_types': list(good_node_types),
            'allowed_edge_types': list(good_edge_types),
            'avoid_regions': avoid_regions[:5],  # Top 5 worst regions
            'semantic_warnings': semantic_warnings[:3],  # Top 3 semantic issues
        }
    
    def get_recent_questions(self, n: int = 20) -> List[str]:
        """
        Get recent questions for diversity in generation.
        
        Args:
            n: Number of recent questions to return
            
        Returns:
            List of recent questions
        """
        return self.recent_questions[-n:]
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self, path: Optional[str] = None):
        """
        Save patterns to JSON file.
        
        Args:
            path: Optional path (uses self.persist_path if not provided)
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified, skipping save")
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'patterns': {
                ptype: [p.to_dict() for p in patterns]
                for ptype, patterns in self.patterns.items()
            },
            'recent_questions': self.recent_questions[-50:],  # Save last 50 (slice)
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self)} patterns to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """
        Load patterns from JSON file.
        
        Args:
            path: Optional path (uses self.persist_path if not provided)
        """
        load_path = Path(path) if path else self.persist_path
        if not load_path or not load_path.exists():
            logger.warning(f"No patterns file at {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Load patterns
        self.patterns = {}
        self.pattern_index = {}
        
        for ptype, pattern_dicts in data.get('patterns', {}).items():
            self.patterns[ptype] = []
            for pdict in pattern_dicts:
                pattern = Pattern.from_dict(pdict)
                self.patterns[ptype].append(pattern)
                self.pattern_index[pattern.pattern_id] = pattern
        
        # Load recent questions
        self.recent_questions = data.get('recent_questions', [])
        
        logger.info(f"Loaded {len(self)} patterns from {load_path}")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns."""
        stats = {
            'total_patterns': len(self),
            'by_type': {k: len(v) for k, v in self.patterns.items()},
            'recent_questions_count': len(self.recent_questions),
        }
        
        # Compute averages
        for ptype, patterns in self.patterns.items():
            if patterns:
                stats[f'{ptype}_avg_reward'] = sum(p.avg_reward for p in patterns) / len(patterns)
                stats[f'{ptype}_avg_confidence'] = sum(p.confidence for p in patterns) / len(patterns)
        
        return stats
    
    def clear(self):
        """Clear all patterns."""
        self.patterns = {
            self.FAST_QUERY: [],
            self.HIGH_YIELD: [],
            self.OPTIMAL_STOPPING: [],
            self.BAD_REGION: [],
            self.SEMANTIC_AMBIGUITY: [],
        }
        self.pattern_index = {}
        self.recent_questions = []
        
        logger.info("ExperienceBuffer cleared")
