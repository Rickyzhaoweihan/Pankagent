"""
Utility functions for reward computation.

Provides helper functions for:
- Cypher syntax validation
- Question diversity scoring
- Data utilization computation
- Reward normalization and clipping
"""

import logging
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def validate_cypher(query: str) -> Dict[str, Any]:
    """
    Validate Cypher query syntax (stub implementation).
    
    This is a basic validation that checks for common Cypher patterns.
    A full implementation would use a proper Cypher parser.
    
    Args:
        query: Cypher query string
        
    Returns:
        Dictionary with:
            - score: int (0-100) indicating syntax quality
            - errors: list of error messages
    """
    errors = []
    score = 100
    
    # Convert to uppercase for checking
    query_upper = query.upper()
    
    # Check for basic Cypher keywords
    has_match = 'MATCH' in query_upper
    has_return = 'RETURN' in query_upper
    
    if not has_match and not has_return:
        errors.append("Query must contain MATCH or RETURN")
        score = 0
        return {'score': score, 'errors': errors}
    
    # Check for relationship variable syntax [r:type]
    # Cypher relationships should have variables
    relationship_pattern = r'\[([a-zA-Z_][a-zA-Z0-9_]*)?:([a-zA-Z_][a-zA-Z0-9_]*)\]'
    relationships = re.findall(relationship_pattern, query)
    
    # Check for relationships without variables [:type]
    bad_relationship_pattern = r'\[:([a-zA-Z_][a-zA-Z0-9_]*)\]'
    bad_relationships = re.findall(bad_relationship_pattern, query)
    
    if bad_relationships:
        errors.append(f"Relationships should have variables: {bad_relationships}")
        score -= 20
    
    # Check for WHERE clause if filtering is needed
    # This is a soft check - not having WHERE is not always wrong
    if has_match and 'WHERE' not in query_upper:
        # Just a warning, not a major error
        score -= 5
    
    # Check for LIMIT clause (good practice for performance)
    if 'LIMIT' not in query_upper:
        score -= 5
    
    # Ensure score is in valid range
    score = max(0, min(100, score))
    
    logger.debug(f"Cypher validation: score={score}, errors={errors}")
    
    return {'score': score, 'errors': errors}


def compute_diversity_score(question: str, recent_questions: List[str]) -> float:
    """
    Compute diversity score for a question compared to recent questions.
    
    Uses simple word overlap (Jaccard similarity) to measure diversity.
    Higher score means more diverse (less similar to recent questions).
    
    Args:
        question: New question to evaluate
        recent_questions: List of recent questions
        
    Returns:
        Float in [0.0, 1.0] where 1.0 is most diverse
    """
    if not recent_questions:
        return 1.0  # Maximally diverse if no history
    
    # Tokenize and normalize
    def tokenize(text: str) -> set:
        # Convert to lowercase and split on non-alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'what', 'which', 'who', 'how', 'where', 'when'}
        return set(w for w in words if w not in stopwords and len(w) > 2)
    
    question_words = tokenize(question)
    
    if not question_words:
        return 0.5  # Neutral score if question has no meaningful words
    
    # Compute Jaccard similarity with each recent question
    similarities = []
    for recent_q in recent_questions:
        recent_words = tokenize(recent_q)
        if not recent_words:
            continue
        
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection = len(question_words & recent_words)
        union = len(question_words | recent_words)
        
        if union > 0:
            similarity = intersection / union
            similarities.append(similarity)
    
    if not similarities:
        return 1.0  # Maximally diverse if no valid comparisons
    
    # Diversity is inverse of maximum similarity
    max_similarity = max(similarities)
    diversity = 1.0 - max_similarity
    
    logger.debug(f"Diversity score: {diversity:.3f} (max_similarity={max_similarity:.3f})")
    
    return diversity


def compute_data_utilization(answer: str, trajectory: List[Dict[str, Any]]) -> float:
    """
    Compute how well the synthesized answer utilizes retrieved data.
    
    Measures the overlap between entities mentioned in the answer and
    entities present in the retrieved data.
    
    Args:
        answer: Synthesized natural language answer
        trajectory: List of trajectory steps with retrieved data
        
    Returns:
        Float in [0.0, 1.0] indicating utilization ratio
    """
    if not answer or not trajectory:
        return 0.0
    
    # Extract words from answer (potential entity mentions)
    def extract_entities(text: str) -> set:
        # Find capitalized words and quoted strings (likely entity names)
        # Also find common biomedical patterns
        words = set()
        
        # Capitalized words (potential gene names, etc.)
        capitalized = re.findall(r'\b[A-Z][A-Z0-9]+\b', text)
        words.update(w.lower() for w in capitalized)
        
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        words.update(w.lower() for w in quoted)
        
        # Numbers (could be counts, IDs, etc.)
        numbers = re.findall(r'\b\d+\b', text)
        words.update(numbers)
        
        return words
    
    answer_entities = extract_entities(answer)
    
    if not answer_entities:
        return 0.0  # No entities in answer
    
    # Extract entities from trajectory data
    data_entities = set()
    
    for step in trajectory:
        result = step.get('result', {})
        
        # Convert result to string and extract entities
        result_str = str(result)
        step_entities = extract_entities(result_str)
        data_entities.update(step_entities)
    
    if not data_entities:
        return 0.0  # No entities in data
    
    # Compute overlap: how many answer entities are in the data
    overlap = len(answer_entities & data_entities)
    
    # Utilization: entities used from data / total entities in data
    # We use min to cap at 1.0 (answer might have more entities than data)
    utilization = min(1.0, overlap / len(data_entities))
    
    logger.debug(
        f"Data utilization: {utilization:.3f} "
        f"(overlap={overlap}, answer_entities={len(answer_entities)}, "
        f"data_entities={len(data_entities)})"
    )
    
    return utilization


def normalize_reward(reward: float, mean: float, std: float) -> float:
    """
    Normalize reward using running statistics.
    
    Args:
        reward: Raw reward value
        mean: Running mean of rewards
        std: Running standard deviation of rewards
        
    Returns:
        Normalized reward
    """
    normalized = (reward - mean) / (std + 1e-8)
    return normalized


def clip_reward(reward: float, min_val: float = -10.0, max_val: float = 10.0) -> float:
    """
    Clip reward to prevent extreme values.
    
    Args:
        reward: Reward value to clip
        min_val: Minimum allowed value (default: -10.0)
        max_val: Maximum allowed value (default: 10.0)
        
    Returns:
        Clipped reward
    """
    return float(np.clip(reward, min_val, max_val))

