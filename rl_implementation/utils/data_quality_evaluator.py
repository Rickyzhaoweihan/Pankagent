"""
Data Quality Evaluator Utilities.

Helper functions for parsing and processing data quality evaluations
from the Orchestrator agent.
"""

import json
import re
from typing import Any


def parse_data_quality_json(response: str) -> dict[str, Any]:
    """
    Parse data quality evaluation JSON from model response.
    
    Looks for JSON in code blocks or raw JSON in the response.
    
    Args:
        response: Model's response string
        
    Returns:
        Dictionary with evaluation scores and metadata
        
    Raises:
        ValueError: If no valid JSON found or required fields missing
    """
    # Try to find JSON in code blocks first
    json_blocks = re.findall(r'```json\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if json_blocks:
        try:
            data = json.loads(json_blocks[-1].strip())
            return _validate_data_quality_json(data)
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON objects
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    for json_str in reversed(json_objects):  # Try from last to first
        try:
            data = json.loads(json_str)
            if 'data_quality_score' in data:  # Basic validation
                return _validate_data_quality_json(data)
        except json.JSONDecodeError:
            continue
    
    # If no JSON found, return default structure
    raise ValueError("No valid data quality JSON found in response")


def _validate_data_quality_json(data: dict) -> dict[str, Any]:
    """
    Validate and fill in missing fields in data quality evaluation.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Validated and complete data quality dict
    """
    # Required fields with defaults
    validated = {
        'data_quality_score': data.get('data_quality_score', 0.5),
        'relevance_score': data.get('relevance_score', 0.5),
        'completeness_score': data.get('completeness_score', 0.5),
        'consistency_score': data.get('consistency_score', 0.5),
        'trajectory_quality_score': data.get('trajectory_quality_score', 0.5),
        'reasoning': data.get('reasoning', ''),
        'semantic_issues': data.get('semantic_issues', []),
        'problematic_regions': data.get('problematic_regions', []),
        'could_answer_question': data.get('could_answer_question', True),
        'doubt_level': data.get('doubt_level', 0.0)
    }
    
    return validated


def parse_answer_quality_json(response: str) -> dict[str, Any]:
    """
    Parse answer quality evaluation JSON from model response.
    
    Args:
        response: Model's response string
        
    Returns:
        Dictionary with answer quality scores
        
    Raises:
        ValueError: If no valid JSON found
    """
    # Try to find JSON in code blocks first
    json_blocks = re.findall(r'```json\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if json_blocks:
        try:
            data = json.loads(json_blocks[-1].strip())
            return _validate_answer_quality_json(data)
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON objects
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    for json_str in reversed(json_objects):
        try:
            data = json.loads(json_str)
            if 'score' in data:  # Basic validation
                return _validate_answer_quality_json(data)
        except json.JSONDecodeError:
            continue
    
    raise ValueError("No valid answer quality JSON found in response")


def _validate_answer_quality_json(data: dict) -> dict[str, Any]:
    """
    Validate and fill in missing fields in answer quality evaluation.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Validated answer quality dict
    """
    validated = {
        'score': data.get('score', 0.5),
        'correctness': data.get('correctness', 0.5),
        'completeness': data.get('completeness', 0.5),
        'clarity': data.get('clarity', 0.5),
        'accuracy': data.get('accuracy', 0.5),
        'reasoning': data.get('reasoning', ''),
        'strengths': data.get('strengths', ''),
        'weaknesses': data.get('weaknesses', '')
    }
    
    return validated


def compute_doubt_level(relevance: float, completeness: float, consistency: float) -> float:
    """
    Compute doubt level based on data quality scores.
    
    High doubt indicates potential semantic ambiguities or data quality issues.
    
    Args:
        relevance: Relevance score (0-1)
        completeness: Completeness score (0-1)
        consistency: Consistency score (0-1)
        
    Returns:
        Doubt level (0-1), where higher means more doubt
    """
    # Doubt is inversely related to quality scores
    # Weight consistency more heavily as it indicates semantic issues
    avg_quality = (0.3 * relevance + 0.3 * completeness + 0.4 * consistency)
    doubt = 1.0 - avg_quality
    
    # Clip to valid range
    return max(0.0, min(1.0, doubt))


def extract_semantic_issues(evaluation: dict) -> list[dict[str, Any]]:
    """
    Extract semantic ambiguity issues from data quality evaluation.
    
    Args:
        evaluation: Data quality evaluation dict
        
    Returns:
        List of semantic issue dicts with edge_type, description, confidence, etc.
    """
    semantic_issues = evaluation.get('semantic_issues', [])
    
    # Filter for high-confidence issues (>0.6)
    high_confidence_issues = [
        issue for issue in semantic_issues
        if isinstance(issue, dict) and issue.get('confidence', 0.0) > 0.6
    ]
    
    return high_confidence_issues


def extract_problematic_regions(evaluation: dict) -> list[dict[str, Any]]:
    """
    Extract problematic data regions from evaluation.
    
    Args:
        evaluation: Data quality evaluation dict
        
    Returns:
        List of problematic region dicts with node_type, edge_type, issue, severity
    """
    return evaluation.get('problematic_regions', [])


def format_semantic_issues_for_prompt(semantic_issues: list[dict]) -> list[str]:
    """
    Format semantic issues as warning strings for prompts.
    
    Args:
        semantic_issues: List of semantic issue dicts
        
    Returns:
        List of formatted warning strings
    """
    warnings = []
    
    for issue in semantic_issues:
        edge_type = issue.get('edge_type', 'unknown')
        description = issue.get('description', '')
        confidence = issue.get('confidence', 0.0)
        recommendation = issue.get('recommendation', '')
        
        warning = f"⚠️ [{edge_type}]: {description}"
        if recommendation:
            warning += f"\n   Recommendation: {recommendation}"
        warning += f" (confidence: {confidence:.1f})"
        
        warnings.append(warning)
    
    return warnings


def compute_data_utilization(answer: str, trajectory_data: list[dict]) -> float:
    """
    Compute how well the answer utilizes retrieved data.
    
    Checks if entities and numbers from trajectory appear in the answer.
    
    Args:
        answer: Synthesized answer text
        trajectory_data: List of data retrieved from queries
        
    Returns:
        Data utilization score (0-1)
    """
    if not trajectory_data or not answer:
        return 0.0
    
    # Extract key information from trajectory
    entities_in_data = set()
    numbers_in_data = set()
    
    for step_data in trajectory_data:
        # Extract entity names (simplified - looks for capitalized words)
        data_str = str(step_data)
        entities_in_data.update(re.findall(r'\b[A-Z][A-Z0-9]+\b', data_str))
        # Extract numbers
        numbers_in_data.update(re.findall(r'\b\d+\b', data_str))
    
    # Check how many appear in answer
    entities_used = sum(1 for entity in entities_in_data if entity in answer)
    numbers_used = sum(1 for num in numbers_in_data if num in answer)
    
    total_items = len(entities_in_data) + len(numbers_in_data)
    if total_items == 0:
        return 0.5  # No data to utilize
    
    utilization = (entities_used + numbers_used) / total_items
    
    return max(0.0, min(1.0, utilization))

