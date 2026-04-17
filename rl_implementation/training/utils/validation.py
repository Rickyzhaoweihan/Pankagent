"""
Validation utilities for drift detection and model evaluation.

Provides:
- Fixed validation set evaluation
- Metric computation
- Train/val comparison for drift detection
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def load_validation_set(filepath: str, num_questions: int = 100) -> List[str]:
    """
    Load fixed validation questions.
    
    Args:
        filepath: Path to validation questions file (JSON)
        num_questions: Number of questions to load
        
    Returns:
        List of validation questions
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Validation file not found: {filepath}")
        # Return empty list if file doesn't exist
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            questions = data[:num_questions]
        elif isinstance(data, dict):
            # Assume format like {'questions': [...]}
            questions = data.get('questions', [])[:num_questions]
        else:
            logger.error(f"Unexpected validation data format: {type(data)}")
            return []
        
        logger.info(f"Loaded {len(questions)} validation questions from {filepath}")
        return questions
        
    except Exception as e:
        logger.error(f"Failed to load validation set: {e}", exc_info=True)
        return []


def validate_on_fixed_set(
    cypher_gen: Any,
    orchestrator_eval: Any,
    questions: List[str],
    env: Any
) -> Dict[str, Any]:
    """
    Run validation on a fixed set of questions.
    
    Args:
        cypher_gen: Cypher Generator agent
        orchestrator_eval: EMA Orchestrator evaluator
        questions: List of validation questions
        env: Graph Reasoning Environment
        
    Returns:
        Dictionary with validation metrics
    """
    if not questions:
        logger.warning("No validation questions provided")
        return {
            'avg_answer_quality': 0.0,
            'avg_data_quality': 0.0,
            'avg_trajectory_length': 0.0,
            'success_rate': 0.0,
            'n_questions': 0
        }
    
    logger.info(f"Running validation on {len(questions)} questions")
    
    results = []
    
    for i, question in enumerate(questions):
        try:
            # Set task and reset environment
            env.task = {'question': question}
            obs, info = env.reset()
            cypher_gen.reset()
            
            # Run episode
            trajectory = []
            done = False
            step = 0
            max_steps = 5
            
            # Ensure observation has question
            if 'question' not in obs:
                obs['question'] = question
            
            while not done and step < max_steps:
                # Agent generates query
                cypher_gen.update_from_env(obs, reward=0.0, done=False, info={})
                
                # Get model response (stub - in real training this would call the model)
                # For validation, we just track the structure
                response = "DONE"  # Placeholder
                action = cypher_gen.update_from_model(response)
                
                # Execute in environment
                obs, reward, done, info = env.step(action.action)
                
                trajectory.append({
                    'query': action.action,
                    'result': obs.get('previous_result', {})
                })
                
                step += 1
                
                if action.action.upper() == "DONE":
                    done = True
            
            # Evaluate data quality (using EMA evaluator)
            # Note: In full implementation, this would call orchestrator_eval
            # For now, we compute basic metrics from trajectory
            data_quality_score = _compute_data_quality_from_trajectory(trajectory)
            
            # Evaluate answer quality
            # Note: In full implementation, this would synthesize answer and evaluate
            answer_quality_score = _estimate_answer_quality(trajectory, data_quality_score)
            
            results.append({
                'question': question,
                'trajectory': trajectory,
                'data_quality_score': data_quality_score,
                'answer_quality_score': answer_quality_score,
                'trajectory_length': len(trajectory),
                'success': answer_quality_score > 0.7
            })
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Validated {i + 1}/{len(questions)} questions")
                
        except Exception as e:
            logger.error(f"Validation failed for question {i}: {e}")
            results.append({
                'question': question,
                'trajectory': [],
                'data_quality_score': 0.0,
                'answer_quality_score': 0.0,
                'trajectory_length': 0,
                'success': False,
                'error': str(e)
            })
    
    # Compute aggregate metrics
    metrics = compute_validation_metrics(results)
    
    logger.info(
        f"Validation complete: "
        f"answer_quality={metrics['avg_answer_quality']:.3f}, "
        f"data_quality={metrics['avg_data_quality']:.3f}, "
        f"success_rate={metrics['success_rate']:.3f}"
    )
    
    return metrics


def compute_validation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from validation results.
    
    Args:
        results: List of validation result dictionaries
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not results:
        return {
            'avg_answer_quality': 0.0,
            'avg_data_quality': 0.0,
            'avg_trajectory_length': 0.0,
            'success_rate': 0.0,
            'n_questions': 0
        }
    
    # Extract scores
    answer_qualities = [r['answer_quality_score'] for r in results]
    data_qualities = [r['data_quality_score'] for r in results]
    trajectory_lengths = [r['trajectory_length'] for r in results]
    successes = [r['success'] for r in results]
    
    # Compute averages
    metrics = {
        'avg_answer_quality': float(np.mean(answer_qualities)),
        'avg_data_quality': float(np.mean(data_qualities)),
        'avg_trajectory_length': float(np.mean(trajectory_lengths)),
        'success_rate': float(np.mean(successes)),
        'n_questions': len(results),
        'std_answer_quality': float(np.std(answer_qualities)),
        'std_data_quality': float(np.std(data_qualities)),
        'min_answer_quality': float(np.min(answer_qualities)),
        'max_answer_quality': float(np.max(answer_qualities))
    }
    
    return metrics


def compare_train_val_metrics(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Compare training and validation metrics to detect drift.
    
    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics
        threshold: Maximum acceptable gap
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'drift_detected': False,
        'gaps': {},
        'warnings': []
    }
    
    # Compare answer quality
    if 'avg_answer_quality' in train_metrics and 'avg_answer_quality' in val_metrics:
        train_aq = train_metrics['avg_answer_quality']
        val_aq = val_metrics['avg_answer_quality']
        gap = train_aq - val_aq
        
        comparison['gaps']['answer_quality'] = gap
        
        if gap > threshold:
            comparison['drift_detected'] = True
            comparison['warnings'].append(
                f"Answer quality gap: {gap:.3f} > {threshold} "
                f"(train={train_aq:.3f}, val={val_aq:.3f})"
            )
    
    # Compare data quality
    if 'avg_data_quality' in train_metrics and 'avg_data_quality' in val_metrics:
        train_dq = train_metrics['avg_data_quality']
        val_dq = val_metrics['avg_data_quality']
        gap = train_dq - val_dq
        
        comparison['gaps']['data_quality'] = gap
        
        if gap > threshold:
            comparison['drift_detected'] = True
            comparison['warnings'].append(
                f"Data quality gap: {gap:.3f} > {threshold} "
                f"(train={train_dq:.3f}, val={val_dq:.3f})"
            )
    
    # Compare success rates
    if 'success_rate' in train_metrics and 'success_rate' in val_metrics:
        train_sr = train_metrics['success_rate']
        val_sr = val_metrics['success_rate']
        gap = train_sr - val_sr
        
        comparison['gaps']['success_rate'] = gap
        
        if gap > threshold:
            comparison['drift_detected'] = True
            comparison['warnings'].append(
                f"Success rate gap: {gap:.3f} > {threshold} "
                f"(train={train_sr:.3f}, val={val_sr:.3f})"
            )
    
    if comparison['drift_detected']:
        logger.warning(f"Drift detected: {len(comparison['warnings'])} warnings")
        for warning in comparison['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.debug("No drift detected in train/val comparison")
    
    return comparison


def _compute_data_quality_from_trajectory(trajectory: List[Dict[str, Any]]) -> float:
    """
    Compute data quality score from trajectory (simplified).
    
    Args:
        trajectory: List of query/result pairs
        
    Returns:
        Data quality score (0-1)
    """
    if not trajectory:
        return 0.0
    
    # Count successful queries with data
    successful_queries = sum(
        1 for step in trajectory
        if step.get('result', {}).get('success', False) and
           step.get('result', {}).get('has_data', False)
    )
    
    # Basic quality score based on success rate
    quality = successful_queries / len(trajectory)
    
    return float(quality)


def _estimate_answer_quality(trajectory: List[Dict[str, Any]], data_quality: float) -> float:
    """
    Estimate answer quality from trajectory and data quality (simplified).
    
    Args:
        trajectory: List of query/result pairs
        data_quality: Data quality score
        
    Returns:
        Estimated answer quality score (0-1)
    """
    if not trajectory:
        return 0.0
    
    # Simple heuristic: answer quality correlates with data quality
    # and trajectory completeness
    trajectory_completeness = min(len(trajectory) / 3.0, 1.0)
    
    # Weighted combination
    answer_quality = 0.7 * data_quality + 0.3 * trajectory_completeness
    
    return float(answer_quality)


def save_validation_results(results: List[Dict[str, Any]], filepath: str):
    """
    Save validation results to file.
    
    Args:
        results: List of validation result dictionaries
        filepath: Path to save results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved validation results to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save validation results: {e}")


def load_validation_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Load validation results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        List of validation result dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Validation results file not found: {filepath}")
        return []
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded validation results from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Failed to load validation results: {e}")
        return []

