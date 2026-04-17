#!/usr/bin/env python3
"""
Training Decision Engine.

Analyzes rollouts and training history to decide which models need training.
Uses reward thresholds and plateau detection for adaptive training decisions.

REWARD EXPLANATION:
===================
The rewards measure model performance during rollout collection:

- cypher_reward: How well the Cypher Generator produces correct, efficient queries
  - Components: execution success, result count, query efficiency, semantic quality
  - Range: 0.0 to 1.0, higher is better
  - Threshold: 0.5 base, adapts up to 0.85 max

- orch_qgen_reward: How good are the Orchestrator's generated questions
  - Components: answerability (did Cypher succeed?), diversity, scope adherence
  - Range: 0.0 to 1.0, higher is better
  - A question is "answerable" if: cypher_success > 0 AND data_richness > 0

- orch_synth_reward: How well the Orchestrator synthesizes answers from data
  - Components: answer quality, data utilization
  - Range: 0.0 to 1.0, higher is better

- orch_avg_reward: Simple average of qgen and synth rewards
  - Threshold: 0.4 base, adapts up to 0.75 max

HOW IMPROVEMENT IS MEASURED:
============================
Iteration N:   Collect rollouts -> Measure rewards -> Train (if below threshold)
Iteration N+1: Collect NEW rollouts -> Measure rewards

Improvement = metrics[N+1] - metrics[N] after training at iteration N.
The shell script tracks this and detects convergence when no improvement
occurs for multiple consecutive training runs.

Usage:
    python training_decision.py --rollouts rollouts_iter_N.jsonl --history training_history.json
    python training_decision.py --rollouts rollouts.jsonl --config dynamic_training_config.yaml

Output (JSON to stdout - SINGLE LINE for shell parsing):
    {"train_cypher": true, "train_orchestrator": false, "metrics": {...}, "reasons": {...}}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_rollouts(rollouts_path: str) -> List[Dict[str, Any]]:
    """Load rollouts from JSONL file."""
    rollouts = []
    with open(rollouts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rollouts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rollouts


def load_history(history_path: str) -> List[Dict[str, Any]]:
    """Load training history from JSON file."""
    path = Path(history_path)
    if not path.exists():
        return []
    
    # Handle empty files
    if path.stat().st_size == 0:
        return []
    
    try:
        with open(history_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception):
        return []


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    defaults = {
        # Reward thresholds (below = needs training)
        'cypher_reward_threshold': 0.5,
        'orch_reward_threshold': 0.4,
        
        # Adaptive thresholds (pursue excellence)
        'adaptive_thresholds': True,
        'threshold_increase_step': 0.05,
        'max_cypher_threshold': 0.85,
        'max_orch_threshold': 0.75,
        'min_expected_improvement': 0.02,
        'force_training_interval': 5,  # Train every N iterations regardless
        
        # Plateau detection
        'plateau_window': 3,  # iterations to check
        'plateau_epsilon': 0.01,  # minimum improvement to not be plateau
        
        # Training cooldown (don't retrain if recently trained)
        'cooldown_iterations': 1,
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            defaults.update(config)
    
    return defaults


def compute_adaptive_threshold(
    history: List[Dict[str, Any]],
    base_threshold: float,
    max_threshold: float,
    increase_step: float,
    metric_key: str,
) -> Tuple[float, str]:
    """
    Compute adaptive threshold based on best historical performance.
    
    The threshold increases as the model improves, always pushing for better.
    
    Args:
        history: Training history
        base_threshold: Starting threshold from config
        max_threshold: Maximum allowed threshold
        increase_step: How much to increase when model exceeds threshold
        metric_key: Which metric to track (e.g., 'cypher_reward')
    
    Returns:
        (effective_threshold, reason_string)
    """
    if not history:
        return base_threshold, f"base={base_threshold:.2f}"
    
    # Find best performance in history
    best_reward = 0.0
    for h in history:
        reward = h.get('metrics', {}).get(metric_key, 0.0)
        if reward > best_reward:
            best_reward = reward
    
    # Adaptive threshold = best_performance + increase_step (capped at max)
    # This means: "you achieved X, now try for X + step"
    if best_reward > base_threshold:
        adaptive = min(best_reward + increase_step, max_threshold)
        return adaptive, f"adaptive={adaptive:.2f} (best={best_reward:.2f}+{increase_step})"
    
    return base_threshold, f"base={base_threshold:.2f}"


def should_force_train(
    history: List[Dict[str, Any]],
    model_type: str,
    force_interval: int,
) -> bool:
    """
    Check if model should be force-trained based on interval.
    
    Returns True if the model hasn't been trained in force_interval iterations.
    """
    if force_interval <= 0:
        return False
    
    if len(history) < force_interval:
        return False
    
    # Count iterations since last training
    iterations_since_train = 0
    for h in reversed(history):
        if h.get('trained', {}).get(model_type, False):
            break
        iterations_since_train += 1
    
    return iterations_since_train >= force_interval


def compute_metrics(rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute average reward metrics from rollouts.
    
    Returns:
        Dictionary with cypher_reward, orch_qgen_reward, orch_synth_reward, orch_avg_reward
    """
    if not rollouts:
        return {
            'cypher_reward': 0.0,
            'orch_qgen_reward': 0.0,
            'orch_synth_reward': 0.0,
            'orch_avg_reward': 0.0,
            'num_rollouts': 0,
        }
    
    cypher_rewards = []
    qgen_rewards = []
    synth_rewards = []
    
    for entry in rollouts:
        traj = entry.get('trajectory', {})
        
        # Extract rewards (handle both old and new formats)
        cypher_reward = traj.get('cypher_reward', traj.get('reward', 0.0))
        qgen_reward = traj.get('orch_qgen_reward', 0.0)
        synth_reward = traj.get('orch_synth_reward', 0.0)
        
        if cypher_reward is not None:
            cypher_rewards.append(cypher_reward)
        if qgen_reward is not None:
            qgen_rewards.append(qgen_reward)
        if synth_reward is not None:
            synth_rewards.append(synth_reward)
    
    # Compute averages
    avg_cypher = sum(cypher_rewards) / len(cypher_rewards) if cypher_rewards else 0.0
    avg_qgen = sum(qgen_rewards) / len(qgen_rewards) if qgen_rewards else 0.0
    avg_synth = sum(synth_rewards) / len(synth_rewards) if synth_rewards else 0.0
    avg_orch = (avg_qgen + avg_synth) / 2 if (qgen_rewards or synth_rewards) else 0.0
    
    return {
        'cypher_reward': avg_cypher,
        'orch_qgen_reward': avg_qgen,
        'orch_synth_reward': avg_synth,
        'orch_avg_reward': avg_orch,
        'num_rollouts': len(rollouts),
    }


def detect_plateau(
    history: List[Dict[str, Any]],
    metric_key: str,
    window: int = 3,
    epsilon: float = 0.01,
) -> Tuple[bool, float]:
    """
    Detect if a metric has plateaued.
    
    Args:
        history: List of past iteration metrics
        metric_key: Key to check (e.g., 'cypher_reward')
        window: Number of iterations to check
        epsilon: Minimum improvement to not be plateau
    
    Returns:
        (is_plateau, improvement_rate)
    """
    if len(history) < window:
        return False, 0.0
    
    # Get recent metrics
    recent = history[-window:]
    values = [h.get('metrics', {}).get(metric_key, 0.0) for h in recent]
    
    if not values or len(values) < 2:
        return False, 0.0
    
    # Compute improvement: (newest - oldest) / window
    improvement = values[-1] - values[0]
    improvement_rate = improvement / window
    
    is_plateau = abs(improvement_rate) < epsilon
    
    return is_plateau, improvement_rate


def check_cooldown(
    history: List[Dict[str, Any]],
    model_type: str,  # 'cypher' or 'orchestrator'
    cooldown: int = 1,
) -> bool:
    """
    Check if a model was trained recently (in cooldown period).
    
    Returns:
        True if in cooldown (should NOT train), False if ready to train
    """
    if not history or cooldown <= 0:
        return False
    
    # Check last N iterations
    recent = history[-cooldown:]
    
    for h in recent:
        trained = h.get('trained', {})
        if model_type == 'cypher' and trained.get('cypher', False):
            return True
        if model_type == 'orchestrator' and trained.get('orchestrator', False):
            return True
    
    return False


def make_decision(
    metrics: Dict[str, float],
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make training decisions based on metrics, history, and config.
    
    Decision rules (priority order):
    1. Don't train if in cooldown (just trained recently)
    2. Train if reward < threshold (adaptive or base)
    3. Train if plateau detected AND below max threshold
    4. Train if force_training_interval reached (pursue excellence)
    5. Otherwise, don't train
    
    Returns:
        Decision dictionary with train flags and reasons
    """
    # Base config values
    base_cypher_threshold = config['cypher_reward_threshold']
    base_orch_threshold = config['orch_reward_threshold']
    plateau_window = config['plateau_window']
    plateau_epsilon = config['plateau_epsilon']
    cooldown = config.get('cooldown_iterations', 1)
    
    # Adaptive threshold config
    adaptive_enabled = config.get('adaptive_thresholds', True)
    increase_step = config.get('threshold_increase_step', 0.05)
    max_cypher = config.get('max_cypher_threshold', 0.85)
    max_orch = config.get('max_orch_threshold', 0.75)
    force_interval = config.get('force_training_interval', 5)
    
    cypher_reward = metrics['cypher_reward']
    orch_reward = metrics['orch_avg_reward']
    
    # Compute effective thresholds (adaptive or base)
    if adaptive_enabled:
        cypher_threshold, cypher_thresh_info = compute_adaptive_threshold(
            history, base_cypher_threshold, max_cypher, increase_step, 'cypher_reward'
        )
        orch_threshold, orch_thresh_info = compute_adaptive_threshold(
            history, base_orch_threshold, max_orch, increase_step, 'orch_avg_reward'
        )
    else:
        cypher_threshold = base_cypher_threshold
        orch_threshold = base_orch_threshold
        cypher_thresh_info = f"fixed={cypher_threshold:.2f}"
        orch_thresh_info = f"fixed={orch_threshold:.2f}"
    
    # Check thresholds
    cypher_below_threshold = cypher_reward < cypher_threshold
    orch_below_threshold = orch_reward < orch_threshold
    
    # Check if at max (room for improvement)
    cypher_at_max = cypher_reward >= max_cypher
    orch_at_max = orch_reward >= max_orch
    
    # Check plateaus
    cypher_plateau, cypher_improvement = detect_plateau(
        history, 'cypher_reward', plateau_window, plateau_epsilon
    )
    orch_plateau, orch_improvement = detect_plateau(
        history, 'orch_avg_reward', plateau_window, plateau_epsilon
    )
    
    # Check cooldowns
    cypher_in_cooldown = check_cooldown(history, 'cypher', cooldown)
    orch_in_cooldown = check_cooldown(history, 'orchestrator', cooldown)
    
    # Check force training intervals
    cypher_force_train = should_force_train(history, 'cypher', force_interval)
    orch_force_train = should_force_train(history, 'orchestrator', force_interval)
    
    # Make decisions
    train_cypher = False
    train_orch = False
    cypher_reason = ""
    orch_reason = ""
    
    # Cypher decision (priority order)
    if cypher_in_cooldown:
        train_cypher = False
        cypher_reason = f"In cooldown (trained recently)"
    elif cypher_at_max:
        train_cypher = False
        cypher_reason = f"At maximum ({cypher_reward:.3f} >= {max_cypher})"
    elif cypher_below_threshold:
        train_cypher = True
        cypher_reason = f"Below threshold ({cypher_reward:.3f} < {cypher_threshold:.2f}, {cypher_thresh_info})"
    elif cypher_force_train and not cypher_at_max:
        train_cypher = True
        cypher_reason = f"Force training (interval={force_interval}, pursuing excellence)"
    elif cypher_plateau and len(history) >= plateau_window and not cypher_at_max:
        train_cypher = True
        cypher_reason = f"Plateau detected (improvement={cypher_improvement:.4f}, pushing higher)"
    else:
        train_cypher = False
        cypher_reason = f"Above threshold ({cypher_reward:.3f} >= {cypher_threshold:.2f})"
    
    # Orchestrator decision (priority order)
    if orch_in_cooldown:
        train_orch = False
        orch_reason = f"In cooldown (trained recently)"
    elif orch_at_max:
        train_orch = False
        orch_reason = f"At maximum ({orch_reward:.3f} >= {max_orch})"
    elif orch_below_threshold:
        train_orch = True
        orch_reason = f"Below threshold ({orch_reward:.3f} < {orch_threshold:.2f}, {orch_thresh_info})"
    elif orch_force_train and not orch_at_max:
        train_orch = True
        orch_reason = f"Force training (interval={force_interval}, pursuing excellence)"
    elif orch_plateau and len(history) >= plateau_window and not orch_at_max:
        train_orch = True
        orch_reason = f"Plateau detected (improvement={orch_improvement:.4f}, pushing higher)"
    else:
        train_orch = False
        orch_reason = f"Above threshold ({orch_reward:.3f} >= {orch_threshold:.2f})"
    
    return {
        'train_cypher': train_cypher,
        'train_orchestrator': train_orch,
        'metrics': metrics,
        'thresholds': {
            'cypher_effective': cypher_threshold,
            'cypher_max': max_cypher,
            'orchestrator_effective': orch_threshold,
            'orchestrator_max': max_orch,
        },
        'reasons': {
            'cypher': cypher_reason,
            'orchestrator': orch_reason,
        },
        'plateau': {
            'cypher': cypher_plateau,
            'cypher_improvement': cypher_improvement,
            'orchestrator': orch_plateau,
            'orchestrator_improvement': orch_improvement,
        },
    }


def update_history(
    history_path: str,
    iteration: int,
    metrics: Dict[str, float],
    trained: Dict[str, bool],
    decision: Dict[str, Any],
) -> None:
    """
    Append new entry to training history.
    
    History Structure (each iteration):
    {
        "iteration": 1,
        "timestamp": "...",
        "metrics": {
            "cypher_reward": 0.42,      # Avg reward from collected rollouts (PRE-training)
            "orch_qgen_reward": 0.38,   # Question generation quality
            "orch_synth_reward": 0.51,  # Answer synthesis quality
            "orch_avg_reward": 0.445,   # Average of qgen + synth
            "num_rollouts": 168         # Number of rollouts in this iteration
        },
        "thresholds": {
            "cypher_effective": 0.5,    # Current threshold (adaptive)
            "cypher_max": 0.85,         # Maximum target
            "orchestrator_effective": 0.4,
            "orchestrator_max": 0.75
        },
        "trained": {"cypher": true, "orchestrator": false},
            # ^ DECISION: whether we DECIDED to train (set by this script)
        "actually_trained": {"cypher": true, "orchestrator": false},
            # ^ OUTCOME: whether training ACTUALLY completed (set by shell script later)
        "reasons": {
            "cypher": "Below threshold (0.42 < 0.50, base=0.50)",
            "orchestrator": "Above threshold (0.445 >= 0.40)"
        }
    }
    
    NOTE: The "metrics" are from rollouts collected BEFORE training in this iteration.
    To measure improvement, compare metrics[i] vs metrics[i-1] after i-1 trained.
    """
    history = load_history(history_path)
    
    entry = {
        'iteration': iteration,
        'timestamp': str(Path(history_path).stat().st_mtime) if Path(history_path).exists() else None,
        'metrics': metrics,  # Pre-training metrics from current rollouts
        'thresholds': decision.get('thresholds', {}),
        'trained': trained,  # Decision (what we want to do)
        # 'actually_trained' will be added by shell script after training completes
        'reasons': decision.get('reasons', {}),
    }
    
    history.append(entry)
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Training Decision Engine')
    
    parser.add_argument('--rollouts', '-r', type=str, required=True,
                        help='Path to rollouts JSONL file')
    parser.add_argument('--history', '-H', type=str, default=None,
                        help='Path to training history JSON file')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--iteration', '-i', type=int, default=0,
                        help='Current iteration number')
    parser.add_argument('--update-history', action='store_true',
                        help='Update history file after decision')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output to stderr')
    
    args = parser.parse_args()
    
    # Load data
    rollouts = load_rollouts(args.rollouts)
    history = load_history(args.history) if args.history else []
    config = load_config(args.config)
    
    if args.verbose:
        print(f"Loaded {len(rollouts)} rollouts from {args.rollouts}", file=sys.stderr)
        print(f"History: {len(history)} iterations", file=sys.stderr)
    
    # Compute metrics
    metrics = compute_metrics(rollouts)
    
    if args.verbose:
        print(f"Metrics: cypher={metrics['cypher_reward']:.3f}, "
              f"qgen={metrics['orch_qgen_reward']:.3f}, "
              f"synth={metrics['orch_synth_reward']:.3f}", file=sys.stderr)
    
    # Make decision
    decision = make_decision(metrics, history, config)
    
    if args.verbose:
        thresholds = decision.get('thresholds', {})
        print(f"Thresholds: cypher={thresholds.get('cypher_effective', 0):.2f} "
              f"(max={thresholds.get('cypher_max', 0):.2f}), "
              f"orch={thresholds.get('orchestrator_effective', 0):.2f} "
              f"(max={thresholds.get('orchestrator_max', 0):.2f})", file=sys.stderr)
        print(f"Decision: train_cypher={decision['train_cypher']}, "
              f"train_orch={decision['train_orchestrator']}", file=sys.stderr)
        print(f"  Cypher: {decision['reasons']['cypher']}", file=sys.stderr)
        print(f"  Orch: {decision['reasons']['orchestrator']}", file=sys.stderr)
    
    # Output decision as SINGLE-LINE JSON (critical for shell parsing!)
    # Multi-line JSON breaks shell variable expansion with triple quotes
    print(json.dumps(decision), flush=True)
    sys.stdout.flush()
    
    # Optionally update history
    # NOTE: We record the DECISION here, not the actual training outcome.
    # The shell script should update the history AFTER training completes
    # to record what actually happened.
    if args.update_history and args.history:
        # Record decision (what we WANT to do) - actual outcome updated by shell
        update_history(args.history, args.iteration, metrics, 
                      {'cypher': decision['train_cypher'], 
                       'orchestrator': decision['train_orchestrator']}, 
                      decision)
        if args.verbose:
            print(f"Updated history: {args.history}", file=sys.stderr)


if __name__ == '__main__':
    main()

