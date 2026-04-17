#!/usr/bin/env python3
"""
LLM Training Controller with Deep Think Mode.

Extends the standard LLM training controller with the ability to analyze
actual rollout data when training is struggling. When triggered, the controller
samples rollouts and asks the LLM to perform root cause analysis.

Deep Think Mode Triggers:
    - Reward drop >15% from best
    - Oscillation detected for 3+ iterations
    - Training health is "critical"
    - Manual --deep-think=always flag

Usage:
    python llm_training_controller_deep.py \
        --history /path/to/training_history.json \
        --rollouts-dir /path/to/rollouts/ \
        --provider claude \
        --deep-think auto \
        --output decision.json \
        --analysis-output analysis.md

Outputs:
    - decision.json: Training decision (same format as standard controller)
    - analysis.md: Detailed analysis report with rollout examples
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging

# Import from original controller
from llm_training_controller import (
    LLMTrainingController,
    LLM_CONTROLLER_SYSTEM_PROMPT,
    create_user_prompt,
    get_fallback_decision,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Deep Think System Prompt Extension
# =============================================================================

DEEP_THINK_SYSTEM_PROMPT = """You are an expert RL training controller performing DEEP ANALYSIS of a Text2Cypher system.

## Context
You have access to:
1. Training history (reward curves, training decisions)
2. ACTUAL ROLLOUT SAMPLES from the latest iteration

This is a DEEP THINK session - the training has been struggling and we need ROOT CAUSE ANALYSIS.

## Your Analysis Tasks

### 1. Rollout Pattern Analysis
Examine the sampled rollouts and identify:
- **Cypher Quality Issues**: Syntax errors, wrong patterns, missing clauses
- **Data Retrieval Issues**: Empty results, wrong data types, irrelevant results
- **Question Quality Issues**: Ambiguous questions, unanswerable questions, out-of-scope
- **Answer Quality Issues**: Poor synthesis, missing information, hallucinations

### 2. Compare High vs Low Reward Examples
- What patterns appear in HIGH reward rollouts?
- What patterns appear in LOW reward rollouts?
- Is there a systematic issue?

### 3. Root Cause Identification
Based on your analysis, identify the PRIMARY cause:
- **Cypher Generator Issue**: Model generating bad queries
- **Orchestrator Question Issue**: Questions are too hard or ambiguous
- **Orchestrator Synthesis Issue**: Answer synthesis is poor
- **Data Region Issue**: Queries hitting bad parts of the knowledge graph
- **Curriculum Issue**: Questions too hard for current model capability

### 4. Specific Recommendations
Provide ACTIONABLE recommendations:
- Should we adjust training epochs?
- Should we focus on one model?
- Should we change the question difficulty?
- Should we recommend rollback?
- Should we recommend early stopping?

### 5. Prompt Hints (NEW)
Based on your analysis, provide SHORT hints/warnings to inject into the Cypher Generator and Orchestrator prompts.
These hints will guide the models to avoid specific mistakes you identified.

Guidelines for hints:
- Keep hints SHORT (1 sentence each, max 100 chars)
- Be SPECIFIC (e.g., "GWAS uses (snp)->disease direction" not "fix relationship direction")
- Max 3 hints per agent
- Use severity: "critical" (must fix), "warning" (should avoid), "info" (helpful tip)

## Output Format
RESPOND WITH ONLY VALID JSON:

{
  "deep_analysis": {
    "triggered_by": "<reason for deep think>",
    "rollout_summary": {
      "total_sampled": <int>,
      "high_reward_count": <int>,
      "low_reward_count": <int>,
      "avg_cypher_reward": <float>,
      "avg_orch_reward": <float>,
      "execution_success_rate": <float>,
      "data_retrieval_rate": <float>
    },
    "pattern_analysis": {
      "cypher_syntax_issues_pct": <float>,
      "empty_results_pct": <float>,
      "question_quality_issues_pct": <float>,
      "answer_quality_issues_pct": <float>
    },
    "high_reward_patterns": ["<pattern1>", "<pattern2>"],
    "low_reward_patterns": ["<pattern1>", "<pattern2>"],
    "root_cause": "<primary cause>",
    "root_cause_confidence": <float 0-1>,
    "detailed_findings": "<2-3 sentences of detailed analysis>"
  },
  "analysis": {
    "cypher_best": <float>,
    "cypher_current": <float>,
    "cypher_gap_percent": <float>,
    "cypher_trend": "<improving|stable|declining|oscillating>",
    "cypher_last_trained_iter": <int or null>,
    "orch_best": <float>,
    "orch_current": <float>,
    "orch_gap_percent": <float>,
    "orch_trend": "<improving|stable|declining|oscillating>",
    "orch_last_trained_iter": <int or null>,
    "bottleneck": "<cypher|orchestrator|none|both>",
    "correlation_healthy": <boolean>,
    "iterations_since_best_improved": <int>,
    "oscillation_detected": <boolean>,
    "training_health": "<healthy|warning|critical>"
  },
  "decision": {
    "train_cypher": <boolean>,
    "train_orchestrator": <boolean>,
    "cypher_epochs": <int: 0-10>,
    "orchestrator_epochs": <int: 0-10>,
    "reasoning": "<1-2 sentences based on deep analysis>"
  },
  "early_stopping": {
    "recommend_stop": <boolean>,
    "reason": "<why stop or continue based on rollout analysis>"
  },
  "rollback": {
    "recommend_rollback": <boolean>,
    "target_iteration": <int or null>,
    "reason": "<why rollback based on rollout analysis>"
  },
  "recommendations": [
    "<specific actionable recommendation 1>",
    "<specific actionable recommendation 2>",
    "<specific actionable recommendation 3>"
  ],
  "prompt_hints": {
    "cypher_generator": [
      {"text": "<short hint for cypher queries>", "severity": "<critical|warning|info>"}
    ],
    "orchestrator": {
      "generation": [
        {"text": "<hint for question generation>", "severity": "<critical|warning|info>"}
      ],
      "synthesis": [
        {"text": "<hint for answer synthesis>", "severity": "<critical|warning|info>"}
      ]
    }
  },
  "confidence": <float: 0.0-1.0>
}"""


def create_deep_think_user_prompt(
    training_history: list,
    sampled_rollouts: list,
    trigger_reason: str,
) -> str:
    """Create the user prompt with training history AND rollout samples."""
    # Format the history
    history_str = json.dumps(training_history, indent=2)
    
    # Compute statistics
    stats_summary = ""
    if training_history:
        cypher_rewards = [h['metrics'].get('cypher_reward', 0) for h in training_history]
        orch_rewards = [h['metrics'].get('orch_avg_reward', 0) for h in training_history]
        
        best_cypher = max(cypher_rewards)
        best_orch = max(orch_rewards)
        current_cypher = cypher_rewards[-1]
        current_orch = orch_rewards[-1]
        
        best_cypher_iter = cypher_rewards.index(best_cypher) + 1
        best_orch_iter = orch_rewards.index(best_orch) + 1
        
        stats_summary = f"""
## Pre-computed Statistics
- **Cypher**: Best={best_cypher:.3f} (iter {best_cypher_iter}), Current={current_cypher:.3f}, Gap={((best_cypher-current_cypher)/max(best_cypher, 0.001)*100):.1f}%
- **Orchestrator**: Best={best_orch:.3f} (iter {best_orch_iter}), Current={current_orch:.3f}, Gap={((best_orch-current_orch)/max(best_orch, 0.001)*100):.1f}%
- **Current iteration**: {len(training_history)}
- **Deep Think Triggered By**: {trigger_reason}
"""
    
    # Format rollout samples
    rollouts_str = format_rollouts_for_display(sampled_rollouts)
    
    return f"""## DEEP THINK SESSION
This is a deep analysis session. Training is struggling and we need to understand WHY.

**Trigger Reason**: {trigger_reason}

---

## Training History
```json
{history_str}
```
{stats_summary}

---

## SAMPLED ROLLOUTS FOR ANALYSIS
Below are {len(sampled_rollouts)} rollouts sampled from the latest iteration.
- First group: HIGH reward examples (what's working)
- Second group: LOW reward examples (what's failing)
- Third group: MEDIUM reward examples (edge cases)
- Fourth group: RANDOM samples (diversity)

{rollouts_str}

---

## Analysis Instructions
1. **Examine rollouts carefully**: Look at questions, Cypher queries, results, and rewards
2. **Identify patterns**: What's common in failures? What's common in successes?
3. **Find root cause**: Is it Cypher, Question generation, or Answer synthesis?
4. **Make recommendations**: Be SPECIFIC about what should change
5. **Training decision**: Based on your analysis, should we train, stop, or rollback?

CRITICAL: Your analysis should be based on the ACTUAL ROLLOUT DATA, not just reward numbers.

Respond with ONLY the JSON object, no other text."""


def format_rollouts_for_display(rollouts: list) -> str:
    """Format rollouts for LLM consumption."""
    output_parts = []
    
    for i, r in enumerate(rollouts):
        category = r.get('_category', 'unknown')
        output_parts.append(f"""
### Rollout {i+1} [{category.upper()}] (Cypher Reward: {r.get('cypher_reward', 0):.3f})

**Question**: {r.get('question', 'N/A')}

**Cypher Query**:
```cypher
{r.get('cypher_query', 'N/A')[:600]}
```

**Execution**: {'✓ Success' if r.get('execution_success') else '✗ Failed'}
**Has Data**: {'✓ Yes' if r.get('has_data') else '✗ No (empty results)'}
**Num Results**: {r.get('num_results', 0)}

**Rewards**:
- Cypher: {r.get('cypher_reward', 0):.3f}
- Orch QGen: {r.get('orch_qgen_reward', 0):.3f}
- Orch Synth: {r.get('orch_synth_reward', 0):.3f}

**Answer Snippet**: {r.get('answer_snippet', 'N/A')[:300]}

{f"**Error**: {r.get('error', 'None')}" if r.get('error') else ""}
---""")
    
    return "\n".join(output_parts)


class DeepThinkController(LLMTrainingController):
    """
    Extended LLM Training Controller with Deep Think capabilities.
    
    When training is struggling, this controller samples actual rollouts
    and asks the LLM to perform root cause analysis.
    """
    
    def __init__(
        self,
        provider: str = 'claude',
        api_key: Optional[str] = None,
        timeout: int = 120,  # Longer timeout for deep analysis
        max_retries: int = 3,
        sample_size: int = 20,
    ):
        """
        Initialize the Deep Think controller.
        
        Args:
            provider: 'claude' or 'openai'
            api_key: API key
            timeout: Request timeout (longer for deep analysis)
            max_retries: Number of retries
            sample_size: Number of rollouts to sample
        """
        super().__init__(provider, api_key, timeout, max_retries)
        self.sample_size = sample_size
        logger.info(f"Deep Think Controller initialized (sample_size={sample_size})")
    
    def should_deep_think(self, history: list) -> Tuple[bool, str]:
        """
        Determine if deep think mode should be triggered.
        
        Returns:
            (should_trigger, reason)
        """
        if not history:
            return False, "first_iteration"
        
        # Get reward history
        cypher_rewards = [h['metrics'].get('cypher_reward', 0) for h in history]
        
        # Condition 1: Reward drop >15% from best
        best = max(cypher_rewards)
        current = cypher_rewards[-1]
        if best > 0 and (best - current) / best > 0.15:
            return True, f"reward_drop_{((best-current)/best*100):.1f}pct"
        
        # Condition 2: Oscillation for 3+ iterations
        if len(history) >= 4:
            last_4 = cypher_rewards[-4:]
            diffs = [last_4[i+1] - last_4[i] for i in range(3)]
            # Check for up-down-up or down-up-down pattern
            if ((diffs[0] > 0.03 and diffs[1] < -0.03 and diffs[2] > 0.03) or
                (diffs[0] < -0.03 and diffs[1] > 0.03 and diffs[2] < -0.03)):
                return True, "oscillation_detected"
        
        # Condition 3: Training health is critical
        last_entry = history[-1]
        if last_entry.get('llm_decision', {}).get('training_health') == 'critical':
            return True, "critical_health"
        
        # Condition 4: No improvement for 5+ iterations
        if len(history) >= 5:
            best_iter = cypher_rewards.index(best) + 1
            if len(history) - best_iter >= 5:
                return True, f"no_improvement_{len(history) - best_iter}_iters"
        
        return False, "normal"
    
    def sample_rollouts(
        self,
        rollouts_file: str,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Sample diverse rollouts from a rollouts file.
        
        Sampling strategy:
        - 5 highest reward rollouts (what's working)
        - 5 lowest reward rollouts (what's failing)
        - 5 medium reward rollouts (edge cases)
        - 5 random rollouts (diversity)
        
        Args:
            rollouts_file: Path to rollouts JSONL file
            n: Total number to sample (default 20)
        
        Returns:
            List of formatted rollout dicts
        """
        if not os.path.exists(rollouts_file):
            logger.error(f"Rollouts file not found: {rollouts_file}")
            return []
        
        # Load all rollouts
        rollouts = []
        with open(rollouts_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        rollouts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not rollouts:
            logger.warning("No rollouts found in file")
            return []
        
        logger.info(f"Loaded {len(rollouts)} rollouts from {rollouts_file}")
        
        # Extract rewards for sorting - use orch_synth_reward as primary metric
        # (cypher_reward is often 0 when queries return no data)
        def get_reward_score(r):
            # Try trajectory dict first (where rewards are stored)
            if 'trajectory' in r and isinstance(r['trajectory'], dict):
                traj = r['trajectory']
                # Use orch_synth_reward as it's more indicative of overall success
                synth = traj.get('orch_synth_reward', 0)
                cypher = traj.get('cypher_reward', 0)
                # Also check reward_metadata inside trajectory
                rm = traj.get('reward_metadata', {})
                if rm:
                    synth = rm.get('orch_synth', {}).get('reward', synth)
                    cypher = rm.get('cypher', {}).get('reward', cypher)
                return synth + cypher  # Combined score
            # Fallback to top-level
            if 'cypher_reward' in r:
                return r['cypher_reward']
            if 'reward' in r:
                return r['reward']
            return 0
        
        # Sort by combined reward score
        sorted_rollouts = sorted(rollouts, key=get_reward_score, reverse=True)
        
        # Calculate group sizes
        group_size = n // 4
        remainder = n % 4
        
        samples = []
        
        # High reward (top)
        high_count = group_size + (1 if remainder > 0 else 0)
        for r in sorted_rollouts[:high_count]:
            samples.append(self._format_rollout(r, 'high_reward'))
        
        # Low reward (bottom)
        low_count = group_size + (1 if remainder > 1 else 0)
        for r in sorted_rollouts[-low_count:]:
            samples.append(self._format_rollout(r, 'low_reward'))
        
        # Medium reward (middle)
        mid_start = len(sorted_rollouts) // 2 - group_size // 2
        mid_count = group_size + (1 if remainder > 2 else 0)
        for r in sorted_rollouts[mid_start:mid_start + mid_count]:
            samples.append(self._format_rollout(r, 'medium_reward'))
        
        # Random (from remaining)
        remaining_indices = set(range(len(sorted_rollouts)))
        used_indices = (
            set(range(high_count)) |
            set(range(len(sorted_rollouts) - low_count, len(sorted_rollouts))) |
            set(range(mid_start, mid_start + mid_count))
        )
        remaining_indices -= used_indices
        
        random_count = group_size
        if remaining_indices:
            random_indices = random.sample(
                list(remaining_indices),
                min(random_count, len(remaining_indices))
            )
            for idx in random_indices:
                samples.append(self._format_rollout(sorted_rollouts[idx], 'random'))
        
        logger.info(f"Sampled {len(samples)} rollouts (high: {high_count}, low: {low_count}, mid: {mid_count}, random: {random_count})")
        return samples
    
    def _format_rollout(self, rollout: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Format a rollout for LLM analysis."""
        # Extract fields from various possible structures
        question = rollout.get('question', '')
        
        # Get trajectory data
        trajectory = rollout.get('trajectory', {})
        if isinstance(trajectory, str):
            try:
                trajectory = json.loads(trajectory.replace("'", '"'))
            except:
                trajectory = {}
        
        # Extract cypher query from steps
        cypher_query = ''
        execution_success = True
        has_data = False
        num_results = 0
        error = None
        
        steps = trajectory.get('steps', []) if isinstance(trajectory, dict) else []
        if steps and isinstance(steps, list):
            first_step = steps[0] if steps else {}
            cypher_query = first_step.get('cypher_query', first_step.get('query', ''))
            execution_success = first_step.get('success', True)
            has_data = first_step.get('has_data', False)
            num_results = first_step.get('num_results', 0)
            error = first_step.get('error')
        
        # Get rewards - check both trajectory dict and top-level for compatibility
        traj_dict = rollout.get('trajectory', {}) if isinstance(rollout.get('trajectory'), dict) else {}
        reward_metadata = traj_dict.get('reward_metadata', rollout.get('reward_metadata', {}))
        
        # Try reward_metadata first, then trajectory dict, then top-level
        cypher_reward = reward_metadata.get('cypher', {}).get('reward', 
                        traj_dict.get('cypher_reward', rollout.get('cypher_reward', 0)))
        orch_qgen_reward = reward_metadata.get('orch_qgen', {}).get('reward', 
                           traj_dict.get('orch_qgen_reward', rollout.get('orch_qgen_reward', 0)))
        orch_synth_reward = reward_metadata.get('orch_synth', {}).get('reward', 
                            traj_dict.get('orch_synth_reward', rollout.get('orch_synth_reward', 0)))
        
        # Get answer snippet
        answer = trajectory.get('synthesized_answer', '') if isinstance(trajectory, dict) else ''
        answer_snippet = answer[:300] if answer else ''
        
        return {
            '_category': category,
            'question': question,
            'cypher_query': cypher_query,
            'execution_success': execution_success,
            'has_data': has_data,
            'num_results': num_results,
            'cypher_reward': cypher_reward,
            'orch_qgen_reward': orch_qgen_reward,
            'orch_synth_reward': orch_synth_reward,
            'answer_snippet': answer_snippet,
            'error': error,
        }
    
    def get_deep_analysis(
        self,
        training_history: list,
        rollouts_file: str,
        trigger_reason: str,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Get deep analysis with rollout examination.
        
        Args:
            training_history: Training history
            rollouts_file: Path to latest rollouts file
            trigger_reason: Why deep think was triggered
        
        Returns:
            (decision_dict, analysis_report_markdown)
        """
        # Sample rollouts
        sampled_rollouts = self.sample_rollouts(rollouts_file, self.sample_size)
        
        if not sampled_rollouts:
            logger.warning("No rollouts sampled, falling back to standard analysis")
            decision = super().get_training_decision(training_history)
            return decision, self._generate_fallback_report(decision, trigger_reason)
        
        # Create deep think prompt
        user_prompt = create_deep_think_user_prompt(
            training_history,
            sampled_rollouts,
            trigger_reason,
        )
        
        logger.info(f"Querying {self.provider} for DEEP analysis...")
        
        # Call the LLM with extended system prompt
        if self.provider == 'claude':
            response = self._call_claude(DEEP_THINK_SYSTEM_PROMPT, user_prompt)
        else:
            response = self._call_openai(DEEP_THINK_SYSTEM_PROMPT, user_prompt)
        
        # Parse response
        decision = self._parse_response(response)
        
        # Generate analysis report
        report = self._generate_analysis_report(
            decision,
            sampled_rollouts,
            trigger_reason,
            len(training_history),
        )
        
        return decision, report
    
    def _generate_analysis_report(
        self,
        decision: Dict[str, Any],
        rollouts: List[Dict[str, Any]],
        trigger_reason: str,
        iteration: int,
    ) -> str:
        """Generate markdown analysis report."""
        deep = decision.get('deep_analysis', {})
        analysis = decision.get('analysis', {})
        d = decision.get('decision', {})
        early_stop = decision.get('early_stopping', {})
        rollback = decision.get('rollback', {})
        recommendations = decision.get('recommendations', [])
        
        # Compute rollout statistics
        high_reward = [r for r in rollouts if r['_category'] == 'high_reward']
        low_reward = [r for r in rollouts if r['_category'] == 'low_reward']
        
        report = f"""# Deep Think Analysis - Iteration {iteration}

Generated: {datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Training Status | {analysis.get('training_health', 'unknown')} |
| Deep Think Triggered By | {trigger_reason} |
| Root Cause | {deep.get('root_cause', 'unknown')} |
| Root Cause Confidence | {deep.get('root_cause_confidence', 0):.0%} |
| Confidence | {decision.get('confidence', 0):.0%} |

## Training Metrics

| Model | Best | Current | Gap |
|-------|------|---------|-----|
| Cypher Generator | {analysis.get('cypher_best', 0):.3f} | {analysis.get('cypher_current', 0):.3f} | {analysis.get('cypher_gap_percent', 0):.1f}% |
| Orchestrator | {analysis.get('orch_best', 0):.3f} | {analysis.get('orch_current', 0):.3f} | {analysis.get('orch_gap_percent', 0):.1f}% |

## Rollout Analysis

### Statistics

| Metric | Value |
|--------|-------|
| Total Sampled | {deep.get('rollout_summary', {}).get('total_sampled', len(rollouts))} |
| Execution Success Rate | {deep.get('rollout_summary', {}).get('execution_success_rate', 0):.1%} |
| Data Retrieval Rate | {deep.get('rollout_summary', {}).get('data_retrieval_rate', 0):.1%} |

### Pattern Analysis

| Issue Type | Percentage |
|------------|------------|
| Cypher Syntax Issues | {deep.get('pattern_analysis', {}).get('cypher_syntax_issues_pct', 0):.1f}% |
| Empty Results | {deep.get('pattern_analysis', {}).get('empty_results_pct', 0):.1f}% |
| Question Quality Issues | {deep.get('pattern_analysis', {}).get('question_quality_issues_pct', 0):.1f}% |
| Answer Quality Issues | {deep.get('pattern_analysis', {}).get('answer_quality_issues_pct', 0):.1f}% |

### High-Reward Patterns (What's Working)
"""
        
        for pattern in deep.get('high_reward_patterns', ['N/A']):
            report += f"- {pattern}\n"
        
        report += """
### Low-Reward Patterns (What's Failing)
"""
        for pattern in deep.get('low_reward_patterns', ['N/A']):
            report += f"- {pattern}\n"
        
        report += f"""
## Root Cause Analysis

{deep.get('detailed_findings', 'No detailed findings provided.')}

## Example Rollouts

### High-Reward Examples
"""
        
        for r in high_reward[:2]:
            report += f"""
**Question**: {r['question'][:200]}...

```cypher
{r['cypher_query'][:400]}
```

- Cypher Reward: {r['cypher_reward']:.3f}
- Has Data: {'Yes' if r['has_data'] else 'No'}

---
"""
        
        report += """
### Low-Reward Examples
"""
        
        for r in low_reward[:2]:
            report += f"""
**Question**: {r['question'][:200]}...

```cypher
{r['cypher_query'][:400]}
```

- Cypher Reward: {r['cypher_reward']:.3f}
- Has Data: {'Yes' if r['has_data'] else 'No'}
- Error: {r.get('error', 'None')}

---
"""
        
        report += f"""
## Recommendations

"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## Training Decision

| Decision | Value |
|----------|-------|
| Train Cypher | {'Yes' if d.get('train_cypher') else 'No'} ({d.get('cypher_epochs', 0)} epochs) |
| Train Orchestrator | {'Yes' if d.get('train_orchestrator') else 'No'} ({d.get('orchestrator_epochs', 0)} epochs) |
| Early Stopping | {'Recommended' if early_stop.get('recommend_stop') else 'Not Recommended'} |
| Rollback | {'Recommended (iter {})'.format(rollback.get('target_iteration')) if rollback.get('recommend_rollback') else 'Not Recommended'} |

**Reasoning**: {d.get('reasoning', 'N/A')}

**Early Stop Reason**: {early_stop.get('reason', 'N/A')}

**Rollback Reason**: {rollback.get('reason', 'N/A')}
"""
        
        return report
    
    def _generate_fallback_report(
        self,
        decision: Dict[str, Any],
        trigger_reason: str,
    ) -> str:
        """Generate a minimal report when rollouts couldn't be sampled."""
        return f"""# Deep Think Analysis (Fallback)

Generated: {datetime.now().isoformat()}

## Warning
Could not sample rollouts for deep analysis. Using standard analysis.

**Trigger Reason**: {trigger_reason}

## Decision
```json
{json.dumps(decision, indent=2)}
```
"""


def find_latest_rollouts_file(rollouts_dir: str, iteration: Optional[int] = None) -> Optional[str]:
    """Find the latest rollouts file in a directory."""
    rollouts_path = Path(rollouts_dir)
    
    if not rollouts_path.exists():
        logger.error(f"Rollouts directory not found: {rollouts_dir}")
        return None
    
    # Look for rollouts files
    patterns = ['rollouts_iter_*.jsonl', 'rollouts_*.jsonl', '*.jsonl']
    
    for pattern in patterns:
        files = list(rollouts_path.glob(pattern))
        if files:
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if iteration is not None:
                # Try to find specific iteration
                for f in files:
                    if f'iter_{iteration:03d}' in f.name or f'iter_{iteration}' in f.name:
                        return str(f)
            
            # Return newest
            return str(files[0])
    
    logger.warning(f"No rollouts files found in {rollouts_dir}")
    return None


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='LLM Training Controller with Deep Think Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Path to training_history.json',
    )
    parser.add_argument(
        '--rollouts-dir',
        type=str,
        required=True,
        help='Directory containing rollouts files',
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['claude', 'openai'],
        default='claude',
        help='LLM provider (default: claude)',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)',
    )
    parser.add_argument(
        '--deep-think',
        type=str,
        choices=['auto', 'always', 'never'],
        default='auto',
        help='Deep think mode: auto (trigger on issues), always, or never',
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of rollouts to sample for deep analysis (default: 20)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for decision JSON (default: stdout)',
    )
    parser.add_argument(
        '--analysis-output',
        type=str,
        default=None,
        help='Output file for analysis report markdown',
    )
    parser.add_argument(
        '--fallback-on-error',
        action='store_true',
        help='Use heuristic fallback if LLM fails',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Load history
    history_path = Path(args.history)
    if not history_path.exists():
        logger.error(f"History file not found: {history_path}")
        sys.exit(1)
    
    with open(history_path, 'r') as f:
        training_history = json.load(f)
    
    # Initialize controller
    try:
        controller = DeepThinkController(
            provider=args.provider,
            api_key=args.api_key,
            sample_size=args.sample_size,
        )
    except Exception as e:
        logger.error(f"Failed to initialize controller: {e}")
        if args.fallback_on_error:
            decision = get_fallback_decision(training_history)
            decision_json = json.dumps(decision, indent=2)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(decision_json)
            else:
                print(decision_json)
            sys.exit(0)
        raise
    
    # Check if deep think should be triggered
    should_deep, trigger_reason = controller.should_deep_think(training_history)
    
    # Override based on flag
    if args.deep_think == 'always':
        should_deep = True
        trigger_reason = 'manual_always'
    elif args.deep_think == 'never':
        should_deep = False
        trigger_reason = 'disabled'
    
    logger.info(f"Deep Think: {should_deep} (reason: {trigger_reason})")
    
    try:
        if should_deep:
            # Find latest rollouts file
            rollouts_file = find_latest_rollouts_file(
                args.rollouts_dir,
                iteration=len(training_history),
            )
            
            if rollouts_file:
                logger.info(f"Using rollouts file: {rollouts_file}")
                decision, analysis_report = controller.get_deep_analysis(
                    training_history,
                    rollouts_file,
                    trigger_reason,
                )
                decision['deep_think_mode'] = True
                decision['trigger_reason'] = trigger_reason
            else:
                logger.warning("No rollouts file found, using standard analysis")
                decision = controller.get_training_decision(training_history)
                decision['deep_think_mode'] = False
                analysis_report = None
        else:
            # Standard analysis
            decision = controller.get_training_decision(training_history)
            decision['deep_think_mode'] = False
            analysis_report = None
        
        decision['source'] = f'llm_{args.provider}_deep' if should_deep else f'llm_{args.provider}'
        
    except Exception as e:
        logger.error(f"LLM controller failed: {e}")
        
        if args.fallback_on_error:
            logger.info("Using fallback heuristic decision")
            decision = get_fallback_decision(training_history)
            decision['deep_think_mode'] = False
            analysis_report = None
        else:
            raise
    
    # Output decision
    decision_json = json.dumps(decision, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(decision_json)
        logger.info(f"Decision written to {args.output}")
    else:
        print(decision_json)
    
    # Output analysis report
    if analysis_report and args.analysis_output:
        with open(args.analysis_output, 'w') as f:
            f.write(analysis_report)
        logger.info(f"Analysis report written to {args.analysis_output}")
    
    # Print summary to stderr
    d = decision.get('decision', {})
    deep = decision.get('deep_analysis', {})
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"🧠 LLM TRAINING CONTROLLER {'(DEEP THINK)' if decision.get('deep_think_mode') else ''}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Provider: {decision.get('source', 'unknown')}", file=sys.stderr)
    if decision.get('deep_think_mode'):
        print(f"Trigger: {decision.get('trigger_reason', 'N/A')}", file=sys.stderr)
        print(f"Root Cause: {deep.get('root_cause', 'N/A')}", file=sys.stderr)
    print(f"Train Cypher: {d.get('train_cypher')} ({d.get('cypher_epochs', 0)} epochs)", file=sys.stderr)
    print(f"Train Orchestrator: {d.get('train_orchestrator')} ({d.get('orchestrator_epochs', 0)} epochs)", file=sys.stderr)
    print(f"Reasoning: {d.get('reasoning', 'N/A')}", file=sys.stderr)
    print(f"Confidence: {decision.get('confidence', 'N/A')}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


if __name__ == '__main__':
    main()

