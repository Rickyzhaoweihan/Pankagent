#!/usr/bin/env python3
"""
LLM Training Controller for Cooperative Multi-Agent RL.

Uses an LLM (Claude or OpenAI) to make intelligent training decisions
based on the training history. The LLM acts as a meta-controller that
analyzes reward curves and decides which model to train and for how many epochs.

Usage:
    python llm_training_controller.py \
        --history /path/to/training_history.json \
        --provider claude \
        --api-key $ANTHROPIC_API_KEY

Supported providers:
    - claude: Uses Claude Opus 4.5 (claude-opus-4-5-20251101) - Most advanced reasoning
    - openai: Uses GPT-4o (gpt-4o) - Latest from OpenAI
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# LLM Training Controller Prompt
# =============================================================================

LLM_CONTROLLER_SYSTEM_PROMPT = """You are an expert RL training controller for a cooperative multi-agent Text2Cypher system.

## System Architecture (from IMPLEMENTATION_OVERVIEW.md)

You manage training for TWO cooperating models in a biomedical knowledge graph QA pipeline:

### 1. Cypher Generator (Qwen2.5-Coder-14B)
- **Purpose**: Multi-step knowledge graph navigator that generates Cypher queries
- **Reward Components**:
  - Answer Quality: 25% (from Orchestrator's evaluation)
  - Data Quality: 25% (from Orchestrator's data evaluation)
  - Trajectory Quality: 20% (search strategy effectiveness)
  - Cypher Correctness: 15% (syntax validity)
  - Retrieval Efficiency: 10% (data per query)
  - Execution Speed: 10% (query performance)

### 2. Orchestrator (Qwen2.5-14B) - Has 4 roles:
- **Question Generation (qgen)**: Creates training questions
  - Reward: Answerability (40%), Difficulty (30%), Diversity (20%), Scope (10%)
- **Data Quality Evaluation**: Assesses retrieved data quality
- **Answer Synthesis (synth)**: Combines data into answers
  - Reward: Answer Quality (70%), Data Utilization (30%)
- **Answer Quality Evaluation**: Evaluates final answer

### Pipeline Flow
Orchestrator generates questions → Cypher retrieves data → Orchestrator evaluates data → Orchestrator synthesizes answer → Orchestrator evaluates answer

## Your Task
Analyze the training history and decide:
1. Should we train the Cypher Generator? How many epochs (1-10)?
2. Should we train the Orchestrator? How many epochs (1-10)?
3. Should we skip training to let models stabilize?
4. Should we recommend EARLY STOPPING or ROLLBACK?

## Critical Training Principles

### 1. Oscillation Detection (HIGHEST PRIORITY)
Watch for HIGH → LOW → HIGH patterns:
- If rewards oscillate by >5% for 3+ iterations without improvement → STOP TRAINING
- Oscillation indicates over-training or learning rate issues
- Recommend: "Consider early stopping - oscillation detected"

### 2. Early Stopping Criteria
RECOMMEND STOPPING when ANY of these occur:
- Best reward not improved for 5+ iterations
- Current reward < 90% of best for 3+ consecutive iterations
- Both models oscillating without improvement
- Total iterations > 10 with no improvement trend

### 3. Rollback Recommendation
RECOMMEND ROLLBACK when:
- Current reward dropped >15% from best
- Sudden collapse (>10% drop in single iteration)
- Include in recommendation: "Rollback to iteration X (best performance)"

### 4. Cooldown Period (STRICT)
After training a model, DO NOT train it again for 2 iterations minimum:
- Check 'actually_trained' field in history
- If trained in last 2 iterations → SKIP that model
- This prevents oscillation and allows consolidation

### 5. Gap-Based Training Decision
| Gap from Best | Action | Epochs |
|--------------|--------|--------|
| < 3% | DON'T TRAIN (risk regression) | 0 |
| 3-5% | Light fine-tuning | 2-3 |
| 5-10% | Moderate training | 4-5 |
| 10-20% | Aggressive training | 6-8 |
| > 20% | Maximum effort | 8-10 |

### 6. Leader-Follower Strategy
When one model has significantly larger gap (>5% difference):
- Train ONLY the bottleneck model
- Don't train the better-performing model (risk destabilizing)

### 7. Never Train Both Aggressively
If training both models:
- Maximum 4 epochs each
- Total epochs ≤ 8 combined
- Prefer training bottleneck more

## Metrics Interpretation
- `cypher_reward`: Query generation quality (0-1). Target: >0.5
- `orch_qgen_reward`: Question generation quality (0-1). Target: >0.5
- `orch_synth_reward`: Answer synthesis quality (0-1). STABLE at ~0.50 is NORMAL
- `orch_avg_reward`: Average of qgen and synth. Target: >0.5

## Pattern Recognition

### Healthy Training Signs
✓ Gradual improvement over iterations
✓ Both rewards moving together (±0.03)
✓ Reward variance decreasing over time
✓ Best reward improving every 3-5 iterations

### Unhealthy Training Signs (TAKE ACTION)
✗ Oscillating rewards (high-low-high pattern)
✗ Rewards diverging (one up, one down)
✗ No improvement for 5+ iterations
✗ Sudden drops >10%
✗ Best reward not improving

## Output Format
RESPOND WITH ONLY VALID JSON. No markdown, no explanation outside JSON.

{
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
    "reasoning": "<1-2 sentences>"
  },
  "early_stopping": {
    "recommend_stop": <boolean>,
    "reason": "<why stop or continue>"
  },
  "rollback": {
    "recommend_rollback": <boolean>,
    "target_iteration": <int or null>,
    "reason": "<why rollback>"
  },
  "confidence": <float: 0.0-1.0>,
  "recommendation": "<actionable suggestion for next steps>"
}"""


def create_user_prompt(training_history: list) -> str:
    """Create the user prompt with training history."""
    # Format the history nicely
    history_str = json.dumps(training_history, indent=2)
    
    # Compute some key statistics to help the LLM
    if training_history:
        cypher_rewards = [h['metrics'].get('cypher_reward', 0) for h in training_history]
        orch_rewards = [h['metrics'].get('orch_avg_reward', 0) for h in training_history]
        
        best_cypher = max(cypher_rewards)
        best_orch = max(orch_rewards)
        current_cypher = cypher_rewards[-1]
        current_orch = orch_rewards[-1]
        
        best_cypher_iter = cypher_rewards.index(best_cypher) + 1
        best_orch_iter = orch_rewards.index(best_orch) + 1
        
        # Find last trained iterations
        cypher_trained_iters = [i+1 for i, h in enumerate(training_history) 
                               if h.get('actually_trained', {}).get('cypher', False)]
        orch_trained_iters = [i+1 for i, h in enumerate(training_history) 
                             if h.get('actually_trained', {}).get('orchestrator', False)]
        
        stats_summary = f"""
## Pre-computed Statistics (for your reference)
- **Cypher**: Best={best_cypher:.3f} (iter {best_cypher_iter}), Current={current_cypher:.3f}, Gap={((best_cypher-current_cypher)/best_cypher*100):.1f}%
- **Orchestrator**: Best={best_orch:.3f} (iter {best_orch_iter}), Current={current_orch:.3f}, Gap={((best_orch-current_orch)/best_orch*100):.1f}%
- **Cypher trained in iterations**: {cypher_trained_iters[-5:] if cypher_trained_iters else 'None'}
- **Orchestrator trained in iterations**: {orch_trained_iters[-5:] if orch_trained_iters else 'None'}
- **Current iteration**: {len(training_history)}
- **Iterations since cypher best**: {len(training_history) - best_cypher_iter}
- **Iterations since orch best**: {len(training_history) - best_orch_iter}
"""
    else:
        stats_summary = "\n## Note: No training history yet - this is the first iteration.\n"
    
    return f"""## Training History
Below is the complete training history. Analyze it carefully.

```json
{history_str}
```
{stats_summary}
## Analysis Instructions
1. **Find best rewards**: Identify peak performance for each model
2. **Calculate gaps**: How far is current from best?
3. **Check trends**: Last 3 iterations - improving, declining, or oscillating?
4. **Check cooldown**: Was model trained in last 2 iterations? (check 'actually_trained')
5. **Detect oscillation**: Is there a high→low→high pattern?
6. **Assess health**: Is training progressing or stuck?
7. **Early stopping check**: Has best not improved for 5+ iterations?
8. **Make decision**: Following ALL Key Principles strictly

CRITICAL REMINDERS:
- If trained recently (last 2 iters), DO NOT train that model
- If gap < 3%, DO NOT train (risk regression)
- If oscillating, recommend STOPPING
- If no improvement for 5+ iters, recommend STOPPING or ROLLBACK

Respond with ONLY the JSON object, no other text."""


class LLMTrainingController:
    """
    LLM-based training controller for cooperative multi-agent RL.
    
    Supports both Claude (Anthropic) and OpenAI APIs.
    """
    
    # Model configurations
    MODELS = {
        'claude': {
            'model': 'claude-opus-4-5-20251101',  # Claude Opus 4.5 - most advanced
            'fallback': 'claude-sonnet-4-20250514',  # Fallback to Claude Sonnet 4
            'api_base': 'https://api.anthropic.com/v1/messages',
            'env_key': 'ANTHROPIC_API_KEY',
        },
        'openai': {
            'model': 'gpt-4o',  # GPT-4o - latest available
            'fallback': 'gpt-4-turbo',  # Fallback to GPT-4 Turbo
            'api_base': 'https://api.openai.com/v1/chat/completions',
            'env_key': 'OPENAI_API_KEY',
        },
    }
    
    def __init__(
        self,
        provider: str = 'claude',
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM training controller.
        
        Args:
            provider: 'claude' or 'openai'
            api_key: API key (if not provided, reads from environment)
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        if provider not in self.MODELS:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'openai'")
        
        self.provider = provider
        self.config = self.MODELS[provider]
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Get API key
        self.api_key = api_key or os.environ.get(self.config['env_key'])
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {self.config['env_key']} environment variable "
                f"or pass api_key parameter."
            )
        
        logger.info(f"LLM Training Controller initialized with provider: {provider}")
        logger.info(f"Model: {self.config['model']}")
    
    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude API."""
        import requests
        
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
        }
        
        data = {
            'model': self.config['model'],
            'max_tokens': 2000,
            'temperature': 0.1,  # Low temperature for consistent decisions
            'system': system_prompt,
            'messages': [
                {'role': 'user', 'content': user_prompt}
            ],
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.config['api_base'],
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['content'][0]['text']
                elif response.status_code == 404 and attempt == 0:
                    # Try fallback model
                    logger.warning(f"Model {self.config['model']} not found, trying fallback")
                    data['model'] = self.config['fallback']
                else:
                    logger.error(f"Claude API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Request failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("Failed to get response from Claude API after retries")
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        data = {
            'model': self.config['model'],
            'max_tokens': 2000,
            'temperature': 0.1,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.config['api_base'],
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 404 and attempt == 0:
                    # Try fallback model
                    logger.warning(f"Model {self.config['model']} not found, trying fallback")
                    data['model'] = self.config['fallback']
                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Request failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        raise RuntimeError("Failed to get response from OpenAI API after retries")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response as JSON."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            raise
    
    def get_training_decision(
        self,
        training_history: list,
    ) -> Dict[str, Any]:
        """
        Get training decision from LLM.
        
        Args:
            training_history: List of training history entries
            
        Returns:
            Decision dict with train_cypher, train_orchestrator, epochs, etc.
        """
        user_prompt = create_user_prompt(training_history)
        
        logger.info(f"Querying {self.provider} for training decision...")
        
        # Call the appropriate API
        if self.provider == 'claude':
            response = self._call_claude(LLM_CONTROLLER_SYSTEM_PROMPT, user_prompt)
        else:
            response = self._call_openai(LLM_CONTROLLER_SYSTEM_PROMPT, user_prompt)
        
        # Parse response
        decision = self._parse_response(response)
        
        return decision
    
    def get_training_decision_from_file(
        self,
        history_path: str,
    ) -> Dict[str, Any]:
        """
        Get training decision from a history file.
        
        Args:
            history_path: Path to training_history.json
            
        Returns:
            Decision dict
        """
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return self.get_training_decision(history)


def get_fallback_decision(training_history: list) -> Dict[str, Any]:
    """
    Fallback heuristic decision when LLM is unavailable.
    
    Uses conservative rules based on reward gaps and cooldown periods.
    """
    if not training_history:
        return {
            'decision': {
                'train_cypher': True,
                'train_orchestrator': True,
                'cypher_epochs': 3,
                'orchestrator_epochs': 3,
                'reasoning': 'First iteration - training both with conservative epochs'
            },
            'analysis': {},
            'early_stopping': {'recommend_stop': False, 'reason': 'Just started'},
            'rollback': {'recommend_rollback': False, 'target_iteration': None, 'reason': 'N/A'},
            'confidence': 0.5,
        }
    
    # Find best rewards and their iterations
    cypher_rewards = [h['metrics'].get('cypher_reward', 0) for h in training_history]
    orch_rewards = [h['metrics'].get('orch_avg_reward', 0) for h in training_history]
    
    best_cypher = max(cypher_rewards)
    best_orch = max(orch_rewards)
    best_cypher_iter = cypher_rewards.index(best_cypher) + 1
    best_orch_iter = orch_rewards.index(best_orch) + 1
    
    # Current rewards
    current = training_history[-1]['metrics']
    current_cypher = current.get('cypher_reward', 0)
    current_orch = current.get('orch_avg_reward', 0)
    current_iter = len(training_history)
    
    # Compute gaps
    cypher_gap = (best_cypher - current_cypher) / max(best_cypher, 0.001)
    orch_gap = (best_orch - current_orch) / max(best_orch, 0.001)
    
    # Check cooldown: was model trained in last 2 iterations?
    cypher_in_cooldown = False
    orch_in_cooldown = False
    for i in range(max(0, len(training_history) - 2), len(training_history)):
        if training_history[i].get('actually_trained', {}).get('cypher', False):
            cypher_in_cooldown = True
        if training_history[i].get('actually_trained', {}).get('orchestrator', False):
            orch_in_cooldown = True
    
    # Iterations since best
    iters_since_cypher_best = current_iter - best_cypher_iter
    iters_since_orch_best = current_iter - best_orch_iter
    
    # Early stopping check
    recommend_stop = False
    stop_reason = "Training progressing normally"
    if iters_since_cypher_best >= 5 and iters_since_orch_best >= 5:
        recommend_stop = True
        stop_reason = f"No improvement for 5+ iterations (best cypher at iter {best_cypher_iter}, best orch at iter {best_orch_iter})"
    
    # Rollback check
    recommend_rollback = False
    rollback_iter = None
    rollback_reason = "N/A"
    if cypher_gap > 0.15 or orch_gap > 0.15:
        recommend_rollback = True
        rollback_iter = max(best_cypher_iter, best_orch_iter)
        rollback_reason = f"Significant regression detected (cypher gap: {cypher_gap*100:.1f}%, orch gap: {orch_gap*100:.1f}%)"
    
    # Decision logic with strict cooldown
    train_cypher = False
    train_orch = False
    cypher_epochs = 0
    orch_epochs = 0
    
    # Only train if NOT in cooldown AND gap > 3%
    if cypher_gap > 0.03 and not cypher_in_cooldown:
        train_cypher = True
        if cypher_gap > 0.20:
            cypher_epochs = 8
        elif cypher_gap > 0.10:
            cypher_epochs = 5
        elif cypher_gap > 0.05:
            cypher_epochs = 3
        else:
            cypher_epochs = 2
    
    if orch_gap > 0.03 and not orch_in_cooldown:
        train_orch = True
        if orch_gap > 0.20:
            orch_epochs = 8
        elif orch_gap > 0.10:
            orch_epochs = 5
        elif orch_gap > 0.05:
            orch_epochs = 3
        else:
            orch_epochs = 2
    
    # If training both, reduce epochs (max 4 each, total 8)
    if train_cypher and train_orch:
        cypher_epochs = min(cypher_epochs, 4)
        orch_epochs = min(orch_epochs, 4)
    
    # If early stopping recommended, don't train
    if recommend_stop:
        train_cypher = False
        train_orch = False
        cypher_epochs = 0
        orch_epochs = 0
    
    reasoning_parts = []
    if cypher_in_cooldown:
        reasoning_parts.append("Cypher in cooldown")
    if orch_in_cooldown:
        reasoning_parts.append("Orch in cooldown")
    reasoning_parts.append(f"Cypher gap: {cypher_gap*100:.1f}%")
    reasoning_parts.append(f"Orch gap: {orch_gap*100:.1f}%")
    reasoning = "; ".join(reasoning_parts)
    
    return {
        'decision': {
            'train_cypher': train_cypher,
            'train_orchestrator': train_orch,
            'cypher_epochs': cypher_epochs,
            'orchestrator_epochs': orch_epochs,
            'reasoning': reasoning,
        },
        'analysis': {
            'cypher_best': best_cypher,
            'cypher_current': current_cypher,
            'cypher_gap_percent': cypher_gap * 100,
            'orch_best': best_orch,
            'orch_current': current_orch,
            'orch_gap_percent': orch_gap * 100,
            'cypher_last_trained_iter': None,
            'orch_last_trained_iter': None,
            'iterations_since_best_improved': max(iters_since_cypher_best, iters_since_orch_best),
            'training_health': 'warning' if recommend_stop else 'healthy',
        },
        'early_stopping': {
            'recommend_stop': recommend_stop,
            'reason': stop_reason,
        },
        'rollback': {
            'recommend_rollback': recommend_rollback,
            'target_iteration': rollback_iter,
            'reason': rollback_reason,
        },
        'confidence': 0.6,
        'source': 'fallback_heuristic',
    }


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='LLM Training Controller for Cooperative Multi-Agent RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Path to training_history.json',
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
        '--output',
        type=str,
        default=None,
        help='Output file for decision JSON (default: stdout)',
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
    
    # Get decision
    try:
        controller = LLMTrainingController(
            provider=args.provider,
            api_key=args.api_key,
        )
        decision = controller.get_training_decision(training_history)
        decision['source'] = f'llm_{args.provider}'
        
    except Exception as e:
        logger.error(f"LLM controller failed: {e}")
        
        if args.fallback_on_error:
            logger.info("Using fallback heuristic decision")
            decision = get_fallback_decision(training_history)
        else:
            raise
    
    # Output
    decision_json = json.dumps(decision, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(decision_json)
        logger.info(f"Decision written to {args.output}")
    else:
        print(decision_json)
    
    # Also print summary to stderr for logging
    d = decision.get('decision', {})
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"🧠 LLM TRAINING CONTROLLER DECISION", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Provider: {decision.get('source', 'unknown')}", file=sys.stderr)
    print(f"Train Cypher: {d.get('train_cypher')} ({d.get('cypher_epochs', 0)} epochs)", file=sys.stderr)
    print(f"Train Orchestrator: {d.get('train_orchestrator')} ({d.get('orchestrator_epochs', 0)} epochs)", file=sys.stderr)
    print(f"Reasoning: {d.get('reasoning', 'N/A')}", file=sys.stderr)
    print(f"Confidence: {decision.get('confidence', 'N/A')}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


if __name__ == '__main__':
    main()

