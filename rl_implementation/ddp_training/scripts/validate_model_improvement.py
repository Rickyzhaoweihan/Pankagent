#!/usr/bin/env python3
"""
Validate Model Improvement After Training.

Re-runs a sample of questions from previous rollouts through the current (trained) models
and compares rewards to measure actual improvement.

Usage:
    python validate_model_improvement.py \
        --rollouts rollouts_iter_001.jsonl \
        --orchestrator-url http://localhost:8001/v1 \
        --cypher-url http://localhost:8002/v1 \
        --sample-size 20

Output (JSON):
    {
        "num_samples": 20,
        "original_cypher_reward": 0.42,
        "new_cypher_reward": 0.55,
        "cypher_improvement": 0.13,
        "original_orch_reward": 0.38,
        "new_orch_reward": 0.45,
        "orch_improvement": 0.07,
        "success": true
    }
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DDP_DIR = SCRIPT_DIR.parent
RL_IMPL_DIR = DDP_DIR.parent
PROJECT_DIR = RL_IMPL_DIR.parent

sys.path.insert(0, str(RL_IMPL_DIR))
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rollouts_sample(rollouts_path: str, sample_size: int = 20) -> List[Dict[str, Any]]:
    """Load a random sample of rollouts from JSONL file."""
    all_rollouts = []
    
    with open(rollouts_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    all_rollouts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if len(all_rollouts) <= sample_size:
        return all_rollouts
    
    return random.sample(all_rollouts, sample_size)


def run_cypher_for_question(
    question: str,
    cypher_url: str,
    schema_path: Optional[str] = None,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """
    Run Cypher generation for a single question.
    
    Returns dict with success, reward, etc.
    """
    try:
        from openai import OpenAI
        from rl_implementation.environments.neo4j_executor import Neo4jExecutor
        from rl_implementation.utils.prompt_builder import PromptBuilder
        from rl_implementation.utils.cypher_auto_fix import auto_fix_cypher
        
        # Initialize
        client = OpenAI(base_url=cypher_url, api_key="dummy")
        executor = Neo4jExecutor()
        
        # Load schema
        schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path) as f:
                schema = json.load(f)
        
        # Build prompt for first step
        prompt_builder = PromptBuilder(tokenizer=None)
        prompt = prompt_builder.build_cypher_prompt(
            question=question,
            schema=schema,
            history=[],
            step=1,
        )
        
        # Generate Cypher
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        
        cypher_text = response.choices[0].message.content
        
        # Extract and fix Cypher
        import re
        cypher_match = re.search(r'```(?:cypher)?\s*(.*?)```', cypher_text, re.DOTALL)
        if cypher_match:
            cypher = cypher_match.group(1).strip()
        else:
            cypher = cypher_text.strip()
        
        cypher = auto_fix_cypher(cypher)
        
        # Execute
        result = executor.execute(cypher)
        
        success = result.get('success', False)
        has_data = result.get('has_data', False)
        num_results = result.get('num_results', 0)
        
        # Compute simple reward
        if success and has_data and num_results > 0:
            data_richness = min(1.0, num_results / 50.0)
            reward = 0.5 + 0.3 * data_richness + 0.2  # Base success + data bonus
        elif success:
            reward = 0.3  # Executed but no data
        else:
            reward = 0.05  # Failed
        
        return {
            'success': success,
            'has_data': has_data,
            'num_results': num_results,
            'reward': reward,
        }
        
    except Exception as e:
        logger.warning(f"Error running Cypher for question: {e}")
        return {
            'success': False,
            'has_data': False,
            'num_results': 0,
            'reward': 0.0,
            'error': str(e),
        }


def validate_improvement(
    rollouts: List[Dict[str, Any]],
    cypher_url: str,
    orchestrator_url: str,
    schema_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate model improvement by re-running questions.
    
    Args:
        rollouts: Sample of original rollouts
        cypher_url: vLLM Cypher generator URL
        orchestrator_url: vLLM Orchestrator URL (not used currently)
        schema_path: Path to KG schema
    
    Returns:
        Validation results with improvement metrics
    """
    original_cypher_rewards = []
    new_cypher_rewards = []
    
    logger.info(f"Validating {len(rollouts)} questions...")
    
    for i, rollout in enumerate(rollouts):
        question = rollout.get('question', '')
        traj = rollout.get('trajectory', {})
        original_reward = traj.get('cypher_reward', traj.get('reward', 0.0))
        
        if not question:
            continue
        
        # Run with current model
        result = run_cypher_for_question(
            question=question,
            cypher_url=cypher_url,
            schema_path=schema_path,
        )
        
        original_cypher_rewards.append(original_reward)
        new_cypher_rewards.append(result['reward'])
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Validated {i + 1}/{len(rollouts)} questions")
    
    # Compute averages
    if original_cypher_rewards:
        avg_original = sum(original_cypher_rewards) / len(original_cypher_rewards)
        avg_new = sum(new_cypher_rewards) / len(new_cypher_rewards)
        improvement = avg_new - avg_original
        
        return {
            'num_samples': len(original_cypher_rewards),
            'original_cypher_reward': round(avg_original, 4),
            'new_cypher_reward': round(avg_new, 4),
            'cypher_improvement': round(improvement, 4),
            'improved_count': sum(1 for o, n in zip(original_cypher_rewards, new_cypher_rewards) if n > o),
            'success': True,
        }
    else:
        return {
            'num_samples': 0,
            'success': False,
            'error': 'No valid questions to validate',
        }


def main():
    parser = argparse.ArgumentParser(description='Validate model improvement after training')
    
    parser.add_argument('--rollouts', '-r', required=True,
                        help='Path to rollouts JSONL file')
    parser.add_argument('--orchestrator-url', default='http://localhost:8001/v1',
                        help='Orchestrator vLLM URL')
    parser.add_argument('--cypher-url', default='http://localhost:8002/v1',
                        help='Cypher Generator vLLM URL')
    parser.add_argument('--schema-path', default=None,
                        help='Path to KG schema JSON')
    parser.add_argument('--sample-size', type=int, default=20,
                        help='Number of questions to validate')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load sample
    logger.info(f"Loading rollouts from {args.rollouts}")
    rollouts = load_rollouts_sample(args.rollouts, args.sample_size)
    logger.info(f"Sampled {len(rollouts)} rollouts for validation")
    
    # Find schema path if not provided
    schema_path = args.schema_path
    if not schema_path:
        default_schema = PROJECT_DIR / "legacy/PankBaseAgent/schemas/kg_schema.json"
        if default_schema.exists():
            schema_path = str(default_schema)
    
    # Run validation
    results = validate_improvement(
        rollouts=rollouts,
        cypher_url=args.cypher_url,
        orchestrator_url=args.orchestrator_url,
        schema_path=schema_path,
    )
    
    # Output results
    print(json.dumps(results))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Log summary
    if results.get('success'):
        improvement = results.get('cypher_improvement', 0)
        if improvement > 0:
            logger.info(f"✓ Model IMPROVED: +{improvement:.4f} cypher reward")
        elif improvement < 0:
            logger.info(f"⚠ Model REGRESSED: {improvement:.4f} cypher reward")
        else:
            logger.info(f"→ No change in reward")
    else:
        logger.error(f"Validation failed: {results.get('error', 'Unknown error')}")
    
    return 0 if results.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())

