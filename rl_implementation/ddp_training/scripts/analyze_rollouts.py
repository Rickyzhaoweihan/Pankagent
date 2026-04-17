#!/usr/bin/env python3
"""
Comprehensive Rollout Analysis Script.

Analyzes rollouts to understand:
1. Overall success/failure rates
2. Common Cypher errors
3. Question quality
4. Reward distributions
5. Comparison with expected performance
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Cypher Error Detection (from legacy cypher_validator.py)
# =============================================================================

def detect_cypher_errors(cypher: str) -> List[str]:
    """Detect common Cypher errors."""
    errors = []
    
    if not cypher:
        return ["Empty query"]
    
    # 1. Missing relationship variable: -[:type]- instead of -[r:type]-
    rel_patterns = re.findall(r'-\[(.*?)\]-', cypher)
    for rel in rel_patterns:
        if rel.strip().startswith(':'):
            rel_type = rel.strip()[1:].split()[0].split('{')[0]
            errors.append(f"Missing relationship variable: [:{rel_type}]")
    
    # 2. Undefined variable in collect()
    # Extract defined variables
    defined_vars = set()
    for match in re.finditer(r'\((\w+)(?::\w+)?[^)]*\)', cypher):
        defined_vars.add(match.group(1))
    for match in re.finditer(r'\[(\w+)(?::\w+)?[^]]*\]', cypher):
        defined_vars.add(match.group(1))
    
    # Check collect() variables
    for match in re.finditer(r'collect\s*\(\s*(?:DISTINCT\s+)?(\w+)\s*\)', cypher, re.IGNORECASE):
        var = match.group(1)
        if var not in defined_vars:
            errors.append(f"Undefined variable in collect(): {var}")
    
    # 3. Missing DISTINCT in collect()
    collect_calls = re.findall(r'collect\s*\([^)]+\)', cypher, re.IGNORECASE)
    for call in collect_calls:
        if 'distinct' not in call.lower():
            errors.append("Missing DISTINCT in collect()")
            break
    
    # 4. Missing WHERE constraint (potential full table scan)
    if 'MATCH' in cypher.upper() and 'WHERE' not in cypher.upper():
        # Check if there's a property filter in the MATCH
        if not re.search(r'\{[^}]+\}', cypher):
            errors.append("No WHERE clause or property filter (full scan risk)")
    
    # 5. Wrong disease naming
    if re.search(r'\bT1D\b', cypher) or re.search(r'Type\s*1\s*Diabetes', cypher):
        errors.append("Wrong disease naming (use 'type 1 diabetes')")
    
    # 6. Missing return format
    if 'RETURN' in cypher.upper():
        if 'nodes' not in cypher.lower() or 'edges' not in cypher.lower():
            errors.append("Missing 'nodes, edges' in RETURN")
    
    return errors


def categorize_error(error_msg: str) -> str:
    """Categorize error message from execution."""
    if not error_msg:
        return "no_error"
    
    error_lower = error_msg.lower()
    
    if 'not defined' in error_lower:
        return "undefined_variable"
    elif 'syntax' in error_lower:
        return "syntax_error"
    elif 'type' in error_lower and 'mismatch' in error_lower:
        return "type_mismatch"
    elif 'property' in error_lower:
        return "property_error"
    elif 'timeout' in error_lower:
        return "timeout"
    elif 'connection' in error_lower:
        return "connection_error"
    else:
        return "other_error"


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_rollouts(rollouts_path: str) -> Dict[str, Any]:
    """Comprehensive rollout analysis."""
    
    results = {
        'total': 0,
        'with_data': 0,
        'without_data': 0,
        'total_steps': 0,
        'steps_with_data': 0,
        'cypher_errors': Counter(),
        'execution_errors': Counter(),
        'error_categories': Counter(),
        'rewards': {
            'cypher': [],
            'orch_qgen': [],
            'orch_synth': [],
        },
        'questions': [],
        'failed_queries': [],
        'successful_queries': [],
        'execution_times': [],
        'num_results_distribution': Counter(),
        'difficulty_distribution': Counter(),
        'data_quality_scores': [],
        'answer_quality_scores': [],
    }
    
    with open(rollouts_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            results['total'] += 1
            
            traj = entry.get('trajectory', {})
            steps = traj.get('steps', [])
            
            # Difficulty
            difficulty = entry.get('difficulty', 'unknown')
            results['difficulty_distribution'][difficulty] += 1
            
            # Question
            question = entry.get('question', '')
            results['questions'].append(question)
            
            # Scores
            results['data_quality_scores'].append(traj.get('data_quality_score', 0))
            results['answer_quality_scores'].append(traj.get('answer_quality_score', 0))
            
            # Rewards
            results['rewards']['cypher'].append(traj.get('cypher_reward', 0))
            results['rewards']['orch_qgen'].append(traj.get('orch_qgen_reward', 0))
            results['rewards']['orch_synth'].append(traj.get('orch_synth_reward', 0))
            
            # Analyze steps
            entry_has_data = False
            for step in steps:
                results['total_steps'] += 1
                
                has_data = step.get('has_data', False)
                if has_data:
                    results['steps_with_data'] += 1
                    entry_has_data = True
                
                # Execution time
                exec_time = step.get('execution_time_ms', 0)
                results['execution_times'].append(exec_time)
                
                # Num results
                num_results = step.get('num_results', 0)
                if num_results == 0:
                    results['num_results_distribution']['0'] += 1
                elif num_results < 10:
                    results['num_results_distribution']['1-9'] += 1
                elif num_results < 100:
                    results['num_results_distribution']['10-99'] += 1
                else:
                    results['num_results_distribution']['100+'] += 1
                
                # Cypher query analysis
                cypher = step.get('cypher_query', '')
                cypher_errors = detect_cypher_errors(cypher)
                for err in cypher_errors:
                    results['cypher_errors'][err] += 1
                
                # Execution errors from prompt (look for error key)
                prompt = step.get('prompt', '')
                if 'error' in prompt.lower():
                    # Try to extract error from prompt
                    pass
                
                # Track queries
                if has_data:
                    results['successful_queries'].append({
                        'question': question[:100],
                        'cypher': cypher,
                        'num_results': num_results,
                    })
                else:
                    results['failed_queries'].append({
                        'question': question[:100],
                        'cypher': cypher,
                        'errors': cypher_errors,
                    })
            
            if entry_has_data:
                results['with_data'] += 1
            else:
                results['without_data'] += 1
    
    return results


def print_analysis(results: Dict[str, Any]):
    """Print comprehensive analysis report."""
    
    print("=" * 80)
    print("ROLLOUT ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # ==========================================================================
    # 1. Overall Statistics
    # ==========================================================================
    print("1. OVERALL STATISTICS")
    print("-" * 40)
    total = results['total']
    with_data = results['with_data']
    without_data = results['without_data']
    
    print(f"Total rollouts: {total}")
    print(f"With data: {with_data} ({100*with_data/total:.1f}%)")
    print(f"Without data: {without_data} ({100*without_data/total:.1f}%)")
    print()
    
    total_steps = results['total_steps']
    steps_with_data = results['steps_with_data']
    print(f"Total steps: {total_steps}")
    print(f"Steps with data: {steps_with_data} ({100*steps_with_data/total_steps:.1f}%)")
    print(f"Avg steps per rollout: {total_steps/total:.2f}")
    print()
    
    # ==========================================================================
    # 2. Cypher Errors Analysis
    # ==========================================================================
    print("2. CYPHER QUERY ERRORS (Detected Issues)")
    print("-" * 40)
    
    cypher_errors = results['cypher_errors']
    if cypher_errors:
        for error, count in cypher_errors.most_common(10):
            pct = 100 * count / total_steps
            print(f"  {error}: {count} ({pct:.1f}%)")
    else:
        print("  No syntax errors detected")
    print()
    
    # ==========================================================================
    # 3. Reward Distribution
    # ==========================================================================
    print("3. REWARD DISTRIBUTION")
    print("-" * 40)
    
    for reward_type, values in results['rewards'].items():
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            zero_count = sum(1 for v in values if v == 0)
            print(f"  {reward_type}:")
            print(f"    Mean: {avg:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}")
            print(f"    Zero rewards: {zero_count}/{len(values)} ({100*zero_count/len(values):.1f}%)")
    print()
    
    # ==========================================================================
    # 4. Quality Scores
    # ==========================================================================
    print("4. QUALITY SCORES")
    print("-" * 40)
    
    dq_scores = results['data_quality_scores']
    aq_scores = results['answer_quality_scores']
    
    if dq_scores:
        avg_dq = sum(dq_scores) / len(dq_scores)
        low_dq = sum(1 for s in dq_scores if s < 0.3)
        print(f"  Data Quality: mean={avg_dq:.3f}, low(<0.3)={low_dq}/{len(dq_scores)}")
    
    if aq_scores:
        avg_aq = sum(aq_scores) / len(aq_scores)
        low_aq = sum(1 for s in aq_scores if s < 0.3)
        print(f"  Answer Quality: mean={avg_aq:.3f}, low(<0.3)={low_aq}/{len(aq_scores)}")
    print()
    
    # ==========================================================================
    # 5. Execution Time Analysis
    # ==========================================================================
    print("5. EXECUTION TIME ANALYSIS")
    print("-" * 40)
    
    exec_times = results['execution_times']
    if exec_times:
        avg_time = sum(exec_times) / len(exec_times)
        fast = sum(1 for t in exec_times if t < 200)
        slow = sum(1 for t in exec_times if t > 1000)
        print(f"  Average: {avg_time:.0f}ms")
        print(f"  Fast (<200ms): {fast}/{len(exec_times)}")
        print(f"  Slow (>1s): {slow}/{len(exec_times)}")
    print()
    
    # ==========================================================================
    # 6. Results Distribution
    # ==========================================================================
    print("6. NUM RESULTS DISTRIBUTION")
    print("-" * 40)
    
    for bucket, count in sorted(results['num_results_distribution'].items()):
        pct = 100 * count / total_steps
        print(f"  {bucket} results: {count} ({pct:.1f}%)")
    print()
    
    # ==========================================================================
    # 7. Sample Failed Queries
    # ==========================================================================
    print("7. SAMPLE FAILED QUERIES (First 5)")
    print("-" * 40)
    
    for i, failed in enumerate(results['failed_queries'][:5]):
        print(f"\n  [{i+1}] Question: {failed['question'][:80]}...")
        cypher = failed['cypher']
        if len(cypher) > 150:
            cypher = cypher[:150] + "..."
        print(f"      Cypher: {cypher}")
        if failed['errors']:
            print(f"      Errors: {', '.join(failed['errors'][:3])}")
    print()
    
    # ==========================================================================
    # 8. Sample Successful Queries
    # ==========================================================================
    print("8. SAMPLE SUCCESSFUL QUERIES (First 5)")
    print("-" * 40)
    
    for i, success in enumerate(results['successful_queries'][:5]):
        print(f"\n  [{i+1}] Question: {success['question'][:80]}...")
        cypher = success['cypher']
        if len(cypher) > 150:
            cypher = cypher[:150] + "..."
        print(f"      Cypher: {cypher}")
        print(f"      Results: {success['num_results']}")
    print()
    
    # ==========================================================================
    # 9. Key Issues Summary
    # ==========================================================================
    print("9. KEY ISSUES SUMMARY")
    print("-" * 40)
    
    issues = []
    
    # Check data retrieval rate
    data_rate = with_data / total
    if data_rate < 0.5:
        issues.append(f"⚠️  Low data retrieval rate: {100*data_rate:.1f}% (target: >50%)")
    
    # Check common errors
    if 'Missing relationship variable' in str(cypher_errors):
        missing_var_count = sum(v for k, v in cypher_errors.items() if 'relationship variable' in k)
        issues.append(f"⚠️  Missing relationship variables: {missing_var_count} queries")
    
    if 'Undefined variable' in str(cypher_errors):
        undef_count = sum(v for k, v in cypher_errors.items() if 'Undefined variable' in k)
        issues.append(f"⚠️  Undefined variables in collect(): {undef_count} queries")
    
    # Check rewards
    cypher_rewards = results['rewards']['cypher']
    if cypher_rewards:
        zero_reward_pct = sum(1 for r in cypher_rewards if r == 0) / len(cypher_rewards)
        if zero_reward_pct > 0.5:
            issues.append(f"⚠️  High zero-reward rate: {100*zero_reward_pct:.1f}%")
    
    # Check steps per rollout
    avg_steps = total_steps / total
    if avg_steps < 1.5:
        issues.append(f"⚠️  Low avg steps per rollout: {avg_steps:.2f} (model stops too early)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No major issues detected")
    print()
    
    # ==========================================================================
    # 10. Recommendations
    # ==========================================================================
    print("10. RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if 'relationship variable' in str(cypher_errors).lower():
        recommendations.append(
            "1. CRITICAL: Update prompt to emphasize relationship variables.\n"
            "   Every relationship MUST have a variable: -[r:type]- not -[:type]-"
        )
    
    if 'Undefined variable' in str(cypher_errors):
        recommendations.append(
            "2. CRITICAL: Fix collect() variable usage.\n"
            "   Only collect() variables that are defined in MATCH clause."
        )
    
    if data_rate < 0.5:
        recommendations.append(
            "3. Improve question-to-query alignment.\n"
            "   Questions may be asking about entities not in the graph."
        )
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}\n")
    else:
        print("  ✓ System performing well, continue training")
    
    print("=" * 80)


def main():
    # Default path
    default_path = Path(__file__).parent.parent.parent.parent / "outputs/stage1_ddp/rollouts_collect.jsonl"
    
    if len(sys.argv) > 1:
        rollouts_path = sys.argv[1]
    else:
        rollouts_path = str(default_path)
    
    if not Path(rollouts_path).exists():
        print(f"Error: Rollouts file not found: {rollouts_path}")
        sys.exit(1)
    
    print(f"Analyzing: {rollouts_path}")
    print()
    
    results = analyze_rollouts(rollouts_path)
    print_analysis(results)


if __name__ == "__main__":
    main()

