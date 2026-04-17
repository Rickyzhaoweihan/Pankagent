#!/usr/bin/env python3
"""
Batch Experience Buffer Evaluator

Evaluates query planning logs using GPT-4 and extracts the best patterns
into the experience buffer for in-context learning.

Usage:
    python batch_evaluator.py --limit 100
    python batch_evaluator.py --all
"""

import json
import argparse
import os
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
from PankBaseAgent.text_to_cypher.src.text2cypher_utils import get_env_variable

# Evaluation prompt template
EVALUATION_PROMPT_TEMPLATE = """You are analyzing query planning executions from a biomedical knowledge graph system.

The system (PankBaseAgent) receives natural language queries and must decompose them into multiple simple sub-queries to gather comprehensive information from a graph database.

Analyze these {count} query executions and evaluate each one:

{queries}

For each query, provide:
1. **Rating (1-10)**: How comprehensive was the planning?
   - 10: Queried all relevant relationship types, very comprehensive
   - 7-9: Good coverage, queried most relationships
   - 4-6: Moderate, missed some important relationships
   - 1-3: Poor, very limited queries

2. **Pattern**: Query type (entity_overview, qtl_query, expression_query, relationship_query, disease_association, function_query, general_query)

3. **Feedback**: Brief explanation of what was good or what was missing

Return a JSON array with one object per query:
[
  {{
    "query_index": 1,
    "rating": 8,
    "pattern": "entity_overview",
    "feedback": "Good comprehensive coverage of most relationships"
  }},
  ...
]

Focus on:
- Did it query ALL 8 relationship types (part_of_QTL_signal, effector_gene_of, DEG_in, expression_level_in, regulation, general_binding, function_annotation, OCR_activity)?
- Did it query supporting entities (cell types, diseases, GO terms)?
- Was planning atomic (simple queries) vs complex?
"""


def load_query_log(log_file: str, last_processed_file: str) -> List[Dict]:
    """
    Load queries from log file that haven't been processed yet
    
    Args:
        log_file: Path to query_log.jsonl
        last_processed_file: Path to file storing last processed timestamp
        
    Returns:
        List of query entries
    """
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    # Load last processed timestamp
    last_timestamp = None
    if os.path.exists(last_processed_file):
        with open(last_processed_file, 'r') as f:
            last_timestamp = f.read().strip()
    
    queries = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Skip if already processed
                if last_timestamp and entry.get('timestamp', '') <= last_timestamp:
                    continue
                queries.append(entry)
            except json.JSONDecodeError:
                continue
    
    return queries


def format_queries_for_evaluation(queries: List[Dict]) -> str:
    """Format queries for GPT-4 evaluation prompt"""
    formatted = ""
    for i, q in enumerate(queries, 1):
        formatted += f"\n{'='*60}\n"
        formatted += f"Query {i}:\n"
        formatted += f"User Question: \"{q['query']}\"\n"
        formatted += f"Planned {q['planning']['num_queries']} sub-queries:\n"
        
        for j, sq in enumerate(q['planning'].get('queries', [])[:10], 1):
            formatted += f"  {j}. {sq.get('input', '')}\n"
        
        if len(q['planning'].get('queries', [])) > 10:
            formatted += f"  ... and {len(q['planning']['queries']) - 10} more queries\n"
        
        formatted += f"Execution time: {q['execution_time_ms']:.0f}ms\n"
        formatted += f"Results: {q['results']}\n"
    
    return formatted


def evaluate_batch_with_gpt4(queries: List[Dict], batch_size: int = 50) -> List[Dict]:
    """
    Evaluate a batch of queries using GPT-4
    
    Args:
        queries: List of query entries
        batch_size: Number of queries to evaluate at once
        
    Returns:
        List of evaluations with ratings and feedback
    """
    client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))
    
    all_evaluations = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        
        print(f"Evaluating batch {i//batch_size + 1} ({len(batch)} queries)...")
        
        # Build evaluation prompt
        formatted_queries = format_queries_for_evaluation(batch)
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            count=len(batch),
            queries=formatted_queries
        )
        
        # Call GPT-4
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating query planning quality for knowledge graph systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON from response (might be wrapped in ```json```)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluations = json.loads(content)
            
            # Merge evaluations with original queries
            for j, eval_data in enumerate(evaluations):
                if j < len(batch):
                    merged = {
                        'query': batch[j]['query'],
                        'planning': batch[j]['planning'],
                        'results': batch[j]['results'],
                        'execution_time_ms': batch[j]['execution_time_ms'],
                        'timestamp': batch[j]['timestamp'],
                        'rating': eval_data.get('rating', 0),
                        'pattern': eval_data.get('pattern', 'general_query'),
                        'feedback': eval_data.get('feedback', '')
                    }
                    all_evaluations.append(merged)
            
            print(f"  ✓ Evaluated {len(evaluations)} queries")
            
        except Exception as e:
            print(f"  ✗ Error evaluating batch: {e}")
            continue
    
    return all_evaluations


def extract_top_patterns(evaluations: List[Dict], top_k: int = 20) -> List[Dict]:
    """
    Extract top K diverse patterns from evaluations
    
    Args:
        evaluations: List of evaluated queries
        top_k: Number of top patterns to extract
        
    Returns:
        List of top patterns
    """
    # Sort by rating
    sorted_evals = sorted(evaluations, key=lambda x: x.get('rating', 0), reverse=True)
    
    # Select top K diverse examples (different patterns)
    top_patterns = []
    pattern_counts = {}
    
    for eval_data in sorted_evals:
        if len(top_patterns) >= top_k:
            break
        
        pattern = eval_data.get('pattern', 'general_query')
        
        # Limit 4 examples per pattern for diversity
        if pattern_counts.get(pattern, 0) < 4:
            top_patterns.append({
                'query': eval_data['query'],
                'planning': eval_data['planning'],
                'rating': eval_data['rating'],
                'pattern': pattern,
                'feedback': eval_data['feedback'],
                'example_rank': len(top_patterns) + 1,
                'timestamp': eval_data['timestamp']
            })
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    return top_patterns


def update_experience_buffer(buffer_file: str, new_patterns: List[Dict]):
    """
    Update experience buffer with new top patterns
    
    Keeps the buffer at a reasonable size by keeping only top-rated examples
    """
    # Load existing patterns
    existing = []
    if os.path.exists(buffer_file):
        with open(buffer_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Merge with new patterns
    all_patterns = existing + new_patterns
    
    # Sort by rating and keep top 50
    all_patterns.sort(key=lambda x: x.get('rating', 0), reverse=True)
    top_patterns = all_patterns[:50]
    
    # Rewrite buffer file
    with open(buffer_file, 'w') as f:
        for pattern in top_patterns:
            f.write(json.dumps(pattern, ensure_ascii=False) + '\n')
    
    print(f"✓ Updated experience buffer: {len(top_patterns)} patterns (added {len(new_patterns)} new)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate query planning logs with GPT-4")
    parser.add_argument('--limit', type=int, default=100, 
                       help='Maximum number of new queries to evaluate (default: 100)')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all unprocessed queries')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of queries per GPT-4 call (default: 50)')
    
    args = parser.parse_args()
    
    log_file = "logs/query_log.jsonl"
    last_processed_file = "logs/last_evaluated.txt"
    buffer_file = "experience_buffer.jsonl"
    
    print("=" * 60)
    print("Batch Experience Buffer Evaluator")
    print("=" * 60)
    
    # Load unprocessed queries
    print(f"\nLoading queries from {log_file}...")
    queries = load_query_log(log_file, last_processed_file)
    
    if not queries:
        print("No new queries to evaluate.")
        return
    
    print(f"Found {len(queries)} new queries to evaluate")
    
    # Limit if specified
    if not args.all and len(queries) > args.limit:
        queries = queries[:args.limit]
        print(f"Limiting to {args.limit} queries")
    
    # Evaluate with GPT-4
    print(f"\nEvaluating queries in batches of {args.batch_size}...")
    evaluations = evaluate_batch_with_gpt4(queries, batch_size=args.batch_size)
    
    if not evaluations:
        print("No evaluations produced.")
        return
    
    print(f"\n✓ Evaluated {len(evaluations)} queries")
    
    # Extract top patterns
    print("\nExtracting top patterns...")
    top_patterns = extract_top_patterns(evaluations, top_k=20)
    
    print(f"Selected {len(top_patterns)} top patterns")
    print("\nTop 5 patterns:")
    for i, p in enumerate(top_patterns[:5], 1):
        print(f"  {i}. Rating {p['rating']}/10 - {p['pattern']}: \"{p['query'][:50]}...\"")
    
    # Update experience buffer
    print(f"\nUpdating {buffer_file}...")
    update_experience_buffer(buffer_file, top_patterns)
    
    # Update last processed timestamp
    if evaluations:
        last_timestamp = max(e['timestamp'] for e in evaluations)
        with open(last_processed_file, 'w') as f:
            f.write(last_timestamp)
        print(f"✓ Updated last processed timestamp: {last_timestamp}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

