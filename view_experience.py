#!/usr/bin/env python3
"""
Experience Buffer Monitoring Dashboard

View and analyze the experience buffer statistics and examples.

Usage:
    python view_experience.py              # Show summary
    python view_experience.py --top 10     # Show top 10 examples
    python view_experience.py --stats      # Detailed statistics
    python view_experience.py --pattern qtl_query  # Show examples for specific pattern
"""

import json
import argparse
import os
from collections import Counter
from datetime import datetime
from typing import List, Dict


def load_experience_buffer(buffer_file: str = "experience_buffer.jsonl") -> List[Dict]:
    """Load all examples from experience buffer"""
    if not os.path.exists(buffer_file):
        return []
    
    examples = []
    with open(buffer_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return examples


def load_query_log(log_file: str = "logs/query_log.jsonl") -> List[Dict]:
    """Load all logged queries"""
    if not os.path.exists(log_file):
        return []
    
    queries = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return queries


def show_summary():
    """Show summary statistics"""
    buffer = load_experience_buffer()
    logs = load_query_log()
    
    print("=" * 60)
    print("Experience Buffer Summary")
    print("=" * 60)
    print()
    print(f"Total queries logged: {len(logs)}")
    print(f"Curated examples in buffer: {len(buffer)}")
    
    if logs:
        total_evaluated = sum(1 for log in logs if 'rating' in log or log.get('timestamp', '') < datetime.now().isoformat())
        print(f"Queries evaluated: {total_evaluated}")
    
    if buffer:
        avg_rating = sum(ex.get('rating', 0) for ex in buffer) / len(buffer)
        print(f"Average rating: {avg_rating:.1f}/10")
        
        # Pattern distribution
        patterns = Counter(ex.get('pattern', 'unknown') for ex in buffer)
        print(f"\nPattern Distribution:")
        for pattern, count in patterns.most_common():
            print(f"  {pattern}: {count}")
        
        # Rating distribution
        ratings = [ex.get('rating', 0) for ex in buffer]
        print(f"\nRating Distribution:")
        print(f"  9-10: {sum(1 for r in ratings if r >= 9)}")
        print(f"  7-8:  {sum(1 for r in ratings if 7 <= r < 9)}")
        print(f"  5-6:  {sum(1 for r in ratings if 5 <= r < 7)}")
        print(f"  1-4:  {sum(1 for r in ratings if r < 5)}")
    
    print()
    print("=" * 60)


def show_top_examples(n: int = 10):
    """Show top N examples"""
    buffer = load_experience_buffer()
    
    if not buffer:
        print("No examples in experience buffer.")
        return
    
    # Sort by rating
    sorted_buffer = sorted(buffer, key=lambda x: x.get('rating', 0), reverse=True)
    
    print("=" * 60)
    print(f"Top {min(n, len(buffer))} Examples")
    print("=" * 60)
    print()
    
    for i, ex in enumerate(sorted_buffer[:n], 1):
        print(f"{i}. Rating: {ex.get('rating', 0)}/10 | Pattern: {ex.get('pattern', 'unknown')}")
        print(f"   Query: \"{ex['query'][:70]}...\"")
        print(f"   Planned {ex.get('planning', {}).get('num_queries', 0)} sub-queries")
        print(f"   Feedback: {ex.get('feedback', 'N/A')[:100]}")
        
        if i < n:
            print()


def show_pattern_examples(pattern: str):
    """Show examples for a specific pattern"""
    buffer = load_experience_buffer()
    
    # Filter by pattern
    examples = [ex for ex in buffer if ex.get('pattern') == pattern]
    
    if not examples:
        print(f"No examples found for pattern: {pattern}")
        return
    
    # Sort by rating
    examples.sort(key=lambda x: x.get('rating', 0), reverse=True)
    
    print("=" * 60)
    print(f"Examples for Pattern: {pattern}")
    print("=" * 60)
    print(f"Total: {len(examples)}")
    print()
    
    for i, ex in enumerate(examples[:10], 1):
        print(f"{i}. Rating: {ex.get('rating', 0)}/10")
        print(f"   Query: \"{ex['query']}\"")
        print(f"   Planned {ex.get('planning', {}).get('num_queries', 0)} sub-queries")
        
        # Show first few queries
        queries = ex.get('planning', {}).get('queries', [])
        if queries:
            print(f"   First queries:")
            for j, q in enumerate(queries[:3], 1):
                print(f"     {j}. {q.get('input', '')[:60]}")
        
        print(f"   Feedback: {ex.get('feedback', 'N/A')}")
        print()


def show_detailed_stats():
    """Show detailed statistics"""
    buffer = load_experience_buffer()
    logs = load_query_log()
    
    print("=" * 60)
    print("Detailed Statistics")
    print("=" * 60)
    print()
    
    # Query log stats
    print("Query Log Statistics:")
    print(f"  Total queries: {len(logs)}")
    
    if logs:
        avg_num_queries = sum(log.get('planning', {}).get('num_queries', 0) for log in logs) / len(logs)
        print(f"  Average sub-queries per query: {avg_num_queries:.1f}")
        
        avg_time = sum(log.get('execution_time_ms', 0) for log in logs) / len(logs)
        print(f"  Average execution time: {avg_time:.0f}ms")
        
        # Time range
        timestamps = [log.get('timestamp', '') for log in logs if log.get('timestamp')]
        if timestamps:
            print(f"  First query: {min(timestamps)}")
            print(f"  Latest query: {max(timestamps)}")
    
    print()
    
    # Buffer stats
    print("Experience Buffer Statistics:")
    print(f"  Total examples: {len(buffer)}")
    
    if buffer:
        # Query count distribution
        query_counts = [ex.get('planning', {}).get('num_queries', 0) for ex in buffer]
        print(f"  Sub-queries per example:")
        print(f"    Min: {min(query_counts)}")
        print(f"    Max: {max(query_counts)}")
        print(f"    Average: {sum(query_counts)/len(query_counts):.1f}")
        
        # Pattern analysis
        patterns = Counter(ex.get('pattern', 'unknown') for ex in buffer)
        print(f"\n  Patterns learned:")
        for pattern, count in patterns.most_common():
            avg_rating = sum(ex.get('rating', 0) for ex in buffer if ex.get('pattern') == pattern) / count
            print(f"    {pattern}: {count} examples (avg rating: {avg_rating:.1f})")
    
    print()
    
    # Evaluation status
    last_evaluated_file = "logs/last_evaluated.txt"
    if os.path.exists(last_evaluated_file):
        with open(last_evaluated_file, 'r') as f:
            last_timestamp = f.read().strip()
        print(f"Last evaluation: {last_timestamp}")
        
        # Count unevaluated queries
        unevaluated = sum(1 for log in logs if log.get('timestamp', '') > last_timestamp)
        print(f"Unevaluated queries: {unevaluated}")
    else:
        print(f"No evaluation done yet")
        print(f"Unevaluated queries: {len(logs)}")
    
    print()
    print("=" * 60)


def list_patterns():
    """List all available patterns"""
    buffer = load_experience_buffer()
    
    patterns = set(ex.get('pattern', 'unknown') for ex in buffer)
    
    print("=" * 60)
    print("Available Patterns")
    print("=" * 60)
    print()
    
    for pattern in sorted(patterns):
        count = sum(1 for ex in buffer if ex.get('pattern') == pattern)
        print(f"  {pattern} ({count} examples)")
    
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="View experience buffer statistics and examples")
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--top', type=int, metavar='N',
                       help='Show top N examples (default: 10)')
    parser.add_argument('--pattern', type=str,
                       help='Show examples for specific pattern')
    parser.add_argument('--list-patterns', action='store_true',
                       help='List all available patterns')
    
    args = parser.parse_args()
    
    if args.stats:
        show_detailed_stats()
    elif args.top is not None:
        show_top_examples(args.top)
    elif args.pattern:
        show_pattern_examples(args.pattern)
    elif args.list_patterns:
        list_patterns()
    else:
        # Default: show summary
        show_summary()


if __name__ == "__main__":
    main()

