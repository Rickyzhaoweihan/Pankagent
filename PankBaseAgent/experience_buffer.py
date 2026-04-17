#!/usr/bin/env python3
"""
Experience Buffer for PankBaseAgent Planning

Stores and retrieves planning examples to improve query decomposition through in-context learning.
Uses simple JSON storage (no vector DB yet, designed for future vectorization).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ExperienceBuffer:
    """
    Manages planning experience storage and retrieval.
    
    Stores two types of data:
    1. query_log.jsonl - All planning executions (raw logs)
    2. experience_buffer.jsonl - Curated top patterns (evaluated by GPT-4)
    """
    
    def __init__(self, 
                 log_file: str = "logs/query_log.jsonl",
                 buffer_file: str = "experience_buffer.jsonl"):
        self.log_file = log_file
        self.buffer_file = buffer_file
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_planning(self, query: str, planning: Dict, results: Dict, 
                    execution_time_ms: float):
        """
        Log a planning execution to query_log.jsonl
        
        Args:
            query: User's natural language query
            planning: Planning details (num_queries, queries list, draft)
            results: Execution results (success, data_count, etc.)
            execution_time_ms: Time taken to execute
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "planning": planning,
            "results": results,
            "execution_time_ms": execution_time_ms
        }
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def load_best_examples(self, max_examples: int = 50) -> List[Dict]:
        """
        Load curated best examples from experience_buffer.jsonl
        
        Returns:
            List of best planning examples, sorted by rating
        """
        if not os.path.exists(self.buffer_file):
            return []
        
        examples = []
        with open(self.buffer_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError:
                        continue
        
        # Sort by rating (descending) and example_rank
        examples.sort(key=lambda x: (x.get('rating', 0), -x.get('example_rank', 999)), 
                     reverse=True)
        
        return examples[:max_examples]
    
    def find_similar(self, query: str, examples: List[Dict], 
                     top_k: int = 3, min_rating: float = 7.0) -> List[Dict]:
        """
        Find similar examples using simple pattern matching
        (No vector search yet, uses keywords and patterns)
        
        Args:
            query: User's query
            examples: List of all loaded examples
            top_k: Number of examples to return
            min_rating: Minimum rating threshold
        
        Returns:
            List of most relevant examples
        """
        query_lower = query.lower()
        
        # Detect query pattern
        query_pattern = self._detect_pattern(query)
        
        # Score each example
        scored_examples = []
        for ex in examples:
            if ex.get('rating', 0) < min_rating:
                continue
            
            score = 0.0
            
            # Pattern match (highest weight)
            if ex.get('pattern') == query_pattern:
                score += 3.0
            
            # Keyword overlap
            ex_query_lower = ex.get('query', '').lower()
            keywords = self._extract_keywords(query_lower)
            for keyword in keywords:
                if keyword in ex_query_lower:
                    score += 0.5
            
            # Rating boost
            score += ex.get('rating', 0) * 0.1
            
            scored_examples.append((score, ex))
        
        # Sort by score and return top k
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:top_k]]
    
    def get_stats(self) -> Dict:
        """
        Get statistics about logged queries and experiences
        
        Returns:
            Dictionary with stats
        """
        stats = {
            "total_logged": 0,
            "total_curated": 0,
            "log_file_exists": os.path.exists(self.log_file),
            "buffer_file_exists": os.path.exists(self.buffer_file)
        }
        
        # Count logged queries
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                stats["total_logged"] = sum(1 for _ in f)
        
        # Count curated examples
        if os.path.exists(self.buffer_file):
            with open(self.buffer_file, 'r') as f:
                stats["total_curated"] = sum(1 for _ in f)
        
        return stats
    
    def _detect_pattern(self, query: str) -> str:
        """
        Detect query pattern for better matching
        
        Args:
            query: User's query string
            
        Returns:
            Pattern name (entity_overview, qtl_query, etc.)
        """
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ["tell me about", "what is", "describe"]):
            return "entity_overview"
        elif "relationship" in query_lower or "relate" in query_lower or "connection" in query_lower:
            return "relationship_query"
        elif "qtl" in query_lower:
            return "qtl_query"
        elif "expression" in query_lower or "expressed" in query_lower or "deg" in query_lower:
            return "expression_query"
        elif "effector" in query_lower or "causal" in query_lower:
            return "disease_association"
        elif "function" in query_lower or "role" in query_lower:
            return "function_query"
        else:
            return "general_query"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: Query string (lowercased)
            
        Returns:
            List of keywords
        """
        # Common stop words to ignore
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                     "for", "of", "with", "by", "from", "is", "are", "was", "were",
                     "what", "how", "does", "do", "tell", "me", "about"}
        
        # Simple tokenization
        tokens = query.split()
        keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return keywords


# Global instance for easy import
_experience_buffer_instance = None

def get_experience_buffer() -> ExperienceBuffer:
    """Get or create global experience buffer instance"""
    global _experience_buffer_instance
    if _experience_buffer_instance is None:
        _experience_buffer_instance = ExperienceBuffer()
    return _experience_buffer_instance

