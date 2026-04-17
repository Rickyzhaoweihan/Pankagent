"""
Utilities module for PanKLLM RL post-training.

Exports:
- PromptBuilder: Prompt construction with token budget management for Cypher Generator
- TokenCounter: Token counting utility with optional tokenizer support
- build_cypher_prompt: Convenience function to build Cypher prompts
- OrchestratorPromptBuilder: Prompt construction for Orchestrator's four roles
- EntityExtractor: Extract entity samples from Neo4j for question generation seeding
- EntitySamples: Container for extracted entity samples
- Data quality evaluation utilities
- Cypher auto-fix utilities for rollout collection
"""

from .prompt_builder import PromptBuilder, TokenCounter, build_cypher_prompt
from .orchestrator_prompt_builder import OrchestratorPromptBuilder
from .entity_extractor import EntityExtractor, EntitySamples
from .data_quality_evaluator import (
    parse_data_quality_json,
    parse_answer_quality_json,
    compute_doubt_level,
    extract_semantic_issues,
    extract_problematic_regions,
    format_semantic_issues_for_prompt,
    compute_data_utilization
)
from .cypher_auto_fix import (
    auto_fix_cypher,
    create_auto_fixer,
    fix_relationship_variables,
    fix_distinct_in_collect,
    fix_return_format,
    fix_missing_collections,
    fix_disease_naming,
    fix_property_names
)

__all__ = [
    'PromptBuilder',
    'TokenCounter',
    'build_cypher_prompt',
    'OrchestratorPromptBuilder',
    'EntityExtractor',
    'EntitySamples',
    'parse_data_quality_json',
    'parse_answer_quality_json',
    'compute_doubt_level',
    'extract_semantic_issues',
    'extract_problematic_regions',
    'format_semantic_issues_for_prompt',
    'compute_data_utilization',
    # Cypher auto-fix
    'auto_fix_cypher',
    'create_auto_fixer',
    'fix_relationship_variables',
    'fix_distinct_in_collect',
    'fix_return_format',
    'fix_missing_collections',
    'fix_disease_naming',
    'fix_property_names'
]

