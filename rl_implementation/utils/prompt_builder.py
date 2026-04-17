"""
Prompt Builder for Cypher Generator Agent.

Constructs prompts with strict token budget management for the Cypher Generator.
Ensures prompts never exceed the budget by tracking actual token usage.

Token Budget (total: 2450 tokens):
- System Rules: 200 tokens
- Learned Rules: 200 tokens  
- Schema: 600 tokens
- History: 1200 tokens
- Question + Instructions: 250 tokens

Based on IMPLEMENTATION_OVERVIEW.md design.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counting utility with optional tokenizer support.
    
    Can use:
    1. Actual tokenizer (most accurate)
    2. tiktoken (fast and accurate for GPT-like models)
    3. Character-based estimation (fallback)
    """
    
    # Default: 1 token ≈ 4 characters (conservative for English)
    DEFAULT_CHARS_PER_TOKEN = 4
    
    def __init__(self, tokenizer=None):
        """
        Initialize token counter.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for accurate counting
        """
        self.tokenizer = tokenizer
        self._tiktoken = None
        
        # Try to load tiktoken for better estimation if no tokenizer provided
        if tokenizer is None:
            try:
                import tiktoken
                self._tiktoken = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except ImportError:
                pass
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        # Method 1: Use actual tokenizer (most accurate)
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        
        # Method 2: Use tiktoken (fast and accurate)
        if self._tiktoken is not None:
            try:
                return len(self._tiktoken.encode(text))
            except Exception:
                pass
        
        # Method 3: Character-based estimation (fallback)
        return len(text) // self.DEFAULT_CHARS_PER_TOKEN
    
    def truncate_to_tokens(self, text: str, max_tokens: int, suffix: str = "\n...(truncated)") -> Tuple[str, bool]:
        """
        Truncate text to fit within token budget.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            suffix: Suffix to add when truncating
            
        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        current_tokens = self.count(text)
        
        if current_tokens <= max_tokens:
            return text, False
        
        # Reserve tokens for suffix
        suffix_tokens = self.count(suffix)
        target_tokens = max_tokens - suffix_tokens
        
        if target_tokens <= 0:
            return suffix, True
        
        # Binary search for the right truncation point
        # Start with character-based estimate
        chars_per_token = len(text) / max(1, current_tokens)
        estimated_chars = int(target_tokens * chars_per_token)
        
        # Fine-tune by checking actual token count
        truncated = text[:estimated_chars]
        while self.count(truncated) > target_tokens and len(truncated) > 0:
            # Reduce by ~10% each iteration
            estimated_chars = int(len(truncated) * 0.9)
            truncated = text[:estimated_chars]
        
        # Try to break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > len(truncated) * 0.8:  # Only if not too much loss
            truncated = truncated[:last_space]
        
        return truncated + suffix, True


class PromptBuilder:
    """
    Builds prompts for Cypher Generator with strict token budget management.
    
    Token Budget (total: 2450 tokens):
    - System Rules: 200 tokens
    - Learned Rules: 200 tokens
    - Schema: 600 tokens
    - History: 1200 tokens
    - Question + Instructions: 250 tokens
    
    Features:
    - Accurate token counting (with optional tokenizer)
    - Strict budget enforcement with truncation
    - Detailed logging of token usage
    - Format validation
    """
    
    # Token budgets for each section
    TOTAL_BUDGET = 2600  # Increased to accommodate deep think hints
    SYSTEM_RULES_TOKENS = 200
    LEARNED_RULES_TOKENS = 200
    DEEP_THINK_HINTS_TOKENS = 150  # New budget for deep think guidance
    SCHEMA_TOKENS = 600
    HISTORY_TOKENS = 1200
    QUESTION_TOKENS = 250
    
    # System rules text (constant) - concise for 14B model, reveals state structure
    SYSTEM_RULES = """You are a Cypher query generator for PankBase (biomedical KG).

TASK: Generate ONE simple Cypher query per step (up to 5 steps total).
- Look at STATE below to see your current step and history
- Use previous results to guide your next query
- Output "DONE" when you have sufficient data

SYNTAX:
- Relationships need variables: [r:type] not [:type]
- Always filter with properties or WHERE - never return all nodes
- Return: WITH collect(DISTINCT x)+collect(DISTINCT y) AS nodes, collect(DISTINCT r) AS edges RETURN nodes, edges;

EXAMPLE:
MATCH (g:gene)-[r:effector_gene_of]->(d:disease {name: 'type 1 diabetes'})
WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;"""

    def __init__(self, tokenizer=None):
        """
        Initialize the PromptBuilder.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for accurate token counting
        """
        self.token_counter = TokenCounter(tokenizer)
        self._system_rules_tokens = self.token_counter.count(self.SYSTEM_RULES)
        
        logger.debug(f"PromptBuilder initialized (system_rules: {self._system_rules_tokens} tokens)")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return self.token_counter.count(text)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        truncated, was_truncated = self.token_counter.truncate_to_tokens(text, max_tokens)
        if was_truncated:
            logger.debug(f"Truncated text from {self.estimate_tokens(text)} to {self.estimate_tokens(truncated)} tokens")
        return truncated
    
    def build_cypher_prompt(
        self,
        question: str,
        schema: Dict[str, Any],
        history: List[Dict[str, Any]],
        learned_rules: List[str],
        step: int,
        max_tokens: int = 2600,
        entity_samples: Optional[Dict[str, Any]] = None,
        deep_think_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build complete Cypher generation prompt with token budget management.
        
        Args:
            question: Natural language question
            schema: Knowledge graph schema
            history: List of previous query steps with results
            learned_rules: Learned patterns and warnings from experience buffer
            step: Current step number (1-5)
            max_tokens: Maximum total tokens (default: 2600)
            entity_samples: Optional dict with sample entity values for each node type
            deep_think_hints: Optional list of hints from Deep Think analysis
            
        Returns:
            Complete formatted prompt string
        """
        # Track token usage
        token_usage = {}
        
        # Section 1: System Rules (fixed budget)
        system_section = self.SYSTEM_RULES
        token_usage['system_rules'] = self._system_rules_tokens
        
        # Section 2: Learned Rules (200 tokens)
        learned_section = self._format_learned_rules(learned_rules)
        token_usage['learned_rules'] = self.estimate_tokens(learned_section)
        
        # Section 3: Deep Think Hints (150 tokens) - NEW
        hints_section = self._format_deep_think_hints(deep_think_hints)
        token_usage['deep_think_hints'] = self.estimate_tokens(hints_section)
        
        # Section 4: Schema (600 tokens)
        schema_section = self._format_schema(schema, question, entity_samples)
        token_usage['schema'] = self.estimate_tokens(schema_section)
        
        # Section 5: History (1200 tokens)
        history_section = self._format_history(history, step)
        token_usage['history'] = self.estimate_tokens(history_section)
        
        # Section 6: Question + Instructions (250 tokens)
        question_section = self._format_question_and_instructions(question, step)
        token_usage['question'] = self.estimate_tokens(question_section)
        
        # Combine all sections (only include non-empty sections)
        sections = [system_section, learned_section]
        if hints_section:
            sections.append(hints_section)
        sections.extend([schema_section, history_section, question_section])
        prompt = "\n\n".join(sections)
        
        # Check total token usage
        total_tokens = self.estimate_tokens(prompt)
        token_usage['total'] = total_tokens
        
        # Log token usage
        logger.debug(f"Token usage: {token_usage}")
        
        # Final check: if over budget, compress history more aggressively
        if total_tokens > max_tokens:
            logger.warning(f"Prompt exceeds budget ({total_tokens} > {max_tokens}), compressing history")
            
            # Reduce history budget
            remaining = max_tokens - (
                token_usage['system_rules'] +
                token_usage['learned_rules'] +
                token_usage['deep_think_hints'] +
                token_usage['schema'] +
                token_usage['question']
            )
            
            history_section = self._format_history(history, step, max_tokens=remaining)
            # Rebuild sections
            sections = [system_section, learned_section]
            if hints_section:
                sections.append(hints_section)
            sections.extend([schema_section, history_section, question_section])
            prompt = "\n\n".join(sections)
            
            final_tokens = self.estimate_tokens(prompt)
            logger.info(f"Compressed prompt to {final_tokens} tokens")
        
        return prompt
    
    def _format_learned_rules(self, learned_rules: List[str]) -> str:
        """
        Format learned patterns and warnings.
        
        Includes:
        - Good patterns (fast queries, high-yield, optimal stopping)
        - Warnings (bad regions, semantic ambiguities)
        
        Args:
            learned_rules: List of learned pattern/warning strings
            
        Returns:
            Formatted learned rules section
        """
        if not learned_rules:
            return "LEARNED PATTERNS:\n(No patterns learned yet - early training phase)"
        
        # Format rules
        lines = ["LEARNED PATTERNS:"]
        
        # Separate good patterns from warnings
        good_patterns = []
        warnings = []
        
        for rule in learned_rules:
            if isinstance(rule, str):
                if rule.startswith("⚠️") or "avoid" in rule.lower() or "warning" in rule.lower():
                    warnings.append(rule)
                else:
                    good_patterns.append(rule)
            elif isinstance(rule, dict):
                desc = rule.get('description', str(rule))
                if rule.get('pattern_type') in ['bad_region', 'semantic_ambiguity']:
                    warnings.append(f"⚠️ {desc}")
                else:
                    good_patterns.append(desc)
        
        # Add good patterns (up to 3)
        if good_patterns:
            lines.append("Good patterns:")
            for i, pattern in enumerate(good_patterns[:3], 1):
                lines.append(f"  {i}. {pattern[:80]}...")
        
        # Add warnings (up to 2)
        if warnings:
            lines.append("Warnings:")
            for warning in warnings[:2]:
                lines.append(f"  {warning[:100]}")
        
        rules_text = "\n".join(lines)
        
        # Truncate to budget
        return self.truncate_to_tokens(rules_text, self.LEARNED_RULES_TOKENS)
    
    def _format_deep_think_hints(
        self,
        hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format Deep Think hints/warnings for inclusion in prompt.
        
        These hints come from Deep Think analysis and provide specific
        guidance to avoid mistakes identified during training.
        
        Args:
            hints: List of hint dicts with 'text' and 'severity' keys
            
        Returns:
            Formatted hints section, or empty string if no hints
        """
        if not hints:
            return ""
        
        # Severity icons
        icons = {
            "critical": "🔴 CRITICAL:",
            "warning": "⚠️ WARNING:",
            "info": "💡 TIP:"
        }
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_hints = sorted(
            hints,
            key=lambda h: severity_order.get(h.get("severity", "info"), 2)
        )
        
        lines = ["🧠 DEEP THINK GUIDANCE (from training analysis):"]
        
        for hint in sorted_hints[:5]:  # Max 5 hints
            severity = hint.get("severity", "info")
            text = hint.get("text", "")
            if not text:
                continue
            icon = icons.get(severity, "💡 TIP:")
            lines.append(f"  {icon} {text}")
        
        if not lines[1:]:  # Only header, no actual hints
            return ""
        
        hints_text = "\n".join(lines)
        
        # Truncate to budget
        return self.truncate_to_tokens(hints_text, self.DEEP_THINK_HINTS_TOKENS)
    
    def _extract_entities_from_question(
        self,
        question: str,
        entity_samples: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Extract entities mentioned in the question by matching against known samples.
        
        Args:
            question: The question text
            entity_samples: Dict with known entity samples
            
        Returns:
            Dict mapping node types to found entity values
        """
        found = {
            'gene': [],
            'disease': [],
            'cell_type': [],
            'snp': [],
            'gene_ontology': [],
        }
        
        if not entity_samples:
            return found
        
        question_lower = question.lower()
        
        # Check for SNPs (rs pattern)
        import re
        snp_matches = re.findall(r'\brs\d+\b', question_lower)
        found['snp'] = snp_matches
        
        # Check for known diseases
        for item in entity_samples.get('diseases', []):
            if isinstance(item, dict):
                name = item.get('name', '')
                if name.lower() in question_lower:
                    found['disease'].append(name)
        
        # Check for known cell types
        for item in entity_samples.get('cell_types', []):
            if isinstance(item, dict):
                name = item.get('name', '')
                if name.lower() in question_lower:
                    found['cell_type'].append(name)
        
        # Check for known gene ontology terms
        for item in entity_samples.get('gene_ontology', []):
            if isinstance(item, dict):
                name = item.get('name', '')
                if name.lower() in question_lower:
                    found['gene_ontology'].append(name)
        
        # Check for known genes (less likely to match, but try)
        for item in entity_samples.get('genes', []):
            if isinstance(item, dict):
                name = item.get('name', '')
                # Gene names are often uppercase, check both
                if name in question or name.lower() in question_lower:
                    found['gene'].append(name)
        
        return found
    
    def _format_schema(
        self,
        schema: Dict[str, Any],
        question: str,
        entity_samples: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format schema with entities extracted from the question.
        
        Shows entities mentioned in the question as the primary examples,
        falling back to random samples if no entities found.
        
        Args:
            schema: Full knowledge graph schema
            question: Question to extract entities from
            entity_samples: Dict with known entity samples
            
        Returns:
            Formatted schema section
        """
        lines = ["SCHEMA:"]
        
        # Handle nested schema structure
        kg_schema = schema.get('knowledge_graph_schema', schema)
        edge_types = kg_schema.get('edge_types', {})
        
        # Extract entities mentioned in the question
        found_entities = self._extract_entities_from_question(question, entity_samples)
        
        # Map node types to entity_samples keys for fallback
        sample_mapping = {
            'gene': ('genes', 'name'),
            'disease': ('diseases', 'name'),
            'cell_type': ('cell_types', 'name'),
            'snp': ('snps', 'id'),
            'gene_ontology': ('gene_ontology', 'name'),
        }
        
        lines.append("Nodes (property format):")
        for node_type, (sample_key, prop_name) in sample_mapping.items():
            # First check if we found entities in the question
            query_entities = found_entities.get(node_type, [])
            
            if query_entities:
                # Show entities from question (highlighted)
                sample_str = ', '.join(f"'{s}'" for s in query_entities[:3])
                lines.append(f"  - {node_type}: {{{prop_name}: {sample_str}}} ← from question")
            else:
                # Fallback to random samples
                samples = []
                if entity_samples and sample_key in entity_samples:
                    sample_list = entity_samples[sample_key]
                    for item in sample_list[:2]:
                        if isinstance(item, dict) and prop_name in item:
                            samples.append(item[prop_name])
                
                if samples:
                    sample_str = ', '.join(f"'{s}'" for s in samples)
                    lines.append(f"  - {node_type}: {{{prop_name}: {sample_str}, ...}}")
                else:
                    lines.append(f"  - {node_type}: {{{prop_name}: '...'}}")
        
        # Format edge types (compact)
        lines.append("\nRelationships:")
        edge_count = 0
        for edge_name, edge_info in edge_types.items():
            if edge_count >= 10:
                break
            
            if isinstance(edge_info, dict):
                source = edge_info.get('source_node_type', '?')
                target = edge_info.get('target_node_type', '?')
                source = source.split(';')[-1] if ';' in source else source
                target = target.split(';')[-1] if ';' in target else target
                lines.append(f"  - (:{source})-[:{edge_name}]->(:{target})")
            else:
                lines.append(f"  - [:{edge_name}]")
            edge_count += 1
        
        schema_text = '\n'.join(lines)
        return self.truncate_to_tokens(schema_text, self.SCHEMA_TOKENS)
    
    def _format_history(
        self,
        history: List[Dict[str, Any]],
        current_step: int,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Format query history with compression for older steps.
        
        Last 2 steps: Full detail with query, results, quality feedback
        Older steps: Summarized (query pattern, result count, time)
        
        Args:
            history: List of previous steps with queries and results
            current_step: Current step number
            max_tokens: Override token budget (optional)
            
        Returns:
            Formatted history section
        """
        budget = max_tokens or self.HISTORY_TOKENS
        
        if not history:
            return f"STATE: Step {current_step}/5 (first query, no history yet)"
        
        lines = [f"STATE: Step {current_step}/5 - History:"]
        
        # Determine detail level based on budget
        if budget < 400:
            detail_steps = 1  # Very compressed
        elif budget < 800:
            detail_steps = min(1, len(history))  # Only last step in detail
        else:
            detail_steps = min(2, len(history))  # Last 2 steps in detail
        
        # Format each step
        for i, step_data in enumerate(history):
            step_num = i + 1
            query = step_data.get('query', '')
            result = step_data.get('result', {})
            
            # Determine if this step should be detailed
            is_recent = i >= len(history) - detail_steps
            
            if is_recent:
                step_text = self._format_step_detailed(step_num, query, result)
            else:
                step_text = self._format_step_summary(step_num, query, result)
            
            lines.append(step_text)
        
        history_text = '\n'.join(lines)
        return self.truncate_to_tokens(history_text, budget)
    
    def _format_step_detailed(
        self,
        step_num: int,
        query: str,
        result: Dict[str, Any],
    ) -> str:
        """Format a single step with full details."""
        lines = [f"Step {step_num}:"]
        
        # Clean query text - remove any contamination from previous formatting
        query = self._clean_history_query(query)
        
        # Query (truncated if needed)
        if len(query) > 200:
            query_display = query[:200] + "..."
        else:
            query_display = query
        lines.append(f"  Query: {query_display}")
        
        # Execution metrics
        exec_time = result.get('execution_time_ms', 0)
        num_results = result.get('num_results', 0)
        success = result.get('success', False)
        has_data = result.get('has_data', False)
        
        # Time indicator
        if exec_time < 100:
            time_icon = "⚡"  # Fast
        elif exec_time < 500:
            time_icon = "○"  # OK
        elif exec_time < 1000:
            time_icon = "△"  # Slow
        else:
            time_icon = "✗"  # Very slow
        
        # Status indicator
        if not success:
            status = "❌ Error"
        elif not has_data:
            status = "⚠️ No data"
        else:
            status = "✓"
        
        lines.append(f"  Results: {num_results} items {time_icon} ({exec_time:.0f}ms) {status}")
        
        # Data quality (if available)
        data_quality = result.get('data_quality_score')
        if data_quality is not None:
            quality_icon = "✓" if data_quality > 0.7 else ("○" if data_quality > 0.4 else "✗")
            lines.append(f"  Data Quality: {quality_icon} ({data_quality:.2f})")
        
        # Data summary (if available)
        data_summary = result.get('data_summary', '')
        if data_summary:
            lines.append(f"  Summary: {data_summary[:100]}...")
        
        # Semantic warning (if present)
        if result.get('semantic_warning'):
            lines.append(f"  ⚠️ Semantic ambiguity: {result['semantic_warning'][:80]}")
        
        return '\n'.join(lines)
    
    def _clean_history_query(self, query: str) -> str:
        """
        Clean query text by removing any contamination from previous formatting.
        
        This prevents formatting corruption from appearing in the history section
        (e.g., "Step 2:\n  Query:" appearing inside a query).
        """
        if not query:
            return ""
        
        # Remove common contamination patterns
        patterns_to_remove = [
            r'Step\s*\d+[:\s]*Query[:\s]*',   # Step 1: Query:
            r'^Step\s*\d+[:\s]*',              # Step 1:
            r'^Query[:\s]*',                   # Query:
            r'^\s*Results?[:\s]*\d+.*$',       # Results: 0 items...
            r'\s*Results?[:\s]*\d+.*$',        # trailing results
            r'^\s*\.\.\.\s*',                  # Leading ...
            r'\s*⚠️.*$',                       # Warning emoji and text
            r'\s*✗.*$',                        # Error emoji and text
            r'\s*❌.*$',                        # Error emoji
        ]
        
        for pattern in patterns_to_remove:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove duplicate newlines
        query = re.sub(r'\n\s*\n', '\n', query)
        
        return query.strip()
    
    def _format_step_summary(
        self,
        step_num: int,
        query: str,
        result: Dict[str, Any],
    ) -> str:
        """Format a single step with summary only."""
        # Clean query text first
        query = self._clean_history_query(query)
        
        num_results = result.get('num_results', 0)
        exec_time = result.get('execution_time_ms', 0)
        success = result.get('success', False)
        
        # Extract key pattern from query (node types, edge types)
        node_types = re.findall(r':(\w+)', query)[:3]
        pattern = ', '.join(node_types) if node_types else "query"
        
        status = "✓" if success else "❌"
        return f"Step {step_num}: {pattern} → {num_results} items ({exec_time:.0f}ms) {status}"
    
    def _format_question_and_instructions(self, question: str, step: int) -> str:
        """
        Format question and current step instructions.
        
        Args:
            question: Natural language question
            step: Current step number
            
        Returns:
            Formatted question section
        """
        lines = [f"QUESTION: {question}", ""]
        
        if step == 1:
            lines.extend([
                "Generate your first Cypher query:",
                "```cypher",
                "MATCH ...",
                "```",
            ])
        elif step < 5:
            lines.extend([
                "Based on state above, either:",
                "- Generate next query (if more data needed)",
                "- Output 'DONE' (if sufficient data)",
            ])
        else:
            lines.extend([
                "Final step! Output query OR 'DONE':",
            ])
        
        question_text = '\n'.join(lines)
        return self.truncate_to_tokens(question_text, self.QUESTION_TOKENS)
    
    def validate_prompt(self, prompt: str, max_tokens: int = 2450) -> Dict[str, Any]:
        """
        Validate a prompt and return token usage details.
        
        Args:
            prompt: The prompt to validate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Dictionary with validation results
        """
        token_count = self.estimate_tokens(prompt)
        
        return {
            'valid': token_count <= max_tokens,
            'token_count': token_count,
            'max_tokens': max_tokens,
            'utilization': token_count / max_tokens,
            'remaining': max_tokens - token_count,
        }


def build_cypher_prompt(
    question: str,
    schema: Dict[str, Any],
    history: List[Dict[str, Any]],
    learned_rules: List[str],
    step: int,
    max_tokens: int = 2600,
    tokenizer=None,
    entity_samples: Optional[Dict[str, Any]] = None,
    deep_think_hints: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Convenience function to build Cypher prompt.
    
    Args:
        question: Natural language question
        schema: Knowledge graph schema
        history: List of previous query steps
        learned_rules: Learned patterns from experience buffer
        step: Current step number
        max_tokens: Maximum tokens (default: 2600)
        tokenizer: Optional tokenizer for accurate counting
        entity_samples: Optional dict with sample entity values
        deep_think_hints: Optional list of hints from Deep Think analysis
        
    Returns:
        Complete formatted prompt
    """
    builder = PromptBuilder(tokenizer)
    return builder.build_cypher_prompt(
        question, schema, history, learned_rules, step, max_tokens, entity_samples, deep_think_hints
    )
