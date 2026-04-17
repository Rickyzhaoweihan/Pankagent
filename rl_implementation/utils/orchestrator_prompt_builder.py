"""
Orchestrator Prompt Builder.

Constructs prompts for the Orchestrator agent's four roles:
1. Question Generation - Generate diverse training questions
2. Data Quality Evaluation - Evaluate retrieved data quality
3. Answer Synthesis - Synthesize natural language answers
4. Answer Quality Evaluation - Evaluate final answer quality

Each role has different token budgets and prompt structures.
Includes token validation and proper truncation to ensure prompts fit.

Based on IMPLEMENTATION_OVERVIEW.md design.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counting utility (same as in prompt_builder.py).
    
    Uses character-based estimation or tiktoken if available.
    """
    
    DEFAULT_CHARS_PER_TOKEN = 4
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._tiktoken = None
        
        if tokenizer is None:
            try:
                import tiktoken
                self._tiktoken = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass
    
    def count(self, text: str) -> int:
        if not text:
            return 0
        
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        
        if self._tiktoken is not None:
            try:
                return len(self._tiktoken.encode(text))
            except Exception:
                pass
        
        return len(text) // self.DEFAULT_CHARS_PER_TOKEN
    
    def truncate(self, text: str, max_tokens: int, suffix: str = "... (truncated)") -> str:
        current = self.count(text)
        if current <= max_tokens:
            return text
        
        suffix_tokens = self.count(suffix)
        target = max_tokens - suffix_tokens
        
        if target <= 0:
            return suffix
        
        # Estimate chars and binary search
        chars_per_token = len(text) / max(1, current)
        estimated = int(target * chars_per_token)
        truncated = text[:estimated]
        
        while self.count(truncated) > target and len(truncated) > 0:
            estimated = int(len(truncated) * 0.9)
            truncated = text[:estimated]
        
        # Break at word boundary if possible
        last_space = truncated.rfind(' ')
        if last_space > len(truncated) * 0.8:
            truncated = truncated[:last_space]
        
        return truncated + suffix


class OrchestratorPromptBuilder:
    """
    Build prompts for Orchestrator's four roles with token budgets.
    
    Role 1 - Question Generation: 1,800 tokens
        - Schema summary: 400 tokens
        - Curriculum: 100 tokens
        - Scope constraints: 200 tokens
        - Recent questions: 600 tokens
        - Instructions: 500 tokens
    
    Role 2 - Data Quality Evaluation: 2,200 tokens
        - Question: 100 tokens
        - Trajectory: 1400 tokens
        - Semantic issues: 200 tokens
        - Criteria: 500 tokens
    
    Role 3 - Answer Synthesis: 3,200 tokens (~8K chars)
        - Question: 100 tokens
        - Data: 2,500 tokens (~50 entities with descriptions)
        - Quality feedback: 150 tokens
        - Instructions: 450 tokens
    
    Role 4 - Answer Quality Evaluation: 1,500 tokens
        - Question: 100 tokens
        - Answer: 500 tokens
        - Criteria: 400 tokens
        - Format: 500 tokens
    """
    
    # Token budgets per role (increased to accommodate deep think hints)
    QUESTION_GEN_BUDGET = 1950
    DATA_EVAL_BUDGET = 2350
    SYNTHESIS_BUDGET = 3350  # ~8000 chars - fits within model context with room for generation
    ANSWER_EVAL_BUDGET = 1650
    
    # Deep Think Hints budget (shared across all roles)
    DEEP_THINK_HINTS_TOKENS = 150
    
    # Question Generation budgets
    QUESTION_GEN_SCHEMA_TOKENS = 400
    QUESTION_GEN_CURRICULUM_TOKENS = 100
    QUESTION_GEN_SCOPE_TOKENS = 200
    QUESTION_GEN_RECENT_TOKENS = 600
    QUESTION_GEN_INSTRUCTION_TOKENS = 500
    
    # Data Quality Evaluation budgets
    DATA_EVAL_QUESTION_TOKENS = 100
    DATA_EVAL_TRAJECTORY_TOKENS = 1400
    DATA_EVAL_SEMANTIC_TOKENS = 200
    DATA_EVAL_CRITERIA_TOKENS = 500
    
    # Answer Synthesis budgets - balance between data and context limits
    SYNTHESIS_QUESTION_TOKENS = 100
    SYNTHESIS_DATA_TOKENS = 2500  # ~6000 chars - enough for 50+ entities, stays under model limits
    SYNTHESIS_QUALITY_TOKENS = 150
    SYNTHESIS_INSTRUCTION_TOKENS = 450
    
    # Answer Quality Evaluation budgets
    ANSWER_EVAL_QUESTION_TOKENS = 100
    ANSWER_EVAL_ANSWER_TOKENS = 500
    ANSWER_EVAL_CRITERIA_TOKENS = 400
    ANSWER_EVAL_FORMAT_TOKENS = 500
    
    def __init__(self, tokenizer=None):
        """
        Initialize the OrchestratorPromptBuilder.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for accurate counting
        """
        self.token_counter = TokenCounter(tokenizer)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self.token_counter.count(text)
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        return self.token_counter.truncate(text, max_tokens)
    
    def _format_deep_think_hints(
        self,
        hints: Optional[List[Dict[str, Any]]] = None,
        role: Optional[str] = None,
    ) -> str:
        """
        Format Deep Think hints for inclusion in orchestrator prompts.
        
        Args:
            hints: List of hint dicts with 'text' and 'severity' keys
            role: Orchestrator role to filter hints for (generation, synthesis, etc.)
            
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
        
        if len(lines) <= 1:  # Only header, no actual hints
            return ""
        
        hints_text = "\n".join(lines)
        
        # Truncate to budget
        return self.truncate_text(hints_text, self.DEEP_THINK_HINTS_TOKENS)
    
    # =========================================================================
    # Role 1: Question Generation (with Entity Seeding)
    # =========================================================================
    
    def build_question_generation_prompt(
        self,
        schema: Dict[str, Any],
        difficulty: str,
        curriculum_constraints: Dict[str, Any],
        scope_constraints: Dict[str, Any],
        recent_questions: List[str],
        # Entity + Relationship seeding for diversity
        seed_entity: Optional[str] = None,
        seed_entity_type: Optional[str] = None,
        seed_relationship: Optional[str] = None,
        # Experience-based keywords from successful patterns
        experience_keywords: Optional[List[str]] = None,
        # Valid entities for grounding (helps avoid non-existent entities)
        valid_entities: Optional[Dict[str, Any]] = None,
        # RL TRAINING FLAG: Whether to include example questions in seed
        # Default OFF - LLM should learn patterns through reward, not imitation
        include_seed_examples: bool = False,
        # Deep Think hints for this role
        deep_think_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build prompt for question generation role with entity + relationship seeding.
        
        Token budget: 1,800 tokens
        
        Entity+relationship seeding ensures each generated question is about a DIFFERENT
        topic AND relationship that exists in the database, significantly improving
        training diversity and answerability.
        
        IMPORTANT FOR RL TRAINING:
        - include_seed_examples=False (default): LLM discovers patterns through reward
        - include_seed_examples=True: Only for debugging, may cause over-reliance on templates
        
        Args:
            schema: KG schema dict
            difficulty: 'easy', 'medium', or 'hard'
            curriculum_constraints: Dict with max_hops, focus_area
            scope_constraints: Dict with allowed/avoided node/edge types
            recent_questions: List of recent questions for diversity
            seed_entity: Specific entity to focus on (e.g., "INS", "beta cell")
            seed_entity_type: Type of the seed entity (e.g., "gene", "cell_type")
            seed_relationship: Relationship type to include (e.g., "expression_level_in")
            experience_keywords: Keywords from successful past patterns
            valid_entities: Dict with cell_types, diseases, go_terms that exist in DB
            include_seed_examples: Whether to include example question patterns (default: False)
            deep_think_hints: Optional list of hints from Deep Think analysis for question generation
            
        Returns:
            Formatted prompt string
        """
        # Build sections with budgets
        schema_text = self._format_schema_summary(schema, self.QUESTION_GEN_SCHEMA_TOKENS)
        curriculum_text = self._format_curriculum(difficulty, curriculum_constraints)
        scope_text = self._format_scope_constraints(scope_constraints, self.QUESTION_GEN_SCOPE_TOKENS)
        recent_text = self._format_recent_questions(recent_questions, self.QUESTION_GEN_RECENT_TOKENS // 2)  # Reduced for seed section
        instruction_text = self._get_question_gen_instruction(difficulty)
        
        # Deep Think hints section (NEW)
        hints_text = self._format_deep_think_hints(deep_think_hints, role='generation')
        
        # Build entity + relationship seed section
        seed_text = self._format_entity_seed(
            seed_entity=seed_entity,
            seed_entity_type=seed_entity_type,
            seed_relationship=seed_relationship,
            experience_keywords=experience_keywords,
            include_examples=include_seed_examples,  # OFF by default for RL
        )
        
        # NEW: Build valid entities section (entity grounding)
        valid_entities_text = self._format_valid_entities(valid_entities)
        
        # Combine into prompt
        hints_section = f"\n{hints_text}\n" if hints_text else ""
        prompt = f"""You are a question generator for a biomedical knowledge graph (PankBase).
Your task is to generate training questions that can be answered using Cypher queries.

KNOWLEDGE GRAPH SCHEMA:
{schema_text}

{valid_entities_text}
{hints_section}
CURRICULUM CONSTRAINTS:
{curriculum_text}

{seed_text}

SCOPE CONSTRAINTS:
{scope_text}

RECENT QUESTIONS (DO NOT repeat or paraphrase these):
{recent_text}
⛔ IMPORTANT: Do NOT generate any question similar to the above! Your question must ask about DIFFERENT entities/topics.

TASK:
{instruction_text}

Generate ONE question:"""
        
        # Validate total tokens
        total_tokens = self.estimate_tokens(prompt)
        if total_tokens > self.QUESTION_GEN_BUDGET:
            logger.warning(f"Question gen prompt exceeds budget: {total_tokens} > {self.QUESTION_GEN_BUDGET}")
            # Compress recent questions if over budget
            recent_text = self._format_recent_questions(
                recent_questions[-5], 
                self.QUESTION_GEN_RECENT_TOKENS // 4
            )
            prompt = f"""You are a question generator for a biomedical knowledge graph (PankBase).

SCHEMA:
{schema_text}

{valid_entities_text}
{hints_section}
CURRICULUM:
{curriculum_text}

{seed_text}

SCOPE:
{scope_text}

RECENT (avoid similar):
{recent_text}

{instruction_text}

Generate ONE question:"""
        
        return prompt
    
    # Relationship descriptions for better prompt guidance
    RELATIONSHIP_DESCRIPTIONS = {
        # Gene relationships (outgoing)
        'expression_level_in': 'gene expression levels in cell types',
        'DEG_in': 'differentially expressed genes in cell types',
        'physical_interaction': 'physical interactions between genes/proteins',
        'regulation': 'gene regulatory relationships',
        'function_annotation': 'gene function annotations (Gene Ontology)',
        'effector_gene_of': 'effector genes of diseases',
        'signal_COLOC_with': 'colocalization signals with diseases',
        # Cell type relationships (incoming)
        'expression_level_in_in': 'genes expressed in this cell type',
        'DEG_in_in': 'differentially expressed genes in this cell type',
        'OCR_activity': 'open chromatin regions active in cell types',
        'OCR_activity_in': 'open chromatin regions active in this cell type',
        # OCR relationships
        'OCR_locate_in': 'OCR regions located in genes',
        # SNP relationships
        'part_of_GWAS_signal': 'SNPs associated with diseases via GWAS',
        'part_of_QTL_signal': 'SNPs associated with gene expression (QTL)',
        # Disease/Ontology relationships (incoming)
        'effector_gene_of_in': 'genes that are effectors of this disease',
        'signal_COLOC_with_in': 'genes with colocalization signals',
        'part_of_GWAS_signal_in': 'SNPs with GWAS signals for this disease',
        'function_annotation_in': 'genes annotated with this GO term',
    }
    
    def _format_entity_seed(
        self,
        seed_entity: Optional[str] = None,
        seed_entity_type: Optional[str] = None,
        seed_relationship: Optional[str] = None,
        experience_keywords: Optional[List[str]] = None,
        include_examples: bool = False,  # Default OFF for RL training
    ) -> str:
        """
        Format the entity + relationship seed section for diverse question generation.
        
        KEY INNOVATION: Sample BOTH entity AND relationship type together.
        This ensures questions are about relationships that actually exist in the database.
        
        IMPORTANT FOR RL TRAINING:
        - include_examples=False (default): LLM must discover good question patterns
          through reward signal. This encourages exploration and learning.
        - include_examples=True: Only for debugging or initial bootstrapping.
          May cause LLM to over-rely on templates instead of learning.
        
        Args:
            seed_entity: Specific entity to focus on (e.g., "INS", "Beta Cell")
            seed_entity_type: Type of the entity (e.g., "gene", "cell_type")
            seed_relationship: Relationship type to include (e.g., "expression_level_in")
            experience_keywords: Keywords from successful patterns
            include_examples: Whether to include example question patterns (default: False)
            
        Returns:
            Formatted seed section string
        """
        lines = []
        
        # CASE 1: Both entity AND relationship provided (from adaptive sampler)
        # This is the PREFERRED case - forces specific, answerable questions
        if seed_entity and seed_relationship:
            lines.append("🎯 QUESTION SEED (MANDATORY - your question MUST be about this):")
            
            # Format entity
            type_display = (seed_entity_type or 'entity').replace('_', ' ')
            lines.append(f"   Entity: {type_display} \"{seed_entity}\"")
            
            # Format relationship with description
            rel_display = seed_relationship.replace('_', ' ').replace(' in', '')
            rel_desc = self.RELATIONSHIP_DESCRIPTIONS.get(seed_relationship, rel_display)
            lines.append(f"   Relationship: {rel_desc}")
            
            # OPTIONAL: Example question (OFF by default for RL training)
            # When OFF, LLM learns question patterns through reward, not imitation
            if include_examples:
                example = self._get_seed_example(seed_entity, seed_entity_type, seed_relationship)
                if example:
                    lines.append(f"   Example pattern: \"{example}\"")
            
            lines.append("")
            lines.append("   ⚠️ Your question MUST involve this entity AND this relationship type!")
        
        # CASE 2: Only entity provided (uniform sampling fallback)
        elif seed_entity and seed_entity_type:
            lines.append("🎯 ENTITY FOCUS (MANDATORY):")
            type_display = seed_entity_type.replace('_', ' ')
            lines.append(f"   Your question MUST be about the {type_display}: \"{seed_entity}\"")
        
        elif seed_entity:
            lines.append("🎯 ENTITY FOCUS (MANDATORY):")
            lines.append(f"   Your question MUST include the entity: \"{seed_entity}\"")
        
        # CASE 3: Only relationship provided
        elif seed_relationship:
            lines.append("🎯 RELATIONSHIP FOCUS (MANDATORY):")
            rel_desc = self.RELATIONSHIP_DESCRIPTIONS.get(seed_relationship, seed_relationship.replace('_', ' '))
            lines.append(f"   Your question MUST be about: {rel_desc}")
        
        # Experience-based keywords (suggests topics that worked well before)
        if experience_keywords:
            if lines:
                lines.append("")
            keywords_str = ", ".join(experience_keywords[:5])
            lines.append(f"💡 Suggested topics (from successful patterns): {keywords_str}")
        
        # If no seed provided, note that any topic is ok
        if not seed_entity and not seed_relationship and not experience_keywords:
            lines = ["🎯 TOPIC: Any biomedical question within scope constraints"]
        
        return '\n'.join(lines)
    
    def _get_seed_example(
        self,
        seed_entity: str,
        seed_entity_type: Optional[str],
        seed_relationship: str,
    ) -> Optional[str]:
        """
        Generate an example question pattern based on the seed.
        
        This helps the LLM understand what kind of question to generate.
        """
        # Gene + relationship examples
        if seed_entity_type == 'gene':
            if 'expression_level_in' in seed_relationship:
                return f"In which cell types is {seed_entity} expressed?"
            elif 'DEG_in' in seed_relationship:
                return f"In which cell types is {seed_entity} differentially expressed?"
            elif 'physical_interaction' in seed_relationship:
                return f"Which genes physically interact with {seed_entity}?"
            elif 'function_annotation' in seed_relationship:
                return f"What are the Gene Ontology annotations for {seed_entity}?"
            elif 'effector_gene_of' in seed_relationship:
                return f"What diseases is {seed_entity} an effector gene of?"
            elif 'signal_COLOC' in seed_relationship:
                return f"Which disease signals colocalize with {seed_entity}?"
        
        # Cell type + relationship examples
        elif seed_entity_type == 'cell_type':
            if 'expression_level_in' in seed_relationship:
                return f"Which genes are expressed in {seed_entity}?"
            elif 'DEG_in' in seed_relationship:
                return f"Which genes are differentially expressed in {seed_entity}?"
            elif 'OCR_activity' in seed_relationship:
                return f"Which open chromatin regions are active in {seed_entity}?"
        
        # SNP + relationship examples
        elif seed_entity_type == 'snp':
            if 'GWAS' in seed_relationship:
                return f"Which diseases is {seed_entity} associated with via GWAS?"
            elif 'QTL' in seed_relationship:
                return f"Which genes does {seed_entity} affect via QTL?"
        
        # Disease + relationship examples
        elif seed_entity_type == 'disease':
            if 'effector_gene' in seed_relationship:
                return f"Which genes are effector genes of {seed_entity}?"
            elif 'GWAS' in seed_relationship:
                return f"Which SNPs are associated with {seed_entity}?"
        
        # Gene ontology examples
        elif seed_entity_type == 'gene_ontology' or seed_entity_type == 'ontology':
            if 'function_annotation' in seed_relationship:
                return f"Which genes are annotated with {seed_entity}?"
            elif 'effector_gene' in seed_relationship:
                return f"Which genes are effector genes related to {seed_entity}?"
            elif 'signal_COLOC' in seed_relationship:
                return f"Which genes have colocalization signals with {seed_entity}?"
        
        # OCR examples
        elif seed_entity_type == 'OCR':
            if 'OCR_activity' in seed_relationship:
                return f"In which cell types is {seed_entity} active?"
            elif 'OCR_locate_in' in seed_relationship:
                return f"Which genes does {seed_entity} regulate?"
        
        return None
    
    def _format_valid_entities(
        self,
        valid_entities: Optional[Dict[str, Any]],
    ) -> str:
        """
        Format valid entities section for question generation.
        
        This is CRITICAL for preventing the model from asking questions about
        entities that don't exist in the database (like "Macrophage Cell").
        
        Args:
            valid_entities: Dict containing cell_types, diseases, go_terms, tips
            
        Returns:
            Formatted valid entities section string
        """
        if not valid_entities:
            return ""
        
        lines = ["⚠️ VALID ENTITIES (use ONLY these exact names in questions):"]
        
        # Cell types - most important for avoiding errors
        cell_types = valid_entities.get('cell_types', [])
        if cell_types:
            ct_str = ", ".join(f'"{ct}"' for ct in cell_types)
            lines.append(f"   Cell types: {ct_str}")
            # Add warning about invalid cell types
            note = valid_entities.get('cell_types_note', '')
            if note:
                lines.append(f"   ⛔ {note}")
        
        # Diseases
        diseases = valid_entities.get('diseases', [])
        if diseases:
            d_str = ", ".join(f'"{d}"' for d in diseases)
            lines.append(f"   Diseases: {d_str}")
        
        # Sample GO terms (not exhaustive, but gives examples)
        go_terms = valid_entities.get('sample_gene_ontology_terms', [])
        if go_terms:
            # Show just a few examples
            go_examples = go_terms[:5]
            go_str = ", ".join(f'"{g}"' for g in go_examples)
            lines.append(f"   Sample GO terms: {go_str}")
            go_note = valid_entities.get('go_terms_note', '')
            if go_note:
                lines.append(f"   💡 {go_note}")
        
        # Tips for generating good questions
        tips = valid_entities.get('tips', [])
        if tips:
            lines.append("   Tips:")
            for tip in tips[:3]:  # Limit to 3 tips
                lines.append(f"     • {tip}")
        
        return '\n'.join(lines)
    
    def _format_curriculum(
        self,
        difficulty: str,
        constraints: Dict[str, Any],
    ) -> str:
        """Format curriculum constraints."""
        max_hops = constraints.get('max_hops', 3)
        node_types = constraints.get('node_types', [])
        relationship_types = constraints.get('relationship_types', [])
        focus_area = constraints.get('focus_area', 'general biomedical queries')
        
        lines = [
            f"Difficulty: {difficulty.upper()}",
            f"Max hops: {max_hops}",
        ]
        
        if node_types:
            lines.append(f"Focus nodes: {', '.join(node_types[:5])}")
        if relationship_types:
            lines.append(f"Focus relationships: {', '.join(relationship_types[:5])}")
        if focus_area:
            lines.append(f"Focus: {focus_area}")
        
        return '\n'.join(lines)
    
    def _format_schema_summary(self, schema: Dict, max_tokens: int) -> str:
        """Format schema as concise summary with descriptions."""
        # Handle nested schema structure
        if 'knowledge_graph_schema' in schema:
            schema = schema['knowledge_graph_schema']
        
        node_types = schema.get('node_types', {})
        edge_types = schema.get('edge_types', {})
        
        lines = []
        
        # Node types
        if node_types:
            lines.append("Node Types:")
            for node_name, node_info in list(node_types.items())[:10]:
                simple_name = node_name.split(';')[-1] if ';' in node_name else node_name
                desc = ''
                if isinstance(node_info, dict):
                    desc = node_info.get('description', '')
                    if desc:
                        desc = f": {desc[:60]}..." if len(desc) > 60 else f": {desc}"
                lines.append(f"  - {simple_name}{desc}")
        
        lines.append("")
        
        # Edge types
        if edge_types:
            lines.append("Relationships:")
            for edge_name, edge_info in list(edge_types.items())[:12]:
                if isinstance(edge_info, dict):
                    source = edge_info.get('source_node_type', '?')
                    target = edge_info.get('target_node_type', '?')
                    source = source.split(';')[-1] if ';' in source else source
                    target = target.split(';')[-1] if ';' in target else target
                    lines.append(f"  - [:{edge_name}]: {source} → {target}")
                else:
                    lines.append(f"  - [:{edge_name}]")
        
        summary = '\n'.join(lines)
        return self.truncate_text(summary, max_tokens)
    
    def _format_scope_constraints(self, scope: Dict, max_tokens: int) -> str:
        """Format scope constraints with allowed/avoided patterns."""
        allowed_nodes = scope.get('allowed_node_types', [])
        allowed_edges = scope.get('allowed_edge_types', [])
        avoid_regions = scope.get('avoid_regions', [])
        semantic_warnings = scope.get('semantic_warnings', [])
        
        lines = []
        
        # Allowed types
        if allowed_nodes:
            lines.append(f"✓ Allowed nodes: {', '.join(allowed_nodes[:8])}")
        if allowed_edges:
            lines.append(f"✓ Allowed relationships: {', '.join(allowed_edges[:8])}")
        
        if not allowed_nodes and not allowed_edges:
            lines.append("✓ All node and relationship types allowed")
        
        # Regions to avoid
        if avoid_regions:
            lines.append("\nAvoid these patterns:")
            for region in avoid_regions[:3]:
                if isinstance(region, dict):
                    desc = region.get('description', str(region))
                    severity = region.get('severity', 'medium')
                    icon = "🔴" if severity == 'high' else "🟡"
                    lines.append(f"  {icon} {desc[:80]}")
                else:
                    lines.append(f"  ⚠️ {str(region)[:80]}")
        
        # Semantic warnings
        if semantic_warnings:
            lines.append("\nSemantic ambiguities (known issues):")
            for warning in semantic_warnings[:2]:
                if isinstance(warning, dict):
                    edge = warning.get('edge_type', '?')
                    desc = warning.get('description', '')
                    rec = warning.get('recommendation', '')
                    lines.append(f"  ⚠️ [:{edge}]: {desc[:60]}")
                    if rec:
                        lines.append(f"      → {rec[:60]}")
                else:
                    lines.append(f"  ⚠️ {str(warning)[:80]}")
        
        text = '\n'.join(lines)
        return self.truncate_text(text, max_tokens)
    
    def _format_recent_questions(self, questions: List[str], max_tokens: int) -> str:
        """Format recent questions for diversity check."""
        if not questions:
            return "(No recent questions - first batch)"
        
        # Show most recent first (reduced from 25 to 15 to save tokens)
        lines = []
        for i, q in enumerate(questions[-15:], 1):  # Last 15 questions
            # Truncate long questions
            q_display = q[:70] + "..." if len(q) > 70 else q
            lines.append(f"{i}. {q_display}")
        
        text = '\n'.join(lines)
        return self.truncate_text(text, max_tokens)
    
    def _get_question_gen_instruction(self, difficulty: str) -> str:
        """Get task instruction for question generation."""
        requirements = {
            'easy': (
                "1-2 hops only, single entity focus, straightforward retrieval.\n"
                "Example: 'What genes are associated with diabetes?' or 'List SNPs on chromosome 6.'"
            ),
            'medium': (
                "2-3 hops, multiple filters, moderate complexity.\n"
                "Example: 'Which genes regulate INS and are associated with cell types in pancreas?'"
            ),
            'hard': (
                "3-5 hops, complex patterns, multiple entity types, aggregation.\n"
                "Example: 'What are the regulatory pathways connecting MAFA to beta cell differentiation markers?'"
            ),
        }
        
        req = requirements.get(difficulty, requirements['medium'])
        
        return f"""Generate ONE biomedical question that:
1. Can be answered using the knowledge graph with Cypher queries
2. Matches {difficulty.upper()} difficulty level:
   {req}
3. Is DIFFERENT from the recent questions listed above
4. Stays within scope constraints (allowed types, avoid bad regions)
5. Is clear, specific, and answerable

CRITICAL OUTPUT FORMAT:
- Output ONLY the question itself, nothing else
- Do NOT add prefixes like "Question:", "###", "**" or markdown formatting
- Do NOT add explanations or follow-up text
- Just the plain question ending with "?"

Example good output: Which genes are expressed in Beta Cells?
Example bad output: **Question:** Which genes are expressed in Beta Cells?"""
    
    # =========================================================================
    # Role 2: Data Quality Evaluation
    # =========================================================================
    
    def build_data_quality_eval_prompt(
        self,
        question: str,
        trajectory: List[Dict[str, Any]],
        known_semantic_issues: List[str],
        deep_think_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build prompt for data quality evaluation role.
        
        Token budget: 2,350 tokens
        
        Args:
            question: Original question
            trajectory: List of query steps with results
            known_semantic_issues: List of known semantic issue warnings
            deep_think_hints: Optional list of hints from Deep Think analysis for data evaluation
            
        Returns:
            Formatted prompt string
        """
        # Build sections
        question_text = self.truncate_text(question, self.DATA_EVAL_QUESTION_TOKENS)
        trajectory_text = self._format_trajectory_for_eval(trajectory, self.DATA_EVAL_TRAJECTORY_TOKENS)
        semantic_text = self._format_semantic_issues(known_semantic_issues, self.DATA_EVAL_SEMANTIC_TOKENS)
        criteria_text = self._get_data_eval_criteria()
        
        # Deep Think hints section (NEW)
        hints_text = self._format_deep_think_hints(deep_think_hints, role='data_eval')
        hints_section = f"\n{hints_text}\n" if hints_text else ""
        
        # Build semantic issues section if available
        semantic_section = ""
        if semantic_text and semantic_text != "(No known semantic issues identified yet)":
            semantic_section = f"""
KNOWN SEMANTIC ISSUES (watch for these):
{semantic_text}

"""

        prompt = f"""You are a data quality evaluator for knowledge graph retrieval.
Assess if the retrieved data can answer the question.

QUESTION:
{question_text}
{hints_section}
QUERY RESULTS:
{trajectory_text}
{semantic_section}
{criteria_text}

IMPORTANT: If no data was retrieved (empty results), scores should be LOW (0.1-0.2).

OUTPUT FORMAT - respond with ONLY this JSON (no other text):
```json
{{
    "data_quality_score": <0.0-1.0>,
    "relevance_score": <0.0-1.0>,
    "completeness_score": <0.0-1.0>,
    "trajectory_quality_score": <0.0-1.0>,
    "could_answer_question": <true or false>,
    "doubt_level": <0.0-1.0>,
    "reasoning": "<one sentence explanation>"
}}
```

Score Guidelines:
- 0.0-0.2: No data or completely irrelevant
- 0.3-0.5: Partial data, some relevance
- 0.6-0.8: Good data, mostly complete
- 0.9-1.0: Excellent, fully answers question"""
        
        return prompt
    
    def _format_trajectory_for_eval(self, trajectory: List[Dict], max_tokens: int) -> str:
        """Format trajectory with queries and results for evaluation."""
        lines = []
        
        for i, step in enumerate(trajectory[:5], 1):  # Max 5 steps
            query = step.get('query', '')
            result = step.get('result', {})
            exec_time = step.get('execution_time_ms', result.get('execution_time_ms', 0))
            num_results = step.get('num_results', result.get('num_results', 0))
            success = step.get('success', result.get('success', True))
            has_data = step.get('has_data', result.get('has_data', True))
            
            lines.append(f"--- Step {i} ---")
            
            # Query (truncate long queries)
            if len(query) > 300:
                query_display = query[:300] + "..."
            else:
                query_display = query
            lines.append(f"Query: {query_display}")
            
            # Execution info
            status = "✓" if success else "❌ Error"
            if success and not has_data:
                status = "⚠️ No results"
            lines.append(f"Execution: {exec_time:.0f}ms, {num_results} results {status}")
            
            # Data summary (if available)
            data_summary = step.get('data_summary', result.get('data_summary', ''))
            if data_summary:
                lines.append(f"Data: {data_summary[:200]}...")
            
            # Actual result data (simplified)
            result_data = result.get('result', {})
            if result_data and isinstance(result_data, dict):
                # Show first few items
                data_preview = str(result_data)[:300]
                lines.append(f"Preview: {data_preview}...")
            
            lines.append("")
        
        text = '\n'.join(lines)
        return self.truncate_text(text, max_tokens)
    
    def _format_semantic_issues(self, issues: List[str], max_tokens: int) -> str:
        """Format known semantic issues."""
        if not issues:
            return "(No known semantic issues identified yet)"
        
        lines = []
        for issue in issues[:5]:
            if isinstance(issue, str):
                lines.append(f"- {issue[:100]}")
            elif isinstance(issue, dict):
                edge = issue.get('edge_type', '?')
                desc = issue.get('description', str(issue))
                lines.append(f"- [:{edge}]: {desc[:80]}")
        
        text = '\n'.join(lines)
        return self.truncate_text(text, max_tokens)
    
    def _get_data_eval_criteria(self) -> str:
        """Get evaluation criteria for data quality."""
        return """EVALUATION CRITERIA:
- RELEVANCE: Does data address the question? (0=unrelated, 1=directly answers)
- COMPLETENESS: Is enough data retrieved? (0=nothing, 1=all needed info)
- TRAJECTORY: Were queries efficient? (0=wasted steps, 1=optimal path)
- DOUBT: How skeptical are you? (0=confident, 1=very suspicious)
- CAN ANSWER: Can the question be answered with this data? (true/false)

CRITICAL: Empty results ([], no data) = LOW scores (0.1-0.2), could_answer_question=false"""
    
    # =========================================================================
    # Role 3: Answer Synthesis
    # =========================================================================
    
    def build_answer_synthesis_prompt(
        self,
        question: str,
        trajectory_data: List[Dict[str, Any]],
        data_quality_feedback: Dict[str, Any],
        deep_think_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build prompt for answer synthesis role.
        
        Token budget: 3,350 tokens
        
        Args:
            question: Original question
            trajectory_data: Retrieved data from queries
            data_quality_feedback: Data quality scores from evaluation
            deep_think_hints: Optional list of hints from Deep Think analysis for synthesis
            
        Returns:
            Formatted prompt string
        """
        question_text = self.truncate_text(question, self.SYNTHESIS_QUESTION_TOKENS)
        data_text = self._format_trajectory_data(trajectory_data, self.SYNTHESIS_DATA_TOKENS)
        quality_text = self._format_quality_feedback(data_quality_feedback, self.SYNTHESIS_QUALITY_TOKENS)
        instruction_text = self._get_synthesis_instruction()
        
        # Deep Think hints section (NEW)
        hints_text = self._format_deep_think_hints(deep_think_hints, role='synthesis')
        hints_section = f"\n{hints_text}\n" if hints_text else ""
        
        prompt = f"""You are a biomedical answer synthesizer.
Your task is to convert retrieved knowledge graph data into a clear, accurate natural language answer.

QUESTION:
{question_text}
{hints_section}
RETRIEVED DATA:
{data_text}

DATA QUALITY ASSESSMENT:
{quality_text}

TASK:
{instruction_text}

Generate a comprehensive answer:"""
        
        return prompt
    
    def _format_trajectory_data(self, trajectory_data: List[Dict], max_tokens: int) -> str:
        """Format trajectory data for synthesis.
        
        Parses Cypher notation strings to extract comprehensive entity data
        including names, types, descriptions, chromosomes, and IDs.
        """
        lines = []
        
        for i, step_data in enumerate(trajectory_data[:5], 1):
            query = step_data.get('query', '')
            result = step_data.get('result', {})
            num_results = step_data.get('num_results', 0)
            
            lines.append(f"--- Query {i} Results ---")
            
            # Show query pattern with semantic meaning
            if query:
                node_types = re.findall(r':(\w+)', query)[:3]
                rel_types = re.findall(r'\[r?[0-9]*:(\w+)\]', query)
                pattern = ', '.join(set(node_types)) if node_types else "query"
                lines.append(f"Pattern: {pattern}")
                
                # Add semantic meaning of relationship types
                rel_meanings = {
                    'DEG_in': 'Differentially Expressed Genes in',
                    'expression_level_in': 'gene expression levels in',
                    'function_annotation': 'Gene Ontology annotations',
                    'physical_interaction': 'protein-protein interactions',
                    'effector_gene_of': 'effector genes of disease',
                    'part_of_GWAS_signal': 'GWAS associations with disease',
                    'part_of_QTL_signal': 'QTL/eQTL effects on genes',
                    'OCR_activity': 'Open Chromatin Regions in',
                    'regulation': 'gene regulatory relationships',
                }
                for rel in rel_types:
                    if rel in rel_meanings:
                        lines.append(f"Relationship meaning: {rel_meanings[rel]}")
            
            lines.append(f"Found: {num_results} items")
            
            # Format result data
            if isinstance(result, dict):
                results_value = result.get('results', '')
                
                # Parse Cypher notation string to extract full entity data
                if isinstance(results_value, str) and results_value.strip():
                    # Pattern matches (:label {properties}) or (:label:label2 {properties})
                    entity_pattern = r'\(:([^\s\{]+)\s*\{([^\}]+)\}\)'
                    entity_matches = re.findall(entity_pattern, results_value)
                    
                    if entity_matches:
                        # Parse each entity's properties
                        entities = []
                        for labels, props_str in entity_matches:
                            entity = {'_labels': labels}
                            # Extract key properties: name, description, type, chr, id
                            name_match = re.search(r'name:\s*["\']([^"\'\,\}]+)["\']', props_str)
                            if name_match:
                                entity['name'] = name_match.group(1)
                            desc_match = re.search(r'description:\s*["\']([^"\'\}]+)["\']', props_str)
                            if desc_match:
                                entity['description'] = desc_match.group(1)
                            type_match = re.search(r'type:\s*["\']([^"\'\,\}]+)["\']', props_str)
                            if type_match:
                                entity['type'] = type_match.group(1)
                            chr_match = re.search(r'chr:\s*["\']?([^"\'\,\}]+)["\']?', props_str)
                            if chr_match:
                                entity['chr'] = chr_match.group(1).strip('"\'')
                            id_match = re.search(r'(?:^|,\s*)id:\s*["\']([^"\'\,\}]+)["\']', props_str)
                            if id_match:
                                entity['id'] = id_match.group(1)
                            entities.append(entity)
                        
                        # Deduplicate by name
                        seen_names = set()
                        unique_entities = []
                        for e in entities:
                            name = e.get('name', '')
                            if name and name not in seen_names:
                                seen_names.add(name)
                                unique_entities.append(e)
                        
                        lines.append(f"Entities ({len(unique_entities)} unique):")
                        
                        # Show up to 50 entities (enough for comprehensive answers)
                        for e in unique_entities[:50]:
                            name = e.get('name', 'Unknown')
                            labels = e.get('_labels', '').split(':')[-1]  # Get last label
                            entity_type = e.get('type', '')
                            chromosome = e.get('chr', '')
                            description = e.get('description', '')
                            entity_id = e.get('id', '')
                            
                            # Build comprehensive entity line
                            details = []
                            if labels:
                                details.append(labels)
                            if entity_type:
                                details.append(entity_type)
                            if chromosome:
                                details.append(f"chr{chromosome}")
                            if entity_id:
                                details.append(f"ID:{entity_id}")
                            
                            detail_str = ', '.join(details) if details else ''
                            
                            if description:
                                # Truncate long descriptions but keep more info
                                desc_short = description[:80] + "..." if len(description) > 80 else description
                                lines.append(f"  - {name} ({detail_str}): {desc_short}")
                            else:
                                lines.append(f"  - {name} ({detail_str})")
                        
                        if len(unique_entities) > 50:
                            lines.append(f"  ... and {len(unique_entities) - 50} more entities")
                    
                    else:
                        # No entity pattern found - show raw data (up to 2000 chars)
                        lines.append("Raw data:")
                        raw_preview = results_value[:2000]
                        lines.append(f"  {raw_preview}")
                        if len(results_value) > 2000:
                            lines.append(f"  ... ({len(results_value) - 2000} more chars)")
                
                # Handle other dict keys (but skip results since we handled it above)
                else:
                    for key, value in list(result.items())[:5]:
                        if key == 'results':
                            continue  # Already handled above
                        if isinstance(value, list):
                            lines.append(f"  {key}: {len(value)} items")
                            for item in value[:3]:
                                item_str = str(item)[:150]
                                lines.append(f"    - {item_str}")
                            if len(value) > 3:
                                lines.append(f"    ... and {len(value) - 3} more")
                        elif isinstance(value, str):
                            lines.append(f"  {key}: {value[:150]}")
                        elif value is not None:
                            lines.append(f"  {key}: {str(value)[:150]}")
            else:
                result_str = str(result)[:500]  # Reasonable limit for non-dict data
                lines.append(f"  Data: {result_str}")
            
            lines.append("")
        
        text = '\n'.join(lines)
        # Skip truncation if max_tokens=0 (unlimited)
        if max_tokens > 0:
            return self.truncate_text(text, max_tokens)
        return text
    
    def _format_quality_feedback(self, feedback: Dict, max_tokens: int) -> str:
        """Format data quality feedback."""
        # Handle None values explicitly (dict.get returns None if key exists but value is None)
        relevance = feedback.get('relevance_score')
        completeness = feedback.get('completeness_score')
        quality = feedback.get('data_quality_score')
        could_answer = feedback.get('could_answer_question')
        doubt = feedback.get('doubt_level')
        
        # Default values for None
        relevance = relevance if relevance is not None else 0.5
        completeness = completeness if completeness is not None else 0.5
        quality = quality if quality is not None else 0.5
        could_answer = could_answer if could_answer is not None else True
        doubt = doubt if doubt is not None else 0.0
        
        # If quality scores are very low, override could_answer
        if quality < 0.2 or relevance < 0.2:
            could_answer = False
        
        def score_label(score: float) -> str:
            if score is None:
                return "Unknown"
            if score >= 0.7:
                return f"High ({score:.2f})"
            elif score >= 0.4:
                return f"Medium ({score:.2f})"
            else:
                return f"Low ({score:.2f})"
        
        lines = [
            f"Relevance: {score_label(relevance)}",
            f"Completeness: {score_label(completeness)}",
            f"Data Quality: {score_label(quality)}",
            f"Can Answer: {'Yes' if could_answer else 'No - insufficient data'}",
        ]
        
        if doubt is not None and doubt > 0.5:
            lines.append(f"⚠️ Doubt Level: {doubt:.2f} - exercise caution")
        
        # Add warning for very low quality
        if quality < 0.2:
            lines.append("\n⚠️ WARNING: Retrieved data is insufficient to answer this question.")
            lines.append("State that the question cannot be answered with available data.")
        
        # Include reasoning if available
        reasoning = feedback.get('reasoning', '')
        if reasoning:
            lines.append(f"\nNote: {reasoning[:100]}")
        
        text = '\n'.join(lines)
        return self.truncate_text(text, max_tokens)
    
    def _get_synthesis_instruction(self) -> str:
        """Get task instruction for answer synthesis."""
        return """KNOWLEDGE GRAPH RELATIONSHIP GLOSSARY:
(The relationship types in the data DIRECTLY answer the question!)
- DEG_in = "Differentially Expressed Gene in" (genes returned via DEG_in ARE differentially expressed)
- expression_level_in = gene expression level in cell type
- function_annotation = Gene Ontology annotation (GO terms)
- physical_interaction = protein-protein interaction
- effector_gene_of = gene is effector of disease
- part_of_GWAS_signal = SNP associated with disease via GWAS
- part_of_QTL_signal = SNP affects gene expression (eQTL)
- OCR_activity = Open Chromatin Region active in cell type

CRITICAL: If the query pattern matches the question topic, the returned data ANSWERS the question!
Example: Question "Which genes are differentially expressed in Alpha Cells?" + Pattern "DEG_in"
         → The returned genes ARE differentially expressed in Alpha Cells. Answer directly!

SYNTHESIS GUIDELINES:

IF DATA QUALITY IS LOW or "Can Answer: No":
- State clearly: "Based on the retrieved data, this question cannot be fully answered."
- Explain what data was missing or why it's insufficient
- Do NOT make up information

IF DATA IS AVAILABLE:
1. Start with direct answer to the question
2. List specific entities found (genes, diseases, etc.)
3. Use bullet points for multiple items
4. Cite numbers from the data (e.g., "Found 42 genes...")
5. The relationship type tells you HOW the entities relate to the question

IMPORTANT: Only use information from the RETRIEVED DATA section above. Do not add external knowledge."""
    
    # =========================================================================
    # Role 4: Answer Quality Evaluation
    # =========================================================================
    
    def build_answer_quality_eval_prompt(
        self,
        question: str,
        answer: str,
        deep_think_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build prompt for answer quality evaluation role.
        
        Token budget: 1,650 tokens
        
        Args:
            question: Original question
            answer: Synthesized answer to evaluate
            deep_think_hints: Optional list of hints from Deep Think analysis for evaluation
            
        Returns:
            Formatted prompt string
        """
        question_text = self.truncate_text(question, self.ANSWER_EVAL_QUESTION_TOKENS)
        answer_text = self.truncate_text(answer, self.ANSWER_EVAL_ANSWER_TOKENS)
        criteria_text = self._get_answer_eval_criteria()
        format_text = self._get_answer_eval_format()
        
        # Deep Think hints section (NEW)
        hints_text = self._format_deep_think_hints(deep_think_hints, role='answer_eval')
        hints_section = f"\n{hints_text}\n" if hints_text else ""
        
        prompt = f"""You are an objective answer quality evaluator.
Your task is to evaluate how well the answer addresses the question.

QUESTION:
{question_text}
{hints_section}
ANSWER TO EVALUATE:
{answer_text}

EVALUATION CRITERIA:
{criteria_text}

{format_text}"""
        
        return prompt
    
    def _get_answer_eval_criteria(self) -> str:
        """Get evaluation criteria for answer quality."""
        return """Evaluate objectively on these dimensions:

1. CORRECTNESS (0-1): Does it answer the question asked?
   - High: Directly addresses the question
   - Low: Off-topic or incorrect

2. COMPLETENESS (0-1): Is important information included?
   - High: Comprehensive coverage
   - Low: Missing key details

3. CLARITY (0-1): Is it well-formatted and understandable?
   - High: Clear, organized, easy to follow
   - Low: Confusing, poorly structured

4. ACCURACY (0-1): Are facts/numbers correct based on the data?
   - High: Accurate, properly cited
   - Low: Contains errors or misinterpretations

OVERALL SCORE: Weighted average (Correctness: 35%, Completeness: 25%, Clarity: 20%, Accuracy: 20%)"""
    
    def _get_answer_eval_format(self) -> str:
        """Get output format for answer evaluation."""
        return """OUTPUT FORMAT - respond with ONLY this JSON (no other text):
```json
{
    "score": <0.0-1.0 overall quality>,
    "correctness": <0.0-1.0>,
    "completeness": <0.0-1.0>,
    "clarity": <0.0-1.0>,
    "accuracy": <0.0-1.0>,
    "reasoning": "<one sentence explanation>"
}
```

Score Guidelines:
- 0.0-0.3: Poor (wrong, unclear, or missing key info)
- 0.4-0.6: Acceptable (partially correct, some issues)
- 0.7-0.8: Good (mostly correct, clear)
- 0.9-1.0: Excellent (fully correct, complete, clear)"""
    
    # =========================================================================
    # Validation Utilities
    # =========================================================================
    
    def validate_prompt(self, prompt: str, role: str) -> Dict[str, Any]:
        """
        Validate a prompt against its role's budget.
        
        Args:
            prompt: The prompt to validate
            role: Role name ('question_gen', 'data_eval', 'synthesis', 'answer_eval')
            
        Returns:
            Dictionary with validation results
        """
        budgets = {
            'question_gen': self.QUESTION_GEN_BUDGET,
            'data_eval': self.DATA_EVAL_BUDGET,
            'synthesis': self.SYNTHESIS_BUDGET,
            'answer_eval': self.ANSWER_EVAL_BUDGET,
        }
        
        budget = budgets.get(role, 2000)
        token_count = self.estimate_tokens(prompt)
        
        # Handle unlimited budget (0)
        if budget == 0:
            return {
                'valid': True,  # Always valid for unlimited
                'token_count': token_count,
                'budget': 'unlimited',
                'utilization': 0,
                'remaining': float('inf'),
                'role': role,
            }
        
        return {
            'valid': token_count <= budget,
            'token_count': token_count,
            'budget': budget,
            'utilization': token_count / budget,
            'remaining': budget - token_count,
            'role': role,
        }
    
    def get_token_stats(self, prompt: str) -> Dict[str, int]:
        """
        Get detailed token statistics for a prompt.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dictionary with section-wise token counts
        """
        # Split by common section markers
        sections = {}
        
        # Try to identify sections
        patterns = [
            (r'QUESTION.*?(?=\n[A-Z]|\Z)', 'question'),
            (r'SCHEMA.*?(?=\n[A-Z]|\Z)', 'schema'),
            (r'TRAJECTORY.*?(?=\n[A-Z]|\Z)', 'trajectory'),
            (r'ANSWER.*?(?=\n[A-Z]|\Z)', 'answer'),
            (r'CRITERIA.*?(?=\n[A-Z]|\Z)', 'criteria'),
        ]
        
        for pattern, name in patterns:
            match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = self.estimate_tokens(match.group())
        
        sections['total'] = self.estimate_tokens(prompt)
        
        return sections
