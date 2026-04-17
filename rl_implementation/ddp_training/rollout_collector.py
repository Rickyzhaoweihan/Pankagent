"""
Rollout Collector for Stage 1 Training.

Collects multi-turn trajectories using:
- Orchestrator for question generation and evaluation (inference only)
- Cypher Generator for query generation
- Neo4j for query execution
- GraphReasoningEnvironment for multi-turn interaction

Reuses existing agents and environments from rl_implementation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

# Import existing components
import sys
import random
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl_implementation.agents import CypherGeneratorAgent, OrchestratorAgent, ExperienceBuffer
from rl_implementation.environments import GraphReasoningEnvironment, Neo4jExecutor
from rl_implementation.utils.prompt_builder import PromptBuilder
from rl_implementation.utils.orchestrator_prompt_builder import OrchestratorPromptBuilder
from rl_implementation.utils.entity_extractor import EntitySamples
from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    prompt: str
    response: str
    cypher_query: str  # Already auto-fixed by CypherGeneratorAgent
    execution_result: Dict[str, Any]
    success: bool
    has_data: bool
    execution_time_ms: float
    num_results: int


@dataclass
class Trajectory:
    """Complete trajectory for one question."""
    question: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    
    # Evaluation results
    data_quality_score: float = 0.0
    answer_quality_score: float = 0.0
    trajectory_quality_score: float = 0.0
    doubt_level: float = 0.0
    
    # Semantic issues identified during evaluation
    semantic_issues: List[Dict[str, Any]] = field(default_factory=list)
    problematic_regions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Synthesized answer
    synthesized_answer: str = ""
    
    # =========================================================================
    # Stored Prompts for Training (to avoid truncation/reconstruction issues)
    # =========================================================================
    # Orchestrator Question Generation prompt (Role 1)
    orch_qgen_prompt: str = ""
    
    # Orchestrator Answer Synthesis prompt (Role 3)
    orch_synth_prompt: str = ""
    
    # Token-level data for training
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    response_mask: Optional[torch.Tensor] = None
    
    # Rewards (separate for each trainable model/role)
    reward: float = 0.0  # Legacy: same as cypher_reward for backwards compatibility
    cypher_reward: float = 0.0  # Reward for Cypher Generator
    orch_qgen_reward: float = 0.0  # Reward for Orchestrator Question Generation (Role 1)
    orch_synth_reward: float = 0.0  # Reward for Orchestrator Answer Synthesis (Role 3)
    
    # Reward metadata (for debugging/analysis)
    reward_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    @property
    def total_results(self) -> int:
        return sum(step.num_results for step in self.steps)
    
    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for step in self.steps if step.success) / len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and persistence."""
        return {
            "question": self.question,
            "num_steps": self.num_steps,
            "total_results": self.total_results,
            "success_rate": self.success_rate,
            "data_quality_score": self.data_quality_score,
            "answer_quality_score": self.answer_quality_score,
            "trajectory_quality_score": self.trajectory_quality_score,
            "doubt_level": self.doubt_level,
            "synthesized_answer": self.synthesized_answer[:200] + "..." if len(self.synthesized_answer) > 200 else self.synthesized_answer,
            # Legacy reward (same as cypher_reward for backwards compatibility)
            "reward": self.reward,
            # Separate rewards for each trainable model/role
            "cypher_reward": self.cypher_reward,
            "orch_qgen_reward": self.orch_qgen_reward,
            "orch_synth_reward": self.orch_synth_reward,
            "reward_metadata": self.reward_metadata,
            "semantic_issues": self.semantic_issues,
            "problematic_regions": self.problematic_regions,
            "steps": [
                {
                    "cypher_query": step.cypher_query[:100] + "..." if len(step.cypher_query) > 100 else step.cypher_query,
                    "success": step.success,
                    "has_data": step.has_data,
                    "num_results": step.num_results,
                    "execution_time_ms": step.execution_time_ms,
                }
                for step in self.steps
            ],
        }


@dataclass
class RolloutCollectorConfig:
    """Configuration for rollout collector."""
    # Paths
    schema_path: str = ""
    connection_config: Optional[Dict[str, Any]] = None  # Neo4j connection config (Bolt or HTTP)
    neo4j_url: Optional[str] = None  # DEPRECATED: Use connection_config instead
    experience_buffer_path: Optional[str] = None  # Path to persist experience buffer
    entity_samples_path: Optional[str] = None  # Path to entity samples for seeding
    
    def __post_init__(self):
        """Handle backward compatibility and set defaults."""
        # Handle backward compatibility with neo4j_url
        if self.neo4j_url is not None and self.connection_config is None:
            self.connection_config = {
                "type": "http",
                "url": self.neo4j_url,
                "kg_name": "PankBase"
            }
        
        # Default to PankBase HTTP API if neither provided
        if self.connection_config is None:
            self.connection_config = {
                "type": "http",
                "url": "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j",
                "kg_name": "PankBase"
            }
    
    # Generation settings
    max_steps: int = 5
    max_prompt_length: int = 4096
    max_response_length: int = 1024
    
    # Batch settings
    batch_size: int = 8
    
    # Experience buffer settings
    max_patterns: int = 100  # Max patterns in experience buffer
    
    # Entity seeding settings
    use_entity_seeding: bool = True  # Enable entity-based question seeding
    use_experience_keywords: bool = True  # Use keywords from successful patterns
    
    # Adaptive entity sampling settings (replaces uniform sampling)
    use_adaptive_sampling: bool = False  # Enable adaptive Thompson Sampling
    entity_degrees_path: Optional[str] = None  # Path to entity degrees JSON
    adaptive_sampler_path: Optional[str] = None  # Path to save/load sampler state
    adaptive_sampler_degree_weight: float = 0.1  # Weight for degree in prior
    adaptive_sampler_slack_threshold: int = 3  # Consecutive failures before full penalty
    
    # Auto-fix settings (auto-fix is built into CypherGeneratorAgent)
    enable_auto_fix: bool = True  # Enable automatic Cypher query fixing in agent
    
    # Temperature for question generation
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Deep Think prompt hints (injected from training analysis)
    prompt_hints_path: Optional[str] = None  # Path to prompt hints JSON


class RolloutCollector:
    """
    Collects rollout trajectories for training.
    
    Uses:
    - InferenceEngine for model inference (both Cypher Generator and Orchestrator)
    - CypherGeneratorAgent for prompt building and response parsing
    - OrchestratorAgent for evaluation prompt building and response parsing
    - GraphReasoningEnvironment for multi-turn interaction logic
    - Neo4jExecutor for query execution
    """
    
    def __init__(
        self,
        config: RolloutCollectorConfig,
        inference_engine,  # InferenceEngine instance
        cypher_tokenizer: AutoTokenizer,
    ):
        """
        Initialize rollout collector.
        
        Args:
            config: RolloutCollectorConfig
            inference_engine: InferenceEngine for model inference
            cypher_tokenizer: Tokenizer for Cypher Generator
        """
        self.config = config
        self.inference_engine = inference_engine
        self.tokenizer = cypher_tokenizer
        
        # Load schema
        with open(config.schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Initialize components
        self.experience_buffer = ExperienceBuffer(
            max_patterns=config.max_patterns,
            persist_path=config.experience_buffer_path,
        )
        # Extract kg_name from schema for logging
        kg_info = self.schema.get('knowledge_graph_schema', {}).get('graph', {})
        kg_name = kg_info.get('name', 'PankBase')
        logger.info(f"Using knowledge graph: {kg_name}")
        
        self.prompt_builder = PromptBuilder(tokenizer=cypher_tokenizer)
        self.orch_prompt_builder = OrchestratorPromptBuilder(tokenizer=cypher_tokenizer)
        
        # Initialize Neo4j executor using connection_config
        # After __post_init__, connection_config is always populated
        conn_config = config.connection_config
        if conn_config.get('type') == 'http':
            neo4j_url = conn_config.get('url')
            kg_name = conn_config.get('kg_name', 'PankBase')
            logger.info(f"Using Neo4j HTTP API for {kg_name}: {neo4j_url}")
            self.neo4j_executor = Neo4jExecutor(api_url=neo4j_url)
        else:
            # For Bolt connections, still use HTTP API for now
            # (Neo4jExecutor only supports HTTP)
            neo4j_url = "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"
            logger.info(f"Using Neo4j HTTP API (Bolt not supported): {neo4j_url}")
            self.neo4j_executor = Neo4jExecutor(api_url=neo4j_url)
        
        # Load Deep Think prompt hints (if available)
        self.prompt_hints = None
        if config.prompt_hints_path and Path(config.prompt_hints_path).exists():
            try:
                from rl_implementation.utils.prompt_hints_manager import PromptHintsManager
                hints_manager = PromptHintsManager(config.prompt_hints_path)
                self.prompt_hints = hints_manager.get_all_hints()
                if hints_manager.has_hints():
                    logger.info(f"Loaded Deep Think prompt hints from {config.prompt_hints_path}")
                    logger.info(f"  Cypher hints: {len(self.prompt_hints.get('cypher_generator', []))}")
                    orch_hints = self.prompt_hints.get('orchestrator', {})
                    orch_count = sum(len(v) for v in orch_hints.values() if isinstance(v, list))
                    logger.info(f"  Orchestrator hints: {orch_count}")
            except Exception as e:
                logger.warning(f"Failed to load prompt hints: {e}")
                self.prompt_hints = None
        
        # Load entity samples for question generation seeding
        self.entity_samples: Optional[EntitySamples] = None
        if config.entity_samples_path and Path(config.entity_samples_path).exists():
            try:
                with open(config.entity_samples_path, 'r') as f:
                    data = json.load(f)
                self.entity_samples = EntitySamples.from_dict(data)
                logger.info(f"Loaded entity samples: {len(self.entity_samples.genes)} genes, "
                           f"{len(self.entity_samples.diseases)} diseases, "
                           f"{len(self.entity_samples.cell_types)} cell types")
            except Exception as e:
                logger.warning(f"Failed to load entity samples from {config.entity_samples_path}: {e}")
        else:
            logger.info("No entity samples loaded - questions will use random topics")
        
        # Load valid entities for question grounding (helps avoid non-existent entities)
        self.valid_entities: Optional[Dict[str, Any]] = None
        valid_entities_path = config.valid_entities_path if hasattr(config, 'valid_entities_path') else None
        if not valid_entities_path:
            # Try default path
            default_path = Path(__file__).parent / "config" / "valid_entities.json"
            if default_path.exists():
                valid_entities_path = str(default_path)
        
        if valid_entities_path and Path(valid_entities_path).exists():
            try:
                with open(valid_entities_path, 'r') as f:
                    self.valid_entities = json.load(f)
                logger.info(f"Loaded valid entities for question grounding: {len(self.valid_entities.get('cell_types', []))} cell types")
            except Exception as e:
                logger.warning(f"Failed to load valid entities from {valid_entities_path}: {e}")
        else:
            logger.info("No valid entities loaded - questions may reference non-existent entities")
        
        # Initialize adaptive entity sampler (replaces uniform sampling when enabled)
        self.adaptive_sampler: Optional[AdaptiveEntitySampler] = None
        if config.use_adaptive_sampling:
            self._init_adaptive_sampler()
        
        # Track current seed relationship for adaptive sampler updates
        self._current_seed_relationships: Dict[str, str] = {}  # question -> relationship
        
        logger.info(f"RolloutCollector initialized with schema from {config.schema_path}")
    
    def generate_questions(
        self,
        num_questions: int,
        difficulty: str = "easy",
        curriculum_constraints: Optional[Dict[str, Any]] = None,
        return_prompts: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Generate training questions using Orchestrator with entity seeding.
        
        Each question gets a DIFFERENT seed entity to ensure diversity:
        - Random entity seeding: picks different genes, diseases, cell types
        - Experience-based keywords: uses keywords from successful past patterns
        
        Args:
            num_questions: Number of questions to generate
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            curriculum_constraints: Optional constraints for curriculum
            return_prompts: If True, return (questions, prompts) tuple for storage
            
        Returns:
            List of generated questions, or (questions, prompts) if return_prompts=True
        """
        logger.info(f"Generating {num_questions} {difficulty} questions with entity seeding...")
        
        # Use experience buffer's recent questions for diversity
        recent_questions = self.experience_buffer.get_recent_questions(n=20)
        scope_constraints = self.experience_buffer.get_scope_constraints()
        
        # Get available entity types for seeding
        entity_types = self._get_available_entity_types()
        
        # Get successful keywords from experience buffer
        experience_keywords_pool = self._get_experience_keywords()
        
        prompts = []
        seeds_used = []  # Track seeds for logging
        
        for i in range(num_questions):
            # Pick entity seed for this question
            seed_entity = None
            seed_entity_type = None
            seed_relationship = None
            experience_keywords = None
            
            if self.config.use_entity_seeding and (self.entity_samples or self.adaptive_sampler):
                # Pick entity seed (uses adaptive sampling when available)
                seed_entity, seed_entity_type, adaptive_relationship = self._pick_random_entity_seed(entity_types)
                
                # Use relationship from adaptive sampler if available
                if adaptive_relationship:
                    seed_relationship = adaptive_relationship
                else:
                    # ALWAYS include relationship for better answerability
                    # Without a relationship, LLM often asks about non-existent connections
                    seed_relationship = self._pick_random_relationship(seed_entity_type)
            
            if self.config.use_experience_keywords and experience_keywords_pool:
                # Sample 2-4 keywords from successful patterns
                num_kw = min(random.randint(2, 4), len(experience_keywords_pool))
                experience_keywords = random.sample(experience_keywords_pool, num_kw)
            
            # Get Deep Think hints for question generation
            qgen_hints = None
            if self.prompt_hints:
                orch_hints = self.prompt_hints.get('orchestrator', {})
                qgen_hints = orch_hints.get('generation', [])
            
            # Build prompt with seeding
            prompt = self.orch_prompt_builder.build_question_generation_prompt(
                schema=self.schema,
                difficulty=difficulty,
                curriculum_constraints=curriculum_constraints or {},
                scope_constraints=scope_constraints,
                recent_questions=recent_questions,
                # Entity seeding
                seed_entity=seed_entity,
                seed_entity_type=seed_entity_type,
                seed_relationship=seed_relationship,
                experience_keywords=experience_keywords,
                # Entity grounding (helps avoid non-existent entities)
                valid_entities=self.valid_entities,
                # Deep Think hints
                deep_think_hints=qgen_hints,
            )
            prompts.append(prompt)
            seeds_used.append(f"{seed_entity_type}:{seed_entity}" if seed_entity else "random")
            
            # Track seed for adaptive sampler update (will be matched to question later)
            if seed_entity and seed_relationship:
                # Store with index since we don't have the question text yet
                self._current_seed_relationships[f"_pending_{i}_entity"] = seed_entity
                self._current_seed_relationships[f"_pending_{i}_relationship"] = seed_relationship
        
        # Log the seeds used
        unique_seeds = len(set(seeds_used))
        logger.info(f"Entity seeds: {unique_seeds} unique seeds for {num_questions} questions")
        logger.debug(f"Seeds used: {seeds_used[:5]}...")
        
        # Generate questions using Orchestrator
        questions = self.inference_engine.generate_questions(prompts)
        
        # Map pending seed tracking to actual questions
        for i, question in enumerate(questions):
            pending_entity = self._current_seed_relationships.pop(f"_pending_{i}_entity", None)
            pending_rel = self._current_seed_relationships.pop(f"_pending_{i}_relationship", None)
            if pending_entity and pending_rel:
                self._current_seed_relationships[f"{question}_entity"] = pending_entity
                self._current_seed_relationships[f"{question}_relationship"] = pending_rel
        
        logger.info(f"Generated {len(questions)} questions")
        
        if return_prompts:
            return questions, prompts
        return questions
    
    def _init_adaptive_sampler(self):
        """Initialize adaptive entity sampler from saved state or degree data."""
        config = self.config
        
        # Try to load saved state first (preserves learned parameters)
        if config.adaptive_sampler_path and Path(config.adaptive_sampler_path).exists():
            try:
                self.adaptive_sampler = AdaptiveEntitySampler.load(config.adaptive_sampler_path)
                logger.info(f"Loaded adaptive sampler from {config.adaptive_sampler_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load adaptive sampler state: {e}")
        
        # Initialize from degree data
        if config.entity_degrees_path and Path(config.entity_degrees_path).exists():
            try:
                self.adaptive_sampler = AdaptiveEntitySampler(
                    entity_degrees_path=config.entity_degrees_path,
                    degree_weight=config.adaptive_sampler_degree_weight,
                    slack_threshold=config.adaptive_sampler_slack_threshold,
                )
                logger.info(f"Initialized adaptive sampler from {config.entity_degrees_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive sampler: {e}")
                self.adaptive_sampler = None
        else:
            logger.warning(f"No entity degrees file found at {config.entity_degrees_path} - adaptive sampling disabled")
            self.adaptive_sampler = None
    
    def _get_available_entity_types(self) -> List[str]:
        """Get list of entity types that have samples available."""
        types = []
        if self.entity_samples:
            if self.entity_samples.genes:
                types.append('gene')
            if self.entity_samples.diseases:
                types.append('disease')
            if self.entity_samples.cell_types:
                types.append('cell_type')
            if self.entity_samples.snps:
                types.append('snp')
            if self.entity_samples.gene_ontology:
                types.append('gene_ontology')
        return types or ['gene', 'disease', 'cell_type']  # Fallback
    
    def _pick_random_entity_seed(self, entity_types: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Pick an entity to use as a question focus seed.
        
        Uses adaptive sampling (Thompson Sampling) when available,
        falls back to uniform sampling otherwise.
        
        Args:
            entity_types: Available entity types
            
        Returns:
            Tuple of (entity_name, entity_type, relationship)
        """
        # Use adaptive sampling when available
        if self.adaptive_sampler is not None:
            # Pick random entity type first
            entity_type = random.choice(entity_types)
            
            # Use Thompson Sampling to pick (entity, relationship) pair
            entity_name, relationship, sampled_type = self.adaptive_sampler.sample_one(
                entity_type=entity_type
            )
            
            if entity_name is not None:
                logger.debug(f"Adaptive sample: {entity_name} + {relationship} (type={sampled_type})")
                return entity_name, sampled_type or entity_type, relationship
            
            # Fall through to uniform if adaptive sampling found nothing
            logger.debug(f"Adaptive sampling found no candidates for {entity_type}, falling back to uniform")
        
        # Uniform sampling fallback
        if not self.entity_samples or not entity_types:
            return None, None, None
        
        # Pick random entity type
        entity_type = random.choice(entity_types)
        
        # Pick random entity of that type
        entity_name = None
        if entity_type == 'gene' and self.entity_samples.genes:
            entity = random.choice(self.entity_samples.genes)
            entity_name = entity.get('name')
        elif entity_type == 'disease' and self.entity_samples.diseases:
            entity = random.choice(self.entity_samples.diseases)
            entity_name = entity.get('name')
        elif entity_type == 'cell_type' and self.entity_samples.cell_types:
            entity = random.choice(self.entity_samples.cell_types)
            entity_name = entity.get('name')
        elif entity_type == 'snp' and self.entity_samples.snps:
            entity = random.choice(self.entity_samples.snps)
            entity_name = entity.get('id')
        elif entity_type == 'gene_ontology' and self.entity_samples.gene_ontology:
            entity = random.choice(self.entity_samples.gene_ontology)
            entity_name = entity.get('name')
        
        return entity_name, entity_type, None  # No relationship from uniform sampling
    
    def _pick_random_relationship(self, entity_type: Optional[str] = None) -> Optional[str]:
        """
        Pick a random relationship type, optionally based on entity type.
        
        Args:
            entity_type: Optional entity type to filter relevant relationships
            
        Returns:
            Relationship type string
        """
        # Relationship types from schema (entity -> relationship)
        relationships = {
            'gene': ['expression_level_in', 'DEG_in', 'physical_interaction', 'function_annotation', 
                     'effector_gene_of', 'signal_COLOC_with'],
            'disease': ['effector_gene_of', 'signal_COLOC_with', 'part_of_GWAS_signal'],
            'cell_type': ['expression_level_in', 'DEG_in', 'OCR_activity'],
            'snp': ['part_of_GWAS_signal', 'part_of_QTL_signal'],
            'gene_ontology': ['function_annotation'],  # GO terms have incoming function_annotation
            'ontology': ['function_annotation'],  # Alias for gene_ontology
        }
        
        if entity_type and entity_type in relationships:
            return random.choice(relationships[entity_type])
        
        # Pick any relationship
        all_rels = [
            'expression_level_in', 'DEG_in', 'physical_interaction', 
            'function_annotation', 'effector_gene_of', 'part_of_GWAS_signal'
        ]
        return random.choice(all_rels)
    
    def _get_experience_keywords(self) -> List[str]:
        """
        Get keywords from successful patterns in experience buffer.
        
        Returns:
            List of keywords that appeared in high-reward episodes
        """
        keywords = set()
        
        # Get keywords from high-reward patterns
        for pattern_type in ['fast_query', 'high_yield', 'optimal_stopping']:
            patterns = self.experience_buffer.patterns.get(pattern_type, [])
            for pattern in patterns:
                if pattern.avg_reward >= 0.6:  # Only from decent patterns
                    keywords.update(pattern.context_keywords)
        
        # Remove common stop words that aren't useful
        stop_words = {'gene', 'disease', 'cell', 'type', 'what', 'which', 'how', 'are', 'the'}
        keywords = keywords - stop_words
        
        # Filter out noise:
        # 1. SNP IDs (rs followed by digits)
        # 2. Partial/incomplete words 
        # 3. Very short terms
        # 4. Chromosome names
        filtered = []
        import re
        snp_pattern = re.compile(r'^rs\d+$', re.IGNORECASE)
        noise_words = {'incompletely', 'differentially', 'expressed', 'associated', 
                       'related', 'involved', 'specifically', 'particularly'}
        
        for kw in keywords:
            # Skip SNP IDs
            if snp_pattern.match(kw):
                continue
            # Skip noise words
            if kw.lower() in noise_words:
                continue
            # Skip very short terms (< 3 chars)
            if len(kw) < 3:
                continue
            # Skip chromosome references
            if kw.lower().startswith('chr') or kw.lower() in ['chromosome']:
                continue
            filtered.append(kw)
        
        return filtered[:50]  # Limit to top 50
    
    def collect_trajectories(
        self,
        questions: List[str],
        qgen_prompts: Optional[List[str]] = None,
    ) -> List[Trajectory]:
        """
        Collect trajectories for a batch of questions using STEP-SYNCHRONIZED BATCHING.
        
        Instead of processing questions sequentially, we process all questions
        at each step in parallel:
        
        Step 1: Batch generate Cypher for ALL questions → Execute all → Get results
        Step 2: Batch generate Cypher for active questions → Execute all → Get results
        ...
        
        This provides ~5-10x speedup compared to sequential processing.
        
        Args:
            questions: List of questions to answer
            qgen_prompts: Optional list of prompts used to generate these questions
                         (for storing in trajectories for training)
            
        Returns:
            List of Trajectory objects
        """
        logger.info(f"Collecting trajectories for {len(questions)} questions (batch mode)...")
        
        # Initialize trajectories and tracking state
        trajectories = [Trajectory(question=q) for q in questions]
        
        # Store question generation prompts if provided
        if qgen_prompts and len(qgen_prompts) == len(questions):
            for traj, prompt in zip(trajectories, qgen_prompts):
                traj.orch_qgen_prompt = prompt
        
        # Track state for each trajectory
        # Each entry: {'history': [], 'prompts': [], 'responses': [], 'active': True, 'agent': CypherGeneratorAgent}
        states = []
        for i, question in enumerate(questions):
            agent = CypherGeneratorAgent(
                schema_path=self.config.schema_path,
                experience_buffer=self.experience_buffer,
                max_steps=self.config.max_steps,
                enable_auto_fix=self.config.enable_auto_fix,
                entity_samples_path=self.config.entity_samples_path,
            )
            agent.reset()
            states.append({
                'history': [],
                'prompts': [],
                'responses': [],
                'active': True,
                'agent': agent,
            })
        
        # Step-synchronized loop
        for step_idx in range(self.config.max_steps):
            # Find active trajectories (not DONE yet)
            active_indices = [i for i, s in enumerate(states) if s['active']]
            
            if not active_indices:
                logger.info(f"All trajectories complete at step {step_idx}")
                break
            
            logger.info(f"Step {step_idx + 1}: Processing {len(active_indices)}/{len(questions)} active trajectories...")
            
            # Build prompts for all active trajectories
            prompts = []
            for i in active_indices:
                question = questions[i]
                history = states[i]['history']
                
                learned_rules = self.experience_buffer.get_relevant_patterns(question)
                semantic_issues = self.experience_buffer.get_semantic_issues_for_prompt(question)
                
                learned_rules_text = []
                for pattern in learned_rules:
                    learned_rules_text.append(pattern.get('description', ''))
                learned_rules_text.extend(semantic_issues)
                
                # Get Deep Think hints for Cypher Generator
                cypher_hints = None
                if self.prompt_hints:
                    cypher_hints = self.prompt_hints.get('cypher_generator', [])
                
                prompt = self.prompt_builder.build_cypher_prompt(
                    question=question,
                    schema=self.schema,
                    history=history,
                    learned_rules=learned_rules_text,
                    step=step_idx + 1,
                    deep_think_hints=cypher_hints,
                )
                prompts.append(prompt)
                states[i]['prompts'].append(prompt)
            
            # BATCH generate Cypher queries for all active trajectories
            responses = self.inference_engine.generate_cypher(prompts)
            
            # Process responses and execute queries
            for idx, i in enumerate(active_indices):
                response = responses[idx]
                states[i]['responses'].append(response)
                
                # Parse response - auto-fix is applied inside the agent's update_from_model
                # which calls _auto_fix() on the parsed query
                cypher_query = states[i]['agent']._parse_cypher_from_response(response)
                
                # Apply auto-fix (same as agent does internally)
                if cypher_query.upper() != "DONE":
                    cypher_query = states[i]['agent']._auto_fix(cypher_query)
                
                # Check for DONE
                if cypher_query.upper() == "DONE":
                    states[i]['active'] = False
                    continue
                
                # Execute query (query is already auto-fixed)
                result = self.neo4j_executor.execute_query(cypher_query)
                
                # Create step
                step = TrajectoryStep(
                    prompt=prompts[idx],
                    response=response,
                    cypher_query=cypher_query,
                    execution_result=result,
                    success=result.get('success', False),
                    has_data=result.get('has_data', False),
                    execution_time_ms=result.get('execution_time_ms', 0),
                    num_results=result.get('num_results', 0),
                )
                trajectories[i].steps.append(step)
                
                # Update history for next step
                states[i]['history'].append({
                    'query': cypher_query,
                    'result': result,
                })
        
        # Tokenize all trajectories
        for i, traj in enumerate(trajectories):
            self._tokenize_trajectory(traj, states[i]['prompts'], states[i]['responses'])
        
        # Log statistics
        avg_steps = sum(t.num_steps for t in trajectories) / len(trajectories)
        logger.info(f"Collected {len(trajectories)} trajectories, avg steps: {avg_steps:.1f}")
        
        return trajectories
    
    def _collect_single_trajectory(self, question: str) -> Trajectory:
        """
        Collect trajectory for a single question (fallback method).
        
        Use collect_trajectories() for batch processing instead.
        
        Args:
            question: The question to answer
            
        Returns:
            Trajectory object with all steps
        """
        # Use batch method with single question
        trajectories = self.collect_trajectories([question])
        return trajectories[0] if trajectories else Trajectory(question=question)
    
    def _tokenize_trajectory(
        self,
        trajectory: Trajectory,
        prompts: List[str],
        responses: List[str],
    ):
        """
        Tokenize trajectory for training.
        
        Creates input_ids, attention_mask, labels, and response_mask.
        
        Args:
            trajectory: Trajectory to tokenize
            prompts: List of prompts
            responses: List of responses
        """
        if not prompts or not responses:
            return
        
        # Concatenate all prompts and responses
        full_text_parts = []
        response_start_positions = []
        current_pos = 0
        
        for prompt, response in zip(prompts, responses):
            # Add prompt
            full_text_parts.append(prompt)
            current_pos += len(prompt)
            
            # Mark response start
            response_start_positions.append(current_pos)
            
            # Add response
            full_text_parts.append(response)
            current_pos += len(response)
        
        full_text = "".join(full_text_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_prompt_length + self.config.max_response_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Create response mask
        # For simplicity, we mask the prompt tokens and only train on response tokens
        # This is a simplified approach - in practice you'd want more precise masking
        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        
        # Find response token positions
        # This is approximate - proper implementation would track token offsets
        total_prompt_len = sum(len(self.tokenizer.encode(p)) for p in prompts)
        response_mask[total_prompt_len:] = 1.0
        
        # Mask padding
        response_mask = response_mask * attention_mask.float()
        
        # Set prompt labels to -100 (ignored in loss)
        labels[:total_prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        trajectory.input_ids = input_ids
        trajectory.attention_mask = attention_mask
        trajectory.labels = labels
        trajectory.response_mask = response_mask
    
    def evaluate_trajectories(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """
        Evaluate trajectories using Orchestrator.
        
        Performs:
        1. Data quality evaluation
        2. Answer synthesis
        3. Answer quality evaluation
        
        Args:
            trajectories: List of trajectories to evaluate
            
        Returns:
            Updated trajectories with evaluation scores
        """
        logger.info(f"Evaluating {len(trajectories)} trajectories...")
        
        # Step 1: Data quality evaluation
        data_eval_hints = None
        if self.prompt_hints:
            orch_hints = self.prompt_hints.get('orchestrator', {})
            data_eval_hints = orch_hints.get('data_eval', [])
        
        data_eval_prompts = []
        for traj in trajectories:
            prompt = self.orch_prompt_builder.build_data_quality_eval_prompt(
                question=traj.question,
                trajectory=[
                    {
                        'query': step.cypher_query,
                        'result': step.execution_result,
                    }
                    for step in traj.steps
                ],
                known_semantic_issues=[],
                deep_think_hints=data_eval_hints,
            )
            data_eval_prompts.append(prompt)
        
        data_evals = self.inference_engine.evaluate_data_quality(data_eval_prompts)
        
        for traj, eval_dict in zip(trajectories, data_evals):
            # Check for empty results - override with low scores
            total_results = sum(step.num_results for step in traj.steps)
            has_any_data = any(step.has_data for step in traj.steps)
            
            if total_results == 0 or not has_any_data:
                # No data retrieved = definitely low quality
                logger.debug(f"Empty results for question: {traj.question[:50]}...")
                traj.data_quality_score = 0.1
                traj.trajectory_quality_score = 0.2
                traj.doubt_level = 0.9
                traj.semantic_issues = []
                traj.problematic_regions = []
            else:
                # Use evaluated scores
                traj.data_quality_score = eval_dict.get('data_quality_score', 0.5)
                traj.trajectory_quality_score = eval_dict.get('trajectory_quality_score', 0.5)
                traj.doubt_level = eval_dict.get('doubt_level', 0.0)
                # Capture semantic issues and problematic regions for experience buffer
                traj.semantic_issues = eval_dict.get('semantic_issues', [])
                traj.problematic_regions = eval_dict.get('problematic_regions', [])
        
        # Step 2: Answer synthesis
        synth_hints = None
        if self.prompt_hints:
            orch_hints = self.prompt_hints.get('orchestrator', {})
            synth_hints = orch_hints.get('synthesis', [])
        
        synthesis_prompts = []
        for traj in trajectories:
            prompt = self.orch_prompt_builder.build_answer_synthesis_prompt(
                question=traj.question,
                trajectory_data=[
                    {
                        'query': step.cypher_query,
                        'result': step.execution_result.get('result', {}),
                    }
                    for step in traj.steps
                ],
                data_quality_feedback={
                    'data_quality_score': traj.data_quality_score,
                    'trajectory_quality_score': traj.trajectory_quality_score,
                },
                deep_think_hints=synth_hints,
            )
            synthesis_prompts.append(prompt)
        
        answers = self.inference_engine.synthesize_answers(synthesis_prompts)
        
        # Store synthesis answers AND prompts for training
        for traj, answer, synth_prompt in zip(trajectories, answers, synthesis_prompts):
            traj.synthesized_answer = answer
            traj.orch_synth_prompt = synth_prompt  # Store full synthesis prompt
        
        # Step 3: Answer quality evaluation
        answer_eval_hints = None
        if self.prompt_hints:
            orch_hints = self.prompt_hints.get('orchestrator', {})
            answer_eval_hints = orch_hints.get('answer_eval', [])
        
        answer_eval_prompts = []
        for traj in trajectories:
            prompt = self.orch_prompt_builder.build_answer_quality_eval_prompt(
                question=traj.question,
                answer=traj.synthesized_answer,
                deep_think_hints=answer_eval_hints,
            )
            answer_eval_prompts.append(prompt)
        
        answer_evals = self.inference_engine.evaluate_answer_quality(answer_eval_prompts)
        
        for traj, eval_dict in zip(trajectories, answer_evals):
            traj.answer_quality_score = eval_dict.get('score', 0.5)
        
        logger.info(f"Evaluation complete for {len(trajectories)} trajectories")
        return trajectories
    
    def compute_rewards(
        self,
        trajectories: List[Trajectory],
        recent_questions: Optional[List[str]] = None,
        target_success_rate: float = 0.6,
    ) -> List[Trajectory]:
        """
        Compute rewards for all trainable models/roles.
        
        Computes three separate rewards:
        1. Cypher Generator reward - based on query success, efficiency, data retrieval
        2. Orchestrator Question Generation reward - based on answerability, difficulty, diversity
        3. Orchestrator Answer Synthesis reward - based on answer quality, data utilization
        
        Args:
            trajectories: List of evaluated trajectories
            recent_questions: List of recent questions for diversity calculation
            target_success_rate: Target success rate for difficulty scoring
            
        Returns:
            Trajectories with all rewards computed
        """
        from rl_implementation.rewards import cypher_generator_reward_fn
        from rl_implementation.rewards.orchestrator_reward import (
            orchestrator_generation_reward_fn,
            orchestrator_synthesis_reward_fn,
        )
        from rl_implementation.rewards.reward_utils import compute_data_utilization
        
        logger.info(f"Computing rewards for {len(trajectories)} trajectories...")
        
        # Use experience buffer's recent questions if not provided
        if recent_questions is None:
            recent_questions = self.experience_buffer.get_recent_questions(n=20)
        
        # Calculate running success rate for difficulty scoring
        if trajectories:
            current_success_rate = sum(t.success_rate for t in trajectories) / len(trajectories)
        else:
            current_success_rate = 0.5
        
        for traj in trajectories:
            # ============================================
            # 1. Cypher Generator Reward
            # ============================================
            cypher_task_info = {
                'question': traj.question,
                'cypher_trajectory': [
                    {
                        'query': step.cypher_query,
                        'result': step.execution_result,
                        'success': step.success,
                        'has_data': step.has_data,
                        'execution_time_ms': step.execution_time_ms,
                        'num_results': step.num_results,
                    }
                    for step in traj.steps
                ],
                'answer_quality_score': traj.answer_quality_score,
                'data_quality_score': traj.data_quality_score,
                'trajectory_quality_score': traj.trajectory_quality_score,
                'doubt_level': traj.doubt_level,
                'num_steps': traj.num_steps,
            }
            
            cypher_reward_output = cypher_generator_reward_fn(cypher_task_info, "")
            traj.cypher_reward = cypher_reward_output.reward
            traj.reward = traj.cypher_reward  # Legacy compatibility
            
            # ============================================
            # 2. Orchestrator Question Generation Reward
            # ============================================
            # Answerability: did the trajectory succeed in getting data?
            answerability = traj.success_rate > 0.5 and traj.total_results > 0
            
            # Compute data richness (normalized: 0 results = 0, 50+ results = 1.0)
            data_richness = min(1.0, traj.total_results / 50.0) if traj.total_results > 0 else 0.0
            
            qgen_task_info = {
                'question': traj.question,
                'answerability': answerability,
                'cypher_reward': traj.cypher_reward,  # NEW: Direct Cypher reward signal
                'data_richness': data_richness,       # NEW: How much data was retrieved
                'success_rate': current_success_rate,
                'target_success_rate': target_success_rate,
                'recent_questions': recent_questions,
                'scope_constraints': self.experience_buffer.get_scope_constraints(),
                'question_used_types': [],  # Could extract from question text if needed
            }
            
            qgen_reward_output = orchestrator_generation_reward_fn(qgen_task_info, traj.question)
            traj.orch_qgen_reward = qgen_reward_output.reward
            
            # ============================================
            # 3. Orchestrator Answer Synthesis Reward
            # ============================================
            # Build trajectory data for data utilization calculation
            trajectory_data = [
                {
                    'query': step.cypher_query,
                    'result': step.execution_result,
                    'num_results': step.num_results,
                }
                for step in traj.steps
            ]
            
            # Compute data utilization
            data_utilization = compute_data_utilization(traj.synthesized_answer, trajectory_data)
            
            synth_task_info = {
                'question': traj.question,
                'answer': traj.synthesized_answer,
                'answer_quality_score': traj.answer_quality_score,
                'trajectory': trajectory_data,
                'data_utilization': data_utilization,
            }
            
            synth_reward_output = orchestrator_synthesis_reward_fn(synth_task_info, traj.synthesized_answer)
            traj.orch_synth_reward = synth_reward_output.reward
            
            # ============================================
            # Store reward metadata for analysis
            # ============================================
            traj.reward_metadata = {
                'cypher': {
                    'reward': traj.cypher_reward,
                    'metadata': cypher_reward_output.metadata if hasattr(cypher_reward_output, 'metadata') else {},
                },
                'orch_qgen': {
                    'reward': traj.orch_qgen_reward,
                    'answerability': answerability,
                    'metadata': qgen_reward_output.metadata if hasattr(qgen_reward_output, 'metadata') else {},
                },
                'orch_synth': {
                    'reward': traj.orch_synth_reward,
                    'data_utilization': data_utilization,
                    'metadata': synth_reward_output.metadata if hasattr(synth_reward_output, 'metadata') else {},
                },
            }
        
        # Log reward statistics for all three reward types
        cypher_rewards = [t.cypher_reward for t in trajectories]
        qgen_rewards = [t.orch_qgen_reward for t in trajectories]
        synth_rewards = [t.orch_synth_reward for t in trajectories]
        
        logger.info(f"Cypher rewards: mean={sum(cypher_rewards)/len(cypher_rewards):.3f}, "
                   f"min={min(cypher_rewards):.3f}, max={max(cypher_rewards):.3f}")
        logger.info(f"Orch QGen rewards: mean={sum(qgen_rewards)/len(qgen_rewards):.3f}, "
                   f"min={min(qgen_rewards):.3f}, max={max(qgen_rewards):.3f}")
        logger.info(f"Orch Synth rewards: mean={sum(synth_rewards)/len(synth_rewards):.3f}, "
                   f"min={min(synth_rewards):.3f}, max={max(synth_rewards):.3f}")
        
        return trajectories
    
    def update_experience_buffer(
        self,
        trajectories: List[Trajectory],
    ):
        """
        Update experience buffer with patterns from completed trajectories.
        
        Extracts:
        - Fast query patterns (high reward, fast execution)
        - High-yield patterns (high reward, many results)
        - Optimal stopping patterns (high reward, good trajectory)
        - Bad data regions (low data quality)
        - Semantic ambiguities (high doubt level)
        
        Args:
            trajectories: List of evaluated trajectories with rewards
        """
        logger.info(f"Updating experience buffer with {len(trajectories)} trajectories...")
        
        patterns_added = 0
        
        for traj in trajectories:
            # Build trajectory list for experience buffer
            trajectory_data = [
                {
                    'query': step.cypher_query,
                    'result': step.execution_result,
                    'success': step.success,
                    'has_data': step.has_data,
                    'execution_time_ms': step.execution_time_ms,
                    'num_results': step.num_results,
                }
                for step in traj.steps
            ]
            
            # Extract semantic issues (if available from evaluation)
            # These would be populated by the data quality evaluation
            semantic_issues = getattr(traj, 'semantic_issues', None) or []
            
            # Add patterns from this episode
            self.experience_buffer.add_from_episode(
                question=traj.question,
                trajectory=trajectory_data,
                reward=traj.reward,
                data_quality=traj.data_quality_score,
                doubt_level=traj.doubt_level,
                semantic_issues=semantic_issues,
            )
            
            patterns_added += 1
        
        # Log buffer stats
        stats = self.experience_buffer.get_stats()
        logger.info(f"Experience buffer updated: {stats['total_patterns']} patterns, "
                   f"by_type: {stats['by_type']}")
    
    def update_adaptive_sampler(
        self,
        trajectories: List[Trajectory],
    ):
        """
        Update adaptive entity sampler based on answerability.
        
        Updates the sampler based on whether questions were answerable:
        - answerable = cypher_reward > 0 AND total_results > 0
        
        Uses slack mechanism: first few failures get soft penalty,
        repeated failures get full penalty.
        
        Args:
            trajectories: List of trajectories with rewards computed
        """
        if self.adaptive_sampler is None:
            return
        
        logger.info(f"Updating adaptive sampler with {len(trajectories)} trajectories...")
        
        updates = []
        for traj in trajectories:
            # Get seed entity and relationship from stored tracking
            seed_entity = self._current_seed_relationships.get(f"{traj.question}_entity")
            seed_relationship = self._current_seed_relationships.get(f"{traj.question}_relationship")
            
            if not seed_entity or not seed_relationship:
                continue
            
            # Answerable = cypher worked AND got results
            answerable = (traj.cypher_reward > 0 and traj.total_results > 0)
            
            updates.append((seed_entity, seed_relationship, answerable))
        
        # Batch update the sampler
        if updates:
            self.adaptive_sampler.batch_update(updates)
            
            # Log summary
            successes = sum(1 for _, _, a in updates if a)
            logger.info(f"Adaptive sampler updated: {successes}/{len(updates)} answerable")
        
        # Persist sampler state
        if self.config.adaptive_sampler_path:
            try:
                self.adaptive_sampler.save(self.config.adaptive_sampler_path)
            except Exception as e:
                logger.warning(f"Failed to save adaptive sampler state: {e}")
        
        # Clear tracking
        self._current_seed_relationships.clear()
    
    def prepare_training_batch(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a training batch from trajectories.
        
        Args:
            trajectories: List of trajectories with tokenized data
            
        Returns:
            Dictionary with batched tensors
        """
        # Filter trajectories with valid tokenized data
        valid_trajs = [t for t in trajectories if t.input_ids is not None]
        
        if not valid_trajs:
            raise ValueError("No valid trajectories with tokenized data")
        
        # Stack tensors
        input_ids = torch.stack([t.input_ids for t in valid_trajs])
        attention_mask = torch.stack([t.attention_mask for t in valid_trajs])
        labels = torch.stack([t.labels for t in valid_trajs])
        response_mask = torch.stack([t.response_mask for t in valid_trajs])
        rewards = torch.tensor([t.reward for t in valid_trajs], dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_mask": response_mask,
            "rewards": rewards,
        }
    
    def close(self):
        """Cleanup resources."""
        self.neo4j_executor.close()
        logger.info("RolloutCollector closed")

