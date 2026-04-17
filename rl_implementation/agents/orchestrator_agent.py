"""
Orchestrator Agent.

Four-role agent for:
1. Question Generation - Generate training questions
2. Data Quality Evaluation - Evaluate retrieved data and identify semantic ambiguities
3. Answer Synthesis - Convert KG data to natural language
4. Answer Quality Evaluation - Evaluate final answer quality

Inherits from rllm's BaseAgent and switches between modes for different roles.
"""

import copy
import json
import logging
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from ..utils.orchestrator_prompt_builder import OrchestratorPromptBuilder
from ..utils.data_quality_evaluator import (
    parse_data_quality_json,
    parse_answer_quality_json
)
from .experience_buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent with four roles.
    
    Modes:
    - 'generation': Generate training questions
    - 'data_eval': Evaluate data quality and identify semantic ambiguities
    - 'synthesis': Synthesize natural language answers
    - 'answer_eval': Evaluate answer quality
    """
    
    VALID_MODES = ['generation', 'data_eval', 'synthesis', 'answer_eval']
    
    def __init__(
        self,
        schema_path: str,
        experience_buffer: ExperienceBuffer | None = None,
        mode: str = 'synthesis'
    ):
        """
        Initialize Orchestrator agent.
        
        Args:
            schema_path: Path to KG schema JSON file
            experience_buffer: Optional experience buffer for learned patterns
            mode: Initial mode (default: 'synthesis')
        """
        # Load schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Initialize components
        self.experience_buffer = experience_buffer if experience_buffer is not None else ExperienceBuffer()
        self.prompt_builder = OrchestratorPromptBuilder()
        
        # Set mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")
        self.mode = mode
        
        # Agent state
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.current_observation: Any = None
        self.context: dict[str, Any] = {}
        
        logger.info(f"OrchestratorAgent initialized in '{mode}' mode with schema from {schema_path}")
    
    def reset(self):
        """Reset agent for new episode. Keeps mode unchanged."""
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None
        self.context = {}
        logger.debug(f"Agent reset in '{self.mode}' mode")
    
    def set_mode(self, mode: str):
        """
        Switch agent to different role.
        
        Args:
            mode: New mode ('generation', 'data_eval', 'synthesis', 'answer_eval')
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")
        
        self.mode = mode
        self.context = {}  # Clear context when switching modes
        logger.debug(f"Agent switched to '{mode}' mode")
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Process observation and build prompt based on current mode.
        
        Args:
            observation: Observation dict (format depends on mode)
            reward: Reward signal (not used during prompt building)
            done: Episode done flag
            info: Additional info dict
            **kwargs: Additional arguments
        """
        if done:
            return
        
        # Store observation
        self.current_observation = observation
        
        # Extract context and build prompt based on mode
        if self.mode == 'generation':
            prompt = self._build_generation_prompt(observation)
        elif self.mode == 'data_eval':
            prompt = self._build_data_eval_prompt(observation)
        elif self.mode == 'synthesis':
            prompt = self._build_synthesis_prompt(observation)
        elif self.mode == 'answer_eval':
            prompt = self._build_answer_eval_prompt(observation)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Add user message
        self.messages.append({"role": "user", "content": prompt})
        logger.debug(f"Built prompt for '{self.mode}' mode ({len(prompt)} chars)")
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Parse model response based on current mode.
        
        Args:
            response: Model's text response
            **kwargs: Additional arguments
            
        Returns:
            Action with parsed output
        """
        # Parse response based on mode
        if self.mode == 'generation':
            action_str = self._parse_question(response)
        elif self.mode == 'data_eval':
            action_str = self._parse_data_quality_eval(response)
        elif self.mode == 'synthesis':
            action_str = self._parse_answer(response)
        elif self.mode == 'answer_eval':
            action_str = self._parse_answer_quality_eval(response)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Create step and add to trajectory
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_str,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)
        
        # Add assistant message
        self.messages.append({"role": "assistant", "content": response})
        
        logger.debug(f"Parsed response in '{self.mode}' mode")
        
        return Action(action=action_str)
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return chat messages for model inference."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Return trajectory for reward computation."""
        return self._trajectory
    
    # Mode-specific prompt builders
    
    def _build_generation_prompt(self, observation: dict) -> str:
        """
        Build prompt for question generation mode.
        
        Expected observation format:
        {
            'schema': dict,  # Can override self.schema
            'difficulty': str,  # 'easy', 'medium', 'hard'
            'curriculum_constraints': dict,
            'scope_constraints': dict,
            'recent_questions': list[str]
        }
        """
        schema = observation.get('schema', self.schema)
        difficulty = observation.get('difficulty', 'medium')
        curriculum_constraints = observation.get('curriculum_constraints', {})
        scope_constraints = observation.get('scope_constraints', 
                                           self.experience_buffer.get_scope_constraints())
        recent_questions = observation.get('recent_questions', [])
        
        # Store context
        self.context = {
            'difficulty': difficulty,
            'curriculum_constraints': curriculum_constraints
        }
        
        return self.prompt_builder.build_question_generation_prompt(
            schema=schema,
            difficulty=difficulty,
            curriculum_constraints=curriculum_constraints,
            scope_constraints=scope_constraints,
            recent_questions=recent_questions
        )
    
    def _build_data_eval_prompt(self, observation: dict) -> str:
        """
        Build prompt for data quality evaluation mode.
        
        Expected observation format:
        {
            'question': str,
            'trajectory': list[dict],  # Cypher queries and results
            'known_semantic_issues': list[dict]  # Optional
        }
        """
        question = observation.get('question', '')
        trajectory = observation.get('trajectory', [])
        
        # Get known semantic issues from experience buffer or observation
        known_semantic_issues = observation.get('known_semantic_issues')
        if known_semantic_issues is None:
            known_semantic_issues = self.experience_buffer.get_semantic_issues_for_prompt(question)
        
        # Store context
        self.context = {
            'question': question,
            'trajectory': trajectory
        }
        
        return self.prompt_builder.build_data_quality_eval_prompt(
            question=question,
            trajectory=trajectory,
            known_semantic_issues=known_semantic_issues
        )
    
    def _build_synthesis_prompt(self, observation: dict) -> str:
        """
        Build prompt for answer synthesis mode.
        
        Expected observation format:
        {
            'question': str,
            'trajectory_data': list[dict],  # Formatted data from queries
            'data_quality_feedback': dict  # From data eval
        }
        """
        question = observation.get('question', '')
        trajectory_data = observation.get('trajectory_data', [])
        data_quality_feedback = observation.get('data_quality_feedback', {})
        
        # Store context
        self.context = {
            'question': question,
            'trajectory_data': trajectory_data,
            'data_quality_feedback': data_quality_feedback
        }
        
        return self.prompt_builder.build_answer_synthesis_prompt(
            question=question,
            trajectory_data=trajectory_data,
            data_quality_feedback=data_quality_feedback
        )
    
    def _build_answer_eval_prompt(self, observation: dict) -> str:
        """
        Build prompt for answer quality evaluation mode.
        
        Expected observation format:
        {
            'question': str,
            'answer': str
        }
        """
        question = observation.get('question', '')
        answer = observation.get('answer', '')
        
        # Store context
        self.context = {
            'question': question,
            'answer': answer
        }
        
        return self.prompt_builder.build_answer_quality_eval_prompt(
            question=question,
            answer=answer
        )
    
    # Mode-specific response parsers
    
    def _parse_question(self, response: str) -> str:
        """
        Parse EXACTLY ONE question from generation response.
        
        Extracts only the FIRST valid biomedical question.
        Avoids returning multiple questions or conversational text.
        
        Args:
            response: Model response
            
        Returns:
            Generated question string (single question only)
        """
        import re
        
        if not response:
            return ""
        
        response = response.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^#+\s*Question:\s*',
            r'^Question:\*\*\s*',
            r'^Question:\s*',
            r'^\*\*Question:\*\*\s*',
        ]
        for prefix in prefixes_to_remove:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Helper to extract only first question
        def extract_first_question(text: str) -> str:
            """Extract only the FIRST question ending with ?"""
            parts = re.split(r'\?(?=\s|$)', text)
            if parts and parts[0].strip():
                return parts[0].strip() + '?'
            return text
        
        # Biomedical keywords for validation
        biomedical_keywords = [
            'gene', 'protein', 'disease', 'cell', 'snp', 'variant', 'expression',
            'diabetes', 'chromosome', 'ontology', 'pathway', 'interaction'
        ]
        
        def is_biomedical(text: str) -> bool:
            text_lower = text.lower()
            return any(kw in text_lower for kw in biomedical_keywords)
        
        # Look for lines with questions
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip().strip('*"\'').strip()
            if not '?' in line:
                continue
            if len(line) < 20:
                continue
            
            # Extract only first question from line
            first_q = extract_first_question(line)
            if first_q.endswith('?') and is_biomedical(first_q):
                logger.debug(f"Parsed question: {first_q[:100]}...")
                return first_q
        
        # Fallback: find first sentence with ?
        q_match = re.search(
            r'((?:Which|What|How|Where|When|Who|Why|Are|Is|Do|Does|Can|Could)[^?]*\?)',
            response,
            re.IGNORECASE
        )
        if q_match:
            question = q_match.group(1).strip()
            logger.debug(f"Parsed question (regex): {question[:100]}...")
            return question
        
        # Last resort: take first line with ? 
        for line in lines:
            line = line.strip().strip('*"\'').strip()
            if '?' in line and len(line) > 15:
                first_q = extract_first_question(line)
                logger.debug(f"Parsed question (fallback): {first_q[:100]}...")
                return first_q
        
        # Absolute fallback
        result = response[:200] if len(response) > 200 else response
        logger.debug(f"Parsed question (full response): {result[:100]}...")
        return result
    
    def _parse_data_quality_eval(self, response: str) -> dict[str, Any]:
        """
        Parse data quality evaluation JSON from response.
        
        Args:
            response: Model response with JSON
            
        Returns:
            Dict with evaluation scores and metadata
        """
        try:
            evaluation = parse_data_quality_json(response)
            logger.debug(f"Parsed data quality eval: score={evaluation.get('data_quality_score', 0):.2f}")
            return evaluation
        except ValueError as e:
            logger.warning(f"Failed to parse data quality JSON: {e}")
            # Return default evaluation
            return {
                'data_quality_score': 0.5,
                'relevance_score': 0.5,
                'completeness_score': 0.5,
                'consistency_score': 0.5,
                'trajectory_quality_score': 0.5,
                'reasoning': 'Failed to parse evaluation',
                'semantic_issues': [],
                'problematic_regions': [],
                'could_answer_question': True,
                'doubt_level': 0.0
            }
    
    def _parse_answer(self, response: str) -> str:
        """
        Parse synthesized answer from response.
        
        Args:
            response: Model response
            
        Returns:
            Synthesized answer string
        """
        # Look for "Generate answer:" or similar markers and take text after
        lines = response.strip().split('\n')
        
        answer_started = False
        answer_lines = []
        
        for line in lines:
            # Skip instruction lines
            if 'generate answer' in line.lower() or line.strip().endswith(':'):
                answer_started = True
                continue
            
            if answer_started or not any(marker in line.lower() for marker in ['question:', 'data:', 'task:']):
                if line.strip():
                    answer_lines.append(line.strip())
        
        # Join answer lines
        if answer_lines:
            answer = ' '.join(answer_lines)
        else:
            # Fallback to full response
            answer = response.strip()
        
        logger.debug(f"Parsed answer: {answer[:100]}...")
        return answer
    
    def _parse_answer_quality_eval(self, response: str) -> dict[str, Any]:
        """
        Parse answer quality evaluation JSON from response.
        
        Args:
            response: Model response with JSON
            
        Returns:
            Dict with answer quality scores
        """
        try:
            evaluation = parse_answer_quality_json(response)
            logger.debug(f"Parsed answer quality eval: score={evaluation.get('score', 0):.2f}")
            return evaluation
        except ValueError as e:
            logger.warning(f"Failed to parse answer quality JSON: {e}")
            # Return default evaluation
            return {
                'score': 0.5,
                'correctness': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'accuracy': 0.5,
                'reasoning': 'Failed to parse evaluation',
                'strengths': '',
                'weaknesses': ''
            }

