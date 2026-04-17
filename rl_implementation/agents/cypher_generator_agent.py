"""
Cypher Generator Agent for multi-step knowledge graph exploration.

Inherits from rllm's BaseAgent to generate Cypher queries iteratively.
"""

import copy
import json
import logging
import re
from typing import Any, Callable, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from ..utils.prompt_builder import PromptBuilder
from ..utils.cypher_auto_fix import auto_fix_cypher, create_auto_fixer
from .experience_buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class CypherGeneratorAgent(BaseAgent):
    """
    Multi-step Cypher query generator for knowledge graph reasoning.
    
    Generates up to 5 Cypher queries iteratively to explore the knowledge graph
    and retrieve relevant data for answering biomedical questions.
    
    Inherits from rllm.agents.agent.BaseAgent and implements:
    - reset(): Initialize new episode
    - update_from_env(): Process environment observations
    - update_from_model(): Parse model responses
    - chat_completions: Return messages for model inference
    - trajectory: Return trajectory for reward computation
    """
    
    def __init__(
        self,
        schema_path: str,
        experience_buffer: ExperienceBuffer | None = None,
        max_steps: int = 5,
        enable_auto_fix: bool = True,
        entity_samples_path: Optional[str] = None
    ):
        """
        Initialize the Cypher Generator Agent.
        
        Args:
            schema_path: Path to knowledge graph schema JSON file
            experience_buffer: ExperienceBuffer instance (creates default if None)
            max_steps: Maximum number of query steps (default: 5)
            enable_auto_fix: Whether to auto-fix Cypher queries (default: True)
            entity_samples_path: Optional path to entity samples JSON for auto-fix
        """
        # Load schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Initialize experience buffer
        self.experience_buffer = experience_buffer if experience_buffer is not None else ExperienceBuffer()
        
        # Configuration
        self.max_steps = max_steps
        self.enable_auto_fix = enable_auto_fix
        
        # Initialize auto-fix function (pre-loads schema/entity samples)
        self._auto_fix: Callable[[str], str] = create_auto_fixer(
            schema_path=schema_path,
            entity_samples_path=entity_samples_path,
            enabled=enable_auto_fix
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        
        # Episode state
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.current_step = 0
        self.current_observation: Any = None
        self.question: str = ""
        self.history: list[dict[str, Any]] = []
        
        logger.info(f"CypherGeneratorAgent initialized with schema from {schema_path}, auto_fix={enable_auto_fix}")
    
    def reset(self):
        """
        Reset agent state for a new episode.
        
        Clears trajectory, messages, step counter, and history.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.current_step = 0
        self.current_observation = None
        self.question = ""
        self.history = []
        
        logger.debug("Agent reset for new episode")
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state after environment step.
        
        Args:
            observation: Observation from environment (dict with question, results, etc.)
            reward: Reward received (not used during rollout)
            done: Whether episode is done
            info: Additional info from environment
        """
        if done:
            logger.debug("Episode done, skipping update")
            return
        
        # Format observation into prompt
        formatted_obs = self._format_observation(observation)
        
        # Extract question on first step
        if self.current_step == 0 and isinstance(observation, dict):
            self.question = observation.get('question', '')
            logger.debug(f"Starting new question: {self.question}")
        
        # Build complete prompt with learned rules
        if self.current_step == 0:
            # First step: no history yet
            learned_rules = self.experience_buffer.get_relevant_patterns(self.question)
            semantic_issues = self.experience_buffer.get_semantic_issues_for_prompt(self.question)
            
            # Combine patterns and semantic issues into learned rules
            learned_rules_text = []
            for pattern in learned_rules:
                learned_rules_text.append(pattern.get('description', ''))
            learned_rules_text.extend(semantic_issues)
            
            prompt = self.prompt_builder.build_cypher_prompt(
                question=self.question,
                schema=self.schema,
                history=[],
                learned_rules=learned_rules_text,
                step=1
            )
        else:
            # Subsequent steps: include history
            learned_rules = self.experience_buffer.get_relevant_patterns(self.question)
            semantic_issues = self.experience_buffer.get_semantic_issues_for_prompt(self.question)
            
            learned_rules_text = []
            for pattern in learned_rules:
                learned_rules_text.append(pattern.get('description', ''))
            learned_rules_text.extend(semantic_issues)
            
            prompt = self.prompt_builder.build_cypher_prompt(
                question=self.question,
                schema=self.schema,
                history=self.history,
                learned_rules=learned_rules_text,
                step=self.current_step + 1
            )
        
        # Add user message to chat history
        self.messages.append({"role": "user", "content": prompt})
        
        # Store observation
        self.current_observation = formatted_obs
        
        logger.debug(f"Updated from env at step {self.current_step}")
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state after model generates response.
        
        Args:
            response: Model's response string
            
        Returns:
            Action object containing parsed Cypher query or "DONE"
        """
        # Parse Cypher query or DONE signal from response
        action_str = self._parse_cypher_from_response(response)
        
        # Apply auto-fix to the query (if enabled and not DONE)
        if action_str.upper() != "DONE":
            action_str = self._auto_fix(action_str)
            logger.debug(f"Auto-fixed query applied")
        
        # Create step object
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_str,
            model_response=response,
            observation=self.current_observation
        )
        
        # Add to trajectory
        self._trajectory.steps.append(new_step)
        
        # Add assistant message to chat history
        self.messages.append({"role": "assistant", "content": response})
        
        # Update history if not DONE
        if action_str.upper() != "DONE":
            self.history.append({
                'query': action_str,
                'result': {}  # Will be filled by environment
            })
        
        # Increment step counter
        self.current_step += 1
        
        logger.debug(f"Parsed action: {action_str[:100]}...")
        
        return Action(action=action_str)
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Return chat messages for model inference.
        
        Returns:
            List of messages in OpenAI chat format
        """
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """
        Return trajectory for reward computation.
        
        Returns:
            Trajectory object with all steps
        """
        return self._trajectory
    
    def _parse_cypher_from_response(self, response: str) -> str:
        """
        Extract Cypher query or DONE signal from model response.
        
        Looks for:
        1. "DONE" keyword (case-insensitive)
        2. Code blocks with ```cypher ... ``` or ``` ... ```
        3. Inline MATCH statements
        4. Detects and rejects garbage responses
        
        IMPORTANT: Never return formatted text or multiple queries.
        Only return a clean, single Cypher query or "DONE".
        
        Args:
            response: Model's response string
            
        Returns:
            Extracted Cypher query or "DONE"
        """
        # Check for DONE signal first
        if re.search(r'\bDONE\b', response, re.IGNORECASE):
            logger.debug("Detected DONE signal")
            return "DONE"
        
        # Look for code blocks
        # Pattern 1: ```cypher ... ```
        cypher_blocks = re.findall(r'```cypher\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if cypher_blocks:
            # Take the FIRST valid block, not the last (to avoid picking up explanations)
            for block in cypher_blocks:
                query = block.strip()
                # Clean up: remove any "Step X:" or formatted text that leaked in
                query = self._clean_query_text(query)
                if query and not self._is_garbage_query(query):
                    logger.debug(f"Extracted Cypher from ```cypher block: {query[:50]}...")
                    return query
            # If all blocks were garbage
            logger.warning(f"All cypher blocks were garbage, returning DONE")
            return "DONE"
        
        # Pattern 2: ``` ... ``` (generic code block)
        code_blocks = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                query = block.strip()
                # Clean up
                query = self._clean_query_text(query)
                if self._is_garbage_query(query):
                    continue
                # Verify it looks like Cypher
                if any(keyword in query.upper() for keyword in ['MATCH', 'RETURN', 'WHERE', 'WITH']):
                    logger.debug(f"Extracted Cypher from generic code block: {query[:50]}...")
                    return query
        
        # Pattern 3: Look for inline MATCH statement (no code blocks)
        # This handles responses like "MATCH (g:gene)... RETURN nodes, edges;"
        match_pattern = re.search(
            r'(MATCH\s*\([^;]+(?:RETURN[^;]+;|RETURN[^;]+))',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if match_pattern:
            query = match_pattern.group(1).strip()
            query = self._clean_query_text(query)
            if not self._is_garbage_query(query):
                logger.debug(f"Extracted inline Cypher: {query[:50]}...")
                return query
        
        # Check if the full response is garbage
        if self._is_garbage_query(response):
            logger.warning("Response appears to be garbage, returning DONE")
            return "DONE"
        
        # Fallback: Return DONE instead of corrupt text
        # Previously we returned the full response which caused history corruption
        logger.warning(f"Could not extract valid Cypher query from response, returning DONE. Response preview: {response[:100]}...")
        return "DONE"
    
    def _clean_query_text(self, query: str) -> str:
        """
        Clean up query text by removing common contamination patterns.
        
        This prevents history corruption from formatted text leaking into queries.
        """
        if not query:
            return ""
        
        # Remove common prefixes that shouldn't be in queries
        prefixes_to_remove = [
            r'^Step\s*\d+[:\s]*',           # Step 1:, Step 2, etc.
            r'^Query[:\s]*',                 # Query:
            r'^\s*Results?[:\s]*',           # Results:
            r'^\s*\.\.\.',                   # Leading ...
            r'^\s*cypher\s*',                # "cypher" language tag
        ]
        
        for pattern in prefixes_to_remove:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove trailing non-query text (like "Results: 0 items")
        suffixes_to_remove = [
            r'\s*Results?:\s*\d+.*$',        # Results: 0 items...
            r'\s*Execution:\s*\d+.*$',       # Execution: 3230ms...
            r'\s*⚠️.*$',                     # Warning emoji and text
            r'\s*✗.*$',                      # X emoji and text
            r'\s*Step\s*\d+:.*$',            # Step N: at the end
        ]
        
        for pattern in suffixes_to_remove:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE | re.DOTALL)
        
        # If query contains multiple MATCH statements separated by newlines,
        # take only the FIRST complete query
        if query.count('MATCH') > 1:
            # Split on semicolons and take first complete query
            parts = query.split(';')
            if parts:
                first_query = parts[0].strip()
                if 'MATCH' in first_query.upper():
                    query = first_query + ';'
        
        return query.strip()
    
    def _is_garbage_query(self, query: str) -> bool:
        """
        Detect garbage/placeholder queries that aren't real Cypher.
        
        Args:
            query: Query string to check
            
        Returns:
            True if the query is garbage
        """
        if not query:
            return True
        
        query_lower = query.lower().strip()
        
        # Known garbage patterns from logs
        garbage_patterns = [
            'your_query_here',
            'your query here',
            'please select',
            'select an option',
            'i cannot',
            'i\'m sorry',
            'i apologize',
            '``` ```',
        ]
        
        for pattern in garbage_patterns:
            if pattern in query_lower:
                return True
        
        # If it doesn't contain any Cypher keywords, it's probably garbage
        cypher_keywords = ['match', 'return', 'where', 'with', 'create', 'merge', 'optional']
        if not any(kw in query_lower for kw in cypher_keywords):
            # Short non-Cypher responses are garbage
            if len(query) < 50:
                return True
        
        return False
    
    def _format_observation(self, obs_dict: Any) -> str:
        """
        Format observation dictionary into human-readable string.
        
        Args:
            obs_dict: Observation from environment
            
        Returns:
            Formatted observation string
        """
        if not isinstance(obs_dict, dict):
            return str(obs_dict)
        
        # Initial observation (just question)
        if 'question' in obs_dict and 'previous_query' not in obs_dict:
            return f"Question: {obs_dict['question']}"
        
        # Subsequent observations (with execution results)
        formatted = []
        
        if 'question' in obs_dict:
            formatted.append(f"Question: {obs_dict['question']}")
        
        if 'previous_query' in obs_dict:
            formatted.append(f"\nPrevious Query: {obs_dict['previous_query']}")
        
        if 'previous_result' in obs_dict:
            result = obs_dict['previous_result']
            
            success = result.get('success', False)
            has_data = result.get('has_data', False)
            num_results = result.get('num_results', 0)
            exec_time = result.get('execution_time_ms', 0)
            
            formatted.append(f"Execution: {'Success' if success else 'Failed'}")
            formatted.append(f"Results: {num_results} items ({exec_time:.1f}ms)")
            
            if 'data_summary' in result:
                formatted.append(f"Summary: {result['data_summary']}")
            
            # Update history with result details
            if self.history and self.history[-1]['query'] == obs_dict['previous_query']:
                self.history[-1]['result'] = result
        
        if 'turn' in obs_dict:
            formatted.append(f"\nTurn: {obs_dict['turn']}/{self.max_steps}")
        
        return "\n".join(formatted)
    
    def get_current_state(self) -> Step | None:
        """
        Get current step/state of the agent.
        
        Returns:
            Current Step object or None if no steps yet
        """
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"CypherGeneratorAgent(step={self.current_step}/{self.max_steps}, question='{self.question[:50]}...')"

