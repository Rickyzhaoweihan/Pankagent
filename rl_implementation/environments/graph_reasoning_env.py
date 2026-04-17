"""
Graph Reasoning Environment for multi-step Cypher query generation.

Executes Cypher queries against Neo4j database and returns structured observations
for the agent to continue exploration.
"""

import logging
from typing import Any, Dict, List, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

from .neo4j_executor import Neo4jExecutor

logger = logging.getLogger(__name__)


class GraphReasoningEnvironment(MultiTurnEnvironment):
    """
    Environment for multi-step knowledge graph exploration.
    
    Executes Cypher queries and returns results as observations.
    Inherits from rllm's MultiTurnEnvironment for RL training integration.
    """
    
    def __init__(
        self,
        task: Dict[str, Any] | None = None,
        max_turns: int = 5,
        api_url: str | None = None,
        schema: Dict[str, Any] | None = None,
        **kwargs
    ):
        """
        Initialize the Graph Reasoning Environment.
        
        Args:
            task: Task dictionary containing at least 'question' field
            max_turns: Maximum number of Cypher queries allowed (default: 5)
            api_url: AWS Lambda endpoint URL (default: production endpoint)
            schema: Knowledge graph schema dictionary
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        
        # Default to production Pankbase endpoint
        self.api_url = api_url or "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"
        
        # Initialize Neo4j executor
        self.executor = Neo4jExecutor(self.api_url)
        
        # Store schema for potential use
        self.schema = schema
        
        # Track query and result history
        self.query_history: List[str] = []
        self.result_history: List[Dict[str, Any]] = []
        
        logger.info(f"GraphReasoningEnvironment initialized with max_turns={max_turns}")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Returns:
            Tuple of (initial_observation, info)
        """
        # Call parent reset to reset done, current_turn, history
        super().reset()
        
        # Clear query and result history
        self.query_history = []
        self.result_history = []
        
        # Build initial observation
        initial_obs = {
            'question': self.task.get('question', '') if self.task else '',
            'turn': 0,
            'history': {
                'queries': [],
                'results': []
            }
        }
        
        logger.debug(f"Environment reset with question: {initial_obs['question']}")
        
        return initial_obs, {}
    
    def get_reward_and_next_obs(
        self,
        task: Dict[str, Any],
        action: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Execute Cypher query and return next observation.
        
        This is the abstract method from MultiTurnEnvironment that must be implemented.
        For Cypher generation, we return 0 reward at each step and compute
        final reward using the reward function after episode completes.
        
        Args:
            task: Task dictionary containing question and other metadata
            action: Cypher query string to execute
            
        Returns:
            Tuple of (reward, next_observation)
        """
        # Handle DONE signal (agent decides to stop)
        if isinstance(action, str) and action.strip().upper() == "DONE":
            logger.info("Agent output DONE signal")
            
            # Return observation indicating completion
            next_obs = {
                'question': task.get('question', ''),
                'previous_query': 'DONE',
                'previous_result': {
                    'success': True,
                    'has_data': False,
                    'result': {},
                    'num_results': 0,
                    'execution_time_ms': 0.0,
                    'data_summary': 'Agent decided to stop exploration'
                },
                'turn': self.current_turn + 1,
                'history': {
                    'queries': self.query_history.copy(),
                    'results': self.result_history.copy()
                }
            }
            
            return 0.0, next_obs
        
        # Execute the Cypher query
        result = self.executor.execute_query(action)
        
        # Store in history
        self.query_history.append(action)
        self.result_history.append(result)
        
        # Generate data summary for prompt
        data_summary = self._generate_data_summary(result)
        
        # Build next observation
        next_obs = {
            'question': task.get('question', ''),
            'previous_query': action,
            'previous_result': {
                'success': result['success'],
                'has_data': result['has_data'],
                'result': result['result'],
                'num_results': result['num_results'],
                'execution_time_ms': result['execution_time_ms'],
                'data_summary': data_summary,
                'error': result.get('error')
            },
            'turn': self.current_turn + 1,
            'history': {
                'queries': self.query_history.copy(),
                'results': self.result_history.copy()
            }
        }
        
        logger.debug(
            f"Turn {self.current_turn + 1}: "
            f"success={result['success']}, "
            f"has_data={result['has_data']}, "
            f"num_results={result['num_results']}, "
            f"time={result['execution_time_ms']:.0f}ms"
        )
        
        # Return 0 reward (final reward computed by reward function)
        return 0.0, next_obs
    
    def _generate_data_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate a brief summary of the query result for prompt inclusion.
        
        Args:
            result: Query execution result dictionary
            
        Returns:
            Brief summary string
        """
        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            return f"Query failed: {error_msg}"
        
        if not result['has_data']:
            return "No results returned"
        
        num_results = result['num_results']
        exec_time = result['execution_time_ms']
        
        # Add time indicator
        if exec_time < 100:
            time_indicator = "⚡"
        elif exec_time < 500:
            time_indicator = "○"
        elif exec_time < 1000:
            time_indicator = "△"
        else:
            time_indicator = "✗"
        
        summary = f"Retrieved {num_results} entities in {exec_time:.0f}ms {time_indicator}"
        
        return summary
    
    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "GraphReasoningEnvironment":
        """
        Create environment from dictionary configuration.
        
        This is required by rllm's AgentTrainer for instantiation from config.
        
        Args:
            env_args: Dictionary with environment configuration
            
        Returns:
            GraphReasoningEnvironment instance
        """
        return GraphReasoningEnvironment(**env_args)
    
    def get_trajectory_data(self) -> Dict[str, Any]:
        """
        Get complete trajectory data for reward computation.
        
        Returns:
            Dictionary containing:
                - question: Original question
                - queries: List of Cypher queries executed
                - results: List of execution results
                - num_steps: Number of steps taken
        """
        return {
            'question': self.task.get('question', '') if self.task else '',
            'queries': self.query_history.copy(),
            'results': self.result_history.copy(),
            'num_steps': len(self.query_history)
        }
    
    def close(self):
        """Clean up resources."""
        self.executor.close()
        logger.info("GraphReasoningEnvironment closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

