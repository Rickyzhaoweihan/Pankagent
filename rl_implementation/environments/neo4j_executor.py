"""
Neo4j Executor for Pankbase API.

Wrapper for executing Cypher queries against the Pankbase Neo4j database
via AWS Lambda endpoint.
"""

import json
import logging
import time
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class Neo4jExecutor:
    """
    Executor for Neo4j Cypher queries via Pankbase AWS Lambda API.
    
    Handles query execution, response parsing, and error handling.
    """
    
    def __init__(self, api_url: str, timeout: int = 60):
        """
        Initialize the Neo4j executor.
        
        Args:
            api_url: AWS Lambda endpoint URL for Pankbase Neo4j API
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        logger.info(f"Neo4jExecutor initialized with endpoint: {api_url}")
    
    def execute_query(self, cypher_query: str) -> Dict[str, Any]:
        """
        Execute a Cypher query against the Neo4j database.
        
        Args:
            cypher_query: Cypher query string to execute
            
        Returns:
            Dictionary containing:
                - success: bool (whether query executed without errors)
                - has_data: bool (whether results contain actual data)
                - result: dict (parsed API response)
                - num_results: int (count of nodes/edges returned)
                - execution_time_ms: float (query execution time)
                - error: str | None (error message if any)
        """
        start_time = time.time()
        
        try:
            # Clean the Cypher query for JSON submission
            cleaned_cypher = self._clean_cypher_for_json(cypher_query)
            
            # Execute the query
            response = self.session.post(
                self.api_url,
                json={'query': cleaned_cypher},
                timeout=self.timeout
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            result = self._parse_response(response)
            
            # Check if the result contains actual data
            has_data = self._check_has_data(result)
            
            # Count results (nodes + edges)
            num_results = self._count_results(result)
            
            logger.debug(f"Query executed in {execution_time_ms:.0f}ms, has_data={has_data}, num_results={num_results}")
            
            return {
                'success': True,
                'has_data': has_data,
                'result': result,
                'num_results': num_results,
                'execution_time_ms': execution_time_ms,
                'error': None
            }
            
        except requests.exceptions.Timeout:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Query timeout after {self.timeout}s"
            logger.warning(error_msg)
            return {
                'success': False,
                'has_data': False,
                'result': {},
                'num_results': 0,
                'execution_time_ms': execution_time_ms,
                'error': error_msg
            }
            
        except requests.exceptions.RequestException as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Request error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'has_data': False,
                'result': {},
                'num_results': 0,
                'execution_time_ms': execution_time_ms,
                'error': error_msg
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'has_data': False,
                'result': {},
                'num_results': 0,
                'execution_time_ms': execution_time_ms,
                'error': error_msg
            }
    
    def _clean_cypher_for_json(self, cypher: str) -> str:
        """
        Clean Cypher query for JSON submission.
        
        Args:
            cypher: Raw Cypher query string
            
        Returns:
            Cleaned Cypher query string
        """
        # Normalize whitespace
        cleaned = ' '.join(cypher.split())
        
        # Escape quotes for JSON
        cleaned = cleaned.replace('"', '\"').replace("'", '\"')
        
        return cleaned
    
    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse the API response.
        
        Args:
            response: HTTP response from API
            
        Returns:
            Parsed response dictionary
            
        Raises:
            ValueError: If response is invalid
        """
        # Check for empty response
        if not response.text.strip():
            raise ValueError("Empty response from Pankbase API")
        
        # Check for error response
        if response.text.strip().startswith("Error:"):
            raise ValueError(f"Pankbase API Error: {response.text}")
        
        # Parse JSON
        try:
            result = response.json()
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from API: {response.text}") from e
    
    def _check_has_data(self, result: Dict[str, Any]) -> bool:
        """
        Check if the result contains actual data.
        
        Args:
            result: Parsed API response
            
        Returns:
            True if result contains data, False otherwise
        """
        if not isinstance(result, dict):
            return False
        
        results_value = result.get('results', '')
        
        # Check if results is "No results" (case-insensitive)
        if isinstance(results_value, str):
            results_lower = results_value.strip().lower()
            
            if results_lower == "no results":
                return False
            
            # Check for empty string
            if not results_value.strip():
                return False
            
            # Check for empty nodes and edges: "nodes, edges\n[], []" or "nodes, edges\n[],[]"
            normalized = ' '.join(results_value.split())
            
            # Check for both "[], []" and "[][]" patterns (with or without space)
            if 'nodes, edges' in normalized.lower():
                if '[], []' in normalized or '[][]' in normalized.replace(' ', ''):
                    return False
        
        # Check for error field
        if result.get('error'):
            return False
        
        return True
    
    def _count_results(self, result: Dict[str, Any]) -> int:
        """
        Count the number of results (nodes + edges) in the response.
        
        Args:
            result: Parsed API response
            
        Returns:
            Count of nodes and edges
        """
        if not isinstance(result, dict):
            return 0
        
        results_value = result.get('results', '')
        
        if not isinstance(results_value, str):
            return 0
        
        # Try to parse the results string to count nodes and edges
        # The format is typically: "nodes, edges\n[...], [...]"
        try:
            # Simple heuristic: count occurrences of common patterns
            # This is a rough estimate - actual parsing would be more complex
            
            # Check if it's empty
            if not self._check_has_data(result):
                return 0
            
            # Count opening braces/brackets as a proxy for entities
            # This is a rough heuristic
            count = results_value.count('{')
            
            return max(0, count)
            
        except Exception:
            return 0
    
    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("Neo4jExecutor session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

