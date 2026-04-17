"""
Entity Degree Extractor for PankBase Knowledge Graph.

Dynamically discovers all node types and relationships from the Neo4j database,
then computes the degree (number of connections) for each entity per relationship type.

This information is used by the AdaptiveEntitySampler to bias sampling toward
entities that are more likely to produce answerable questions.

Usage:
    python -m rl_implementation.utils.degree_extractor --output outputs/entity_degrees.json
    
    # Or from Python:
    from rl_implementation.utils.degree_extractor import DegreeExtractor
    extractor = DegreeExtractor()
    degrees = extractor.extract_all_degrees()
    extractor.save(degrees, 'outputs/entity_degrees.json')
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


DEFAULT_API_URL = "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"


class DegreeExtractor:
    """
    Extracts entity degrees from the PankBase Neo4j database.
    
    Dynamically discovers:
    1. All node labels in the graph
    2. All relationship types
    3. Source/target node types for each relationship
    4. Degree (connection count) for each entity per relationship
    """
    
    def __init__(
        self,
        api_url: str = None,
        timeout: int = 120,
    ):
        """
        Initialize the degree extractor.
        
        Args:
            api_url: Neo4j API endpoint URL
            timeout: Request timeout in seconds (longer for degree queries)
        """
        self.api_url = api_url or DEFAULT_API_URL
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        logger.info(f"DegreeExtractor initialized with API: {self.api_url}")
    
    def _execute_query(self, cypher: str) -> Dict[str, Any]:
        """Execute a Cypher query and return results."""
        try:
            response = self.session.post(
                self.api_url,
                json={'query': cypher},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Query timed out after {self.timeout}s: {cypher[:100]}...")
            return {'error': 'timeout', 'results': ''}
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {'error': str(e), 'results': ''}
    
    def _parse_results_to_list(self, result: Dict[str, Any]) -> List[str]:
        """Parse result string into list of values."""
        results_str = result.get('results', '')
        if not results_str or results_str.lower() == 'no results':
            return []
        
        lines = []
        for line in results_str.strip().split('\n'):
            line = line.strip().strip('"')
            if line and line.lower() != 'no results':
                lines.append(line)
        
        # Skip header row if present
        if lines and lines[0].lower() in ['label', 'relationshiptype', 'name', 'degree']:
            lines = lines[1:]
        
        return lines
    
    def _parse_degree_results(self, result: Dict[str, Any]) -> List[Tuple[str, int]]:
        """
        Parse degree query results into list of (entity_name, degree) tuples.
        
        Expected format: "name, degree" or "name, id, degree"
        """
        results_str = result.get('results', '')
        if not results_str or results_str.lower() == 'no results':
            return []
        
        entries = []
        lines = results_str.strip().split('\n')
        
        # Skip header
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split(', ')
                if len(parts) >= 2:
                    name = parts[0].strip().strip('"')
                    # Degree is last column
                    try:
                        degree = int(parts[-1].strip().strip('"'))
                        if name and degree > 0:
                            entries.append((name, degree))
                    except ValueError:
                        continue
        
        return entries
    
    def discover_node_labels(self) -> List[str]:
        """Discover all node labels in the graph."""
        logger.info("Discovering node labels...")
        
        query = "CALL db.labels() YIELD label RETURN label"
        result = self._execute_query(query)
        
        labels = self._parse_results_to_list(result)
        logger.info(f"Found {len(labels)} node labels: {labels}")
        
        return labels
    
    def discover_relationship_types(self) -> List[str]:
        """Discover all relationship types in the graph."""
        logger.info("Discovering relationship types...")
        
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        result = self._execute_query(query)
        
        rel_types = self._parse_results_to_list(result)
        logger.info(f"Found {len(rel_types)} relationship types: {rel_types}")
        
        return rel_types
    
    def discover_relationship_schema(self, rel_type: str) -> Optional[Dict[str, str]]:
        """
        Discover source and target node types for a relationship.
        
        Args:
            rel_type: Relationship type name
            
        Returns:
            Dict with 'source' and 'target' node labels, or None if not found
        """
        query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        RETURN DISTINCT labels(a)[0] AS source, labels(b)[0] AS target
        LIMIT 1
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        if not results_str or results_str.lower() == 'no results':
            return None
        
        lines = results_str.strip().split('\n')
        if len(lines) > 1:
            # Parse "source, target" line
            parts = lines[1].split(', ')
            if len(parts) >= 2:
                source = parts[0].strip().strip('"')
                target = parts[1].strip().strip('"')
                return {'source': source, 'target': target}
        
        return None
    
    def discover_schema(self) -> Dict[str, Any]:
        """
        Dynamically discover the complete schema from the graph.
        
        Returns:
            Dict with node_labels and relationships (with source/target info)
        """
        logger.info("Discovering complete schema...")
        
        node_labels = self.discover_node_labels()
        rel_types = self.discover_relationship_types()
        
        # Get source/target for each relationship
        relationships = {}
        for rel_type in rel_types:
            schema = self.discover_relationship_schema(rel_type)
            if schema:
                relationships[rel_type] = schema
                logger.debug(f"  {rel_type}: {schema['source']} -> {schema['target']}")
        
        schema = {
            'node_labels': node_labels,
            'relationships': relationships,
        }
        
        logger.info(f"Schema discovery complete: {len(node_labels)} node types, {len(relationships)} relationship types")
        
        return schema
    
    def extract_degrees_for_relationship(
        self,
        rel_type: str,
        source_label: str,
        target_label: str,
    ) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
        """
        Extract degrees for all entities involved in a relationship.
        
        Args:
            rel_type: Relationship type
            source_label: Source node label
            target_label: Target node label
            
        Returns:
            Dict with 'source_degrees' and 'target_degrees' lists
        """
        logger.info(f"Extracting degrees for {rel_type} ({source_label} -> {target_label})...")
        
        # Outgoing degree for source nodes
        # Use name if available, fall back to id
        source_query = f"""
        MATCH (n:{source_label})-[r:{rel_type}]->()
        RETURN COALESCE(n.name, n.id) AS name, count(r) AS degree
        ORDER BY degree DESC
        """
        
        source_result = self._execute_query(source_query)
        source_degrees = self._parse_degree_results(source_result)
        
        # Incoming degree for target nodes (only if different from source)
        target_degrees = []
        if target_label != source_label:
            target_query = f"""
            MATCH ()-[r:{rel_type}]->(n:{target_label})
            RETURN COALESCE(n.name, n.id) AS name, count(r) AS degree
            ORDER BY degree DESC
            """
            
            target_result = self._execute_query(target_query)
            target_degrees = self._parse_degree_results(target_result)
        
        logger.info(f"  {rel_type}: {len(source_degrees)} source entities, {len(target_degrees)} target entities")
        
        return {
            'source_degrees': source_degrees,
            'target_degrees': target_degrees,
        }
    
    def extract_all_degrees(self) -> Dict[str, Any]:
        """
        Extract degrees for all entities in all relationships.
        
        Returns:
            Complete degree data structure ready for saving
        """
        logger.info("Starting full degree extraction...")
        
        # Discover schema first
        schema = self.discover_schema()
        
        # Initialize degrees structure by node type
        degrees: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # Extract degrees for each relationship
        for rel_type, info in schema['relationships'].items():
            source_label = info['source']
            target_label = info['target']
            
            rel_degrees = self.extract_degrees_for_relationship(
                rel_type, source_label, target_label
            )
            
            # Add source degrees
            if source_label not in degrees:
                degrees[source_label] = {}
            
            for entity_name, degree in rel_degrees['source_degrees']:
                if entity_name not in degrees[source_label]:
                    degrees[source_label][entity_name] = {}
                # Use relationship name for outgoing
                degrees[source_label][entity_name][rel_type] = degree
            
            # Add target degrees (with _in suffix to indicate incoming)
            if rel_degrees['target_degrees']:
                if target_label not in degrees:
                    degrees[target_label] = {}
                
                for entity_name, degree in rel_degrees['target_degrees']:
                    if entity_name not in degrees[target_label]:
                        degrees[target_label][entity_name] = {}
                    # Use relationship name with _in suffix for incoming
                    degrees[target_label][entity_name][f"{rel_type}_in"] = degree
        
        # Build final output
        output = {
            'metadata': {
                'extracted_at': datetime.now().isoformat(),
                'api_url': self.api_url,
                'discovered_schema': schema,
            },
            'degrees': degrees,
        }
        
        # Log summary
        total_entities = sum(len(entities) for entities in degrees.values())
        logger.info(f"Extraction complete:")
        logger.info(f"  Node types: {len(degrees)}")
        logger.info(f"  Total entities with degrees: {total_entities}")
        for node_type, entities in degrees.items():
            logger.info(f"    {node_type}: {len(entities)} entities")
        
        return output
    
    def save(self, degrees: Dict[str, Any], path: str):
        """Save degree data to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(degrees, f, indent=2)
        
        logger.info(f"Saved degree data to {path}")
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load degree data from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded degree data from {path}")
        return data
    
    def close(self):
        """Close the session."""
        self.session.close()


def main():
    """CLI entry point for degree extraction."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Extract entity degrees from PankBase Neo4j database'
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs/entity_degrees.json',
        help='Output file path (default: outputs/entity_degrees.json)'
    )
    parser.add_argument(
        '--api-url',
        default=None,
        help='Neo4j API URL (default: PankBase production)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds (default: 120)'
    )
    
    args = parser.parse_args()
    
    # Extract degrees
    extractor = DegreeExtractor(
        api_url=args.api_url,
        timeout=args.timeout,
    )
    
    try:
        degrees = extractor.extract_all_degrees()
        extractor.save(degrees, args.output)
        
        print(f"\n✅ Degree extraction complete!")
        print(f"   Output: {args.output}")
        
        # Show summary
        print(f"\nSummary:")
        for node_type, entities in degrees['degrees'].items():
            print(f"  {node_type}: {len(entities)} entities")
            
            # Show top 3 by total degree
            if entities:
                sorted_entities = sorted(
                    entities.items(),
                    key=lambda x: sum(x[1].values()),
                    reverse=True
                )[:3]
                for name, rels in sorted_entities:
                    total = sum(rels.values())
                    print(f"    - {name}: {total} total connections")
        
    finally:
        extractor.close()


if __name__ == '__main__':
    main()

