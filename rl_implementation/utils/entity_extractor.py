"""
Entity Extractor for PankBase Knowledge Graph.

Queries the Neo4j database to extract sample entities for each node type.
These entities can be used as seeds for diverse question generation.

Usage:
    python -m rl_implementation.utils.entity_extractor
    
    # Or from Python:
    from rl_implementation.utils.entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    entities = extractor.extract_all()
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


@dataclass
class EntitySamples:
    """Container for entity samples from the knowledge graph."""
    
    # Gene entities
    genes: List[Dict[str, str]] = field(default_factory=list)  # [{name, id, description}, ...]
    
    # Disease entities
    diseases: List[Dict[str, str]] = field(default_factory=list)  # [{name, id, definition}, ...]
    
    # Cell type entities
    cell_types: List[Dict[str, str]] = field(default_factory=list)  # [{name, id, definition}, ...]
    
    # SNP entities
    snps: List[Dict[str, str]] = field(default_factory=list)  # [{id, chr, type}, ...]
    
    # Gene ontology terms
    gene_ontology: List[Dict[str, str]] = field(default_factory=list)  # [{name, id, description}, ...]
    
    # Relationship samples (to understand what connections exist)
    relationship_samples: Dict[str, List[str]] = field(default_factory=dict)
    
    # Relationship-aware entity lists (entities that HAVE specific relationships)
    # These are used by the question generator to ensure questions can be answered
    genes_with_physical_interaction: List[str] = field(default_factory=list)
    genes_with_function_annotation: List[str] = field(default_factory=list)
    genes_with_DEG_in: List[str] = field(default_factory=list)
    genes_with_effector_gene_of: List[str] = field(default_factory=list)
    genes_with_signal_COLOC: List[str] = field(default_factory=list)
    cell_types_with_DEG: List[str] = field(default_factory=list)
    cell_types_with_expression: List[str] = field(default_factory=list)
    snps_with_GWAS: List[str] = field(default_factory=list)
    snps_with_QTL: List[str] = field(default_factory=list)
    
    # Metadata
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntitySamples":
        return cls(**data)
    
    def get_random_gene(self) -> Optional[str]:
        """Get a random gene name."""
        if not self.genes:
            return None
        return random.choice(self.genes).get('name')
    
    def get_random_disease(self) -> Optional[str]:
        """Get a random disease name."""
        if not self.diseases:
            return None
        return random.choice(self.diseases).get('name')
    
    def get_random_cell_type(self) -> Optional[str]:
        """Get a random cell type name."""
        if not self.cell_types:
            return None
        return random.choice(self.cell_types).get('name')
    
    def get_random_entity(self, entity_type: Optional[str] = None) -> Optional[str]:
        """
        Get a random entity, optionally of a specific type.
        
        Args:
            entity_type: 'gene', 'disease', 'cell_type', or None for any
        """
        if entity_type == 'gene':
            return self.get_random_gene()
        elif entity_type == 'disease':
            return self.get_random_disease()
        elif entity_type == 'cell_type':
            return self.get_random_cell_type()
        else:
            # Random from any type
            all_names = []
            for gene in self.genes:
                if gene.get('name'):
                    all_names.append(('gene', gene['name']))
            for disease in self.diseases:
                if disease.get('name'):
                    all_names.append(('disease', disease['name']))
            for cell in self.cell_types:
                if cell.get('name'):
                    all_names.append(('cell_type', cell['name']))
            
            if not all_names:
                return None
            return random.choice(all_names)[1]


class EntityExtractor:
    """
    Extracts entity samples from the PankBase Neo4j database.
    
    Queries for sample entities of each node type to enable
    entity-seeded question generation for training diversity.
    """
    
    DEFAULT_API_URL = "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"
    
    def __init__(
        self,
        api_url: str = None,
        timeout: int = 60,
        sample_size: int = 100,
    ):
        """
        Initialize the entity extractor.
        
        Args:
            api_url: Neo4j API endpoint URL
            timeout: Request timeout in seconds
            sample_size: Max number of entities to sample per type
        """
        self.api_url = api_url or self.DEFAULT_API_URL
        self.timeout = timeout
        self.sample_size = sample_size
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
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
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {'error': str(e), 'results': ''}
    
    def _parse_results(self, result: Dict[str, Any]) -> List[str]:
        """Parse result string into list of values."""
        results_str = result.get('results', '')
        if not results_str or results_str.lower() == 'no results':
            return []
        
        # Parse the Neo4j result format
        lines = []
        for line in results_str.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('(') and line != 'No results':
                lines.append(line)
        
        return lines
    
    def extract_genes(self) -> List[Dict[str, str]]:
        """Extract sample genes from the database."""
        logger.info("Extracting gene samples...")
        
        query = f"""
        MATCH (g:gene)
        WHERE g.name IS NOT NULL
        RETURN DISTINCT g.name AS name, g.id AS id, g.description AS description
        LIMIT {self.sample_size}
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        genes = []
        if results_str and results_str.lower() != 'no results':
            # Parse results - format: "name, id, description\nvalue1, value2, value3\n..."
            lines = results_str.strip().split('\n')
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    parts = line.split(', ', 2)  # Split into max 3 parts
                    if len(parts) >= 2:
                        genes.append({
                            'name': parts[0].strip().strip('"'),
                            'id': parts[1].strip().strip('"') if len(parts) > 1 else '',
                            'description': parts[2].strip().strip('"') if len(parts) > 2 else '',
                        })
        
        logger.info(f"Extracted {len(genes)} genes")
        return genes
    
    def extract_diseases(self) -> List[Dict[str, str]]:
        """Extract sample diseases from the database."""
        logger.info("Extracting disease samples...")
        
        query = f"""
        MATCH (d:disease)
        WHERE d.name IS NOT NULL
        RETURN DISTINCT d.name AS name, d.id AS id, d.definition AS definition
        LIMIT {self.sample_size}
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        diseases = []
        if results_str and results_str.lower() != 'no results':
            lines = results_str.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(', ', 2)
                    if parts:
                        diseases.append({
                            'name': parts[0].strip().strip('"'),
                            'id': parts[1].strip().strip('"') if len(parts) > 1 else '',
                            'definition': parts[2].strip().strip('"')[:200] if len(parts) > 2 else '',
                        })
        
        logger.info(f"Extracted {len(diseases)} diseases")
        return diseases
    
    def extract_cell_types(self) -> List[Dict[str, str]]:
        """Extract sample cell types from the database."""
        logger.info("Extracting cell type samples...")
        
        query = f"""
        MATCH (c:cell_type)
        WHERE c.name IS NOT NULL
        RETURN DISTINCT c.name AS name, c.id AS id, c.definition AS definition
        LIMIT {self.sample_size}
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        cell_types = []
        if results_str and results_str.lower() != 'no results':
            lines = results_str.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(', ', 2)
                    if parts:
                        cell_types.append({
                            'name': parts[0].strip().strip('"'),
                            'id': parts[1].strip().strip('"') if len(parts) > 1 else '',
                            'definition': parts[2].strip().strip('"')[:200] if len(parts) > 2 else '',
                        })
        
        logger.info(f"Extracted {len(cell_types)} cell types")
        return cell_types
    
    def extract_snps(self) -> List[Dict[str, str]]:
        """Extract sample SNPs from the database."""
        logger.info("Extracting SNP samples...")
        
        query = f"""
        MATCH (s:snp)
        WHERE s.id IS NOT NULL
        RETURN DISTINCT s.id AS id, s.chr AS chr, s.type AS type
        LIMIT {self.sample_size}
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        snps = []
        if results_str and results_str.lower() != 'no results':
            lines = results_str.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(', ', 2)
                    if parts:
                        snps.append({
                            'id': parts[0].strip().strip('"'),
                            'chr': parts[1].strip().strip('"') if len(parts) > 1 else '',
                            'type': parts[2].strip().strip('"') if len(parts) > 2 else '',
                        })
        
        logger.info(f"Extracted {len(snps)} SNPs")
        return snps
    
    def extract_gene_ontology(self) -> List[Dict[str, str]]:
        """Extract sample gene ontology terms from the database."""
        logger.info("Extracting gene ontology samples...")
        
        query = f"""
        MATCH (go:gene_ontology)
        WHERE go.name IS NOT NULL
        RETURN DISTINCT go.name AS name, go.id AS id, go.description AS description
        LIMIT {self.sample_size}
        """
        
        result = self._execute_query(query)
        results_str = result.get('results', '')
        
        go_terms = []
        if results_str and results_str.lower() != 'no results':
            lines = results_str.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(', ', 2)
                    if parts:
                        go_terms.append({
                            'name': parts[0].strip().strip('"'),
                            'id': parts[1].strip().strip('"') if len(parts) > 1 else '',
                            'description': parts[2].strip().strip('"')[:200] if len(parts) > 2 else '',
                        })
        
        logger.info(f"Extracted {len(go_terms)} gene ontology terms")
        return go_terms
    
    def extract_relationship_samples(self) -> Dict[str, List[str]]:
        """Extract sample relationships to understand what connections exist."""
        logger.info("Extracting relationship samples...")
        
        samples = {}
        
        # Gene-Disease relationships (effector genes)
        query = """
        MATCH (g:gene)-[r:effector_gene_of]->(d:disease)
        RETURN g.name + ' -> ' + d.name AS sample
        LIMIT 20
        """
        result = self._execute_query(query)
        samples['gene_effector_disease'] = self._parse_simple_list(result)
        
        # Gene-CellType relationships (expression)
        query = """
        MATCH (g:gene)-[r:expression_level_in]->(c:cell_type)
        RETURN g.name + ' -> ' + c.name AS sample
        LIMIT 20
        """
        result = self._execute_query(query)
        samples['gene_expression_celltype'] = self._parse_simple_list(result)
        
        # Gene-Gene relationships (physical interaction)
        query = """
        MATCH (g1:gene)-[r:physical_interaction]->(g2:gene)
        RETURN g1.name + ' -> ' + g2.name AS sample
        LIMIT 20
        """
        result = self._execute_query(query)
        samples['gene_interaction_gene'] = self._parse_simple_list(result)
        
        # SNP-Disease relationships (GWAS)
        query = """
        MATCH (s:snp)-[r:part_of_GWAS_signal]->(d:disease)
        RETURN s.id + ' -> ' + d.name AS sample
        LIMIT 20
        """
        result = self._execute_query(query)
        samples['snp_gwas_disease'] = self._parse_simple_list(result)
        
        # Gene-GeneOntology relationships (function)
        query = """
        MATCH (g:gene)-[r:function_annotation]->(go:gene_ontology)
        RETURN g.name + ' -> ' + go.name AS sample
        LIMIT 20
        """
        result = self._execute_query(query)
        samples['gene_function_go'] = self._parse_simple_list(result)
        
        logger.info(f"Extracted relationship samples for {len(samples)} types")
        return samples
    
    def _parse_simple_list(self, result: Dict[str, Any]) -> List[str]:
        """Parse a simple list result."""
        results_str = result.get('results', '')
        if not results_str or results_str.lower() == 'no results':
            return []
        
        lines = results_str.strip().split('\n')
        if len(lines) > 1:
            return [line.strip().strip('"') for line in lines[1:] if line.strip()]
        return []
    
    def extract_all(self) -> EntitySamples:
        """
        Extract all entity samples from the database.
        
        Returns:
            EntitySamples object with all extracted data
        """
        logger.info("Starting full entity extraction...")
        
        samples = EntitySamples(
            genes=self.extract_genes(),
            diseases=self.extract_diseases(),
            cell_types=self.extract_cell_types(),
            snps=self.extract_snps(),
            gene_ontology=self.extract_gene_ontology(),
            relationship_samples=self.extract_relationship_samples(),
            source_url=self.api_url,
        )
        
        # Log summary
        logger.info(f"Extraction complete:")
        logger.info(f"  - Genes: {len(samples.genes)}")
        logger.info(f"  - Diseases: {len(samples.diseases)}")
        logger.info(f"  - Cell types: {len(samples.cell_types)}")
        logger.info(f"  - SNPs: {len(samples.snps)}")
        logger.info(f"  - GO terms: {len(samples.gene_ontology)}")
        logger.info(f"  - Relationship types: {len(samples.relationship_samples)}")
        
        return samples
    
    def save(self, samples: EntitySamples, path: str):
        """Save entity samples to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(samples.to_dict(), f, indent=2)
        
        logger.info(f"Saved entity samples to {path}")
    
    def load(self, path: str) -> EntitySamples:
        """Load entity samples from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded entity samples from {path}")
        return EntitySamples.from_dict(data)
    
    def close(self):
        """Close the session."""
        self.session.close()


def main():
    """CLI entry point for entity extraction."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Extract entity samples from PankBase')
    parser.add_argument(
        '--output', '-o',
        default='outputs/entity_samples.json',
        help='Output file path (default: outputs/entity_samples.json)'
    )
    parser.add_argument(
        '--sample-size', '-n',
        type=int,
        default=100,
        help='Max entities per type (default: 100)'
    )
    parser.add_argument(
        '--api-url',
        default=None,
        help='Neo4j API URL (default: PankBase production)'
    )
    
    args = parser.parse_args()
    
    # Extract entities
    extractor = EntityExtractor(
        api_url=args.api_url,
        sample_size=args.sample_size,
    )
    
    try:
        samples = extractor.extract_all()
        extractor.save(samples, args.output)
        
        print(f"\n✅ Entity extraction complete!")
        print(f"   Output: {args.output}")
        print(f"\nSummary:")
        print(f"  - Genes: {len(samples.genes)}")
        print(f"  - Diseases: {len(samples.diseases)}")
        print(f"  - Cell types: {len(samples.cell_types)}")
        print(f"  - SNPs: {len(samples.snps)}")
        print(f"  - GO terms: {len(samples.gene_ontology)}")
        
        # Show some examples
        print(f"\nSample entities:")
        if samples.genes:
            print(f"  Gene: {samples.genes[0].get('name', 'N/A')}")
        if samples.diseases:
            print(f"  Disease: {samples.diseases[0].get('name', 'N/A')}")
        if samples.cell_types:
            print(f"  Cell Type: {samples.cell_types[0].get('name', 'N/A')}")
        
    finally:
        extractor.close()


if __name__ == '__main__':
    main()

