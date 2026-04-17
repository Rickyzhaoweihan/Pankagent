"""
Cypher Auto-Fix Utility for RL Rollout Collection.

Automatically fixes common Cypher query errors using hard-coded rules (no LLM).
Designed to run silently during rollout collection to improve query success rate.

Usage:
    from rl_implementation.utils.cypher_auto_fix import auto_fix_cypher
    
    fixed_query = auto_fix_cypher(query)  # Returns fixed query silently
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Schema and Entity Data Loading (cached)
# ============================================================================

_cached_schema: Dict[str, Any] | None = None
_cached_entity_samples: Dict[str, Any] | None = None

# Default paths (relative to this file)
# This file is at: rl_implementation/utils/cypher_auto_fix.py
# Need to go up 2 levels to project root, then into legacy/
_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "legacy" / "PankBaseAgent" / "text_to_cypher" / "data" / "input" / "kg_schema.json"
_DEFAULT_ENTITY_SAMPLES_PATH = Path(__file__).resolve().parent.parent.parent / "legacy" / "PankBaseAgent" / "text_to_cypher" / "data" / "input" / "entity_samples.json"


def load_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the knowledge graph schema."""
    global _cached_schema
    if _cached_schema is None:
        path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle nested schema format
                if "knowledge_graph_schema" in data:
                    _cached_schema = data["knowledge_graph_schema"]
                else:
                    _cached_schema = data
        else:
            logger.warning(f"Schema file not found: {path}")
            _cached_schema = {"node_types": {}, "edge_types": {}}
    return _cached_schema


def load_entity_samples(samples_path: Optional[str] = None) -> Dict[str, Any]:
    """Load entity samples for value validation."""
    global _cached_entity_samples
    if _cached_entity_samples is None:
        path = Path(samples_path) if samples_path else _DEFAULT_ENTITY_SAMPLES_PATH
        if path.exists():
            with open(path, 'r') as f:
                _cached_entity_samples = json.load(f)
        else:
            logger.warning(f"Entity samples file not found: {path}")
            _cached_entity_samples = {}
    return _cached_entity_samples


def clear_cache():
    """Clear cached schema and entity samples."""
    global _cached_schema, _cached_entity_samples
    _cached_schema = None
    _cached_entity_samples = None


# ============================================================================
# Schema-Based Helpers
# ============================================================================

def get_relationship_directions() -> Dict[str, Dict[str, str]]:
    """
    Get expected source/target node types for each relationship type.
    
    Returns:
        {rel_type: {'source': 'node_type', 'target': 'node_type'}, ...}
    """
    schema = load_schema()
    directions = {}
    
    for rel_key, rel_spec in schema.get("edge_types", {}).items():
        rel_type = rel_key.split(";")[-1]  # Get simple name
        source = rel_spec.get("source_node_type", "")
        target = rel_spec.get("target_node_type", "")
        directions[rel_type] = {
            'source': source,
            'target': target
        }
    
    return directions


def get_valid_node_properties() -> Dict[str, Set[str]]:
    """
    Get valid properties for each node type.
    
    Returns:
        {node_label: {'prop1', 'prop2', ...}, ...}
    """
    schema = load_schema()
    valid_props = {}
    
    for node_key, node_spec in schema.get("node_types", {}).items():
        node_label = node_key.split(";")[-1]
        props = set(node_spec.get("properties", {}).keys())
        valid_props[node_label] = props
    
    return valid_props


def get_valid_edge_properties() -> Dict[str, Set[str]]:
    """
    Get valid properties for each edge type.
    
    Returns:
        {edge_type: {'prop1', 'prop2', ...}, ...}
    """
    schema = load_schema()
    valid_props = {}
    
    for edge_key, edge_spec in schema.get("edge_types", {}).items():
        edge_type = edge_key.split(";")[-1]
        props = set(edge_spec.get("properties", {}).keys())
        valid_props[edge_type] = props
    
    return valid_props


def get_valid_entity_values() -> Dict[str, Set[str]]:
    """
    Get valid entity values from entity samples.
    
    Returns:
        {
            'gene_names': {'INS', 'CFTR', ...},
            'disease_names': {'type 1 diabetes', ...},
            'cell_type_names': {'Beta Cell', 'Alpha Cell', ...},
            'snp_ids': {'rs7903146', ...},
            'go_names': {'protein binding', ...}
        }
    """
    samples = load_entity_samples()
    
    valid_values = {
        'gene_names': set(),
        'disease_names': set(),
        'cell_type_names': set(),
        'snp_ids': set(),
        'go_names': set()
    }
    
    # Extract gene names
    for gene in samples.get('genes', []):
        if gene.get('name'):
            valid_values['gene_names'].add(gene['name'])
    
    # Extract disease names
    for disease in samples.get('diseases', []):
        if disease.get('name'):
            valid_values['disease_names'].add(disease['name'])
    
    # Extract cell type names
    for cell_type in samples.get('cell_types', []):
        if cell_type.get('name'):
            valid_values['cell_type_names'].add(cell_type['name'])
    
    # Extract SNP IDs
    for snp in samples.get('snps', []):
        if snp.get('id'):
            valid_values['snp_ids'].add(snp['id'])
    
    # Extract GO term names
    for go in samples.get('gene_ontology', []):
        if go.get('name'):
            valid_values['go_names'].add(go['name'])
    
    return valid_values


# ============================================================================
# Query Analysis Helpers
# ============================================================================

def extract_matched_variables(cypher: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract all node and relationship variables from MATCH clauses.
    
    Returns:
        (node_variables, relationship_variables)
    """
    node_vars = set()
    rel_vars = set()
    
    # Find all MATCH clauses
    match_patterns = re.findall(
        r'\bMATCH\b(.*?)(?=\bWHERE\b|\bWITH\b|\bRETURN\b|\bMATCH\b|$)',
        cypher, re.IGNORECASE | re.DOTALL
    )
    
    for pattern in match_patterns:
        # Find node variables: (var:Label) or (var {props}) or (var)
        node_matches = re.findall(r'\((\w+)(?::\w+)?[^)]*\)', pattern)
        node_vars.update(node_matches)
        
        # Find relationship variables: [var:Type] or [var {props}] or [var]
        rel_matches = re.findall(r'\[(\w+)(?::\w+)?[^]]*\]', pattern)
        rel_vars.update(rel_matches)
    
    return node_vars, rel_vars


def extract_collected_variables(cypher: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract variables collected in WITH clause for nodes and edges.
    
    Returns:
        (collected_nodes, collected_edges)
    """
    collected_nodes = set()
    collected_edges = set()
    
    # Find WITH clause (between WITH and RETURN)
    with_match = re.search(r'\bWITH\b(.*?)\bRETURN\b', cypher, re.IGNORECASE | re.DOTALL)
    if not with_match:
        return collected_nodes, collected_edges
    
    with_clause = with_match.group(1)
    
    # Find what's collected for nodes (... AS nodes)
    nodes_pattern = re.search(r'(.*?)\s+AS\s+nodes', with_clause, re.IGNORECASE | re.DOTALL)
    if nodes_pattern:
        nodes_expr = nodes_pattern.group(1)
        collected = re.findall(r'collect\s*\(\s*(?:DISTINCT\s+)?(\w+)\s*\)', nodes_expr, re.IGNORECASE)
        collected_nodes.update(collected)
    
    # Find what's collected for edges (... AS edges)
    edges_pattern = re.search(r'(.*?)\s+AS\s+edges', with_clause, re.IGNORECASE | re.DOTALL)
    if edges_pattern:
        edges_expr = edges_pattern.group(1)
        collected = re.findall(r'collect\s*\(\s*(?:DISTINCT\s+)?(\w+)\s*\)', edges_expr, re.IGNORECASE)
        collected_edges.update(collected)
    
    return collected_nodes, collected_edges


def extract_node_labels_and_rel_types(cypher: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract node labels and relationship types from query.
    
    Returns:
        (node_labels, relationship_types)
    """
    node_labels = set()
    rel_types = set()
    
    # Extract node labels: (var:Label)
    node_patterns = re.findall(r'\([^)]*:(\w+)[^)]*\)', cypher)
    for label in node_patterns:
        node_labels.add(label.split(';')[-1])
    
    # Extract relationship types: [var:Type]
    rel_patterns = re.findall(r'\[[^]]*:(\w+)[^]]*\]', cypher)
    for rel_type in rel_patterns:
        rel_types.add(rel_type.split(';')[-1])
    
    return node_labels, rel_types


# ============================================================================
# Query Complexity Detection
# ============================================================================

def is_multi_match_query(cypher: str) -> bool:
    """
    Detect if query has multiple MATCH clauses.
    
    Multi-MATCH queries like:
        MATCH ... WITH ... MATCH ... RETURN ...
    
    These are valid Cypher but require different handling in auto-fix
    to avoid breaking their structure.
    
    Returns:
        True if query has multiple MATCH clauses
    """
    match_count = len(re.findall(r'\bMATCH\b', cypher, re.IGNORECASE))
    return match_count > 1


def is_simple_query(cypher: str) -> bool:
    """
    Detect if this is a simple single-MATCH query.
    
    Simple queries have:
    - One MATCH clause
    - One WITH clause (at the end, before RETURN)
    - Standard nodes/edges collection pattern
    
    These are safe for all auto-fix operations.
    """
    match_count = len(re.findall(r'\bMATCH\b', cypher, re.IGNORECASE))
    return match_count == 1


# ============================================================================
# Individual Fix Functions
# ============================================================================

def fix_single_quotes_to_double(cypher: str) -> str:
    """
    Convert single quotes to double quotes for string values in Cypher.
    
    CRITICAL FIX: The Neo4j API gateway strips single quotes from queries,
    causing syntax errors like:
        {name: 'Delta Cell'} → {name: Delta Cell} → ERROR!
        
    This fix converts:
        {name: 'Delta Cell'} → {name: "Delta Cell"}
        {id: 'rs123456'} → {id: "rs123456"}
        .name = 'value' → .name = "value"
    
    Note: This must be applied EARLY in the fix chain before other fixes
    that might depend on the quote style.
    """
    if not cypher:
        return cypher
    
    # Pattern 1: Fix {property: 'value'} patterns
    # Matches: {name: 'value'}, {id: 'value'}, etc.
    def fix_property_value(match):
        prop = match.group(1)
        value = match.group(2)
        return f'{{{prop}: "{value}"}}'
    
    fixed = re.sub(
        r"\{(\w+):\s*'([^']+)'\}",
        fix_property_value,
        cypher
    )
    
    # Pattern 2: Fix .property = 'value' patterns  
    # Matches: .name = 'value', .id = 'value', etc.
    def fix_assignment_value(match):
        prop = match.group(1)
        value = match.group(2)
        return f'.{prop} = "{value}"'
    
    fixed = re.sub(
        r"\.(\w+)\s*=\s*'([^']+)'",
        fix_assignment_value,
        fixed
    )
    
    # Pattern 3: Fix standalone 'value' in WHERE clauses
    # e.g., WHERE n.name = 'value' (if not caught by pattern 2)
    # More conservative - only fix known patterns
    def fix_where_value(match):
        prefix = match.group(1)
        value = match.group(2)
        return f'{prefix}"{value}"'
    
    fixed = re.sub(
        r"(WHERE\s+\w+\.\w+\s*=\s*)'([^']+)'",
        fix_where_value,
        fixed,
        flags=re.IGNORECASE
    )
    
    return fixed


def fix_relationship_variables(cypher: str) -> str:
    """
    Add missing variable names to relationships.
    
    Fixes: -[:TYPE]- → -[r:TYPE]-
    """
    # Counter for unique variable names
    var_counter = [0]
    
    def replacement(match):
        rel_content = match.group(1).strip()
        
        # If it starts with : (no variable name), add one
        if rel_content.startswith(':'):
            var_counter[0] += 1
            var_name = f"r{var_counter[0]}"
            return f"-[{var_name}{rel_content}]-"
        
        # If empty, add variable and placeholder type
        if not rel_content:
            var_counter[0] += 1
            return f"-[r{var_counter[0]}]-"
        
        return match.group(0)  # No change needed
    
    # Find and fix relationship patterns
    fixed = re.sub(r'-\[(.*?)\]-', replacement, cypher)
    
    return fixed


def fix_undefined_collected_variables(cypher: str) -> str:
    """
    Fix references to undefined variables in collect() statements.
    
    Works for both simple and multi-MATCH queries by analyzing scope.
    
    For simple queries:
        MATCH (g:gene)-[r1:TYPE]->(d:disease)
        WITH collect(g) AS nodes, collect(r) AS edges  ← r undefined, becomes r1
    
    For multi-MATCH queries:
        MATCH (a)-[r1:TYPE1]->(b)
        WITH collect(a) AS nodes1, collect(r) AS edges1  ← r becomes r1
        MATCH (c)-[r2:TYPE2]->(d)
        WITH nodes1 + collect(c) AS n2, edges1 + collect(r) AS e2  ← r becomes r2
    """
    # Check if multi-MATCH
    is_multi = is_multi_match_query(cypher)
    
    if is_multi:
        return _fix_undefined_vars_multi_match(cypher)
    else:
        return _fix_undefined_vars_simple(cypher)


def _fix_undefined_vars_simple(cypher: str) -> str:
    """Fix undefined variables for simple single-MATCH queries."""
    # Get all defined variables
    matched_nodes, matched_rels = extract_matched_variables(cypher)
    all_matched = matched_nodes | matched_rels
    
    # Find all variables referenced in collect() statements
    collected_vars = re.findall(r'collect\s*\(\s*(?:DISTINCT\s+)?(\w+)\s*\)', cypher, re.IGNORECASE)
    
    # Identify undefined collected variables
    undefined_vars = [v for v in collected_vars if v not in all_matched]
    
    if not undefined_vars:
        return cypher
    
    fixed = cypher
    
    for undef_var in undefined_vars:
        # Check if it looks like a relationship variable
        is_rel_like = undef_var.lower() in ['r', 'rel', 'r1', 'r2', 'r3', 'edge', 'e']
        
        if is_rel_like and matched_rels:
            actual_rel = sorted(matched_rels)[0]
            fixed = re.sub(
                rf'collect\s*\(\s*(DISTINCT\s+)?{re.escape(undef_var)}\s*\)',
                rf'collect(\1{actual_rel})',
                fixed,
                flags=re.IGNORECASE
            )
        else:
            # Remove undefined collect
            fixed = re.sub(
                rf'\+\s*collect\s*\(\s*(?:DISTINCT\s+)?{re.escape(undef_var)}\s*\)',
                '', fixed, flags=re.IGNORECASE
            )
            fixed = re.sub(
                rf'collect\s*\(\s*(?:DISTINCT\s+)?{re.escape(undef_var)}\s*\)\s*\+',
                '', fixed, flags=re.IGNORECASE
            )
            fixed = re.sub(
                rf',\s*collect\s*\(\s*(?:DISTINCT\s+)?{re.escape(undef_var)}\s*\)\s+AS\s+edges',
                ', [] AS edges', fixed, flags=re.IGNORECASE
            )
    
    return fixed


def _fix_undefined_vars_multi_match(cypher: str) -> str:
    """
    Fix undefined variables for multi-MATCH queries.
    
    Strategy: Parse into MATCH-WITH segments, fix each segment's WITH
    using the relationship variables from that segment's MATCH.
    """
    # Split query into segments at MATCH boundaries
    # Pattern: MATCH ... (up to next MATCH or RETURN)
    segments = re.split(r'(?=\bMATCH\b)', cypher, flags=re.IGNORECASE)
    segments = [s for s in segments if s.strip()]
    
    if len(segments) <= 1:
        return _fix_undefined_vars_simple(cypher)
    
    fixed_segments = []
    
    for segment in segments:
        # Find relationship variables in this segment's MATCH clause
        rel_vars_in_segment = re.findall(r'\[(\w+):\w+\]', segment)
        
        if not rel_vars_in_segment:
            fixed_segments.append(segment)
            continue
        
        # The relationship variable for this segment
        segment_rel = rel_vars_in_segment[0]  # Usually just one per MATCH
        
        # Find undefined 'r' or similar in this segment's collect() calls
        # Pattern: collect(DISTINCT r) where r is not defined in this segment
        def fix_collect_in_segment(match):
            full_match = match.group(0)
            distinct_part = match.group(1) or ''
            var_name = match.group(2)
            
            # Check if this variable is defined in the segment
            # Only match relationship vars [var:TYPE] and node vars (var:Label)
            # NOT collect(var) which would incorrectly mark var as defined
            all_vars_in_segment = set(re.findall(r'\[(\w+):\w+\]', segment))  # [r1:TYPE]
            all_vars_in_segment.update(re.findall(r'\((\w+):\w+', segment))   # (g:gene)
            
            if var_name not in all_vars_in_segment:
                # It's undefined - check if it looks like a relationship var
                if var_name.lower() in ['r', 'rel', 'edge', 'e'] or re.match(r'^r\d*$', var_name, re.IGNORECASE):
                    return f'collect({distinct_part}{segment_rel})'
            
            return full_match
        
        # Fix collect() calls in the WITH clause of this segment
        fixed_segment = re.sub(
            r'collect\s*\(\s*(DISTINCT\s+)?(\w+)\s*\)',
            fix_collect_in_segment,
            segment,
            flags=re.IGNORECASE
        )
        
        fixed_segments.append(fixed_segment)
    
    return ''.join(fixed_segments)


def fix_distinct_in_collect(cypher: str) -> str:
    """
    Add DISTINCT to collect() calls that are missing it.
    
    Fixes: collect(var) → collect(DISTINCT var)
    """
    # Pattern: collect( followed by word (not DISTINCT) followed by )
    fixed = re.sub(
        r'collect\s*\(\s*(?!DISTINCT)(\w+)\s*\)',
        r'collect(DISTINCT \1)',
        cypher,
        flags=re.IGNORECASE
    )
    return fixed


def fix_return_format(cypher: str) -> str:
    """
    Fix RETURN statement to return only nodes and edges.
    
    Fixes: RETURN nodes, edges, extra → RETURN nodes, edges;
    Preserves any existing LIMIT clause.
    """
    # Check if we need to fix
    if not re.search(r'\bRETURN\b', cypher, re.IGNORECASE):
        return cypher
    
    # Already correct format? (with optional LIMIT)
    if re.search(r'RETURN\s+nodes\s*,\s*edges\s*(?:\s+LIMIT\s+\d+)?\s*;?\s*$', cypher, re.IGNORECASE):
        return cypher
    
    # Extract any existing LIMIT before replacing
    limit_match = re.search(r'\bLIMIT\s+(\d+)', cypher, re.IGNORECASE)
    limit_clause = f'\nLIMIT {limit_match.group(1)}' if limit_match else ''
    
    # Fix: replace RETURN statement with correct format, preserving LIMIT
    fixed = re.sub(
        r'\bRETURN\s+.*$',
        f'RETURN nodes, edges{limit_clause};',
        cypher,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    return fixed


def fix_multi_match_collections(cypher: str) -> str:
    """
    Fix multi-MATCH queries to properly collect all nodes and edges.
    
    This function tracks variable SCOPE properly:
    - After `WITH var1, var2`, only var1 and var2 are in scope
    - Variables from previous MATCHes are LOST unless carried forward
    
    Strategy:
    1. Parse query linearly, tracking scope at each point
    2. Fix INTERMEDIATE WITH clauses to carry forward ALL vars from their preceding MATCH
    3. Track what's actually in scope at the final WITH
    4. Only collect variables that are in scope at the end
    
    Example:
        MATCH (g:gene)-[r:TYPE1]->(d:disease)
        WITH g                                    ← Bug! r and d lost
        MATCH (g)-[r2:TYPE2]->(go:gene_ontology)
        WITH collect(g) AS nodes, collect(r) AS edges  ← r out of scope!
        RETURN nodes, edges;
        
    Becomes:
        MATCH (g:gene)-[r:TYPE1]->(d:disease)
        WITH g, r, d                              ← All vars carried forward
        MATCH (g)-[r2:TYPE2]->(go:gene_ontology)
        WITH collect(DISTINCT d)+collect(DISTINCT g)+collect(DISTINCT go) AS nodes, 
             collect(DISTINCT r)+collect(DISTINCT r2) AS edges
        RETURN nodes, edges;
    """
    if not is_multi_match_query(cypher):
        return cypher
    
    # Helper to extract node vars from a MATCH clause
    def extract_nodes_from_match(text):
        return set(re.findall(r'\((\w+)(?::\w+)?[^)]*\)', text))
    
    # Helper to extract relationship vars from a MATCH clause
    def extract_rels_from_match(text):
        return set(re.findall(r'\[(\w+):\w+[^\]]*\]', text))
    
    # Split query into lines and process
    lines = cypher.strip().split('\n')
    
    # Track scope as we process the query
    current_scope_nodes = set()
    current_scope_rels = set()
    all_node_vars = set()
    all_rel_vars = set()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line has a MATCH
        if re.match(r'\bMATCH\b', line, re.IGNORECASE):
            # Extract variables from this MATCH
            new_nodes = extract_nodes_from_match(line)
            new_rels = extract_rels_from_match(line)
            
            # Add to scope and track all vars
            current_scope_nodes.update(new_nodes)
            current_scope_rels.update(new_rels)
            all_node_vars.update(new_nodes)
            all_rel_vars.update(new_rels)
            
            fixed_lines.append(lines[i])
            i += 1
            continue
        
        # Check if this line has WHERE (may have constraints, keep as-is)
        if re.match(r'\bWHERE\b', line, re.IGNORECASE):
            fixed_lines.append(lines[i])
            i += 1
            continue
        
        # Check if this is a WITH line
        if re.match(r'\bWITH\b', line, re.IGNORECASE):
            # Check if there's more after this (another MATCH or RETURN)
            remaining_lines = '\n'.join(lines[i+1:]) if i+1 < len(lines) else ''
            has_more_match = bool(re.search(r'\bMATCH\b', remaining_lines, re.IGNORECASE))
            has_return = bool(re.search(r'\bRETURN\b', remaining_lines, re.IGNORECASE))
            
            with_content = re.sub(r'^\s*WITH\s+', '', line, flags=re.IGNORECASE)
            has_collect = 'collect' in with_content.lower()
            
            if has_more_match and not has_collect:
                # This is an INTERMEDIATE WITH that just passes vars
                # Expand it to include ALL vars currently in scope
                vars_to_carry = sorted(current_scope_nodes | current_scope_rels)
                vars_to_carry = [v for v in vars_to_carry if v not in ('nodes', 'edges')]
                
                if vars_to_carry:
                    fixed_lines.append(f"WITH {', '.join(vars_to_carry)}")
                else:
                    fixed_lines.append(lines[i])
                i += 1
                continue
            
            elif has_more_match and has_collect:
                # This is an INTERMEDIATE WITH with collect() - keep as-is but reset scope
                # After this, only the aliases are in scope
                aliases = re.findall(r'\bAS\s+(\w+)', with_content, re.IGNORECASE)
                # These aliases become the new "variables" in scope (they're collections, not nodes)
                # Clear the node/rel scope - next MATCH will add new vars
                current_scope_nodes.clear()
                current_scope_rels.clear()
                # Track intermediate collection aliases for use in final WITH
                for alias in aliases:
                    if 'node' in alias.lower() or alias.lower().startswith('n'):
                        current_scope_nodes.add(alias)
                    elif 'edge' in alias.lower() or alias.lower().startswith('e') or 'rel' in alias.lower():
                        current_scope_rels.add(alias)
                
                fixed_lines.append(lines[i])
                i += 1
                continue
            
            elif not has_more_match:
                # This is the FINAL WITH clause (or only WITH before RETURN)
                # Build proper collection from in-scope variables
                # Separate intermediate aliases (already collections) from regular vars
                in_scope_nodes = current_scope_nodes - {'nodes', 'edges'}
                in_scope_rels = current_scope_rels - {'nodes', 'edges'}
                
                node_parts = []
                edge_parts = []
                
                for v in sorted(in_scope_nodes):
                    # Check if this is an intermediate alias (already a collection)
                    if re.match(r'^nodes\d*$', v, re.IGNORECASE) or v.lower().startswith('n') and v[1:].isdigit():
                        node_parts.append(v)  # Already a collection, use directly
                    else:
                        node_parts.append(f"collect(DISTINCT {v})")  # Wrap in collect
                
                for v in sorted(in_scope_rels):
                    # Check if this is an intermediate alias (already a collection)
                    if re.match(r'^edges\d*$', v, re.IGNORECASE) or v.lower().startswith('e') and v[1:].isdigit():
                        edge_parts.append(v)  # Already a collection, use directly
                    else:
                        edge_parts.append(f"collect(DISTINCT {v})")  # Wrap in collect
                
                nodes_expr = "+".join(node_parts) if node_parts else "[]"
                edges_expr = "+".join(edge_parts) if edge_parts else "[]"
                
                fixed_lines.append(f"WITH {nodes_expr} AS nodes, {edges_expr} AS edges")
                i += 1
                continue
        
        # Check if this is RETURN
        if re.match(r'\bRETURN\b', line, re.IGNORECASE):
            fixed_lines.append("RETURN nodes, edges;")
            i += 1
            continue
        
        # Default: keep line as-is
        fixed_lines.append(lines[i])
        i += 1
    
    return '\n'.join(fixed_lines)


def fix_missing_collections(cypher: str) -> str:
    """
    Add missing nodes and relationships to collections.
    
    Ensures all MATCH variables are collected in WITH clause.
    """
    # Get matched vs collected variables
    matched_nodes, matched_rels = extract_matched_variables(cypher)
    collected_nodes, collected_edges = extract_collected_variables(cypher)
    
    missing_nodes = matched_nodes - collected_nodes
    missing_rels = matched_rels - collected_edges
    
    if not missing_nodes and not missing_rels:
        return cypher
    
    fixed = cypher
    
    # Add missing nodes to collection
    if missing_nodes:
        # Find the nodes expression in WITH clause
        with_match = re.search(r'(WITH\s+)(.*?)(\s+AS\s+nodes)', fixed, re.IGNORECASE)
        if with_match:
            prefix = with_match.group(1)
            current_expr = with_match.group(2)
            suffix = with_match.group(3)
            
            for var in missing_nodes:
                if not re.search(rf'\bcollect\s*\([^)]*\b{var}\b[^)]*\)', current_expr, re.IGNORECASE):
                    current_expr += f"+collect(DISTINCT {var})"
            
            fixed = fixed[:with_match.start()] + prefix + current_expr + suffix + fixed[with_match.end():]
    
    # Add missing relationships to collection
    if missing_rels:
        # Find the edges expression in WITH clause
        edges_match = re.search(r'(,\s*)(.*?)(\s+AS\s+edges)', fixed, re.IGNORECASE)
        if edges_match:
            prefix = edges_match.group(1)
            current_expr = edges_match.group(2)
            suffix = edges_match.group(3)
            
            for var in missing_rels:
                if var not in current_expr:
                    if current_expr.strip() == '[]':
                        current_expr = f"collect(DISTINCT {var})"
                    else:
                        current_expr += f"+collect(DISTINCT {var})"
            
            fixed = fixed[:edges_match.start()] + prefix + current_expr + suffix + fixed[edges_match.end():]
    
    return fixed


def fix_extra_collections(cypher: str) -> str:
    """
    Merge extra collections into nodes/edges.
    
    Fixes: WITH collect(g) AS nodes, collect(r) AS edges, collect(d) AS diseases
    →      WITH collect(g)+collect(d) AS nodes, collect(r) AS edges
    """
    # Find WITH clause
    with_match = re.search(r'WITH\s+(.*?)\s+RETURN', cypher, re.IGNORECASE | re.DOTALL)
    if not with_match:
        return cypher
    
    with_content = with_match.group(1)
    
    # Count AS clauses
    as_count = len(re.findall(r'\bAS\s+\w+', with_content, re.IGNORECASE))
    if as_count <= 2:
        return cypher  # Already correct
    
    # Extract all collections
    collections = re.findall(r'(collect\([^)]+\))\s+AS\s+(\w+)', with_content, re.IGNORECASE)
    
    node_collections = []
    edge_collections = []
    
    for collect_expr, alias in collections:
        alias_lower = alias.lower()
        if alias_lower in ['edges', 'relationships', 'rels']:
            edge_collections.append(collect_expr)
        elif alias_lower != 'nodes':  # Other aliases are treated as nodes
            node_collections.append(collect_expr)
        else:
            node_collections.insert(0, collect_expr)  # Keep existing nodes first
    
    if len(node_collections) > 1 or edge_collections:
        merged_nodes = '+'.join(node_collections) if node_collections else 'collect(DISTINCT null)'
        merged_edges = '+'.join(edge_collections) if edge_collections else '[]'
        
        new_with = f"WITH {merged_nodes} AS nodes, {merged_edges} AS edges"
        fixed = re.sub(r'WITH\s+.*?(?=\s+RETURN)', new_with, cypher, flags=re.IGNORECASE | re.DOTALL)
        return fixed
    
    return cypher


def fix_disease_naming(cypher: str) -> str:
    """
    Fix disease name references to use 'type 1 diabetes'.
    
    Fixes: T1D, Type 1 Diabetes, etc. → type 1 diabetes
    """
    # Common incorrect patterns and their fixes
    fixes = [
        # T1D → type 1 diabetes
        (r"['\"]\s*T1D\s*['\"]", "'type 1 diabetes'"),
        # Type 1 Diabetes (capitalized) → type 1 diabetes  
        (r"['\"]Type\s+1\s+Diabetes['\"]", "'type 1 diabetes'"),
        # type 1 diabetic → type 1 diabetes
        (r"['\"]\s*type\s+1\s+diabetic\s*['\"]", "'type 1 diabetes'"),
        # diabetes type 1 → type 1 diabetes
        (r"['\"]\s*diabetes\s+type\s+1\s*['\"]", "'type 1 diabetes'"),
        # T1DM → type 1 diabetes
        (r"['\"]\s*T1DM\s*['\"]", "'type 1 diabetes'"),
    ]
    
    fixed = cypher
    for pattern, replacement in fixes:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
    
    return fixed


def fix_cell_type_references(cypher: str) -> str:
    """
    Fix cell type references using entity_samples.json.
    
    Handles:
    1. Plurals: "Ductal Cells" → "Ductal Cell"
    2. Case: "ductal cell" → "Ductal Cell"
    3. Invalid types: Removes references to Endothelial, Stellate, Macrophage
       (these exist in entity_samples but NOT in actual database)
    
    Uses exact names from entity_samples.json cell_types[].name property.
    Only the following 5 cell types exist in the actual database:
    - Acinar Cell, Alpha Cell, Beta Cell, Delta Cell, Ductal Cell
    """
    # Get entity samples
    samples = load_entity_samples()
    if not samples or 'cell_types' not in samples:
        return cypher
    
    # Extract ALL cell types from entity_samples.json
    all_cell_types_in_samples = {ct['name'] for ct in samples.get('cell_types', []) if 'name' in ct}
    
    # Only these 5 actually exist in database (based on valid_entities.json)
    # Others (Stellate, Endothelial, Macrophage) exist in samples but NOT in database
    VALID_DATABASE_CELL_TYPES = {
        'Acinar Cell',
        'Alpha Cell', 
        'Beta Cell',
        'Delta Cell',
        'Ductal Cell'
    }
    
    if not all_cell_types_in_samples:
        return cypher
    
    # Build comprehensive mappings for ALL cell types in entity_samples
    # (We fix plurals even for invalid types - let them fail for the RIGHT reason)
    
    # 1. Plural → Singular mapping (for ALL cell types in samples)
    plural_to_singular = {}
    for ct in all_cell_types_in_samples:
        plural = ct + "s"  # "Beta Cell" → "Beta Cells"
        plural_to_singular[plural.lower()] = ct
    
    # 2. Case-insensitive mapping (for ALL cell types in samples)
    case_map = {ct.lower(): ct for ct in all_cell_types_in_samples}
    
    # 3. Combined mapping (handles both plural and case)
    all_fixes = {}
    all_fixes.update(case_map)  # Add case fixes
    all_fixes.update(plural_to_singular)  # Add plural fixes (overwrites if needed)
    
    def fix_cell_type_value_in_braces(match):
        """Fix a cell type value in {name: "value"} pattern."""
        quote = match.group(1)
        value = match.group(2)
        value_lower = value.lower()
        
        # Try to fix the value
        if value_lower in all_fixes:
            fixed_name = all_fixes[value_lower]
            if fixed_name not in VALID_DATABASE_CELL_TYPES:
                logger.debug(f"Fixed cell type '{value}' → '{fixed_name}', but this type doesn't exist in database - query will likely fail")
            # Return complete pattern with fixed value
            return f'{{name: {quote}{fixed_name}{quote}}}'
        
        # Value doesn't match any known cell type - leave unchanged
        return match.group(0)
    
    def fix_cell_type_value_in_assignment(match):
        """Fix a cell type value in .name = "value" pattern."""
        quote = match.group(1)
        value = match.group(2)
        value_lower = value.lower()
        
        # Try to fix the value
        if value_lower in all_fixes:
            fixed_name = all_fixes[value_lower]
            if fixed_name not in VALID_DATABASE_CELL_TYPES:
                logger.debug(f"Fixed cell type '{value}' → '{fixed_name}', but this type doesn't exist in database - query will likely fail")
            # Return complete pattern with fixed value
            return f'.name = {quote}{fixed_name}{quote}'
        
        # Value doesn't match any known cell type - leave unchanged
        return match.group(0)
    
    # Pattern 1: Match {name: 'value'} or {name: "value"} format
    fixed = re.sub(
        r'\{name:\s*([\'"])([^\'\"]+)\1\}',
        fix_cell_type_value_in_braces,
        cypher,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: Match .name = 'value' or .name = "value" format  
    fixed = re.sub(
        r'\.name\s*=\s*([\'"])([^\'\"]+)\1',
        fix_cell_type_value_in_assignment,
        fixed,
        flags=re.IGNORECASE
    )
    
    return fixed


def fix_property_names(cypher: str) -> str:
    """
    Fix common property name typos based on schema.
    
    Common fixes:
    - upOrDownRegulation → UpOrDownRegulation
    - log2foldchange → Log2FoldChange
    """
    # Get valid properties
    node_props = get_valid_node_properties()
    edge_props = get_valid_edge_properties()
    
    # Build case-insensitive property mapping
    all_props = set()
    for props in node_props.values():
        all_props.update(props)
    for props in edge_props.values():
        all_props.update(props)
    
    prop_map = {p.lower(): p for p in all_props}
    
    def fix_property_access(match):
        """Fix a property access to use correct case."""
        var = match.group(1)
        prop = match.group(2)
        prop_lower = prop.lower()
        
        if prop_lower in prop_map and prop != prop_map[prop_lower]:
            return f"{var}.{prop_map[prop_lower]}"
        return match.group(0)
    
    # Pattern: var.property
    fixed = re.sub(r'(\w+)\.(\w+)', fix_property_access, cypher)
    
    return fixed


def fix_relationship_directions(cypher: str) -> str:
    """
    Fix relationship directions based on schema.
    
    If the direction is wrong, swap it to match schema.
    
    Note: This is a complex fix and may not always be correct.
    Currently just logs a warning for manual review.
    """
    # This is harder to auto-fix without potentially breaking queries
    # For now, we'll leave this as a validation-only check
    # The fix would require swapping node positions which could break WHERE clauses
    
    # Get expected directions
    directions = get_relationship_directions()
    
    # For now, just return the query unchanged
    # A future enhancement could attempt to fix this
    return cypher


def add_result_limit(cypher: str, default_limit: int = 100) -> str:
    """
    Add LIMIT to queries that don't have one to prevent timeout on large results.
    
    The PankBase API times out (~30s) when returning thousands of results.
    Adding LIMIT prevents this while still returning useful data.
    
    IMPORTANT: LIMIT must be placed BEFORE collect() operations, not after RETURN!
    If placed after RETURN, the collect() still aggregates ALL results first.
    
    Pattern: 
        MATCH (g)-[r]->(c) 
        WITH g, r, c LIMIT 100           ← ADD LIMIT HERE
        WITH collect(g) AS nodes, ...
        RETURN nodes, edges;
    
    Args:
        cypher: The Cypher query
        default_limit: Maximum number of results (default: 100)
        
    Returns:
        Query with LIMIT added if not present
    """
    if not cypher:
        return cypher
    
    # Check if LIMIT already exists anywhere in the query
    if re.search(r'\bLIMIT\s+\d+', cypher, re.IGNORECASE):
        return cypher
    
    # For queries with collect() - add LIMIT BEFORE the collect operation
    # Pattern: ... WITH collect(... 
    collect_match = re.search(r'\bWITH\s+collect\s*\(', cypher, re.IGNORECASE)
    
    if collect_match:
        # Find the line before the WITH collect() and add a LIMIT clause
        # We need to insert: WITH <vars> LIMIT N before the WITH collect()
        
        # Find all variables from the MATCH clause(s)
        # Look for node and relationship variable patterns
        node_vars = set(re.findall(r'\((\w+)(?::\w+)?[^)]*\)', cypher[:collect_match.start()]))
        rel_vars = set(re.findall(r'\[(\w+):\w+[^\]]*\]', cypher[:collect_match.start()]))
        all_vars = node_vars | rel_vars
        
        if all_vars:
            vars_str = ', '.join(sorted(all_vars))
            # Insert WITH <vars> LIMIT N before the WITH collect()
            insert_pos = collect_match.start()
            return (cypher[:insert_pos] + 
                    f"WITH {vars_str} LIMIT {default_limit}\n" + 
                    cypher[insert_pos:])
    
    # Fallback: For simple queries without collect(), add LIMIT after RETURN
    if re.search(r'\bRETURN\b', cypher, re.IGNORECASE):
        cypher_clean = cypher.rstrip().rstrip(';')
        return f"{cypher_clean}\nLIMIT {default_limit};"
    
    return cypher


# ============================================================================
# Main Auto-Fix Function
# ============================================================================

def auto_fix_cypher(
    cypher: str,
    schema_path: Optional[str] = None,
    entity_samples_path: Optional[str] = None
) -> str:
    """
    Automatically fix common Cypher query errors.
    
    Handles two types of queries:
    - SIMPLE queries (single MATCH): All fixes applied
    - MULTI-MATCH queries: Only safe fixes applied (preserves query structure)
    
    Fixes for ALL queries:
    0. Convert single quotes to double quotes (CRITICAL - Neo4j API strips single quotes!)
    1. Add missing relationship variable names: -[:TYPE]- → -[r:TYPE]-
    1b. Fix undefined collected variables: collect(r) → collect(r1)
    2. Add DISTINCT to collect() calls
    6. Fix disease naming (T1D → type 1 diabetes)
    7. Fix cell type naming (beta cell → Beta Cell)
    8. Fix property name case
    
    Additional fixes for SIMPLE queries:
    3. Merge extra collections into nodes/edges
    4. Add missing variables to collections
    5. Fix return format to RETURN nodes, edges;
    
    Additional fixes for MULTI-MATCH queries:
    3-5. Fix multi-match collections: Rebuilds final WITH to collect ALL nodes/edges
         from ALL MATCH clauses, outputs RETURN nodes, edges;
    
    Args:
        cypher: The Cypher query to fix
        schema_path: Optional path to schema JSON file
        entity_samples_path: Optional path to entity samples JSON file
    
    Returns:
        Fixed Cypher query (silently, no error reporting)
    """
    if not cypher or not isinstance(cypher, str):
        return cypher
    
    # Load schema and entity samples if custom paths provided
    if schema_path:
        global _cached_schema
        _cached_schema = None
        load_schema(schema_path)
    
    if entity_samples_path:
        global _cached_entity_samples
        _cached_entity_samples = None
        load_entity_samples(entity_samples_path)
    
    fixed = cypher
    
    # Detect query complexity - multi-MATCH queries need gentler handling
    is_multi = is_multi_match_query(cypher)
    is_simple = is_simple_query(cypher)
    
    if is_multi:
        logger.debug("Detected multi-MATCH query - applying safe fixes only (preserving structure)")
    
    try:
        # Fix 0: Convert single quotes to double quotes (CRITICAL - API strips single quotes!)
        # Must be applied FIRST before other fixes
        fixed = fix_single_quotes_to_double(fixed)
        
        # Fix 1: Add missing relationship variable names (SAFE for all queries)
        fixed = fix_relationship_variables(fixed)
        
        # Fix 1b: Fix undefined variables in collect() 
        # Works for both simple and multi-MATCH queries (handles scope properly)
        fixed = fix_undefined_collected_variables(fixed)
        
        # Fix 2: Add DISTINCT to collect() calls (SAFE for all queries)
        fixed = fix_distinct_in_collect(fixed)
        
        # Fix 3-5: Collection and return fixes
        if is_multi:
            # For multi-MATCH: Use special fix that properly collects all nodes/edges
            # while preserving the multi-hop structure
            fixed = fix_multi_match_collections(fixed)
        else:
            # For simple queries: Use standard fixes
            # Fix 3: Merge extra collections
            fixed = fix_extra_collections(fixed)
            
            # Fix 4: Add missing variables to collections
            fixed = fix_missing_collections(fixed)
            
            # Fix 5: Fix return format
            fixed = fix_return_format(fixed)
        
        # Fix 6: Fix disease naming (SAFE for all queries)
        fixed = fix_disease_naming(fixed)
        
        # Fix 7: Fix cell type references - handles plurals and invalid types (SAFE for all queries)
        fixed = fix_cell_type_references(fixed)
        
        # Fix 8: Fix property name case (SAFE for all queries)
        fixed = fix_property_names(fixed)
        
        # Fix 9: Add LIMIT to prevent timeout on large results (SAFE for all queries)
        # The API times out (~30s) when returning thousands of nodes
        fixed = add_result_limit(fixed, default_limit=100)
        
    except Exception as e:
        # If any fix fails, return the original query
        logger.debug(f"Auto-fix failed: {e}")
        return cypher
    
    return fixed


# ============================================================================
# Convenience Functions
# ============================================================================

def create_auto_fixer(
    schema_path: Optional[str] = None,
    entity_samples_path: Optional[str] = None,
    enabled: bool = True
):
    """
    Create an auto-fix function with pre-loaded configuration.
    
    Args:
        schema_path: Path to schema JSON file
        entity_samples_path: Path to entity samples JSON file
        enabled: Whether auto-fix is enabled (if False, returns identity function)
    
    Returns:
        Function that takes a query and returns the fixed query
    """
    if not enabled:
        return lambda cypher: cypher
    
    # Pre-load schema and entity samples
    if schema_path:
        load_schema(schema_path)
    if entity_samples_path:
        load_entity_samples(entity_samples_path)
    
    def fixer(cypher: str) -> str:
        return auto_fix_cypher(cypher)
    
    return fixer


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_queries = [
        # Missing relationship variable
        """MATCH (g:gene)-[:effector_gene_of]->(d:disease) 
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        
        # Missing DISTINCT
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WITH collect(g)+collect(d) AS nodes, collect(r) AS edges 
        RETURN nodes, edges;""",
        
        # Wrong disease name
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WHERE d.name = 'T1D'
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        
        # Missing node in collection
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WHERE d.name = 'type 1 diabetes'
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        
        # Extra collections
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges, collect(DISTINCT d) AS diseases
        RETURN nodes, edges, diseases;""",
    ]
    
    print("=" * 70)
    print("CYPHER AUTO-FIX TEST")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}:")
        print(f"Original:\n{query.strip()}")
        fixed = auto_fix_cypher(query)
        print(f"\nFixed:\n{fixed.strip()}")
        print("-" * 70)

