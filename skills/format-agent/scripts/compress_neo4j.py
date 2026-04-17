"""Compress raw Neo4j results into compact JSON for the FormatAgent/ReasoningAgent.

Fixes over the original version:
- Edges with no properties (e.g. [:effector_gene_of]) are now captured.
- ALL key-value properties are extracted generically instead of a hardcoded list.
- Nodes without properties are captured (label-only nodes).
- Noisy metadata keys (data_version, data_source, link, etc.) are stripped to save tokens.
"""
from __future__ import annotations

import re
from typing import List

# Properties to EXCLUDE from output — they waste tokens and carry no analytical value.
_SKIP_PROPS = frozenset({
    'data_version', 'data_source', 'data_source_url',
    'link', 'gencode_annotation',
    'start', 'end',  # internal Neo4j edge start/end node IDs, not genomic coords
})


def compress_neo4j_results(neo4j_results: List[dict]) -> List[dict]:
    """
    Compress raw Neo4j results into a compact format.

    The raw Neo4j results contain massive text blobs with full node properties
    repeated for every node.  This function parses the text and extracts ALL
    key-value properties generically, skipping only noisy metadata fields.

    Args:
        neo4j_results: List of dicts with 'query' and 'result' keys.

    Returns:
        List of dicts with 'query', 'node_count', 'edge_count', 'nodes', 'edges'.
    """
    compressed = []
    for entry in neo4j_results:
        query = entry.get('query', '')
        result = entry.get('result', {})

        parsed_nodes = []
        parsed_edges = []

        # New API format: structured records with typed nodes/edges
        if isinstance(result, dict) and "records" in result:
            for record in result.get("records", []):
                if not isinstance(record, dict):
                    continue
                for node in record.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    props = {
                        k: v for k, v in node.get("properties", {}).items()
                        if k not in _SKIP_PROPS
                    }
                    labels = ', '.join(node.get("labels", []))
                    props['_labels'] = labels
                    parsed_nodes.append(props)
                for edge in record.get("edges", []):
                    if not isinstance(edge, dict):
                        continue
                    props = {
                        k: v for k, v in edge.get("properties", {}).items()
                        if k not in _SKIP_PROPS
                    }
                    props['_type'] = edge.get("type", "")
                    parsed_edges.append(props)
        else:
            # Old API format: results is a plain string
            results_text = (result.get('results', '')
                            if isinstance(result, dict) else str(result))

            if isinstance(results_text, str):
                # NODES — three patterns in decreasing specificity
                for m in re.finditer(r'\(:([^{)]+)\{([^}]+)\}\)', results_text):
                    labels = _clean_labels(m.group(1))
                    props = _extract_all_props(m.group(2))
                    props['_labels'] = labels
                    parsed_nodes.append(props)

                for m in re.finditer(r'\(:([^{)]+)\{\s*\}\)', results_text):
                    labels = _clean_labels(m.group(1))
                    parsed_nodes.append({'_labels': labels})

                for m in re.finditer(r'\(:([^{)]+)\)(?!\s*\{)', results_text):
                    labels = _clean_labels(m.group(1))
                    parsed_nodes.append({'_labels': labels})

                # EDGES — three patterns
                for m in re.finditer(r'\[:([^\s{\]]+)\s*\{([^}]+)\}\]', results_text):
                    rel_type = m.group(1)
                    props = _extract_all_props(m.group(2))
                    props['_type'] = rel_type
                    parsed_edges.append(props)

                for m in re.finditer(r'\[:([^\s{\]]+)\s*\{\s*\}\]', results_text):
                    rel_type = m.group(1)
                    parsed_edges.append({'_type': rel_type})

                for m in re.finditer(r'\[:([^\s{\]]+)\](?!\s*\{)', results_text):
                    rel_type = m.group(1)
                    parsed_edges.append({'_type': rel_type})

        # ----------------------------------------------------------
        # Deduplicate nodes by (_labels, name, id) tuple
        # ----------------------------------------------------------
        seen = set()
        unique_nodes = []
        for n in parsed_nodes:
            key = (n.get('_labels', ''), n.get('name', ''), n.get('id', ''))
            if key not in seen:
                seen.add(key)
                unique_nodes.append(n)

        # Deduplicate edges by full content (convert to frozen items)
        seen_edges = set()
        unique_edges = []
        for e in parsed_edges:
            key = tuple(sorted(e.items()))
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        compressed.append({
            'query': query,
            'node_count': len(unique_nodes),
            'edge_count': len(unique_edges),
            'nodes': unique_nodes,
            'edges': unique_edges,
        })

    return compressed


# ── helpers ──────────────────────────────────────────────────────────

def _clean_labels(raw: str) -> str:
    """Convert ':gene:coding_elements ' → 'gene, coding_elements'."""
    return ', '.join(
        part.strip() for part in raw.replace(':', ' ').split() if part.strip()
    )


def _extract_all_props(props_str: str) -> dict:
    """
    Generically extract ALL key-value pairs from a Neo4j property string.

    Handles:
      - Quoted strings:    name: "CFTR"
      - Integers:          start_loc: 117287120
      - Floats / sci:      pip: 0.95,  pval: 1.23E-5
      - Unquoted tokens:   strand: +
      - Nested quotes:     description: "some \\"escaped\\" text"
    """
    props: dict = {}

    # Match  key: "value"  (quoted string — greedy within quotes)
    for m in re.finditer(r'(\w[\w.]*)\s*:\s*"((?:[^"\\]|\\.)*)"', props_str):
        key, val = m.group(1), m.group(2)
        if key not in _SKIP_PROPS:
            props[key] = val

    # Match  key: <number>  (int / float / scientific notation)
    for m in re.finditer(r'(\w[\w.]*)\s*:\s*(-?[\d][\d.]*(?:[eE][+-]?\d+)?)', props_str):
        key, val = m.group(1), m.group(2)
        if key not in _SKIP_PROPS and key not in props:
            # Try to keep the most compact numeric representation
            try:
                if '.' in val or 'e' in val.lower():
                    props[key] = float(val)
                else:
                    props[key] = int(val)
            except ValueError:
                props[key] = val

    # Match  key: <bare_token>  (unquoted non-numeric, e.g. strand: +)
    # Use a lookbehind to ensure the key starts after a boundary (comma, space,
    # or start of string) so we don't accidentally match inside quoted URLs
    # like "http://..." where "http" looks like a key.
    for m in re.finditer(r'(?:^|[,\s])(\w[\w.]*)\s*:\s*([^\s",}{:]+)', props_str):
        key, val = m.group(1), m.group(2)
        if key not in _SKIP_PROPS and key not in props:
            # Skip if it looks like a number (already handled above)
            if not re.match(r'^-?[\d]', val):
                props[key] = val

    return props
