"""
ssGSEA REST API client for the PanKgraph immune-cell enrichment server.

The server runs ssGSEA (single-sample Gene Set Enrichment Analysis) on
pseudo-bulk immune cell data from 112 HPAP donors.

Base URL: http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com

Endpoints:
    GET  /genes?search=X  — search available gene names
    GET  /donors          — donor metadata
    POST /ssgsea          — run ssGSEA with {"genes": [...]}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic
import requests

logger = logging.getLogger(__name__)

SSGSEA_BASE_URL = (
    "http://Robject-PanKgraph-ALB-1292067250.us-east-1.elb.amazonaws.com"
)

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type": "application/json"})
    return _session


def search_genes(query: str, timeout: int = 10) -> list[str]:
    """Search available genes by prefix.

    >>> search_genes("INS")
    ['INS', 'INS-IGF2', 'INSC', 'INSIG1', ...]
    """
    resp = _get_session().get(
        f"{SSGSEA_BASE_URL}/genes",
        params={"search": query},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("genes", [])


def get_donors(timeout: int = 10) -> list[dict]:
    """Return donor metadata (one dict per donor)."""
    resp = _get_session().get(
        f"{SSGSEA_BASE_URL}/donors",
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("donors", [])


def run_ssgsea(genes: list[str], timeout: int = 30) -> dict:
    """Run ssGSEA with a custom gene set.

    Args:
        genes: List of gene symbols (e.g. ["INS", "GCG", "SST"]).

    Returns:
        {
            "genes_submitted": int,
            "genes_used": int,
            "genes_not_found": list[str],
            "scores": [{"donor_id": str, "score": float}, ...]
        }
    """
    resp = _get_session().post(
        f"{SSGSEA_BASE_URL}/ssgsea",
        json={"genes": genes},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_gene_names(question: str) -> list[str]:
    """Extract gene symbols from a natural language question using Claude.

    Falls back to regex extraction if Claude is unavailable.
    """
    # Try regex first for simple cases like "Run ssGSEA for INS, GCG, SST, PPY"
    genes = _extract_genes_regex(question)
    if genes:
        return genes

    # Use Claude for complex cases
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=(
                "Extract gene symbols from the user's question. "
                "Return ONLY a JSON array of uppercase gene symbols, nothing else. "
                "Example: [\"INS\", \"GCG\", \"SST\"]\n"
                "If no specific genes are mentioned, return []."
            ),
            messages=[{"role": "user", "content": question}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning(f"Claude gene extraction failed: {exc}")
        return []


def _extract_genes_regex(text: str) -> list[str]:
    """Extract gene symbols from text using regex heuristics.

    Looks for patterns like:
    - "genes INS, GCG, SST, PPY"
    - "INS and GCG"
    - "for INS, GCG"
    """
    # Common words that look like gene symbols but aren't
    _SKIP = {
        "THE", "AND", "FOR", "ARE", "NOT", "ALL", "CAN", "HAS", "RUN",
        "GET", "SET", "USE", "WITH", "FROM", "THAT", "THIS", "WHAT",
        "GENE", "GENES", "GSEA", "CELL", "TYPE", "IMMUNE", "DONOR",
        "SHOW", "FIND", "LIST", "GIVE", "TELL", "ABOUT", "ACROSS",
    }

    # Find sequences of uppercase gene-like tokens
    # Pattern: 2-10 uppercase letters/digits, optionally with hyphen
    tokens = re.findall(r'\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b', text)
    genes = [t for t in tokens if t not in _SKIP and len(t) >= 2]

    # Only return if we found a reasonable number (likely an explicit gene list)
    if len(genes) >= 2:
        return list(dict.fromkeys(genes))  # deduplicate, preserve order
    return []
