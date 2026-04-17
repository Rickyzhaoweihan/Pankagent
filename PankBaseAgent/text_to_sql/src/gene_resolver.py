"""
gene_resolver.py
Resolve human-readable gene names (e.g. INS, CFTR) to Ensembl IDs
(e.g. ENSG00000254647) using the RDS Lambda gene_name table.

The same endpoint and fuzzy-match logic used by TemplateToolAgent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

RDS_LAMBDA_URL = (
    "https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/RDSLambda"
)

# Cache resolved names across calls (gene names don't change within a session)
_cache: dict[str, str | None] = {}

# Matches tokens that look like gene symbols: 2-10 uppercase letters/digits,
# optionally followed by a hyphen and more chars (e.g. HLA-A, TP53, INS).
_GENE_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b")

# Ensembl ID pattern — skip these, they're already resolved
_ENSEMBL_RE = re.compile(r"ENSG\d{11}")


def resolve_gene_name(name: str, timeout: float = 8) -> Optional[str]:
    """Return the Ensembl ID for a gene symbol, or None if not found.

    Uses the RDS Lambda ``gene_name`` table with trigram fuzzy matching,
    identical to TemplateToolAgent/ai_assistant.py.
    """
    key = name.upper().strip()
    if key in _cache:
        return _cache[key]

    safe = key.replace("'", "''")
    sql = (
        f"SELECT id, name FROM gene_name "
        f"WHERE name % '{safe}' "
        f"ORDER BY similarity(name, '{safe}') DESC LIMIT 1"
    )

    try:
        resp = requests.post(
            RDS_LAMBDA_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": sql}),
            timeout=timeout,
        )
        if resp.status_code == 200:
            results = resp.json().get("results") or []
            if results:
                ensembl_id = results[0].get("id")
                if ensembl_id:
                    _cache[key] = ensembl_id
                    logger.info(f"Gene resolved: {name} → {ensembl_id}")
                    return ensembl_id
    except Exception as exc:
        logger.warning(f"Gene resolution failed for '{name}': {exc}")

    _cache[key] = None
    return None


def resolve_gene_names_in_text(text: str) -> tuple[str, dict[str, str]]:
    """Find gene symbols in *text* and replace them with Ensembl IDs.

    Returns ``(new_text, resolved)`` where *resolved* maps each original
    symbol to its Ensembl ID.  Symbols that cannot be resolved are left
    unchanged.

    Only resolves tokens that look like gene symbols (2-10 uppercase
    letters/digits).  Skips tokens that are already Ensembl IDs.
    """
    # Common English words that match the gene-symbol regex but aren't genes
    _SKIP = {
        "INS",  # Keep this — it's a real gene (insulin)
        "THE", "AND", "FOR", "ARE", "NOT", "ALL", "CAN", "HAS", "HER",
        "WAS", "ONE", "OUR", "OUT", "HAD", "HIS", "HOW", "ITS", "MAY",
        "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "LET",
        "SAY", "SHE", "TOO", "USE", "WHAT", "WHICH", "WHERE", "WHEN",
        "FROM", "THAT", "THIS", "THAN", "FIND", "WITH", "GENE", "GENES",
        "WHAT", "GWAS", "SNPS", "TYPE", "CELL", "BETA", "ALSO", "DOES",
        "HAVE", "MANY", "MUCH", "NEAR", "SHOW", "TELL", "GIVE", "LIST",
        "LIKE", "EACH", "BOTH", "MOST", "ONLY", "SOME", "MORE", "OVER",
        "SAME", "SUCH", "VERY", "JUST", "WILL", "THEM", "BEEN", "LONG",
        "MAKE", "BETWEEN", "ABOUT", "AFTER", "COULD", "THESE", "OTHER",
        "THEIR", "WHICH", "WOULD", "THERE", "FIRST", "KNOWN", "EVERY",
        "THOSE", "UNDER", "STILL", "WHILE", "MIGHT", "BEING", "ALONG",
        "LARGE", "SMALL", "TOTAL", "COUNT", "CHECK", "PEAKS", "OVERLAP",
        "OCR", "QTL", "SNP",
    }
    # But keep known real gene names that overlap with English words
    _FORCE_RESOLVE = {"INS", "MAFA", "CFTR", "TP53", "BRCA1", "BRCA2"}

    candidates = _GENE_SYMBOL_RE.findall(text)
    resolved: dict[str, str] = {}

    for symbol in set(candidates):
        if _ENSEMBL_RE.match(symbol):
            continue
        if symbol in _SKIP and symbol not in _FORCE_RESOLVE:
            continue
        ensembl_id = resolve_gene_name(symbol)
        if ensembl_id:
            resolved[symbol] = ensembl_id

    # Replace in text — use word boundaries to avoid partial matches
    new_text = text
    for symbol, ensembl_id in resolved.items():
        new_text = re.sub(
            rf"\b{re.escape(symbol)}\b",
            ensembl_id,
            new_text,
        )

    return new_text, resolved
