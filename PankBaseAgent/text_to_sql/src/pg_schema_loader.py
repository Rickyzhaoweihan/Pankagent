"""
pg_schema_loader.py
Compact PostgreSQL schema string for the text-to-SQL LLM prompt.
Mirrors schema_loader.py from text_to_cypher but for the genomic_interval table.
"""

from __future__ import annotations

_cached_schema: str | None = None


def get_pg_schema_for_llm() -> str:
    """Return ultra-compact PostgreSQL schema string optimized for small models.

    Keeps token usage low (~200 tokens) while providing all necessary context
    for generating correct SQL against the genomic_interval table.
    """
    global _cached_schema
    if _cached_schema is None:
        _cached_schema = (
            'Table: genomic_interval(id text, entity_type text, "chr" text, "start" bigint, "end" bigint)\n'
            "PK: (entity_type, id). Index: (chr, start, end).\n"
            "Entity types:\n"
            "  Ensembl_genes.node (78687 rows) — Ensembl gene IDs, e.g. ENSG00000254647\n"
            "  GWAS_snp_id.node (1615 rows) — GWAS SNP rsIDs, e.g. rs1050976\n"
            "  ocr_peak.node (5294421 rows) — Open Chromatin Region peaks, e.g. CL_0000169_1_100008394_100008769\n"
            "  QTL_snp.node (19422 rows) — QTL SNP rsIDs, e.g. rs10004120\n"
            "Chromosomes: 1, 2, 3, ..., 22, X, Y\n"
            "\n"
            "Notes:\n"
            '- ALWAYS double-quote "chr", "start", "end" — they are PostgreSQL reserved words\n'
            "- Overlap test: a.\"chr\" = b.\"chr\" AND a.\"start\" <= b.\"end\" AND a.\"end\" >= b.\"start\"\n"
            "- Filter entity type: WHERE entity_type = 'Ensembl_genes.node'\n"
            "- Always add LIMIT for exploratory queries (default LIMIT 100)\n"
        )
    return _cached_schema
