---
name: reasoning-agent
description: >-
  This skill should be used when the user's biomedical query is COMPLEX and
  requires multi-hop reasoning across multiple data types (QTL, OCR, expression,
  GO, PPI, colocalization, disease associations). The PlannerAgent classifies
  question complexity at the start and routes complex questions here instead of
  to the FormatAgent. The ReasoningAgent decomposes the question, reasons
  step-by-step through the data, and produces a transparent reasoning trace
  alongside the final summary.
---

# ReasoningAgent — Multi-Hop Biomedical Reasoning Engine

Perform multi-hop reasoning over raw sub-agent data (Neo4j results, HIRN literature, Cypher queries) to answer complex biomedical questions that require cross-data-source integration.

## When to Use This Skill (vs FormatAgent)

| Question Type | Agent | Example |
|---|---|---|
| Simple gene lookup | FormatAgent | "What is CFTR?" |
| Simple QTL query | FormatAgent | "Which SNP serves as the lead QTL for CFTR?" |
| Multi-hop reasoning | **ReasoningAgent** | "Which genes are QTL targets of variants that lie in Beta-cell OCR, and are also differentially expressed in Beta cells?" |
| Cross-entity integration | **ReasoningAgent** | "For T1D, which genes colocalize with disease signals and are also QTL targets of fine-mapped variants?" |
| Set operations | **ReasoningAgent** | "Which GO terms are shared across T1D colocalized genes and T2D GWAS-to-QTL mapped genes?" |
| Network reasoning | **ReasoningAgent** | "What PPI subnetwork is induced by starting from T1D colocalized genes and expanding one hop?" |

## Pipeline Overview

```
Raw Sub-Agent Data
    |
    v
 Compress Neo4j -------> Parse raw Neo4j text into compact JSON (deduplicated nodes/edges)
    |
    v
 Reasoning Response ----> Claude Opus 4.6 decomposes question, reasons step-by-step,
    |                      and synthesizes data into structured JSON with reasoning_trace
    v
 Check Hallucination ---> Verify GO terms and PubMed IDs exist in retrieved data
    |
    v
 Clean Summary ---------> Auto-remove any hallucinated IDs from the final text
    |
    v
 Structured JSON output with reasoning_trace, summary, Cypher queries, and hallucination report
```

## Output Format

The ReasoningAgent produces JSON with an additional `reasoning_trace` field:

```json
{
  "to": "user",
  "text": {
    "template_matching": "agent_answer",
    "cypher": ["MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' RETURN g, r, d;"],
    "reasoning_trace": "## Question Decomposition\n- Sub-Q1: Which genes colocalize with T1D?\n- Sub-Q2: Which of those are also QTL targets?\n\n## Data Mapping\n- Sub-Q1 → Coloc edges in Neo4j results\n- Sub-Q2 → QTL edges in Neo4j results\n\n## Reasoning Chain\nStep 1: From coloc data, genes X, Y, Z colocalize with T1D\nStep 2: From QTL data, genes Y, Z, W are QTL targets\nStep 3: Intersection: Y, Z are both colocalized AND QTL targets\n\n## Conclusion\nGenes Y and Z have convergent evidence.",
    "summary": "Answer\nGenes Y (ENSG...) and Z (ENSG...) both colocalize with T1D GWAS signals and are QTL targets of fine-mapped variants.\n\nEvidence and Data\n..."
  }
}
```

## Prompt Modes

### WITH-LITERATURE mode (`use_literature=True`)
- Allows PubMed ID citations from HIRN data
- Citations must appear inline as `[PubMed ID: <id>]`

### NO-LITERATURE mode (`use_literature=False`)
- **ZERO PubMed IDs allowed**
- Only Neo4j database results are used

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | None | Anthropic API key for Claude Opus 4.6 |

## Shared Components

The ReasoningAgent reuses these modules from the FormatAgent skill:
- `compress_neo4j.py` — Neo4j result compression
- `hallucination_checker.py` — Hallucination detection and cleanup

