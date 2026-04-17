---
name: format-agent
description: >-
  This skill should be used when the user's biomedical query has been answered
  by sub-agents (PankBase, GLKB, TemplateToolAgent) and the raw results need
  to be formatted into a final structured JSON response. Triggers after all
  data retrieval is complete and the system needs to produce the final user-facing
  summary with Cypher queries, GO terms, SNP data, expression values, and
  optional PubMed citations.
---

# FormatAgent — Biomedical Response Formatter

Format raw sub-agent data (Neo4j results, GLKB abstracts, Cypher queries) into a structured JSON response with hallucination checking.

## Pipeline Overview

```
Raw Sub-Agent Data
    |
    v
 Compress Neo4j -------> Parse raw Neo4j text into compact JSON (deduplicated nodes/edges)
    |
    v
 Format Response -------> Claude Opus 4.6 synthesizes data into structured JSON summary
    |
    v
 Check Hallucination ---> Verify GO terms and PubMed IDs exist in retrieved data
    |
    v
 Clean Summary ---------> Auto-remove any hallucinated IDs from the final text
    |
    v
 Structured JSON output with summary, Cypher queries, and hallucination report
```

## Workflow

### Step 1: Compress Neo4j results

```bash
cd {SKILL_DIR}
python -c "
import json, sys
sys.path.insert(0, 'scripts')
from compress_neo4j import compress_neo4j_results

# neo4j_results is a list of dicts with 'query' and 'result' keys
raw_results = json.loads(open('/dev/stdin').read())
compressed = compress_neo4j_results(raw_results)
print(json.dumps(compressed, indent=2))
" <<< 'NEO4J_RESULTS_JSON_HERE'
```

Parses raw Neo4j text blobs into compact JSON with deduplicated nodes and edges. Extracts gene properties, GO terms, SNP/QTL data, expression values, and disease relationships.

### Step 2: Format the response using Claude

```bash
cd {SKILL_DIR}
python -c "
import json, sys
sys.path.insert(0, 'scripts')
from format_response import format_response

result = format_response(
    human_query='USER_QUERY_HERE',
    compressed_neo4j=json.loads('COMPRESSED_JSON'),
    cypher_queries=['CYPHER1', 'CYPHER2'],
    glkb_text='OPTIONAL_GLKB_OUTPUT',
    use_glkb=False
)
print(json.dumps(result, indent=2))
"
```

Calls Claude Opus 4.6 with the appropriate system prompt (WITH-LITERATURE or NO-LITERATURE mode) to synthesize all data into a structured JSON response.

### Step 3: Check for hallucinations

```bash
cd {SKILL_DIR}
python -c "
import json, sys
sys.path.insert(0, 'scripts')
from hallucination_checker import check_hallucination, remove_hallucinated_ids

summary = 'THE_SUMMARY_TEXT'
neo4j_results = json.loads('NEO4J_RESULTS_JSON')
raw_agent_output = 'RAW_AGENT_OUTPUT'

report = check_hallucination(summary, neo4j_results, raw_agent_output)
print(json.dumps(report, indent=2))

if not report['is_clean']:
    cleaned = remove_hallucinated_ids(summary, report['hallucinated_go_terms'], report['hallucinated_pubmed_ids'])
    print('CLEANED:', cleaned)
"
```

### Complete End-to-End Workflow

For the full format pipeline, run all steps together:

```bash
cd {SKILL_DIR}
python -c "
import json, sys
sys.path.insert(0, 'scripts')
from format_response import run_format_pipeline

result = run_format_pipeline(
    human_query='USER_QUERY_HERE',
    neo4j_results=json.loads('RAW_NEO4J_JSON'),
    cypher_queries=['CYPHER1'],
    functions_result='RAW_AGENT_OUTPUT',
    use_glkb=False
)
print(json.dumps(result, indent=2))
"
```

## Output Format

The skill produces a JSON object:

```json
{
  "to": "user",
  "text": {
    "template_matching": "agent_answer",
    "cypher": ["MATCH (g:gene {name:'CFTR'}) RETURN g;"],
    "summary": "Answer\nCFTR (ENSG00000001626) is a protein-coding gene...\n\nGene overview\n...\n\nQTL overview\n...\n\nSpecific relation to Type 1 Diabetes\n..."
  }
}
```

If hallucinations were detected and cleaned:

```json
{
  "to": "user",
  "text": {
    "template_matching": "agent_answer",
    "cypher": [...],
    "summary": "...(cleaned)...",
    "hallucination_check": {
      "is_clean": false,
      "removed_go_terms": ["GO_9999999"],
      "removed_pubmed_ids": ["99999999"],
      "note": "Fake IDs were automatically removed from the summary"
    }
  }
}
```

## Prompt Modes

The skill has two prompt modes selected by the `use_glkb` flag:

### WITH-LITERATURE mode (`use_glkb=True`)
- Allows PubMed ID citations from GLKB/HIRN data
- Citations must appear inline as `[PubMed ID: <id>]`
- Only IDs present in the input data are allowed

### NO-LITERATURE mode (`use_glkb=False`)
- **ZERO PubMed IDs allowed**
- No literature references of any kind
- Only Neo4j database results are used

## Data Utilization Requirements

The FormatAgent is evaluated on **data utilization rate**:

1. **Gene Information**: Ensembl ID, chromosomal location, strand, description, GC%
2. **GO Terms**: List ALL individually with IDs (at least 10-15 if available)
3. **QTL/SNP Data**: List EVERY SNP with rsID, position, PIP score, tissue, slope
4. **Expression Data**: EXACT values for NonDiabetic, T1D, Log2FoldChange, p-values
5. **Disease Relationships**: All associations with relationship types

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | None | Anthropic API key for Claude Opus 4.6 |

## Limitations

- **Token limits**: Claude Opus 4.6 has a 200K context window but output is capped at 4096 tokens. Very large datasets may be truncated in the summary.
- **JSON extraction**: Claude may wrap JSON in markdown code blocks; the skill auto-extracts it.
- **Hallucination checking**: Only catches GO terms and PubMed IDs; other fabricated data (e.g., expression values) requires manual review.

