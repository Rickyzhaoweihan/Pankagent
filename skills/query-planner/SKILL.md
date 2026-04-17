# Query Planner Skill

## Overview

A Claude skill that generates **parallel or chained** natural-language query plans
for the PanKgraph knowledge graph. Each plan step is a simple one-hop NL query
that the existing **text2cypher** vLLM model translates into Cypher.

For **chain** plans the individual Cypher outputs are merged into a single
compound Cypher query (using `WITH` carry-forward) by a hardcoded Python
combiner before execution, so Neo4j performs the joins natively.

## Directory Layout

```
skills/query-planner/
  SKILL.md              # this file
  data/                 # (reserved for caches / artefacts)
  scripts/
    __init__.py
    prompts.py          # Claude system prompt (schema + few-shot examples)
    cypher_combiner.py  # pure-Python combiner: individual Cyphers -> compound
    query_planner.py    # main entry: plan_query, translate_plan, execute_plan, pipeline
```

## Data Flow

```
User question
  -> plan_query()        [Claude]  produces plan JSON (parallel | chain)
  -> translate_plan()    [vLLM text2cypher, ALL steps in parallel]
  -> if chain: combine_chain()  [Python combiner -> compound Cypher]
     else:     keep individual Cyphers
  -> execute each Cypher against Neo4j API
  -> return list[dict] of results
```

## Plan JSON Schema

```json
{
  "plan_type": "chain" | "parallel",
  "reasoning": "...",
  "steps": [
    {
      "id": 1,
      "natural_language": "Get OCRs that have OCR_activity in Beta Cell",
      "join_var": "o",
      "depends_on": null
    },
    ...
  ]
}
```

## Key Insight: Two-Phase Compound Cypher

Instead of 3 separate queries, the combiner produces ONE query in two phases:

**Phase 1 — Filter chain:** Clean MATCH chain with only join variables in WITH.
Filters down to the matching rows cheaply.

**Phase 2 — Return block:** Re-MATCHes each hop using the now-constrained join
variables, then collects all nodes and edges. The re-MATCHes are cheap because
the join variables are already filtered to ≤ LIMIT rows.

```cypher
-- Phase 1: filter chain
MATCH (o:OCR)-[r1:OCR_activity]->(ct:cell_type) WHERE ct.name = "Beta Cell"
WITH o LIMIT 200
MATCH (o)-[r2:OCR_locate_in]->(g:gene)
WITH o, g LIMIT 200
-- Phase 2: re-expand and collect all nodes/edges
MATCH (o)-[r1:OCR_activity]->(ct:cell_type) WHERE ct.name = "Beta Cell"
MATCH (o)-[r2:OCR_locate_in]->(g)
MATCH (g)-[r3:function_annotation]->(fo:gene_ontology)
WITH collect(DISTINCT o) + collect(DISTINCT ct) + collect(DISTINCT g) + collect(DISTINCT fo) AS nodes,
     collect(DISTINCT r1) + collect(DISTINCT r2) + collect(DISTINCT r3) AS edges
RETURN nodes, edges
```

Phase 1 avoids cartesian explosion (only join vars carried). Phase 2 brings
all variables back into scope for the final aggregation.

## Usage

```python
from query_planner import run_query_planner_pipeline

results, plan = run_query_planner_pipeline(
    "Which GO terms are associated with genes that have OCR active in Beta cells?"
)
# results: list[dict] with 'query' and 'result' keys
# plan: dict with 'plan_type', 'reasoning', 'steps' (each step has 'cypher')
```

## Integration

The skill is imported by:
- `PankBaseAgent/ai_assistant.py` — calls `run_query_planner_pipeline()` directly
- `claude.py` (root) — exposes `run_query_planner_pipeline` in `__all__`

The PlannerAgent in `main.py` dispatches to `pankbase_chat_one_round` (in `utils.py`),
which calls the updated `PankBaseAgent/ai_assistant.py`, which uses this skill.
