# Rigor Format Agent

Strict evidence-only variant of the FormatAgent for simple questions.

## Key Differences from FormatAgent

- **Zero tolerance for unsupported claims** — every sentence must trace to input data
- **Short, direct answers** — no mandatory "Gene overview / QTL overview / T1D" sections
- **Present data, don't interpret** — tables and raw values over narrative prose
- **No background knowledge** — if it's not in the input, it's not in the answer

## When to Use

Activated via `--rigor` flag in `main.py`. Routes simple questions through this agent
instead of the standard FormatAgent.

## Pipeline

1. Receive raw Neo4j results (no compression)
2. Call Claude with strict evidence-only system prompt
3. Hallucination check + auto-cleanup
4. Return JSON
