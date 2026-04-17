# Rigor Reasoning Agent

Strict evidence-only variant of the ReasoningAgent for complex questions.

## Key Differences from ReasoningAgent

- **Zero tolerance for unsupported claims** — every reasoning step must cite specific data
- **Tight reasoning traces** — no verbose decomposition, just data-backed steps
- **Short synthesis** — direct answer with supporting evidence, no essays
- **No speculative interpretation** — if the data doesn't show causality, don't claim it

## When to Use

Activated via `--rigor` flag in `main.py`. Routes complex questions through this agent
instead of the standard ReasoningAgent.

## Pipeline

1. Receive raw Neo4j results (no compression)
2. Call Claude with strict evidence-only reasoning prompt
3. Hallucination check + auto-cleanup
4. Return JSON with reasoning_trace + summary
