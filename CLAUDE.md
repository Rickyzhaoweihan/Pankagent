# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PanKgraph AI Assistant — a multi-agent system for querying a Type 1 Diabetes (T1D) biomedical knowledge graph using natural language. A PlannerAgent orchestrates specialized sub-agents, uses a fine-tuned local LLM (via vLLM) for text-to-Cypher translation, and Claude Opus 4 for orchestration/formatting/reasoning.

## Commands

### Setup
```bash
pip install -r requirements.txt
pip install -r requirements-server.txt   # for API server mode
# Create config.py from config.py.example with API_KEY and OPENAI_API_KEY
# Create .env from .env.example
```

### Running
```bash
python3 main.py                          # interactive REPL
python3 main.py "your question here"     # single question
python3 server.py                        # FastAPI server on port 8080
python3 server.py 9000                   # custom port
PORT=5000 python3 server.py              # port via env var
```

### vLLM (text-to-Cypher model, requires GPU)
```bash
sbatch host_vllm.sbatch                  # HPC/Slurm
# or manually:
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/cypher_model --served-model-name text2cypher_merged \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 --max-model-len 8192 \
  --max-num-seqs 32                      # multi-user support
```

### Tests
```bash
# Cypher validator unit tests (pytest)
pytest PankBaseAgent/text_to_cypher/test_dynamic_enrichment.py
pytest PankBaseAgent/text_to_cypher/test_relationship_directions.py
pytest PankBaseAgent/text_to_cypher/test_where_constraints.py
pytest PankBaseAgent/text_to_cypher/test_with_clause_validation.py

# HIRN publication retrieval tests (43 tests, mocked HTTP)
pytest hirn_publication_retrieval/tests/

# Integration tests (require running server)
python3 test_server.py [port]
python3 test_server_stream.py            # streaming endpoint tests

# Batch evaluation against full pipeline
python3 batch_evaluator.py
```

## Architecture

### Agent Orchestration Flow

```
User Question → PlannerAgent (main.py)
  ├─ test-time scaling: N parallel planner candidates → select best by result quality
  ├─ classifies complexity: "simple" | "complex"
  ├─ dispatches in parallel (threads):
  │   ├─ PankBaseAgent → query-planner skill → vLLM text2cypher → Neo4j API
  │   ├─ GLKBAgent/HIRN Literature → semantic search + PubMed/PMC retrieval
  │   ├─ TemplateToolAgent → rule-based triple extraction → RDS Lambda
  │   └─ HPAPAgent → HPAP MySQL metadata (optional)
  └─ routes to final pipeline:
      ├─ FormatAgent (simple) → compress + format + hallucination check
      └─ ReasoningAgent (complex) → multi-hop reasoning + hallucination check
```

`PLANNER_CANDIDATES` (default 5) controls parallel planner instances. `RIGOR_MODE` (default `True` in server.py) switches to stricter `rigor-format-agent` / `rigor-reasoning-agent` variants.

### Agent Module Pattern

Every agent follows the same layout:
- `ai_assistant.py` — entry point function (`chat_one_round_*`)
- `claude.py` — LLM client wrapper
- `utils.py` — tool/API functions
- `prompts/` — system prompt templates (`general_prompt.txt`, `error_prompt.txt`)

### Skills (modular pipeline components, in `skills/`)
- `query-planner/` — QueryPlanner: plan → translate → combine → execute Cypher chains
- `format-agent/` — FormatAgent: compresses Neo4j results, formats final answer
- `reasoning-agent/` — ReasoningAgent: multi-hop reasoning for complex questions
- `rigor-format-agent/` — stricter FormatAgent (used in production when `RIGOR_MODE=True`)
- `rigor-reasoning-agent/` — stricter ReasoningAgent (used in production when `RIGOR_MODE=True`)
- `hpap-database-metadata/` — queries HPAP (Human Pancreas Atlas Project) MySQL metadata

### LLM Response Contract

All Claude LLM calls produce JSON with this structure:
```json
{"draft": "...", "to": "system"|"user", "complexity": "simple"|"complex", "functions": [...], "text": "..."}
```
- `to: "system"` → function dispatch (functions array has `{name, arguments}`)
- `to: "user"` → final response (text field has the answer)

### Function Dispatch

`utils.py:run_functions()` dispatches function calls in parallel threads. The three top-level callable functions are: `pankbase_chat_one_round`, `hirn_chat_one_round`, `template_chat_one_round`.

### Text-to-Cypher Pipeline (`PankBaseAgent/text_to_cypher/`)

1. **Schema loading** (`src/schema_loader.py`) — loads/caches Neo4j schema, produces minimal schema string for the LLM
2. **Text2CypherAgent** (`src/text2cypher_agent.py`) — LangChain agent backed by local vLLM (`localhost:8002`), lazy singleton (initialized once on first call)
3. **Cypher validator** (`src/cypher_validator.py`, ~2000 lines) — scores queries 0-100, auto-fixes: quote conversion, relationship variables, DISTINCT in collect(), direction validation, property/value normalization, LIMIT injection, etc.
4. **Refinement loop** — if score < 90, iterates up to 5 times with feedback

### Experience Buffer

`PankBaseAgent/experience_buffer.py` stores past successful planning examples for in-context learning. Raw logs go to `query_log.jsonl`; curated top patterns go to `experience_buffer.jsonl` (root level). The query-planner skill reads from this buffer to guide future plans.

### Server Endpoints (`server.py`)

All agents are pre-initialized at startup via FastAPI lifespan hook for fast response times.

- `POST /query` — synchronous query
- `POST /query/stream` — streaming NDJSON response
- `POST /plan/start`, `POST /plan/revise`, `POST /plan/confirm` — interactive user-guided planning with revision loop

### Streaming Protocol

`stream_events.py` emits NDJSON to stdout: `{"event": str, "ts": float, "data": dict}` per line, for real-time frontend rendering. Toggle with `set_streaming_enabled(bool)`. Event name prefixes: `planner_*`, `text2cypher_*`, `cypher_*`, `hirn_*`, `reasoning_*`, `format_*`, `hallucination_check_*`.

### FormatAgent Data Requirements

The format prompt (`prompts/format_prompt.txt`) enforces exhaustive data extraction — this is intentional and critical to output quality:
- List **ALL** GO terms individually (10-15+), grouped by category — never summarize
- List **ALL** SNPs with rsID, chromosome, PIP, tissue, effect size
- Use **exact** numeric values, not summaries
- Zero fabricated PubMed IDs (only cite IDs present in retrieved data)

### Hallucination Checker

`skills/format-agent/scripts/hallucination_checker.py` validates GO_XXXXXXX and PubMed IDs in the output against retrieved data via regex. `remove_hallucinated_ids()` strips fake IDs from the final text. Used in both FormatAgent and ReasoningAgent.

## External Services

- **PanKgraph Neo4j API**: `https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j` — all Cypher queries
- **RDS Lambda**: `https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/RDSLambda` — gene name → Ensembl ID
- **HIRN Abstracts API**: `https://glkb.dcmb.med.umich.edu/api/external/search_hirn_abstracts` — semantic search
- **NCBI E-utilities / PMC Open Access** — PubMed/PMC article retrieval
- **vLLM** (self-hosted) — fine-tuned cypher-writer model on port 8002

## Configuration

Two config sources (API key resolution: env var → config.py):
- `config.py` (from `config.py.example`): `API_KEY`, `OPENAI_API_KEY`
- `.env` (from `.env.example`): `NEO4J_SCHEMA_PATH`, `SCHEMA_HINTS_PATH`, `VLLM_PORT`, `OPENAI_API_*`, `CLAUDE_API_KEY`

Key env vars: `PORT` (server), `VLLM_PORT` (default 8002), `ANTHROPIC_API_KEY`, `HIRN_ABSTRACT_SEARCH_URL`, `NCBI_API_KEY`.

## Conventions

- PEP 8, 4-space indentation, snake_case functions, PascalCase classes
- Commit messages: short, present-tense subject lines ("Add format agent", "reduce amount of iterations")
- Prompt templates: `<agent>/prompts/<purpose>_prompt.txt`
- Thread-based parallelism with `_thread.start_new_thread` + `Queue` for sub-agent calls; `multi_thread_workers.py` provides `map_once()` and `map_infinite_retry()` helpers
- `performance_monitor.py` wraps functions with timing decorators, logs to `logs/performance.log`
- Hallucination checker validates GO terms and PubMed IDs in outputs against retrieved data

## RL Training (`rl_implementation/`)

Reinforcement learning pipeline for improving Cypher generation quality. `CypherGeneratorAgent` runs multi-turn episodes (max 5 steps) against `GraphReasoningEnvironment` (wraps Neo4j executor). `train_collaborative_system.py` orchestrates training via the `rllm` framework. `DualResourcePoolManager` manages separate GPU pools for the Cypher Generator and Orchestrator models.
