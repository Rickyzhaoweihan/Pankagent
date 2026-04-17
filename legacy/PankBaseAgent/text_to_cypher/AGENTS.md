# Repository Guidelines

## Project Structure & Module Organization
text2cypher is a lightweight LangChain agent packaged under `src/`. `text2cypher_agent.py` exposes the CLI REPL and LLM pipeline, `schema_loader.py` manages cached access to `data/input/neo4j_schema.json` and optional `schema_hints.json`, and `text2cypher_utils.py` centralizes environment lookups. Treat `src/` as the only importable module; keep notebooks or exploratory scripts under `experiments/` (gitignored) if needed. Refresh schemas atomically and commit both JSON files together.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` sets up an isolated interpreter.
- `pip install -r ../requirements.txt` installs LangChain, OpenAI, Google, and dotenv dependencies required by the agent.
- `python3 src/text2cypher_agent.py` opens the interactive REPL; set the provider via environment variables before launching.
- `python3 -m pytest` runs the test suite once `tests/` cases exist; add `-k text2cypher` to target agent-only coverage.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and snake_case helpers. Keep agent classes in PascalCase (`Text2CypherAgent`) and constants like `SYSTEM_RULES` uppercase. Include explicit type hints (`dict[str, str]`, `list[dict]`) to mirror the current modules, and prefer `pathlib.Path` for filesystem work. Log meaningful events through the module-level logger instead of prints.

## Testing Guidelines
Favor `pytest` with fixtures that load `neo4j_schema.json` from a temporary directory. Mock `ChatOpenAI`/`ChatGoogleGenerativeAI` so tests never call external APIs. Cover schema parsing edge cases (missing hints, malformed properties) and prompt assembly, and document any manual REPL smoke tests in `logs/README.md`.

## Commit & Pull Request Guidelines
Use short, present-tense subjects (`refine schema loader`). PRs should summarize changes, explain why the agent behavior improves, list exact setup or pytest commands run, and attach trimmed transcripts when sharing REPL sessions or Cypher outputs.

## Security & Configuration
Place secrets in `.env`; at minimum supply `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, and `OPENAI_API_MODEL`. Optional providers need `GOOGLE_API_KEY` and `GOOGLE_MODEL`. The agent resolves data paths relative to the nearest `.env`, so keep that file at the repository root. Never commit schema files containing proprietary data.
