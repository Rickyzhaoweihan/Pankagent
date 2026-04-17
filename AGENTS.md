# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the PlannerAgent entry point; it brokers calls to helper agents and writes conversational traces to `log.txt`.
- `FormatAgent/`, `PankBaseAgent/`, `TemplateToolAgent/`, and `GLKB_agent_ai_assistant/GLKB_agent_ai_assistant/` follow the same layout (`ai_assistant.py`, `claude.py`, `utils.py`, `prompts/`, `logs/`). Keep shared logic in `utils.py` or `multi_thread_workers.py` to avoid copy-paste fixes.
- Reusable prompt templates live in `prompts/`; update companion JSON schemas when you change prompt contracts.
- Use `logs/` only for temporary run artifacts. Trim or gitignore large outputs before opening a PR.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` creates an isolated dev environment.
- `pip install -r requirements.txt` installs Anthropic, LangChain, Neo4j, and OpenAI bindings referenced across agents.
- `python3 main.py` launches the PlannerAgent REPL from the repository root.
- `python3 GLKB_agent_ai_assistant/GLKB_agent_ai_assistant/test.py` drives the GLKB integration exercise; set dataset paths inside the script before invoking it.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case module-level functions. Keep agent classes or data holders in PascalCase.
- Prefer explicit type hints (`list[dict]`, `Tuple[...]`) as shown in `main.py`.
- Log external calls and function dispatch decisions; the existing `PRINT_FUNC_*` flags help keep traces reproducible.
- When extending prompts, mirror the naming pattern `<agent>/prompts/<purpose>_prompt.txt`.

## Testing Guidelines
- Scripts under `GLKB_agent_ai_assistant/GLKB_agent_ai_assistant/` and `PankBaseAgent/` act as integration tests. They assume local JSON datasets; update the file paths or expose them via env vars before sharing.
- Add lightweight unit tests for new helpers using `pytest` or `unittest` under a `tests/` package, and mock network calls to Anthropic/OpenAI/Neo4j.
- Document any long-running evaluations in `logs/README.md` or the PR description so reviewers can replay them.

## Commit & Pull Request Guidelines
- Match the existing history: short, present-tense subject lines such as “reduce amount of iterations” or “Add format agent”. Keep them under ~60 characters and avoid trailing punctuation.
- Every PR should include: what changed, why it matters, manual or automated test evidence (exact commands), and links to any tracking issues or datasets. Include redacted screenshots or transcripts if UI or dialog output changed.

## Configuration & Secrets
- Copy `.env.example` to `.env` and create `config.py` with the required API keys (`API_KEY`, `OPENAI_API_KEY`) before running any agent.
- Never hard-code secrets or absolute dataset paths; prefer environment variables and document expected locations in the README or PR notes.
