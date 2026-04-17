"""Structured streaming event emitter for the PanKLLM pipeline.

Every pipeline stage emits JSON events (one per line, NDJSON format) so a
frontend can parse and render progress in real time.

Usage
-----
    from stream_events import emit

    emit("plan_start", {"question": q})
    emit("cypher_executing", {"index": 1, "cypher": cypher})
    ...

Event Schema
------------
Each line printed to stdout is a JSON object with these fields:

    {
        "event":     str,       # event type (see EVENT_TYPES below)
        "ts":        float,     # Unix epoch timestamp
        "data":      dict,      # event-specific payload
    }

EVENT_TYPES (non-exhaustive — new events can be added freely):

  Stage: Planner
    planner_decision          – PlannerAgent decided which sub-agents to call
    complexity_classified     – question classified as simple/complex

  Stage: Query Planner
    pipeline_start            – query-planner pipeline begins
    plan_generated            – Claude produced the plan JSON
    text2cypher_start         – vLLM translation of one step started
    text2cypher_done          – one step translated (or failed)
    translate_done            – all steps translated
    cypher_executing          – a Cypher query is being sent to Neo4j
    cypher_result             – Neo4j returned a result for one query
    pipeline_done             – query-planner pipeline finished

  Stage: PankBase Agent
    pankbase_summary          – PankBaseAgent wrapper summary

  Stage: HIRN Literature
    hirn_search_start         – HIRN skill begins
    hirn_publications_loaded  – publication index loaded
    hirn_matches_found        – title search done
    hirn_pmcids_resolved      – PMCIDs resolved
    hirn_result               – final HIRN result ready

  Stage: Reasoning / Format Agent
    reasoning_start           – ReasoningAgent pipeline begins
    reasoning_compress        – compression step done (or skipped)
    reasoning_claude_start    – calling Claude for reasoning
    reasoning_claude_done     – Claude responded
    reasoning_raw_output      – raw Claude output (truncated)
    reasoning_halluc_check    – hallucination check result
    reasoning_done            – ReasoningAgent pipeline finished
    format_start / format_compress / format_claude_start / format_claude_done
    format_raw_output / format_halluc_check / format_done

  Stage: Main
    main_routing              – routing to reasoning or format agent
    final_response            – final answer ready
"""

from __future__ import annotations

import json
import sys
import time
import threading

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_enabled: bool = True          # flip to False to suppress all events
_pretty: bool = False          # True → indent JSON (for debugging)


def set_streaming_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def set_streaming_pretty(pretty: bool) -> None:
    global _pretty
    _pretty = pretty


# ---------------------------------------------------------------------------
# Core emitter
# ---------------------------------------------------------------------------

def emit(event: str, data: dict | None = None) -> None:
    """Print one JSON event line to stdout (thread-safe).

    Parameters
    ----------
    event : str
        The event type identifier.
    data : dict, optional
        Arbitrary payload.  ``None`` is normalised to ``{}``.
    """
    if not _enabled:
        return

    payload = {
        "event": event,
        "ts": time.time(),
        "data": data or {},
    }

    indent = 2 if _pretty else None
    line = json.dumps(payload, ensure_ascii=False, default=str, indent=indent)

    with _lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

