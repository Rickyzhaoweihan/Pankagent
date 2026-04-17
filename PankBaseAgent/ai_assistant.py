"""PankBaseAgent — thin wrapper around the query-planner skill.

The query-planner skill handles:
  1. Claude-based planning (chain vs parallel)
  2. text2cypher translation (vLLM, all steps in parallel)
  3. Cypher combination (WITH carry-forward for chains)
  4. Neo4j API execution

This module exposes the same ``chat_one_round_pankbase`` interface that
``utils.py`` and ``main.py`` expect.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Tuple

# Structured streaming events
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from stream_events import emit

# ---------------------------------------------------------------------------
# Import the query-planner skill
# ---------------------------------------------------------------------------

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_qp_scripts_dir = os.path.join(_repo_root, "skills", "query-planner", "scripts")
if _qp_scripts_dir not in sys.path:
    sys.path.insert(0, _qp_scripts_dir)

from qp_query_planner import run_query_planner_pipeline  # noqa: E402


def chat_one_round_pankbase(
    messages_history: list[dict],
    question: str,
) -> Tuple[list[dict], str, Dict]:
    """Run the query-planner pipeline and return results in the legacy format.

    Returns:
        (messages_history, response_json_str, planning_data)

    ``response_json_str`` is a JSON string with keys:
        - source: "pankbase"
        - planning: reasoning from the plan
        - plan_type: "chain" | "parallel"
        - queries_executed: list of Cypher queries that were sent to Neo4j
        - raw_results: list of {query, result} dicts
    """
    question = question.strip() or "<empty>"

    # Run the full pipeline
    results, plan = run_query_planner_pipeline(question)

    # Build planning_data for the experience buffer / logging
    planning_data: Dict = {
        "draft": plan.get("reasoning", ""),
        "plan_type": plan.get("plan_type", "unknown"),
        "num_queries": len(results),
        "queries": [
            {"name": "neo4j_compound_cypher", "input": r.get("query", "")}
            for r in results
        ],
        "steps": plan.get("steps", []),
    }

    # Emit structured summary event
    emit("pankbase_summary", {
        "plan_type": plan.get("plan_type", "unknown"),
        "reasoning": plan.get("reasoning", ""),
        "num_steps": len(plan.get("steps", [])),
        "num_queries": len(results),
        "queries": [r.get("query", "")[:200] for r in results],
        "test_time_scaling": True,
    })

    # Build response in the same shape the downstream FormatAgent / main.py expects
    raw_response = {
        "source": "pankbase",
        "planning": plan.get("reasoning", ""),
        "plan_type": plan.get("plan_type", "unknown"),
        "queries_executed": [r.get("query", "") for r in results],
        "raw_results": results,
    }

    # Build a minimal messages list so callers that inspect messages still work
    messages = list(messages_history)
    messages.append({"role": "user", "content": f"====== From User ======\n{question}"})
    messages.append({"role": "assistant", "content": json.dumps(raw_response, ensure_ascii=False)})

    return messages, json.dumps(raw_response, ensure_ascii=False), planning_data


def chat_forever():
    messages: list[dict] = []
    while True:
        question = input("Your question: ")
        messages, response, _ = chat_one_round_pankbase(messages, question)
        emit("pankbase_interactive_response", {"response": response[:2000]})


if __name__ == "__main__":
    chat_forever()
