"""Expand a natural-language question into focused keyword queries for HIRN search.

Uses a lightweight Claude call to extract biomedical entities and generate
multiple short keyword queries that maximize recall when matched against
publication titles and full-text passages.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a biomedical search-query generator for the HIRN (Human Islet Research \
Network) publication database. The database contains papers about Type 1 \
Diabetes, pancreatic islets, beta cells, immune mechanisms, and related topics.

Given a user question, produce exactly 3 short keyword search queries that \
together maximise the chance of finding relevant HIRN publications.

Rules:
- Each query should be 2-5 words of precise biomedical keywords (gene names, \
  cell types, pathways, diseases, techniques).
- Strip filler words (give, me, some, information, about, related, to, etc.).
- Query 1: the most specific/literal extraction from the question.
- Query 2: a synonym or closely related concept that might appear in titles.
- Query 3: a broader contextual angle (e.g. pathway, disease, tissue).
- Keep gene/protein names EXACTLY as given (do NOT expand abbreviations).

Return ONLY a JSON array of 3 strings. No explanation, no markdown fences.

Examples:
User: "Can you give me some literature related to gene CFTR?"
["CFTR gene", "CFTR cystic fibrosis transmembrane", "CFTR pancreas diabetes"]

User: "What papers discuss the role of IL-2 in T1D?"
["IL-2 T1D", "interleukin-2 type 1 diabetes", "IL-2 immune regulation islet"]

User: "Tell me about beta cell dedifferentiation"
["beta cell dedifferentiation", "islet cell identity loss", "beta cell transcription factor"]
"""


def expand_query(question: str) -> list[str]:
    """Return a list of ~3 keyword queries derived from *question*.

    Falls back to a simple keyword extraction if Claude is unavailable.
    """
    try:
        return _expand_with_claude(question)
    except Exception as exc:
        logger.warning("Claude query expansion failed (%s), using fallback", exc)
        return _expand_fallback(question)


def _expand_with_claude(question: str) -> list[str]:
    """Call Claude to generate 3 search queries."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import importlib
            import sys
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            config = importlib.import_module("config")
            api_key = getattr(config, "ANTHROPIC_API_KEY", None) or getattr(config, "API_KEY", None)
        except (ImportError, AttributeError):
            pass

    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not available for query expansion")

    client = anthropic.Anthropic(api_key=api_key.strip())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0.0,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    text = response.content[0].text.strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    queries = json.loads(text)
    if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
        queries = [q.strip() for q in queries if q.strip()]
        if queries:
            logger.info("Expanded query into %d sub-queries: %s", len(queries), queries)
            return queries

    raise ValueError(f"Unexpected Claude response format: {text!r}")


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in on at to for with by from about into "
    "through during before after above below between out off over under again "
    "further then once and but or nor not so yet both either neither each every "
    "all any few more most other some such no only own same than too very just "
    "also how what which who whom when where why how give me you your tell "
    "please show find get some information related literature papers discuss "
    "role describe explain".split()
)


def _expand_fallback(question: str) -> list[str]:
    """Simple keyword extraction when Claude is unavailable."""
    tokens = re.findall(r"\w+", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    if not keywords:
        keywords = re.findall(r"\w+", question)

    if len(keywords) <= 3:
        return [" ".join(keywords)]

    return [
        " ".join(keywords),
        " ".join(keywords[:3]),
        " ".join(keywords[-3:]),
    ]
