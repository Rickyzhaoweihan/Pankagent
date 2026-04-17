"""BM25 search and ranking for text chunks."""
from __future__ import annotations

import math
import re


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return re.findall(r"\w+", text.lower())


def _bm25_score_single(
    chunks: list[dict],
    query_terms: list[str],
    doc_tokens: list[list[str]],
    avgdl: float,
    k1: float,
    b: float,
) -> dict[int, float]:
    """Return {chunk_index: bm25_score} for a single set of query terms."""
    n = len(doc_tokens)
    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(1 for d in doc_tokens if term in d)

    scores: dict[int, float] = {}
    for i, tokens in enumerate(doc_tokens):
        score = 0.0
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term in query_terms:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_norm

        if score > 0:
            scores[i] = score
    return scores


def search_chunks(
    chunks: list[dict],
    query: str | list[str],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """Rank chunks by BM25 relevance to one or more queries.

    Args:
        chunks: List of chunk dicts (must have 'text' key).
        query: A single search string **or** a list of keyword queries.
            When a list is given each query is scored independently and
            the best score per chunk is kept before ranking.
        top_k: Number of top results to return.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        Top-k chunks sorted by score (descending), each with added 'score' key.
    """
    queries: list[str] = query if isinstance(query, list) else [query]
    queries = [q for q in queries if q and q.strip()]
    if not chunks or not queries:
        return []

    doc_tokens = [_tokenize(c.get("text", "")) for c in chunks]
    n = len(doc_tokens)
    avgdl = sum(len(d) for d in doc_tokens) / n if n else 1

    best: dict[int, float] = {}
    for q in queries:
        terms = _tokenize(q)
        if not terms:
            continue
        for idx, sc in _bm25_score_single(chunks, terms, doc_tokens, avgdl, k1, b).items():
            if sc > best.get(idx, 0):
                best[idx] = sc

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        chunk = dict(chunks[idx])
        chunk["score"] = round(score, 4)
        results.append(chunk)

    return results
