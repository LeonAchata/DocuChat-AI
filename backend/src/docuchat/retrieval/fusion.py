"""Rank fusion and diversification."""

from collections.abc import Mapping

import numpy as np

from docuchat.index.store import Scored
from docuchat.index.text import terms
from docuchat.models import Hit


def reciprocal_rank_fusion(
    ranked: Mapping[str, Scored], weights: Mapping[str, float], k: int = 60
) -> list[Hit]:
    """Weighted RRF: score(d) = sum_i w_i / (k + rank_i(d)).

    ``ranked`` keys look like ``"dense:0"`` (retriever:query-variant); the weight is looked up by
    the retriever prefix. RRF only uses ranks, so incomparable score scales (cosine, BM25,
    SPLADE dot products) never need calibrating.
    """
    hits: dict[str, Hit] = {}
    for name, results in ranked.items():
        weight = weights.get(name.split(":", 1)[0], 1.0)
        for rank, (chunk, _score) in enumerate(results, start=1):
            hit = hits.setdefault(chunk.id, Hit(chunk=chunk))
            if hit.chunk.dense is None and chunk.dense is not None:
                hit.chunk = chunk
            hit.score += weight / (k + rank)
            hit.ranks[name] = rank
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)


def maximal_marginal_relevance(hits: list[Hit], *, top_k: int, lambda_: float) -> list[Hit]:
    """Greedy MMR: trade relevance against similarity to what is already selected, so the
    context window is not spent on near-duplicate chunks."""
    if len(hits) <= 1 or lambda_ >= 1:
        return hits[:top_k]
    relevance = np.array(
        [h.rerank_score if h.rerank_score is not None else h.score for h in hits], dtype=float
    )
    span = relevance.max() - relevance.min()
    relevance = (relevance - relevance.min()) / span if span > 0 else np.ones_like(relevance)
    sim = _similarity(hits)

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(hits)) if i != selected[0]]
    while remaining and len(selected) < top_k:
        scores = [
            lambda_ * relevance[i] - (1 - lambda_) * max(sim[i, j] for j in selected)
            for i in remaining
        ]
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [hits[i] for i in selected]


def _similarity(hits: list[Hit]) -> np.ndarray:
    if all(h.chunk.dense is not None for h in hits):
        matrix = np.stack([h.chunk.dense for h in hits])  # type: ignore[misc]
        return np.asarray(matrix @ matrix.T, dtype=float)
    sets = [set(terms(h.chunk.text)) for h in hits]
    n = len(sets)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            union = sets[i] | sets[j]
            sim[i, j] = len(sets[i] & sets[j]) / len(union) if union else 0.0
    return sim
