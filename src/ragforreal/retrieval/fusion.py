"""Reciprocal Rank Fusion: merge ranked lists using ranks only, never raw scores.

Cosine similarity and ``ts_rank_cd`` live on unrelated scales, so adding them is meaningless;
ranks are comparable. Each list contributes ``1 / (k + rank)`` per item (Cormack et al. 2009).
k=60 damps the head: rank 1 vs rank 2 matters little, appearing in both lists matters a lot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ragforreal.store.pgvector import RetrievedChunk


def rrf(lists: Sequence[Sequence[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.id, chunk)
    order = sorted(scores, key=lambda cid: -scores[cid])  # stable: ties keep first-seen order
    return [replace(first_seen[cid], score=scores[cid]) for cid in order]
