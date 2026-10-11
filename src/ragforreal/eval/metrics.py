"""Retrieval metrics against FinanceBench gold evidence pages. Pure functions, no LLM needed.

A retrieved chunk "covers" a gold page when it comes from the same document and its page span
includes that page. Caveat: a bigger chunk spans more pages and so covers more gold pages "for
free" - always read hit@k alongside ``context_tokens@k`` when comparing chunk sizes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class _Span(Protocol):
    doc_name: str
    page_start: int
    page_end: int


GoldPage = tuple[str, int]


def covers(chunk: _Span, gold: GoldPage) -> bool:
    doc, page = gold
    return chunk.doc_name == doc and chunk.page_start <= page <= chunk.page_end


def hit_at_k(retrieved: Sequence[_Span], gold: set[GoldPage], k: int) -> float:
    return float(any(covers(c, g) for c in retrieved[:k] for g in gold))


def page_recall_at_k(retrieved: Sequence[_Span], gold: set[GoldPage], k: int) -> float:
    if not gold:
        return 0.0
    found = {g for g in gold if any(covers(c, g) for c in retrieved[:k])}
    return len(found) / len(gold)


def doc_hit_at_k(retrieved: Sequence[_Span], gold: set[GoldPage], k: int) -> float:
    gold_docs = {d for d, _ in gold}
    return float(any(c.doc_name in gold_docs for c in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[_Span], gold: set[GoldPage]) -> float:
    for rank, c in enumerate(retrieved, 1):
        if any(covers(c, g) for g in gold):
            return 1.0 / rank
    return 0.0


def retrieval_metrics(retrieved: Sequence[_Span], gold: set[GoldPage], ks: Sequence[int]
                      ) -> dict[str, float]:
    out: dict[str, float] = {"mrr": reciprocal_rank(retrieved, gold)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        out[f"page_recall@{k}"] = page_recall_at_k(retrieved, gold, k)
        out[f"doc_hit@{k}"] = doc_hit_at_k(retrieved, gold, k)
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]
