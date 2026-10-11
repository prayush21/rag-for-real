"""Retrieval metrics against FinanceBench gold evidence pages. Pure functions, no LLM needed.

A retrieved chunk "covers" a gold page when it comes from the same document and its page span
includes that page. Caveat: a bigger chunk spans more pages and so covers more gold pages "for
free" - always read hit@k alongside ``context_tokens@k`` when comparing chunk sizes.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------- evidence-text matching
#
# Page provenance can drift: a parser assigns a table to the next page, or the same table is
# printed twice in a filing. A chunk that contains most of the gold evidence text is relevant
# wherever it sits, so ``ev_*`` metrics count it as a hit too. Still deterministic. They do NOT
# credit a *different* passage that also answers the question (e.g. a tax note stating the
# effective rate when the gold evidence is the income statement) - nothing cheap does.

_TOKEN = re.compile(r"[a-z]+|\d[\d,]*(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    return {t.replace(",", "") for t in _TOKEN.findall(text.lower()) if len(t) > 1 or t.isdigit()}


def evidence_overlap(chunk_text: str, evidence_text: str) -> float:
    """Shared distinct tokens over the smaller side's count: 1.0 when one contains the other.
    Min-normalised because a gold page is often longer than a chunk (or vice versa)."""
    c, e = _tokens(chunk_text), _tokens(evidence_text)
    if not c or not e:
        return 0.0
    return len(c & e) / min(len(c), len(e))


class _TextSpan(_Span, Protocol):
    text: str


Evidence = tuple[str, str]  # (doc_name, evidence_text)


def relevance(retrieved: Sequence[_TextSpan], gold: set[GoldPage],
              evidence: Sequence[Evidence], threshold: float = 0.8, min_shared: int = 20
              ) -> list[bool]:
    """Per chunk: on a gold page, or (same document) containing most of a gold evidence text.
    ``min_shared`` stops a tiny chunk from "containing" a long evidence passage by accident."""
    ev_tokens = [(doc, _tokens(text)) for doc, text in evidence]
    out = []
    for c in retrieved:
        hit = any(covers(c, g) for g in gold)
        if not hit:
            ct = _tokens(c.text)
            for doc, et in ev_tokens:
                shared = len(ct & et)
                if (c.doc_name == doc and ct and et and shared >= min_shared
                        and shared / min(len(ct), len(et)) >= threshold):
                    hit = True
                    break
        out.append(hit)
    return out


def evidence_metrics(flags: Sequence[bool], ks: Sequence[int]) -> dict[str, float]:
    first = next((i for i, f in enumerate(flags, 1) if f), 0)
    out = {"ev_mrr": 1.0 / first if first else 0.0}
    for k in ks:
        out[f"ev_hit@{k}"] = float(any(flags[:k]))
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]
