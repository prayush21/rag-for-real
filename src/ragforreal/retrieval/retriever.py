"""Retrieval as composable stages, each configured in YAML (``RetrievalConfig``):

    filter (company named in the question)
    -> candidates (dense | sparse: Postgres FTS or BM25 | hybrid via RRF)
    -> rerank (cross-encoder) -> top k

``retrieve_traced`` also returns the pre-rerank candidate list and per-stage timings, so the
eval can score every stage and a regression is attributable to the stage that caused it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ragforreal.config import ExperimentConfig, Settings
from ragforreal.ingest.embed import Embedder
from ragforreal.retrieval.bm25 import BM25Search
from ragforreal.retrieval.filters import filter_docs
from ragforreal.retrieval.fusion import rrf
from ragforreal.retrieval.rerank import Reranker
from ragforreal.store.pgvector import PgVectorStore, RetrievedChunk


@dataclass
class RetrievalTrace:
    final: list[RetrievedChunk]
    candidates: list[RetrievedChunk]  # after fusion, before reranking
    doc_filter: list[str] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg: ExperimentConfig, settings: Settings,
                 store: PgVectorStore | None = None):
        self.cfg = cfg
        self.rcfg = cfg.retrieval
        self.index = cfg.index_name(settings.corpus)
        self.store = store or PgVectorStore(settings.database_url)
        if self.store.index_status(self.index) != "ready":
            raise RuntimeError(f"index {self.index} is not built - run `rag ingest` first")
        self.embedder = Embedder(cfg.ingest.embedding, settings.cache_dir,
                                 settings.embed_rpm, settings.embed_tpm)
        self.bm25: BM25Search | None = None
        if self.rcfg.strategy in ("sparse", "hybrid"):
            if self.rcfg.sparse == "bm25":
                self.bm25 = BM25Search(self.store.all_chunks(self.index))
            else:
                self.store.ensure_fulltext(self.index)  # upgrades pre-Phase-3 indexes in place
        self.reranker = (Reranker(self.rcfg.reranker, settings.cache_dir)
                         if self.rcfg.reranker else None)
        self.doc_companies = (self.store.doc_companies(self.index)
                              if self.rcfg.company_filter else {})

    def retrieve_traced(self, query: str, query_vec: np.ndarray | None = None,
                        k: int | None = None) -> RetrievalTrace:
        k = k or self.rcfg.top_k
        n = max(self.rcfg.candidate_k, k)
        timings: dict[str, float] = {}
        docs = filter_docs(query, self.doc_companies) if self.rcfg.company_filter else None

        lists = []
        if self.rcfg.strategy in ("dense", "hybrid"):
            if query_vec is None:
                t = time.perf_counter()
                query_vec = self.embedder.embed_queries([query])[0]
                timings["embed"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            lists.append(self.store.dense_search(self.index, query_vec, n, doc_names=docs))
            timings["dense"] = (time.perf_counter() - t) * 1000
        if self.rcfg.strategy in ("sparse", "hybrid"):
            t = time.perf_counter()
            lists.append(self.bm25.search(query, n, doc_names=docs) if self.bm25 else
                         self.store.sparse_search(self.index, query, n, doc_names=docs))
            timings["sparse"] = (time.perf_counter() - t) * 1000
        # Fused list is cut to candidate_k so reranking costs the same for every strategy.
        candidates = lists[0] if len(lists) == 1 else rrf(lists, self.rcfg.rrf_k)[:n]

        final = candidates
        if self.reranker:
            t = time.perf_counter()
            final = self.reranker.rerank(query, candidates)
            timings["rerank"] = (time.perf_counter() - t) * 1000
        return RetrievalTrace(final=final[:k], candidates=candidates, doc_filter=docs,
                              timings_ms=timings)

    def retrieve_many(self, queries: list[str], k: int | None = None) -> list[list[RetrievedChunk]]:
        # One batched embedding call for all queries: matters a lot under tight rate limits.
        vecs = (self.embedder.embed_queries(queries) if self.rcfg.strategy != "sparse"
                else [None] * len(queries))
        return [self.retrieve_traced(q, v, k).final
                for q, v in zip(queries, vecs, strict=True)]

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_many([query], k)[0]
