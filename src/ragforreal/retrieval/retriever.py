"""Retrieval entry point. Phase 1 is dense-only; hybrid search and reranking slot in here."""

from __future__ import annotations

from ragforreal.config import ExperimentConfig, Settings
from ragforreal.ingest.embed import Embedder
from ragforreal.store.pgvector import PgVectorStore, RetrievedChunk


class Retriever:
    def __init__(self, cfg: ExperimentConfig, settings: Settings,
                 store: PgVectorStore | None = None):
        self.cfg = cfg
        self.index = cfg.index_name(settings.corpus)
        self.store = store or PgVectorStore(settings.database_url)
        if self.store.index_status(self.index) != "ready":
            raise RuntimeError(f"index {self.index} is not built - run `rag ingest` first")
        self.embedder = Embedder(cfg.ingest.embedding, settings.cache_dir,
                                 settings.embed_rpm, settings.embed_tpm)

    def retrieve_many(self, queries: list[str], k: int | None = None) -> list[list[RetrievedChunk]]:
        k = k or self.cfg.retrieval.top_k
        if self.cfg.retrieval.strategy != "dense":
            raise ValueError(f"unknown retrieval strategy: {self.cfg.retrieval.strategy}")
        # One batched embedding call for all queries: matters a lot under tight rate limits.
        vecs = self.embedder.embed_queries(queries)
        return [self.store.dense_search(self.index, v, k) for v in vecs]

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_many([query], k)[0]
