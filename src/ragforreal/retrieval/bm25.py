"""In-process BM25 (``bm25s``) over every chunk of an index.

Postgres ``ts_rank_cd`` has no IDF: with an OR-ed natural-language question, a page that
repeats "United States" fifty times outranks the one table that matches the rare terms. BM25
weights each term by rarity and saturates repeated terms. The whole dev index (~8K chunks)
fits in memory and indexes in about a second, so no extra infrastructure is needed - a
production deployment would use a Postgres BM25 extension (ParadeDB, VectorChord-BM25).
"""

from __future__ import annotations

from dataclasses import replace

import bm25s
import numpy as np
import Stemmer

from ragforreal.store.pgvector import RetrievedChunk

_STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: list[str], return_ids: bool = True):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, return_ids=return_ids,
                          show_progress=False)


class BM25Search:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self.model = bm25s.BM25()
        self.model.index(_tokenize([c.text for c in chunks]), show_progress=False)
        self.doc_names = np.array([c.doc_name for c in chunks])

    def search(self, query: str, k: int, doc_names: list[str] | None = None
               ) -> list[RetrievedChunk]:
        scores = self.model.get_scores(_tokenize([query], return_ids=False)[0])
        if doc_names:
            scores = np.where(np.isin(self.doc_names, doc_names), scores, 0.0)
        top = np.argsort(-scores, kind="stable")[:k]
        return [replace(self.chunks[i], score=float(scores[i])) for i in top if scores[i] > 0]
