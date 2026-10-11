"""Cross-encoder reranking via LiteLLM, so the provider is a config string
(``voyage/rerank-2.5``, ``cohere/rerank-v3.5``, a self-hosted model behind an OpenAI-style proxy).

A bi-encoder embeds question and chunk separately; a cross-encoder reads them together and
scores relevance directly - slower, so it only re-orders a short candidate list.

Scores are cached in SQLite by (model, query, chunk text): eval re-runs cost nothing and are
deterministic.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ragforreal.store.pgvector import RetrievedChunk

_RETRYABLE = (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
              litellm.InternalServerError, litellm.ServiceUnavailableError)


class Reranker:
    def __init__(self, model: str, cache_dir: Path):
        self.model = model
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_dir / "rerank.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS score (key TEXT PRIMARY KEY, value REAL)")
        self.lock = threading.Lock()
        self.tokens_billed = 0
        self.cost_usd = 0.0

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{query}|{text}".encode()).hexdigest()

    @retry(retry=retry_if_exception_type(_RETRYABLE),
           wait=wait_exponential_jitter(multiplier=2, max=60), stop=stop_after_attempt(6),
           reraise=True)
    def _call(self, query: str, texts: list[str]) -> list[float]:
        resp = litellm.rerank(model=self.model, query=query, documents=texts,
                              top_n=len(texts), return_documents=False)
        scores = [0.0] * len(texts)
        for r in resp.results:
            scores[r["index"]] = float(r["relevance_score"])
        with self.lock:
            billed = (resp.meta or {}).get("billed_units") or {}
            self.tokens_billed += int(billed.get("total_tokens") or 0)
            self.cost_usd += float((resp._hidden_params or {}).get("response_cost") or 0.0)
        return scores

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """All candidates, re-ordered by cross-encoder score (score field replaced)."""
        if not chunks:
            return []
        keys = [self._key(query, c.text) for c in chunks]
        with self.lock:
            q = f"SELECT key, value FROM score WHERE key IN ({','.join('?' * len(keys))})"
            cached = dict(self.db.execute(q, keys).fetchall())
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            new = self._call(query, [chunks[i].text for i in missing])
            fresh = {keys[i]: s for i, s in zip(missing, new, strict=True)}
            with self.lock:
                self.db.executemany("INSERT OR REPLACE INTO score VALUES (?, ?)", fresh.items())
                self.db.commit()
            cached.update(fresh)
        scored = [replace(c, score=cached[k]) for c, k in zip(chunks, keys, strict=True)]
        # Stable sort: equal scores keep candidate order.
        return sorted(scored, key=lambda c: -c.score)
