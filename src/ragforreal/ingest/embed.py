"""Embedding via LiteLLM, so the provider is a config string (``voyage/voyage-4``,
``gemini/gemini-embedding-001``, ``openai/text-embedding-3-large``...).

Two production concerns handled here:

* **Caching** - embeddings are keyed by (model, dims, input_type, text) in SQLite. Re-running an
  experiment, or a new experiment that reuses chunks, costs zero API tokens.
* **Rate limiting** - a client-side RPM/TPM throttle plus retry with exponential backoff on
  429s, so a free-tier key degrades to "slow" instead of "crashes halfway through indexing".
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import litellm
import numpy as np
from rich.progress import Progress
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ragforreal.config import EmbeddingConfig
from ragforreal.ingest.chunking import count_tokens

log = logging.getLogger(__name__)

# Stay well under provider per-request limits (Voyage: 1000 inputs, 120K-320K tokens).
MAX_BATCH_TOKENS = 100_000

_RETRYABLE = (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
              litellm.InternalServerError, litellm.ServiceUnavailableError)


class Throttle:
    """Sliding-window limiter over recent requests and tokens.

    The window is a little longer than a minute: provider clocks and network latency mean a
    request sent exactly 60s after another can still land in the same provider-side window.
    """

    def __init__(self, rpm: int, tpm: int, window_s: float = 65.0):
        self.rpm, self.tpm, self.window_s = rpm, tpm, window_s
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def wait(self, tokens: int) -> None:
        if not self.rpm and not self.tpm:
            return
        with self.lock:
            while True:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > self.window_s:
                    self.events.popleft()
                used_tokens = sum(t for _, t in self.events)
                ok_r = not self.rpm or len(self.events) < self.rpm
                ok_t = not self.tpm or used_tokens + tokens <= self.tpm or not self.events
                if ok_r and ok_t:
                    self.events.append((now, tokens))
                    return
                time.sleep(max(0.5, self.window_s - (now - self.events[0][0])))


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL)")

    def get_meta(self, key: str, default: float) -> float:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self.db.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
            for k, blob in self.db.execute(q, part):
                out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.db.commit()


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, cache_dir: Path, rpm: int = 0, tpm: int = 0):
        self.cfg = cfg
        self.cache = EmbeddingCache(cache_dir / "embeddings.sqlite")
        self.throttle = Throttle(rpm, tpm)
        self.tpm = tpm
        # Provider tokens per local (cl100k) token. Our tokenizer is only a proxy - number-heavy
        # financial tables cost noticeably more provider tokens - so learn the real ratio from
        # the usage each response reports, and size batches and the throttle with it. Persisted
        # per model so a restarted run doesn't re-learn it by tripping the rate limit.
        self._ratio_key = f"token_ratio|{cfg.model}"
        self.token_ratio = self.cache.get_meta(self._ratio_key, 1.0)
        self.tokens_billed = 0

    @property
    def max_batch_tokens(self) -> int:
        """Batch budget in local tokens. One request can never exceed the per-minute token
        budget, and providers appear to check it with a conservative pre-count (Voyage rejects
        batches its own billing later counts at ~85% of the limit), so stay at 60% after
        converting with the learned ratio."""
        if not self.tpm:
            return MAX_BATCH_TOKENS
        return min(MAX_BATCH_TOKENS, int(self.tpm * 0.6 / self.token_ratio))

    def _key(self, text: str, input_type: str | None) -> str:
        raw = f"{self.cfg.model}|{self.cfg.dimensions}|{input_type}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @retry(retry=retry_if_exception_type(_RETRYABLE),
           wait=wait_exponential_jitter(multiplier=2, max=60), stop=stop_after_attempt(8),
           before_sleep=before_sleep_log(log, logging.WARNING), reraise=True)
    def _call(self, texts: list[str], input_type: str | None, n_tokens: int) -> list[np.ndarray]:
        self.throttle.wait(int(n_tokens * self.token_ratio))
        kwargs = {"dimensions": self.cfg.dimensions}
        if input_type:
            kwargs["input_type"] = input_type
        resp = litellm.embedding(model=self.cfg.model, input=texts, **kwargs)
        used = resp.usage.total_tokens if resp.usage else 0
        self.tokens_billed += used
        log.info("embedded %d texts: ~%d local tokens, %d billed (ratio %.2f)",
                 len(texts), n_tokens, used, used / n_tokens if n_tokens else 0)
        if used and n_tokens and used / n_tokens > self.token_ratio:
            self.token_ratio = used / n_tokens
            self.cache.set_meta(self._ratio_key, self.token_ratio)
        data = sorted(resp.data, key=lambda d: d["index"])
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]

    def _batches(self, texts: list[str]):
        batch, batch_tokens = [], 0
        for t in texts:
            n = count_tokens(t)
            if batch and (len(batch) >= self.cfg.batch_size
                          or batch_tokens + n > self.max_batch_tokens):
                yield batch, batch_tokens
                batch, batch_tokens = [], 0
            batch.append(t)
            batch_tokens += n
        if batch:
            yield batch, batch_tokens

    def _embed_batch(self, batch: list[str], input_type: str | None, n_tokens: int
                     ) -> list[np.ndarray]:
        """Embed a batch; if it is still rate-limited after all retries, it is probably larger
        than the per-minute budget on its own, so split it in half instead of failing."""
        try:
            return self._call(batch, input_type, n_tokens)
        except litellm.RateLimitError:
            if len(batch) == 1:
                raise
            log.warning("batch of %d still rate-limited after retries; splitting", len(batch))
            halves = batch[: len(batch) // 2], batch[len(batch) // 2 :]
            return [v for half in halves
                    for v in self._embed_batch(half, input_type,
                                               sum(count_tokens(t) for t in half))]

    def embed(self, texts: list[str], input_type: str | None, show_progress: bool = False
              ) -> np.ndarray:
        keys = [self._key(t, input_type) for t in texts]
        found = self.cache.get_many(list(set(keys)))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in found))

        with Progress(disable=not show_progress) as progress:
            task = progress.add_task(f"Embedding ({self.cfg.model})", total=len(missing))
            for batch, n_tokens in self._batches(missing):
                vecs = self._embed_batch(batch, input_type, n_tokens)
                new = {self._key(t, input_type): v for t, v in zip(batch, vecs, strict=True)}
                self.cache.put_many(new)
                found.update(new)
                progress.advance(task, len(batch))
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.cfg.dimensions))

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self.embed(texts, self.cfg.document_input_type, show_progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, self.cfg.query_input_type)
