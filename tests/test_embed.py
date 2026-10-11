import time
from types import SimpleNamespace

import numpy as np
import pytest

from ragforreal.config import EmbeddingConfig
from ragforreal.ingest import embed as embed_mod
from ragforreal.ingest.embed import Embedder, Throttle


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_embedding(model, input, **kwargs):
        calls.append({"input": list(input), **kwargs})
        data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=len(input)))

    monkeypatch.setattr(embed_mod.litellm, "embedding", fake_embedding)
    return calls


def test_embed_batches_caches_and_passes_input_type(tmp_path, fake_api):
    cfg = EmbeddingConfig(model="voyage/voyage-4", dimensions=3, batch_size=2)
    e = Embedder(cfg, tmp_path)

    vecs = e.embed_documents(["a", "bb", "ccc"], show_progress=False)
    assert vecs.shape == (3, 3)
    assert vecs[:, 0].tolist() == [1.0, 2.0, 3.0]  # order preserved
    assert [len(c["input"]) for c in fake_api] == [2, 1]  # batch_size respected
    assert all(c["input_type"] == "document" and c["dimensions"] == 3 for c in fake_api)

    # Second call is served entirely from the SQLite cache.
    fake_api.clear()
    again = Embedder(cfg, tmp_path).embed_documents(["ccc", "a"], show_progress=False)
    assert fake_api == []
    assert again[:, 0].tolist() == [3.0, 1.0]

    # Same text as a query is a different cache entry (asymmetric embeddings).
    e.embed_queries(["a"])
    assert fake_api[0]["input_type"] == "query"


def test_duplicate_texts_embedded_once(tmp_path, fake_api):
    e = Embedder(EmbeddingConfig(dimensions=3), tmp_path)
    vecs = e.embed_documents(["same", "same", "other"], show_progress=False)
    assert sum(len(c["input"]) for c in fake_api) == 2
    assert np.array_equal(vecs[0], vecs[1])


def test_throttle_disabled_is_free():
    t = Throttle(0, 0)
    start = time.monotonic()
    for _ in range(1000):
        t.wait(10_000)
    assert time.monotonic() - start < 0.5


def test_throttle_allows_oversized_single_request():
    t = Throttle(rpm=0, tpm=100)
    t.wait(500)  # bigger than the whole budget: must not deadlock on an empty window
    assert len(t.events) == 1


def test_batches_fit_the_tpm_budget(tmp_path):
    e = Embedder(EmbeddingConfig(dimensions=3, batch_size=64), tmp_path, rpm=0, tpm=1000)
    texts = [f"chunk {i} " + "word " * 200 for i in range(10)]  # ~200 tokens each
    batches = list(e._batches(texts))
    assert all(n_tokens <= 600 for _, n_tokens in batches)  # 60% of the TPM budget
    assert [t for b, _ in batches for t in b] == texts


def test_token_ratio_learned_from_usage_shrinks_batches(tmp_path, monkeypatch):
    def fake_embedding(model, input, **kwargs):
        # Provider counts 2x our local tokens.
        used = 2 * sum(embed_mod.count_tokens(t) for t in input)
        return SimpleNamespace(data=[{"index": i, "embedding": [1.0, 0.0, 0.0]}
                                     for i in range(len(input))],
                               usage=SimpleNamespace(total_tokens=used))

    monkeypatch.setattr(embed_mod.litellm, "embedding", fake_embedding)
    e = Embedder(EmbeddingConfig(dimensions=3), tmp_path, rpm=0, tpm=100_000)
    before = e.max_batch_tokens
    e.embed_documents(["some words here"], show_progress=False)
    assert e.token_ratio == pytest.approx(2.0)
    assert e.max_batch_tokens == before // 2
    # A fresh Embedder (e.g. a restarted run) starts from the persisted ratio.
    assert Embedder(EmbeddingConfig(dimensions=3), tmp_path, tpm=100_000).token_ratio == \
        pytest.approx(2.0)


def test_persistently_rate_limited_batch_is_split(tmp_path, monkeypatch):
    from tenacity import stop_after_attempt

    def fake_embedding(model, input, **kwargs):
        if len(input) > 1:
            raise embed_mod.litellm.RateLimitError("too big", llm_provider="voyage", model=model)
        return SimpleNamespace(data=[{"index": 0, "embedding": [float(len(input[0])), 0.0, 0.0]}],
                               usage=SimpleNamespace(total_tokens=1))

    monkeypatch.setattr(embed_mod.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(Embedder._call.retry, "stop", stop_after_attempt(1))
    e = Embedder(EmbeddingConfig(dimensions=3, batch_size=8), tmp_path)
    vecs = e.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"], show_progress=False)
    assert vecs[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
