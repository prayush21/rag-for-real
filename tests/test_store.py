"""Integration tests against a throwaway embedded Postgres + pgvector (no Docker needed)."""

import numpy as np
import pytest

from ragforreal.ingest.chunking import Chunk
from ragforreal.store.pgvector import PgVectorStore

pgserver = pytest.importorskip("pgserver")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    yield PgVectorStore(srv.get_uri())
    srv.cleanup()


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_index_lifecycle_and_search(store):
    chunks = [
        Chunk("ACME_2023_10K", 0, "revenue table", 3, 4, 10, {"company": "ACME"}),
        Chunk("ACME_2023_10K", 1, "risk factors", 10, 10, 10, {"company": "ACME"}),
        Chunk("BETA_2022_10K", 0, "cash flow", 7, 7, 10, {"company": "BETA"}),
    ]
    vecs = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0.9, 0.1, 0])])

    store.create_index("t_idx", "dev", {"k": "v"}, dimensions=3)
    assert store.index_status("t_idx") == "building"
    store.insert_chunks("t_idx", chunks, vecs)
    store.finalize_index("t_idx")
    assert store.index_status("t_idx") == "ready"
    assert [i["n_chunks"] for i in store.list_indexes() if i["name"] == "t_idx"] == [3]

    hits = store.dense_search("t_idx", _unit([1, 0, 0]), k=2)
    assert [h.id for h in hits] == ["ACME_2023_10K:0", "BETA_2022_10K:0"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert (hits[0].page_start, hits[0].page_end) == (3, 4)
    assert hits[0].metadata == {"company": "ACME"}

    filtered = store.dense_search("t_idx", _unit([1, 0, 0]), k=5, doc_names=["BETA_2022_10K"])
    assert [h.doc_name for h in filtered] == ["BETA_2022_10K"]


def test_rebuild_replaces_rows(store):
    c = [Chunk("X", 0, "a", 0, 0, 1)]
    for _ in range(2):
        store.create_index("t_rebuild", "dev", {}, dimensions=3)
        store.insert_chunks("t_rebuild", c, np.stack([_unit([1, 1, 0])]))
        store.finalize_index("t_rebuild")
    assert len(store.dense_search("t_rebuild", _unit([1, 1, 0]), k=10)) == 1
