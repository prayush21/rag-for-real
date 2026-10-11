from pathlib import Path

import pytest

from ragforreal.config import ExperimentConfig
from ragforreal.store.pgvector import _table

CONFIGS = sorted(Path(__file__).parents[1].glob("configs/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_load_and_name_a_safe_table(path):
    cfg = ExperimentConfig.load(path)
    assert _table(cfg.index_name("dev"))  # raises on unsafe names


def test_index_name_depends_only_on_ingest_settings():
    base = ExperimentConfig(name="a")
    same_ingest = ExperimentConfig(name="b", retrieval={"top_k": 3},
                                   generation={"model": "x/y"})
    other_chunking = ExperimentConfig(name="c", ingest={"chunking": {"chunk_size": 256}})
    assert base.index_name("dev") == same_ingest.index_name("dev")
    assert base.index_name("dev") != other_chunking.index_name("dev")
    assert base.index_name("dev") != base.index_name("full")


def test_unsafe_table_name_rejected():
    with pytest.raises(ValueError):
        _table("x; DROP TABLE users")
