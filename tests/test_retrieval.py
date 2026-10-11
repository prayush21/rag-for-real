from types import SimpleNamespace

import pytest

from ragforreal.config import ExperimentConfig, RetrievalConfig, Settings
from ragforreal.retrieval.filters import companies_in, filter_docs
from ragforreal.retrieval.fusion import rrf
from ragforreal.retrieval.rerank import Reranker
from ragforreal.retrieval.retriever import Retriever
from ragforreal.store.pgvector import RetrievedChunk


def _c(cid: str, score: float = 0.0, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(cid, cid.split(":")[0], 0, 0, 0, text or cid, {}, score)


# ---------------------------------------------------------------- RRF


def test_rrf_rewards_agreement_over_one_top_rank():
    dense = [_c("A:1"), _c("B:1"), _c("C:1")]
    sparse = [_c("D:1"), _c("B:1"), _c("A:1")]
    fused = rrf([dense, sparse], k=60)
    # A: 1/61 + 1/63, B: 2/62, D: 1/61 -> A and B (in both lists) beat D (top of one list).
    assert [c.id for c in fused] == ["A:1", "B:1", "D:1", "C:1"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_single_list_keeps_order():
    ranked = [_c("A:1", 0.1), _c("B:1", 0.9)]  # raw scores are ignored, only ranks count
    assert [c.id for c in rrf([ranked])] == ["A:1", "B:1"]


# ---------------------------------------------------------------- company filter

COMPANIES = {"3M", "American Express", "Johnson & Johnson", "AES Corporation", "Pepsico",
             "MGM Resorts", "Block", "Boeing", "AMD", "Coca-Cola"}


@pytest.mark.parametrize("question,expected", [
    ("Does AMEX have an improving operating margin profile?", {"American Express"}),
    ("under American Express' name as of 2022?", {"American Express"}),
    ("Are JnJ's FY2022 financials that of a high growth company?", {"Johnson & Johnson"}),
    ("What is the FY2018 capex for 3M?", {"3M"}),
    ("restructuring costs in AES Corporation's income statements", {"AES Corporation"}),
    ("Roughly how many times has AES sold its inventory?", {"AES Corporation"}),
    ("What is the FY2022 EBITDA for PepsiCo?", {"Pepsico"}),
    ("Which region had the Highest EBITDAR Contribution for MGM?", {"MGM Resorts"}),
    ("Compare Boeing and AMD revenue growth", {"Boeing", "AMD"}),
    ("Coca Cola's dividend payout", {"Coca-Cola"}),
    ("What drove the reduction in SG&A expense in FY2023?", set()),
    ("Does the company block share repurchases?", set()),  # lowercase "block" is a word
    ("Did Block report Cash App revenue?", {"Block"}),
    ("What is 3Mx?", set()),  # word boundaries
])
def test_companies_in(question, expected):
    assert companies_in(question, COMPANIES) == expected


def test_filter_docs_returns_all_filings_of_the_company():
    docs = {"3M_2018_10K": "3M", "3M_2022_10K": "3M", "BOEING_2022_10K": "Boeing"}
    assert filter_docs("3M capex trend?", docs) == ["3M_2018_10K", "3M_2022_10K"]
    assert filter_docs("Which segment grew most?", docs) is None


# ---------------------------------------------------------------- reranker


def test_reranker_orders_by_score_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_rerank(model, query, documents, **kw):
        calls.append(list(documents))
        return SimpleNamespace(
            results=[{"index": i, "relevance_score": float(len(d))}
                     for i, d in enumerate(documents)],
            meta={"billed_units": {"total_tokens": 7}}, _hidden_params={})

    monkeypatch.setattr("ragforreal.retrieval.rerank.litellm.rerank", fake_rerank)
    r = Reranker("voyage/rerank-2.5", tmp_path)
    cands = [_c("A:1", text="aa"), _c("B:1", text="bbbb"), _c("C:1", text="c")]
    assert [c.id for c in r.rerank("q", cands)] == ["B:1", "A:1", "C:1"]
    assert r.rerank("q", cands)[0].score == 4.0
    assert len(calls) == 1  # second call fully cached
    r.rerank("q", [*cands, _c("D:1", text="ddddd")])
    assert calls[-1] == ["ddddd"]  # only the uncached chunk is sent
    assert r.tokens_billed == 14


# ---------------------------------------------------------------- stage composition


class FakeStore:
    def __init__(self):
        self.calls = []

    def index_status(self, name):
        return "ready"

    def ensure_fulltext(self, name):
        self.calls.append("ensure_fulltext")

    def doc_companies(self, name):
        return {"A": "Acme", "B": "Beta"}

    def dense_search(self, name, vec, k, doc_names=None):
        self.calls.append(("dense", k, doc_names))
        return [_c("A:1"), _c("B:1"), _c("A:2")][:k]

    def sparse_search(self, name, query, k, doc_names=None):
        self.calls.append(("sparse", k, doc_names))
        return [_c("B:2"), _c("A:2")][:k]


def _retriever(tmp_path, **retrieval) -> tuple[Retriever, FakeStore]:
    cfg = ExperimentConfig(name="t", retrieval=RetrievalConfig(**retrieval))
    store = FakeStore()
    return Retriever(cfg, Settings(cache_dir=tmp_path), store=store), store


def test_hybrid_fuses_both_generators(tmp_path):
    r, store = _retriever(tmp_path, strategy="hybrid", candidate_k=3, top_k=2)
    trace = r.retrieve_traced("q", query_vec=[0.0])
    assert "ensure_fulltext" in store.calls
    assert ("dense", 3, None) in store.calls and ("sparse", 3, None) in store.calls
    assert [c.id for c in trace.candidates][:2] == ["A:2", "A:1"]  # A:2 is in both lists
    assert len(trace.candidates) == 3  # fused list cut to candidate_k
    assert len(trace.final) == 2
    assert set(trace.timings_ms) == {"dense", "sparse"}


def test_company_filter_reaches_every_generator(tmp_path):
    r, store = _retriever(tmp_path, strategy="hybrid", company_filter=True)
    trace = r.retrieve_traced("What did Acme spend?", query_vec=[0.0])
    assert trace.doc_filter == ["A"]
    searches = [c for c in store.calls if isinstance(c, tuple)]
    assert len(searches) == 2 and all(c[2] == ["A"] for c in searches)


def test_rerank_stage_reorders_candidates(tmp_path, monkeypatch):
    r, _ = _retriever(tmp_path, strategy="dense", reranker="voyage/rerank-2.5", top_k=2)
    monkeypatch.setattr(r.reranker, "rerank", lambda q, cs: list(reversed(cs)))
    trace = r.retrieve_traced("q", query_vec=[0.0])
    assert [c.id for c in trace.candidates] == ["A:1", "B:1", "A:2"]
    assert [c.id for c in trace.final] == ["A:2", "B:1"]
    assert "rerank" in trace.timings_ms


# ---------------------------------------------------------------- BM25


def test_bm25_prefers_rare_terms_over_repetition():
    from ragforreal.retrieval.bm25 import BM25Search

    # "United States" is everywhere in a filing (low IDF); "property"/"equipment" are not.
    filler = [_c(f"C:{i}", text=f"Operations in the United States, segment {i}")
              for i in range(20)]
    chunks = [
        _c("A:1", text="United States, Delaware. " * 30 + "purchases"),
        _c("A:2", text="Purchases of property, plant and equipment were 1,577 million"),
        _c("B:1", text="Purchases of property and equipment at Beta"),
        _c("B:2", text="quick ratio and current liabilities"),
        *filler,
    ]
    bm25 = BM25Search(chunks)
    hits = bm25.search("What were the purchases of property and equipment in the United States?",
                       k=3)
    assert hits[0].id in {"A:2", "B:1"}
    assert "B:2" not in {h.id for h in hits}  # no shared term: never returned
    assert [h.id for h in bm25.search("purchases of property", k=5, doc_names=["B"])] == ["B:1"]
