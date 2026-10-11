from dataclasses import dataclass

from ragforreal.eval.metrics import (
    doc_hit_at_k,
    hit_at_k,
    page_recall_at_k,
    percentile,
    reciprocal_rank,
    retrieval_metrics,
)


@dataclass
class C:
    doc_name: str
    page_start: int
    page_end: int


GOLD = {("A", 10), ("A", 12)}
RETRIEVED = [C("B", 10, 10), C("A", 1, 2), C("A", 9, 10), C("A", 12, 13)]


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, GOLD, 2) == 0.0
    assert hit_at_k(RETRIEVED, GOLD, 3) == 1.0


def test_wrong_doc_same_page_is_not_a_hit():
    assert hit_at_k([C("B", 10, 10)], GOLD, 1) == 0.0


def test_page_recall():
    assert page_recall_at_k(RETRIEVED, GOLD, 3) == 0.5
    assert page_recall_at_k(RETRIEVED, GOLD, 4) == 1.0


def test_doc_hit():
    assert doc_hit_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert doc_hit_at_k(RETRIEVED, GOLD, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, GOLD) == 1 / 3
    assert reciprocal_rank([C("Z", 0, 0)], GOLD) == 0.0


def test_retrieval_metrics_keys():
    m = retrieval_metrics(RETRIEVED, GOLD, [1, 5])
    assert set(m) == {"mrr", "hit@1", "hit@5", "page_recall@1", "page_recall@5",
                      "doc_hit@1", "doc_hit@5"}


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_judge_parses_verdict_from_malformed_json(monkeypatch):
    from types import SimpleNamespace

    from ragforreal.eval import judge as judge_mod

    raw = '{"verdict": "correct", "reason": "it says "yes" to the "legal" question"}'
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
    monkeypatch.setattr(judge_mod.litellm, "completion", lambda **kw: resp)
    monkeypatch.setattr(judge_mod.litellm, "completion_cost", lambda **kw: 0.0)
    assert judge_mod.judge_answer("q", "gold", "cand", "m").verdict == "correct"


def test_evidence_relevance_credits_matching_text_on_another_page():
    from ragforreal.eval.metrics import evidence_metrics, relevance

    table = " ".join(f"item{i} {1000 + i:,}" for i in range(30))
    chunks = [
        SimpleChunk("A", 5, 5, "unrelated risk factors " * 10),
        SimpleChunk("A", 11, 11, "Consolidated statement " + table),  # gold is page 10
        SimpleChunk("B", 11, 11, "Consolidated statement " + table),  # wrong document
        SimpleChunk("A", 10, 10, "anything"),  # on the gold page
    ]
    flags = relevance(chunks, {("A", 10)}, [("A", table)])
    assert flags == [False, True, False, True]
    m = evidence_metrics(flags, [1, 2])
    assert m == {"ev_mrr": 0.5, "ev_hit@1": 0.0, "ev_hit@2": 1.0}


def test_evidence_relevance_needs_enough_shared_tokens():
    from ragforreal.eval.metrics import relevance

    # A short chunk fully contained in the evidence still needs min_shared tokens in common.
    assert relevance([SimpleChunk("A", 1, 1, "total revenue")], set(),
                     [("A", "total revenue grew 5% to 1,234 million")]) == [False]


@dataclass
class SimpleChunk:
    doc_name: str
    page_start: int
    page_end: int
    text: str
