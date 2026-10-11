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
