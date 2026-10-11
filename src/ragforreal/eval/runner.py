"""Run an experiment against FinanceBench and write a reproducible result bundle.

results/runs/<experiment>/<timestamp>/
    config.json         full experiment config + index name + git commit
    per_question.jsonl  retrieved pages, metrics, answer, verdict for every question
    summary.json        aggregate metrics
results/leaderboard.md  regenerated from the latest run of every experiment
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.progress import track
from rich.table import Table

from ragforreal.config import ROOT, ExperimentConfig, Settings
from ragforreal.data import financebench as fb
from ragforreal.eval.judge import judge_answer
from ragforreal.eval.metrics import (
    evidence_metrics,
    percentile,
    relevance,
    retrieval_metrics,
)
from ragforreal.generation.answer import generate_answer
from ragforreal.ingest.chunking import count_tokens
from ragforreal.retrieval.retriever import Retriever

console = Console()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _mean_by(rows: list[dict], key: str, group: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[r[group]].append(r["retrieval"][key])
    return {g: round(mean(v), 4) for g, v in sorted(buckets.items())}


def _stage_latency(rows: list[dict]) -> dict[str, float]:
    stages = sorted({k for r in rows for k in r["timings_ms"]})
    out = {}
    for st in stages:
        vals = [r["timings_ms"][st] for r in rows if st in r["timings_ms"]]
        out[f"{st}_p50"] = round(percentile(vals, 50), 1)
        out[f"{st}_p95"] = round(percentile(vals, 95), 1)
    return out


def run_eval(cfg: ExperimentConfig, settings: Settings, generate: bool = False,
             limit: int | None = None, workers: int = 4, repeats: int = 1) -> dict:
    questions = fb.load_questions(settings.data_dir, settings.corpus)[:limit]
    retriever = Retriever(cfg, settings)
    ks = sorted(cfg.eval.ks)
    k_max = max(max(ks), cfg.retrieval.top_k)
    cand_ks = sorted({*ks, cfg.retrieval.candidate_k})

    t0 = time.perf_counter()
    qvecs = (retriever.embedder.embed_queries([q.question for q in questions])
             if cfg.retrieval.strategy != "sparse" else [None] * len(questions))
    embed_ms = (time.perf_counter() - t0) * 1000 / max(1, len(questions))

    rows: list[dict] = []
    for q, vec in track(list(zip(questions, qvecs, strict=True)), description="Retrieving"):
        trace = retriever.retrieve_traced(q.question, vec, k_max)
        hits = trace.final
        gold = {(e.doc_name, e.page) for e in q.evidence}
        evidence = [(e.doc_name, e.text) for e in q.evidence]
        context = hits[: cfg.retrieval.top_k]
        rows.append({
            "id": q.id,
            "question": q.question,
            "gold_answer": q.answer,
            "question_type": q.question_type,
            "gold_pages": sorted(gold),
            "doc_filter": trace.doc_filter,
            "retrieved": [[h.doc_name, h.page_start, h.page_end, round(h.score, 4)] for h in hits],
            "retrieval": {**retrieval_metrics(hits, gold, ks),
                          **evidence_metrics(relevance(hits, gold, evidence), ks)},
            # Before reranking: the reranker can only promote what is in here.
            "retrieval_candidates": retrieval_metrics(trace.candidates, gold, cand_ks),
            "context_tokens": sum(count_tokens(h.text) for h in context),
            "timings_ms": {k: round(v, 1) for k, v in trace.timings_ms.items()},
            "_context": context,
        })

    if generate:
        def answer_and_judge(row: dict) -> dict:
            t = time.perf_counter()
            try:
                ans = generate_answer(row["question"], row["_context"], cfg.generation)
                verdict = judge_answer(row["question"], row["gold_answer"], ans.text,
                                       cfg.eval.judge_model)
            except Exception as e:  # one failed question shouldn't sink the whole run
                return {"answer": "", "citations": [], "verdict": "error",
                        "judge_reason": repr(e)[:300], "latency_ms": 0,
                        "cost_usd": 0.0, "prompt_tokens": 0}
            return {
                "answer": ans.text,
                "citations": ans.citations,
                "verdict": verdict.verdict,
                "judge_reason": verdict.reason,
                "latency_ms": round((time.perf_counter() - t) * 1000),
                "cost_usd": ans.cost_usd + verdict.cost_usd,
                "prompt_tokens": ans.prompt_tokens,
            }

        # Repeats measure the noise floor: identical configs move several points run to run.
        jobs = [r for r in rows for _ in range(repeats)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(track(pool.map(answer_and_judge, jobs), total=len(jobs),
                                 description=f"Generating + judging (x{repeats})"))
        for i, row in enumerate(rows):
            row["generations"] = results[i * repeats : (i + 1) * repeats]
            row["generation"] = row["generations"][0]

    summary = _summarize(cfg, settings, rows, ks, embed_ms, generate)
    if retriever.reranker:
        summary["rerank"] = {"model": retriever.reranker.model,
                             "tokens_billed": retriever.reranker.tokens_billed,
                             "cost_usd": round(retriever.reranker.cost_usd, 4)}
    _write_bundle(cfg, settings, retriever.index, rows, summary)
    _print_summary(summary, ks)
    return summary


def _generation_stats(gens: list[dict]) -> dict:
    n = len(gens)
    lat = [g["latency_ms"] for g in gens]
    return {
        "accuracy": round(sum(g["verdict"] == "correct" for g in gens) / n, 4),
        "incorrect_rate": round(sum(g["verdict"] == "incorrect" for g in gens) / n, 4),
        "refusal_rate": round(sum(g["verdict"] == "refused" for g in gens) / n, 4),
        # Failed LLM calls (quota, outage). Non-zero means accuracy is NOT trustworthy.
        "error_rate": round(sum(g["verdict"] in ("error", "unparseable") for g in gens) / n, 4),
        "citation_rate": round(sum(bool(g["citations"]) for g in gens) / n, 4),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "cost_per_question_usd": round(sum(g["cost_usd"] for g in gens) / n, 6),
    }


def _summarize(cfg: ExperimentConfig, settings: Settings, rows: list[dict], ks: list[int],
               embed_ms: float, generate: bool) -> dict:
    metric_keys = rows[0]["retrieval"].keys() if rows else []
    cand_keys = rows[0]["retrieval_candidates"].keys() if rows else []
    summary: dict = {
        "experiment": cfg.name,
        "description": cfg.description,
        "corpus": settings.corpus,
        "index": cfg.index_name(settings.corpus),
        "n_questions": len(rows),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrieval": {k: round(mean(r["retrieval"][k] for r in rows), 4) for k in metric_keys},
        "retrieval_candidates": {k: round(mean(r["retrieval_candidates"][k] for r in rows), 4)
                                 for k in cand_keys},
        "hit@5_by_question_type": _mean_by(rows, "hit@5", "question_type") if 5 in ks else {},
        "context_tokens@top_k": round(mean(r["context_tokens"] for r in rows)) if rows else 0,
        "top_k": cfg.retrieval.top_k,
        "candidate_k": cfg.retrieval.candidate_k,
        "filtered_share": round(mean(r["doc_filter"] is not None for r in rows), 4) if rows else 0,
        "latency_ms": {"query_embed_avg": round(embed_ms, 1), **_stage_latency(rows)},
    }
    if generate and rows:
        repeats = len(rows[0]["generations"])
        runs = [_generation_stats([r["generations"][i] for r in rows]) for i in range(repeats)]
        pooled = _generation_stats([g for r in rows for g in r["generations"]])
        summary["generation"] = {
            "model": cfg.generation.model,
            "judge_model": cfg.eval.judge_model,
            "repeats": repeats,
            **pooled,  # means over all repeats
            "accuracy_runs": [s["accuracy"] for s in runs],
            "accuracy_spread": round(max(s["accuracy"] for s in runs)
                                     - min(s["accuracy"] for s in runs), 4),
        }
    return summary


def _write_bundle(cfg: ExperimentConfig, settings: Settings, index: str, rows: list[dict],
                  summary: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = settings.results_dir / "runs" / cfg.name / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"experiment": cfg.model_dump(), "index": index, "corpus": settings.corpus,
         "git_commit": _git_commit()}, indent=2))
    with (out / "per_question.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_leaderboard(settings.results_dir)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    console.print(f"Results written to [bold]{shown}[/]")
    return out


def write_leaderboard(results_dir: Path) -> None:
    latest: dict[tuple[str, str], dict] = {}
    for path in sorted((results_dir / "runs").glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        latest[(s["corpus"], s["experiment"])] = s  # sorted by timestamp dir, so last wins

    lines = [
        "# Leaderboard",
        "",
        "Latest run of each experiment. Retrieval metrics are scored against FinanceBench "
        "gold evidence pages; accuracy is LLM-judged against the gold answer.",
        "",
        "Ev hit@5 also counts chunks containing most of the gold evidence text (blank for runs "
        "before that metric existed). Accuracy ± is the max-min spread over repeated runs.",
        "",
        "| Corpus | Experiment | N | Hit@1 | Hit@5 | Hit@10 | Ev hit@5 | MRR "
        "| Ctx tokens | Accuracy | Refusals | $/question |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (corpus, name), s in sorted(latest.items(), key=lambda kv: kv[1]["timestamp"]):
        r, g = s["retrieval"], s.get("generation", {})

        def pct(d: dict, k: str) -> str:
            return f"{d[k]:.1%}" if k in d else "-"

        cost = f"${g['cost_per_question_usd']:.4f}" if g else "-"
        acc = pct(g, "accuracy")
        if g.get("repeats", 1) > 1:
            acc += f" ±{g['accuracy_spread'] / 2:.1%} (x{g['repeats']})"
        if g.get("error_rate"):
            name = f"{name} ⚠️ {g['error_rate']:.0%} errors"
        lines.append(
            f"| {corpus} | {name} | {s['n_questions']} | {pct(r, 'hit@1')} | {pct(r, 'hit@5')} "
            f"| {pct(r, 'hit@10')} | {pct(r, 'ev_hit@5')} | {r.get('mrr', 0):.3f} "
            f"| {s['context_tokens@top_k']:,} | {acc} | {pct(g, 'refusal_rate')} "
            f"| {cost} |"
        )
    (results_dir / "leaderboard.md").write_text("\n".join(lines) + "\n")


def _print_summary(summary: dict, ks: list[int]) -> None:
    table = Table(title=f"{summary['experiment']} ({summary['corpus']}, "
                        f"n={summary['n_questions']})")
    table.add_column("metric")
    for k in ks:
        table.add_column(f"@{k}", justify="right")
    r = summary["retrieval"]
    c = summary["retrieval_candidates"]
    for m in ("hit", "ev_hit", "page_recall", "doc_hit"):
        table.add_row(m, *[f"{r[f'{m}@{k}']:.1%}" for k in ks])
    table.add_row("hit (pre-rerank)", *[f"{c[f'hit@{k}']:.1%}" for k in ks])
    console.print(table)
    n_cand = summary.get("candidate_k")
    console.print(f"MRR {r['mrr']:.3f} (ev {r['ev_mrr']:.3f}) · candidates hit@{n_cand} "
                  f"{c.get(f'hit@{n_cand}', 0):.1%} · filtered {summary['filtered_share']:.0%} · "
                  f"context tokens@top_k {summary['context_tokens@top_k']:,}")
    console.print(f"Latency: {summary['latency_ms']}")
    if "rerank" in summary:
        console.print(f"Rerank: {summary['rerank']}")
    if "generation" in summary:
        console.print(f"Generation: {summary['generation']}")
        if summary["generation"]["error_rate"]:
            console.print(f"[bold red]{summary['generation']['error_rate']:.0%} of questions "
                          "failed (see per_question.jsonl) - accuracy is not trustworthy[/]")
