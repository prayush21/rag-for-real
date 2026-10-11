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
from ragforreal.eval.metrics import percentile, retrieval_metrics
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


def run_eval(cfg: ExperimentConfig, settings: Settings, generate: bool = False,
             limit: int | None = None, workers: int = 4) -> dict:
    questions = fb.load_questions(settings.data_dir, settings.corpus)[:limit]
    retriever = Retriever(cfg, settings)
    ks = sorted(cfg.eval.ks)
    k_max = max(max(ks), cfg.retrieval.top_k)

    t0 = time.perf_counter()
    qvecs = retriever.embedder.embed_queries([q.question for q in questions])
    embed_ms = (time.perf_counter() - t0) * 1000 / max(1, len(questions))

    rows: list[dict] = []
    for q, vec in track(list(zip(questions, qvecs, strict=True)), description="Retrieving"):
        t = time.perf_counter()
        hits = retriever.store.dense_search(retriever.index, vec, k_max)
        search_ms = (time.perf_counter() - t) * 1000
        gold = {(e.doc_name, e.page) for e in q.evidence}
        context = hits[: cfg.retrieval.top_k]
        rows.append({
            "id": q.id,
            "question": q.question,
            "gold_answer": q.answer,
            "question_type": q.question_type,
            "gold_pages": sorted(gold),
            "retrieved": [[h.doc_name, h.page_start, h.page_end, round(h.score, 4)] for h in hits],
            "retrieval": retrieval_metrics(hits, gold, ks),
            "context_tokens": sum(count_tokens(h.text) for h in context),
            "search_ms": round(search_ms, 1),
            "_context": context,
        })

    if generate:
        def answer_and_judge(row: dict) -> None:
            t = time.perf_counter()
            try:
                ans = generate_answer(row["question"], row["_context"], cfg.generation)
                verdict = judge_answer(row["question"], row["gold_answer"], ans.text,
                                       cfg.eval.judge_model)
            except Exception as e:  # one failed question shouldn't sink the whole run
                row["generation"] = {"answer": "", "citations": [], "verdict": "error",
                                     "judge_reason": repr(e)[:300], "latency_ms": 0,
                                     "cost_usd": 0.0, "prompt_tokens": 0}
                return
            row["generation"] = {
                "answer": ans.text,
                "citations": ans.citations,
                "verdict": verdict.verdict,
                "judge_reason": verdict.reason,
                "latency_ms": round((time.perf_counter() - t) * 1000),
                "cost_usd": ans.cost_usd + verdict.cost_usd,
                "prompt_tokens": ans.prompt_tokens,
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(track(pool.map(answer_and_judge, rows), total=len(rows),
                       description="Generating + judging"))

    summary = _summarize(cfg, settings, rows, ks, embed_ms, generate)
    _write_bundle(cfg, settings, retriever.index, rows, summary)
    _print_summary(summary, ks)
    return summary


def _summarize(cfg: ExperimentConfig, settings: Settings, rows: list[dict], ks: list[int],
               embed_ms: float, generate: bool) -> dict:
    metric_keys = rows[0]["retrieval"].keys() if rows else []
    search = [r["search_ms"] for r in rows]
    summary: dict = {
        "experiment": cfg.name,
        "description": cfg.description,
        "corpus": settings.corpus,
        "index": cfg.index_name(settings.corpus),
        "n_questions": len(rows),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrieval": {k: round(mean(r["retrieval"][k] for r in rows), 4) for k in metric_keys},
        "hit@5_by_question_type": _mean_by(rows, "hit@5", "question_type") if 5 in ks else {},
        "context_tokens@top_k": round(mean(r["context_tokens"] for r in rows)) if rows else 0,
        "top_k": cfg.retrieval.top_k,
        "latency_ms": {
            "query_embed_avg": round(embed_ms, 1),
            "search_p50": round(percentile(search, 50), 1),
            "search_p95": round(percentile(search, 95), 1),
        },
    }
    if generate and rows:
        gens = [r["generation"] for r in rows]
        n = len(gens)
        lat = [g["latency_ms"] for g in gens]
        summary["generation"] = {
            "model": cfg.generation.model,
            "judge_model": cfg.eval.judge_model,
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
        "| Corpus | Experiment | N | Hit@1 | Hit@5 | Hit@10 | Page recall@10 | MRR "
        "| Ctx tokens | Accuracy | Refusals | $/question |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (corpus, name), s in sorted(latest.items(), key=lambda kv: kv[1]["timestamp"]):
        r, g = s["retrieval"], s.get("generation", {})

        def pct(d: dict, k: str) -> str:
            return f"{d[k]:.1%}" if k in d else "-"

        cost = f"${g['cost_per_question_usd']:.4f}" if g else "-"
        if g.get("error_rate"):
            name = f"{name} ⚠️ {g['error_rate']:.0%} errors"
        lines.append(
            f"| {corpus} | {name} | {s['n_questions']} | {pct(r, 'hit@1')} | {pct(r, 'hit@5')} "
            f"| {pct(r, 'hit@10')} | {pct(r, 'page_recall@10')} | {r.get('mrr', 0):.3f} "
            f"| {s['context_tokens@top_k']:,} | {pct(g, 'accuracy')} | {pct(g, 'refusal_rate')} "
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
    for m in ("hit", "page_recall", "doc_hit"):
        table.add_row(m, *[f"{r[f'{m}@{k}']:.1%}" for k in ks])
    console.print(table)
    console.print(f"MRR {r['mrr']:.3f} · context tokens@top_k "
                  f"{summary['context_tokens@top_k']:,} · latency {summary['latency_ms']}")
    if "generation" in summary:
        console.print(f"Generation: {summary['generation']}")
        if summary["generation"]["error_rate"]:
            console.print(f"[bold red]{summary['generation']['error_rate']:.0%} of questions "
                          "failed (see per_question.jsonl) - accuracy is not trustworthy[/]")
