"""`rag` command-line entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragforreal.config import ExperimentConfig, get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v", help="Log every API call")):
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(level=logging.WARNING, format="%(message)s",
                        handlers=[RichHandler(console=console, show_path=False)])
    logging.getLogger("ragforreal").setLevel(logging.INFO if verbose else logging.WARNING)
    # LiteLLM is very chatty on errors we already handle with retries.
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

ConfigOpt = typer.Option(Path("configs/p1_baseline.yaml"), "--config", "-c",
                         help="Experiment config YAML")


@app.command()
def fetch() -> None:
    """Download FinanceBench questions and the PDFs for the current CORPUS (dev|full)."""
    from ragforreal.data import financebench as fb

    s = get_settings()
    fb.fetch_metadata(s.data_dir)
    paths = fb.fetch_pdfs(s.data_dir, s.corpus)
    n_q = len(fb.load_questions(s.data_dir, s.corpus))
    console.print(f"[green]{s.corpus} corpus:[/] {len(paths)} PDFs, {n_q} questions")


@app.command()
def chunks(config: Path = ConfigOpt,
           sample: int = typer.Option(0, help="Print N random chunks to eyeball them"),
           seed: int = typer.Option(0, help="Random seed for --sample")) -> None:
    """Parse + chunk only (no API calls) and print chunk statistics."""
    import random
    from statistics import mean, median

    from rich.panel import Panel

    from ragforreal.ingest.pipeline import build_chunks

    cfg, s = ExperimentConfig.load(config), get_settings()
    cs = build_chunks(cfg, s)
    toks = [c.n_tokens for c in cs]
    console.print(f"{len(cs):,} chunks from {len({c.doc_name for c in cs})} docs · tokens: "
                  f"total {sum(toks):,}, mean {mean(toks):.0f}, median {median(toks):.0f}, "
                  f"min {min(toks)}, max {max(toks)}")
    n_tables = sum("|---" in c.text or "| ---" in c.text for c in cs)
    if n_tables:
        console.print(f"{n_tables:,} chunks contain a Markdown table")
    console.print(f"Index name: [bold]{cfg.index_name(s.corpus)}[/]")
    for c in random.Random(seed).sample(cs, min(sample, len(cs))):
        title = (f"{c.id} · pages {c.page_start + 1}-{c.page_end + 1} · {c.n_tokens} tokens"
                 + (f" · {c.metadata['section']}" if c.metadata.get("section") else ""))
        console.print(Panel(c.text[:1500], title=title, title_align="left"))


@app.command()
def ingest(config: Path = ConfigOpt,
           force: bool = typer.Option(False, help="Rebuild even if the index exists")) -> None:
    """Parse, chunk, embed and index the corpus for an experiment."""
    from ragforreal.ingest.pipeline import ingest as run_ingest

    run_ingest(ExperimentConfig.load(config), get_settings(), force=force)


@app.command("eval")
def evaluate(
    config: Path = ConfigOpt,
    generate: bool = typer.Option(False, "--generate", "-g",
                                  help="Also generate answers and LLM-judge them"),
    limit: int | None = typer.Option(None, help="Only the first N questions"),
    workers: int = typer.Option(4, help="Parallel generation workers"),
) -> None:
    """Evaluate an experiment on FinanceBench (retrieval metrics; --generate for answers)."""
    from ragforreal.eval.runner import run_eval

    run_eval(ExperimentConfig.load(config), get_settings(), generate=generate, limit=limit,
             workers=workers)


@app.command()
def ask(question: str, config: Path = ConfigOpt) -> None:
    """Answer one question with citations."""
    from ragforreal.generation.answer import generate_answer
    from ragforreal.retrieval.retriever import Retriever

    cfg = ExperimentConfig.load(config)
    chunks = Retriever(cfg, get_settings()).retrieve(question)
    ans = generate_answer(question, chunks, cfg.generation)
    console.print(ans.text)
    console.print("\n[bold]Sources[/]")
    for i, c in enumerate(chunks, 1):
        mark = "*" if i in ans.citations else " "
        console.print(f"{mark}[{i}] {c.doc_name} p.{c.page_start + 1}-{c.page_end + 1} "
                      f"(score {c.score:.3f})")


@app.command()
def indexes() -> None:
    """List vector indexes in the database."""
    from ragforreal.store.pgvector import PgVectorStore

    rows = PgVectorStore(get_settings().database_url).list_indexes()
    table = Table("name", "corpus", "status", "chunks", "dims", "created")
    for r in rows:
        table.add_row(r["name"], r["corpus"], r["status"], str(r["n_chunks"]),
                      str(r["dimensions"]), r["created_at"].strftime("%Y-%m-%d %H:%M"))
    console.print(table)


@app.command()
def leaderboard() -> None:
    """Regenerate results/leaderboard.md from all saved runs."""
    from ragforreal.eval.runner import write_leaderboard

    s = get_settings()
    write_leaderboard(s.results_dir)
    console.print((s.results_dir / "leaderboard.md").read_text())


if __name__ == "__main__":
    app()
