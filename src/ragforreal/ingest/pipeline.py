"""parse -> chunk -> embed -> store, for one experiment config."""

from __future__ import annotations

import json

from rich.console import Console

from ragforreal.config import ExperimentConfig, Settings
from ragforreal.data import financebench as fb
from ragforreal.ingest.chunking import Chunk, chunk_document
from ragforreal.ingest.embed import Embedder
from ragforreal.ingest.parse import parse_corpus
from ragforreal.store.pgvector import PgVectorStore

console = Console()


def _doc_metadata(settings: Settings) -> dict[str, dict]:
    path = settings.data_dir / "financebench" / fb.DOC_INFO_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        r["doc_name"]: {"company": r["company"], "doc_type": r["doc_type"],
                        "doc_period": r["doc_period"], "sector": r.get("gics_sector")}
        for r in rows
    }


def build_chunks(cfg: ExperimentConfig, settings: Settings) -> list[Chunk]:
    doc_names = fb.corpus_doc_names(settings.data_dir, settings.corpus)
    pdfs = [fb.pdf_dir(settings.data_dir) / f"{d}.pdf" for d in doc_names]
    missing = [p.name for p in pdfs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} PDFs missing - run `rag fetch` first")

    docs = parse_corpus(pdfs, settings.cache_dir, cfg.ingest.parser)
    meta = _doc_metadata(settings)
    chunks: list[Chunk] = []
    for name, elements in docs.items():
        chunks.extend(chunk_document(elements, cfg.ingest.chunking, meta.get(name, {})))
    return chunks


def ingest(cfg: ExperimentConfig, settings: Settings, force: bool = False) -> str:
    index = cfg.index_name(settings.corpus)
    store = PgVectorStore(settings.database_url)
    if store.index_status(index) == "ready" and not force:
        console.print(f"[green]Index {index} already built[/] (use --force to rebuild)")
        return index

    chunks = build_chunks(cfg, settings)
    n_tokens = sum(c.n_tokens for c in chunks)
    console.print(f"{len(chunks):,} chunks, ~{n_tokens:,} tokens -> index [bold]{index}[/]")

    embedder = Embedder(cfg.ingest.embedding, settings.cache_dir,
                        settings.embed_rpm, settings.embed_tpm)
    vectors = embedder.embed_documents([c.text for c in chunks])
    console.print(f"Embedding tokens billed this run: {embedder.tokens_billed:,} "
                  "(the rest came from cache)")

    store.create_index(index, settings.corpus, cfg.ingest.model_dump(),
                       cfg.ingest.embedding.dimensions)
    store.insert_chunks(index, chunks, vectors)
    store.finalize_index(index)
    console.print(f"[green]Index {index} ready[/]")
    return index
