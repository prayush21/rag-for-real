# rag-for-real

A production-style RAG system over SEC filings (10-Ks, 10-Qs, earnings reports), built one
technique at a time, with every change **measured** against the
[FinanceBench](https://github.com/patronus-ai/financebench) benchmark: expert-written questions
with gold answers and gold evidence pages.

The point is not "a chatbot over PDFs". It's the engineering loop: change one thing → re-run the
eval → keep it only if the numbers say so. See [`results/leaderboard.md`](results/leaderboard.md).

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | **Postgres + pgvector** on [Neon](https://neon.com) (free tier) | One database for vectors, metadata and full-text search. Versioned indexes. |
| Embeddings | **Voyage `voyage-4`** via LiteLLM (200M free tokens) | Provider is a config string; swap without code changes. |
| LLM (answers + judge) | Gemini or Claude via **LiteLLM** | Same: provider-agnostic. |
| Heavy parsing (Phase 2) | **Docling on Modal** (serverless GPU) | Keeps torch/models off the laptop. |
| API | **FastAPI** | `/query` returns answer + cited sources. |
| Reranking (Phase 3) | **Voyage `rerank-2.5`** via LiteLLM | Cross-encoder over the dense top-50; scores cached in SQLite. |
| Eval | Custom retrieval metrics + LLM judge | Retrieval scoring needs no LLM → cheap, deterministic. |
| Tests / CI | pytest (embedded Postgres via `pgserver`), ruff, GitHub Actions | |

## Setup

```bash
make setup                      # venv + deps (~550 MB, no torch)
cp .env.example .env            # then fill in DATABASE_URL, VOYAGE_API_KEY, GEMINI/ANTHROPIC key
make fetch                      # FinanceBench questions + PDFs (dev corpus: 20 docs, 37 MB)
```

Accounts you need (all free): Neon (copy the connection string), Voyage AI (API key; add a card
to lift the 3 RPM / 10K TPM trial limit — usage stays free within 200M tokens), and Gemini
(AI Studio) or Anthropic for the LLM. Without a Voyage card, set `EMBED_RPM=3` and
`EMBED_TPM=10000` in `.env`; indexing then takes hours instead of minutes.

## Usage

```bash
rag chunks  -c configs/p1_baseline.yaml   # parse + chunk stats, no API calls
rag ingest  -c configs/p1_baseline.yaml   # embed + build the index (~2.6M tokens for dev)
rag eval    -c configs/p1_baseline.yaml   # retrieval metrics only (cheap)
rag eval    -c configs/p1_baseline.yaml --generate   # + answers + LLM judge
rag ask "What was 3M's FY2022 capital expenditure?"
rag indexes                               # list versioned indexes in Postgres
uvicorn ragforreal.api.main:app --reload  # POST /query {"question": "..."}
```

## Phase 2: better ingestion

```bash
pip install modal && modal setup          # once: browser login, token saved locally
make parse-docling                        # Docling on Modal L4 GPUs, ~$1 for the dev corpus
rag chunks -c configs/p2b_docling_structure.yaml --sample 3   # look at the chunks!
make phase2                               # ingest + eval p2a..p2d, then print the leaderboard
```

## Phase 3: retrieval

Every Phase 3 experiment reuses the p2c index (no re-embedding). Retrieval is a chain of
stages configured in YAML: company filter → candidates (dense | Postgres full-text | BM25 |
hybrid via RRF) → cross-encoder reranker → top k.

```bash
make phase3                               # retrieval eval of every p3* config
rag eval -c configs/p3b_rerank.yaml --generate --repeats 3   # accuracy with a noise band
```

The eval scores both the candidate list and the final ranking ("hit (pre-rerank)"), so a
reranker's ceiling and its gain are visible separately, and reports per-stage latency.

Hypotheses and what to look for in each experiment: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## How it's measured

FinanceBench gives, for each question, the document and **page** holding the evidence. A retrieved
chunk is a hit when it's from that document and its page span covers that page. Metrics:

- **hit@k** — any gold page in the top k chunks. **page_recall@k** — fraction of gold pages found
  (multi-page questions). **doc_hit@k** — right document at all. **MRR**.
- **ev_hit@k** — also counts a chunk from the gold document that contains most of the gold
  evidence *text* (filings repeat tables and passages on several pages).
- **context tokens** — tokens sent to the LLM. Bigger chunks inflate hit@k "for free", so always
  read hit@k next to this.
- With `--generate`: **accuracy / incorrect / refusal rate** (LLM judge vs. gold answer — a refusal
  is scored separately because in finance "I don't know" is far cheaper than a wrong number),
  citation rate, p50/p95 latency, $/question. `--repeats N` re-runs generation N times and
  reports the spread: identical configs move several points run to run at n=74.

Each run writes `results/runs/<experiment>/<timestamp>/` with the full config, git commit, and
per-question results, so every number is reproducible.

**Corpora.** `CORPUS=dev`: the 20 docs with the most questions (74 questions, 3,048 pages).
`CORPUS=full`: all 84 question docs (150 questions). All docs share one index, so other
companies' filings act as distractors.

## Design notes

- **Index versioning.** The index name is a hash of everything that changes stored vectors
  (parser, chunking, embedding model) + corpus. A `rag_indexes` registry records each config.
  Experiments that only change retrieval/generation reuse an index; old indexes stay queryable.
- **Embedding cache.** SQLite, keyed by (model, dims, input type, text). Re-runs cost zero tokens.
- **Rate limits.** Client-side RPM/TPM throttle + exponential backoff on 429s.
- **Asymmetric embeddings.** Queries and documents are embedded with Voyage's `input_type`.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Baseline: pypdf, fixed 512-token chunks, dense search, eval harness | ✅ built |
| 2 | Ingestion: Docling on Modal (tables, headings), structure-aware + contextual chunks, metadata | ✅ done — headers win; p2c is the new base |
| 3 | Retrieval: hybrid (BM25 + dense, RRF), reranking (Voyage rerank-2.5), metadata filters | ✅ dev done — reranking +12 pts accuracy; hybrid rejected; p3b is the new base |
| 4 | Query transformation: multi-query, HyDE, decomposition | |
| 5 | Agentic: LangGraph router → SQL / vector, self-correcting retrieval | |
| 6 | Production: Langfuse tracing, semantic cache, model routing, guardrails, eval gate in CI | |
| 7 | Stretch: embedding fine-tuning, GraphRAG, ColPali | |

FinanceBench data is CC-BY-NC-4.0 (Patronus AI).
