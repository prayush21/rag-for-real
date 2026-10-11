# Experiment log

One change per experiment, measured against the previous best on the same corpus. Write the
hypothesis *before* running, then the result and what you learned. A technique that doesn't
move the numbers is a finding too.

Corpus for all rows below: `dev` (20 docs, 74 questions, 3,048 pages).

---

## Phase 1 — baseline

**`p1_baseline`** — pypdf text, fixed 512-token windows (64 overlap), voyage-4, dense top-10.

- Purpose: the reference point. Deliberately naive, but done properly (asymmetric query/doc
  embeddings, HNSW index, cached embeddings) so later gains aren't just bug fixes.
- Expect weak spots on `metrics-generated` questions: their evidence is mostly financial-
  statement tables, which pypdf flattens into a stream of numbers with no row/column structure.
- Look at: `hit@k` by `question_type`; `doc_hit@10` vs `hit@10` (right filing but wrong page
  means the failure is *within* the document); `rag chunks --sample 5` to see what the
  embedder actually sees.

Result (2026-10-08, retrieval only — answer accuracy pending an LLM with enough quota):

| hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | doc_hit@1 | MRR | context tokens |
|---|---|---|---|---|---|---|---|
| 33.8% | 55.4% | 67.6% | 75.7% | 89.2% | 90.5% | 0.476 | 5,119 |

hit@5 by type: metrics-generated **100%**, novel-generated 67.9%, domain-relevant 61.5%.

- **Hypothesis wrong.** Table questions were the *easiest*, not the hardest: they name the
  statement and line item ("...relying on the cash flow statement"), which lexically matches the
  flattened table text well even without structure.
- **The right filing is found almost always** (doc_hit@1 90.5%); misses are *within* the
  document. Conceptual questions ("does AMEX have an improving operating margin profile?")
  don't share vocabulary with the page that answers them.
- **Company confusion:** "Boeing's effective tax rate FY2022 vs FY2021" retrieves J&J, AMD and 3M
  tax-rate chunks — nothing in a mid-document chunk says whose filing it is. Direct target for
  contextual headers (p2c/p2d).
- **Ops finding:** search p50 was 730 ms, almost all TLS connection setup to Neon per query.
  Connection pooling brought it to ~160 ms.

Generation (Claude Haiku 5.5 answers + Haiku 5.5 judge, top-10 context, 2026-10-10):

| accuracy | incorrect | refused | cited | p50 latency | $/question |
|---|---|---|---|---|---|
| 63.5% | 21.6% | 14.9% | 100% | 5.9 s | $0.0016 |

- **Noise floor:** two identical runs gave the same accuracy but incorrect/refused moved by
  ~3–4 points (17.6/17.6 → 21.6/14.9). With n=74, one question is 1.35 points — treat
  generation deltas under ~5 points as noise; retrieval metrics are deterministic.
- **Retrieval isn't the only bottleneck.** When the gold page *was* in the top 10 (56 qs), 10
  answers were still wrong — mostly metric-definition disagreements (quick ratio components,
  average vs. ending inventory for turnover), not missing evidence. That's a prompt / domain
  problem for a later phase.
- **Refusals are mostly honest.** When retrieval missed the gold page, the model refused 7 of
  18 times rather than guessing — the behavior we want in finance.

---

## Phase 2 — ingestion

**Docling parse (Modal, L4, 2026-10-10):** 20 docs / 3,048 pages in ~4.5 min wall clock
(10 containers in parallel, ~32 GPU-minutes, ≈ $0.50). 1,860 tables extracted. All 93 gold
evidence passages land on the correct page (≥80% word overlap) — page provenance survives the
parser swap, so retrieval metrics stay comparable with p1.

Fixes found by looking at the output before embedding it:
- Docling pads Markdown table cells for alignment: compacting them saves ~15% of table tokens
  (~5% of the corpus) with identical content.
- Headings directly before an oversized paragraph were emitted as standalone one-line chunks;
  they now ride along with the paragraph's first window (as table headings already did).

Chunk shapes: p1/p2a/p2d ≈ 5,000 chunks × 512 tokens; p2b/p2c 8,027 chunks, mean 281 tokens
(p2c 313 with headers), 1,791 contain a table.

**`p2a_docling_fixed`** — only the parser changes: Docling (layout model + TableFormer), tables
as Markdown, page headers/footers dropped. Same fixed 512-token windows.

- Hypothesis: better table text alone lifts `hit@k` on `metrics-generated` questions; little
  change on `domain-relevant` / `novel-generated` prose questions.

**`p2b_docling_structure`** — p2a + structure-aware chunking: split on headings, never cut a
paragraph, every table its own chunk with its intro heading attached.

- Hypothesis: higher `hit@1`/`MRR` (chunks are about one thing, so they rank more sharply), at
  the cost of more, smaller chunks. Watch `context tokens`: smaller chunks deliver less text per
  top-10, which can *lower* hit@10 even if precision rises.

**`p2c_docling_structure_ctx`** — p2b + contextual headers
(`[3M | 10k | 2018 | Item 8 > Consolidated Statement of Cash Flows]`).

- Hypothesis: the biggest single jump in `doc_hit@k`. A table chunk with no company name in it
  is nearly unfindable in a 20-company index: every 10-K has a "Purchases of property, plant
  and equipment" row. The header tells the embedder whose table it is.

**`p2d_pypdf_ctx`** — ablation: p1 + document-level headers, no Docling.

- Purpose: separates "headers help" from "structure helps". If p2d ≈ p2c, the parser was not
  what mattered; if p2c ≫ p2d, structure and section paths earn their cost.

Results (2026-10-10, dev, n=74; generation = Claude Haiku 5.5 answer + judge):

| experiment | hit@1 | hit@5 | hit@10 | MRR | right-doc share@10 | ctx tokens | accuracy | refused | $/q |
|---|---|---|---|---|---|---|---|---|---|
| p1 baseline | 33.8% | 67.6% | 75.7% | 0.476 | 76.4% | 5,119 | 63.5% | 14.9% | .0016 |
| p2d pypdf + headers | 33.8% | 68.9% | **79.7%** | **0.485** | **85.8%** | 5,239 | 67.6% | 16.2% | .0016 |
| p2a Docling, fixed | 31.1% | 62.2% | 78.4% | 0.457 | 76.5% | 5,116 | 67.6% | 17.6% | .0015 |
| p2b Docling, structure | 32.4% | 63.5% | 73.0% | 0.457 | 70.1% | 3,263 | 60.8% | 20.3% | .0012 |
| p2c Docling, structure + headers | 32.4% | **73.0%** | **79.7%** | 0.482 | 84.7% | **3,311** | 64.9% | 17.6% | **.0013** |

*right-doc share@10 = fraction of the top-10 chunks that come from a gold document.*

What we learned:
- **Contextual headers are the clear win.** Wrong-company chunks in the top 10 drop from 24% to
  14% (p1→p2d) and 30% to 15% (p2b→p2c). Cheapest technique of the phase: no parser, no model,
  ~2% more tokens.
- **Structure-aware chunking only works *with* headers.** Alone (p2b) it is the worst run:
  281-token chunks carry even less "whose filing is this" signal than 512-token ones (right-doc
  share 70%). With headers (p2c) it gives the best hit@5 (+5.4 pts) while sending **35% fewer
  context tokens** to the LLM (19% cheaper per question) for the same hit@10.
- **Docling alone moved nothing at hit@5** (p2a vs p1). Better tables don't matter until the chunk
  can be attributed to the right company; `metrics-generated` questions were already at ceiling.
- **Accuracy differences are within noise.** Every run lands at 61–68%, inside the ~5-point
  noise band measured in Phase 1. Retrieval gains this size don't yet show up in answers.
- **The metric has a blind spot.** Both "effective tax rate" questions "miss" in every run: the
  gold page is the income statement (rate computed from pre-tax income and tax expense), while
  retrieval finds the income-tax note, which states the rate directly. AMEX's answer was judged
  correct anyway. Page-level hit@k is a lower bound when an answer lives on more than one page.
- **Small n.** p1→p2c at hit@5 is +9 / −5 questions. Directionally solid (the right-doc share
  shift is large and consistent), but confirm on the full 150-question corpus before trusting
  the exact deltas.

**Decision:** carry **p2c** (Docling + structure + headers) forward as the Phase 3 base: best
hit@5, tied-best hit@10, and the smallest, cheapest context.

---

## Up next (Phase 3)

Hybrid search (BM25 via Postgres full-text + dense, fused with RRF) and a reranker
(Voyage `rerank-2.5`). Exact terms like "PP&E", "Item 7A" or a ticker are where BM25 beats
embeddings.
