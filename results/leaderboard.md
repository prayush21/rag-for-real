# Leaderboard

Latest run of each experiment. Retrieval metrics are scored against FinanceBench gold evidence pages; accuracy is LLM-judged against the gold answer.

Ev hit@5 also counts chunks containing most of the gold evidence text (blank for runs before that metric existed). Accuracy ± is the max-min spread over repeated runs.

| Corpus | Experiment | N | Hit@1 | Hit@5 | Hit@10 | Ev hit@5 | MRR | Ctx tokens | Accuracy | Refusals | $/question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | p1_baseline | 74 | 33.8% | 67.6% | 75.7% | - | 0.476 | 5,119 | 63.5% | 14.9% | $0.0016 |
| dev | p2d_pypdf_ctx | 74 | 33.8% | 68.9% | 79.7% | - | 0.485 | 5,239 | 67.6% | 16.2% | $0.0016 |
| dev | p2b_docling_structure | 74 | 32.4% | 63.5% | 73.0% | - | 0.457 | 3,263 | 60.8% | 20.3% | $0.0012 |
| dev | p2a_docling_fixed | 74 | 31.1% | 62.2% | 78.4% | - | 0.457 | 5,116 | 67.6% | 17.6% | $0.0015 |
| dev | p3a_hybrid | 74 | 25.7% | 48.6% | 71.6% | 50.0% | 0.370 | 4,487 | - | - | - |
| dev | p3c_hybrid_rerank | 74 | 54.0% | 86.5% | 91.9% | 86.5% | 0.686 | 3,891 | - | - | - |
| dev | p3d_company_filter | 74 | 32.4% | 74.3% | 81.1% | 75.7% | 0.490 | 3,357 | - | - | - |
| dev | p3f_hybrid_bm25 | 74 | 18.9% | 48.6% | 64.9% | 51.3% | 0.337 | 3,232 | - | - | - |
| dev | p2c_docling_structure_ctx | 74 | 32.4% | 73.0% | 79.7% | 74.3% | 0.482 | 3,311 | 64.4% ±2.7% (x3) | 19.8% | $0.0013 |
| dev | p3b_rerank | 74 | 52.7% | 86.5% | 93.2% | 86.5% | 0.680 | 3,951 | 76.6% ±3.4% (x3) | 5.9% | $0.0014 |
| dev | p3g_rerank_filter | 74 | 52.7% | 87.8% | 94.6% | 87.8% | 0.683 | 4,008 | 73.9% ±1.4% (x3) | 7.2% | $0.0014 |
| dev | p3e_rerank_top3 | 74 | 52.7% | 86.5% | 93.2% | 86.5% | 0.680 | 1,218 | 69.8% ±0.7% (x3) | 18.0% | $0.0008 |
| dev | p3e_rerank_top5 | 74 | 52.7% | 86.5% | 93.2% | 86.5% | 0.680 | 2,006 | 73.4% ±1.4% (x3) | 10.4% | $0.0010 |
