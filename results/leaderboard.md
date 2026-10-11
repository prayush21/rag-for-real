# Leaderboard

Latest run of each experiment. Retrieval metrics are scored against FinanceBench gold evidence pages; accuracy is LLM-judged against the gold answer.

| Corpus | Experiment | N | Hit@1 | Hit@5 | Hit@10 | Page recall@10 | MRR | Ctx tokens | Accuracy | Refusals | $/question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | p1_baseline | 74 | 33.8% | 67.6% | 75.7% | 74.3% | 0.476 | 5,119 | 63.5% | 14.9% | $0.0016 |
| dev | p2d_pypdf_ctx | 74 | 33.8% | 68.9% | 79.7% | 76.1% | 0.485 | 5,239 | 67.6% | 16.2% | $0.0016 |
| dev | p2c_docling_structure_ctx | 74 | 32.4% | 73.0% | 79.7% | 74.3% | 0.482 | 3,311 | 64.9% | 17.6% | $0.0013 |
| dev | p2b_docling_structure | 74 | 32.4% | 63.5% | 73.0% | 72.3% | 0.457 | 3,263 | 60.8% | 20.3% | $0.0012 |
| dev | p2a_docling_fixed | 74 | 31.1% | 62.2% | 78.4% | 74.1% | 0.457 | 5,116 | 67.6% | 17.6% | $0.0015 |
