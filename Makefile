CONFIG ?= configs/p1_baseline.yaml
RAG = .venv/bin/rag

.PHONY: setup fetch parse-docling chunks ingest eval eval-gen phase2 test lint
setup:     ## create venv and install
	python3 -m venv .venv && .venv/bin/pip install -e '.[dev,modal]'
fetch:     ## download FinanceBench questions + PDFs
	$(RAG) fetch
parse-docling:  ## Docling on Modal GPUs -> .cache/parsed/docling
	.venv/bin/modal run modal_apps/parse_docling.py
chunks:    ## parse + chunk stats, no API calls
	$(RAG) chunks -c $(CONFIG)
ingest:    ## embed + index
	$(RAG) ingest -c $(CONFIG)
eval:      ## retrieval metrics (cheap, no LLM)
	$(RAG) eval -c $(CONFIG)
eval-gen:  ## retrieval + answers + LLM judge
	$(RAG) eval -c $(CONFIG) --generate
phase2:    ## index + eval every Phase 2 experiment (run parse-docling first)
	for c in configs/p2*.yaml; do $(RAG) ingest -c $$c && $(RAG) eval -c $$c || exit 1; done
	$(RAG) leaderboard
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
