"""Parse the corpus PDFs with Docling on Modal GPUs, one container per document in parallel.

    modal run modal_apps/parse_docling.py                 # dev corpus, skip already-parsed docs
    modal run modal_apps/parse_docling.py --corpus full
    modal run modal_apps/parse_docling.py --force --limit 1   # re-parse one doc (smoke test)

Writes .cache/parsed/docling/<doc>.json (elements for chunking) and <doc>.md (for reading).
Docling's models are baked into the image at build time, so containers start warm and the
models are never re-downloaded. Only PDFs go up and a few MB of JSON comes back.
"""

from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path

import modal

DOCLING_VERSION = "2.135.0"
GPU = "L4"
MAX_CONTAINERS = 10  # Modal Starter plan GPU concurrency limit


def _download_models() -> None:
    from docling.utils.model_downloader import download_models

    download_models(with_code_formula=False, with_picture_classifier=False, with_rapidocr=False)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(f"docling=={DOCLING_VERSION}")
    .run_function(_download_models)
    .add_local_python_source("ragforreal")
)

app = modal.App("rag-for-real-docling", image=image)


@app.cls(gpu=GPU, timeout=30 * 60, max_containers=MAX_CONTAINERS, scaledown_window=60)
class DoclingParser:
    @modal.enter()
    def load(self) -> None:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.datamodel.settings import settings
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(
            artifacts_path=settings.cache_dir / "models",
            do_ocr=False,  # SEC filings have a text layer; OCR only adds time and errors
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE, do_cell_matching=True),
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    @modal.method()
    def parse(self, name: str, pdf: bytes) -> dict:
        from docling.datamodel.base_models import DocumentStream

        from ragforreal.ingest.docling_elements import docling_to_elements

        t0 = time.perf_counter()
        result = self.converter.convert(DocumentStream(name=f"{name}.pdf", stream=BytesIO(pdf)))
        doc = result.document
        return {
            "doc_name": name,
            "docling_version": DOCLING_VERSION,
            "n_pages": doc.num_pages(),
            "seconds": round(time.perf_counter() - t0, 1),
            "elements": docling_to_elements(doc),
            "markdown": doc.export_to_markdown(),
        }


@app.local_entrypoint()
def main(corpus: str = "dev", force: bool = False, limit: int = 0) -> None:
    from ragforreal.config import get_settings
    from ragforreal.data import financebench as fb
    from ragforreal.ingest.parse import docling_cache_dir

    s = get_settings()
    out = docling_cache_dir(s.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in fb.corpus_doc_names(s.data_dir, corpus)
             if force or not (out / f"{n}.json").exists()]
    if limit:
        names = names[:limit]
    if not names:
        print("Nothing to parse - all docs already in cache (use --force to re-parse).")
        return

    pdfs = [fb.pdf_dir(s.data_dir) / f"{n}.pdf" for n in names]
    missing = [p.name for p in pdfs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing PDFs {missing} - run `rag fetch` first")

    print(f"Parsing {len(names)} docs on {GPU} (up to {MAX_CONTAINERS} in parallel)...")
    t0, pages, failures = time.perf_counter(), 0, []
    args = [(n, p.read_bytes()) for n, p in zip(names, pdfs, strict=True)]
    for name, res in zip(names, DoclingParser().parse.starmap(args, return_exceptions=True),
                         strict=True):
        if isinstance(res, BaseException):
            failures.append(name)
            print(f"  FAILED {name}: {res!r}")
            continue
        (out / f"{name}.md").write_text(res.pop("markdown"))
        (out / f"{name}.json").write_text(json.dumps(res))
        pages += res["n_pages"]
        n_tables = sum(e["type"] == "table" for e in res["elements"])
        print(f"  {name}: {res['n_pages']} pages, {len(res['elements'])} elements, "
              f"{n_tables} tables, {res['seconds']}s")

    print(f"Done: {len(names) - len(failures)}/{len(names)} docs, {pages} pages in "
          f"{time.perf_counter() - t0:.0f}s wall clock. Output: {Path(out)}")
    if failures:
        raise SystemExit(f"{len(failures)} failed: {failures}")
