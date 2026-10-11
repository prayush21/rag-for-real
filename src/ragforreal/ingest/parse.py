"""PDF -> a flat, ordered list of document elements.

* ``pypdf`` (Phase 1): one ``text`` element per page. Fast and layout-blind: tables come out
  as a stream of cells with no structure.
* ``docling`` (Phase 2): layout-aware elements - headings, paragraphs, list items, and tables
  rendered as Markdown. Docling needs torch + vision models, so it runs on a Modal GPU
  (``modal run modal_apps/parse_docling.py``) and writes its output into the parse cache; this
  module only reads it.

Parsed output is cached as JSON(L) so re-chunking never re-parses.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader
from rich.progress import track


@dataclass(frozen=True)
class Element:
    doc_name: str
    page: int  # 0-based page the element starts on
    text: str
    type: str = "text"  # text | heading | list_item | table
    page_end: int | None = None  # set when an element (e.g. a table) spans pages
    level: int = 0  # heading depth; 0 = top level (e.g. "Item 8.")

    @property
    def last_page(self) -> int:
        return self.page if self.page_end is None else self.page_end


def clean_text(text: str) -> str:
    # Some PDFs yield NUL bytes, which Postgres TEXT columns reject.
    return text.replace("\x00", "")


def parse_pdf_pypdf(path: Path) -> list[Element]:
    reader = PdfReader(path)
    return [
        Element(doc_name=path.stem, page=i, text=clean_text(page.extract_text() or ""))
        for i, page in enumerate(reader.pages)
    ]


_SEPARATOR_CELL = re.compile(r":?-+:?")


def compact_markdown_table(md: str) -> str:
    """Strip Docling's column-alignment padding from Markdown tables (~15% fewer table
    tokens, identical content). Non-table lines (e.g. a caption) pass through unchanged."""
    out = []
    for line in md.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and all(_SEPARATOR_CELL.fullmatch(c) for c in cells if c):
                cells = ["---"] * len(cells)
            line = "| " + " | ".join(cells) + " |"
        out.append(line)
    return "\n".join(out)


def docling_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "parsed" / "docling"


def load_docling(name: str, cache_dir: Path) -> list[Element]:
    path = docling_cache_dir(cache_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No Docling output for {name}. Run `modal run modal_apps/parse_docling.py` first."
        )
    data = json.loads(path.read_text())
    elements = []
    for e in data["elements"]:
        text = clean_text(e["text"])
        if e["type"] == "table":
            text = compact_markdown_table(text)
        elements.append(Element(doc_name=name, **{**e, "text": text}))
    return elements


def parse_corpus(pdf_paths: list[Path], cache_dir: Path, parser: str = "pypdf"
                 ) -> dict[str, list[Element]]:
    if parser == "docling":
        return {p.stem: load_docling(p.stem, cache_dir) for p in pdf_paths}
    if parser != "pypdf":
        raise ValueError(f"unknown parser: {parser}")

    out_dir = cache_dir / "parsed" / parser
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: dict[str, list[Element]] = {}
    for path in track(pdf_paths, description=f"Parsing ({parser})"):
        cached = out_dir / f"{path.stem}.jsonl"
        if cached.exists():
            pages = [Element(**json.loads(line)) for line in cached.read_text().splitlines()]
        else:
            pages = parse_pdf_pypdf(path)
            cached.write_text("\n".join(json.dumps(asdict(p)) for p in pages))
        docs[path.stem] = pages
    return docs
