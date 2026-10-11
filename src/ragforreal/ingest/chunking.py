"""Chunking strategies. Every chunk keeps the page span it came from, which is what lets us
score retrieval against FinanceBench's gold evidence pages and cite pages in answers.

* ``fixed_tokens`` - sliding token window over the whole document. Cuts through sentences,
  tables and sections without noticing. The Phase 1 baseline.
* ``structure`` - follows the document's own structure (needs a structured parser like
  Docling): a chunk never spans two sections unless the first is tiny, paragraphs are never
  split unless they alone exceed the chunk size, and every table is its own chunk (split by
  rows with the header repeated when too big), prefixed by the heading that introduces it.

Either can add a *contextual header* - "[3M | 10k | 2018 | Item 8 > Cash Flows]" - so a chunk
that just says "Purchases of PP&E (1,577)" is still findable by "3M 2018 capex".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import tiktoken

from ragforreal.config import ChunkingConfig
from ragforreal.ingest.parse import Element

# cl100k is not Voyage's tokenizer, but it is a close, fast proxy for sizing chunks.
_ENC = tiktoken.get_encoding("cl100k_base")

MAX_SECTION_DEPTH = 3  # headings kept in the section path


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    doc_name: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    n_tokens: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_name}:{self.chunk_index}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


# ---------------------------------------------------------------- fixed_tokens


def _token_windows(text_pages: list[tuple[str, int]], size: int, overlap: int
                   ) -> list[tuple[str, int, int, int]]:
    """Sliding window over concatenated texts -> (text, page_start, page_end, n_tokens)."""
    tokens: list[int] = []
    token_page: list[int] = []  # page index of every token
    for text, page in text_pages:
        toks = _ENC.encode(text + "\n\n", disallowed_special=())
        tokens.extend(toks)
        token_page.extend([page] * len(toks))

    out = []
    stride = size - overlap
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        text = _ENC.decode(tokens[start:end]).strip()
        if text:
            out.append((text, token_page[start], token_page[end - 1], end - start))
        if end == len(tokens):
            break
        start += stride
    return out


def chunk_fixed_tokens(elements: list[Element], size: int, overlap: int) -> list[Chunk]:
    doc = elements[0].doc_name
    windows = _token_windows([(e.text, e.page) for e in elements], size, overlap)
    return [Chunk(doc, i, text, ps, pe, n) for i, (text, ps, pe, n) in enumerate(windows)]


# ---------------------------------------------------------------- structure


def split_table(markdown: str, max_tokens: int) -> list[str]:
    """Split a Markdown table by rows, repeating the caption/header lines in every piece."""
    if count_tokens(markdown) <= max_tokens:
        return [markdown]
    lines = markdown.split("\n")
    sep = next((i for i, ln in enumerate(lines) if ln.startswith("|") and set(ln) <= set("|-: ")),
               None)
    if sep is None:  # not a recognisable Markdown table: fall back to token windows
        return [t for t, *_ in _token_windows([(markdown, 0)], max_tokens, 0)]
    header, rows = lines[: sep + 1], lines[sep + 1 :]
    header_tokens = count_tokens("\n".join(header))
    pieces, current, current_tokens = [], [], header_tokens
    for row in rows:
        n = count_tokens(row) + 1
        if current and current_tokens + n > max_tokens:
            pieces.append("\n".join(header + current))
            current, current_tokens = [], header_tokens
        current.append(row)
        current_tokens += n
    if current:
        pieces.append("\n".join(header + current))
    return pieces


@dataclass
class _Draft:
    text: str
    page_start: int
    page_end: int
    section: str


def chunk_structure(elements: list[Element], cfg: ChunkingConfig) -> list[Chunk]:
    drafts: list[_Draft] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)
    buf: list[Element] = []
    buf_tokens = 0
    buf_section = ""

    def section_path() -> str:
        return " > ".join(t for _, t in stack[-MAX_SECTION_DEPTH:])

    def emit(text: str, first: int, last: int, section: str) -> None:
        drafts.append(_Draft(text, first, last, section))

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            emit("\n\n".join(e.text for e in buf), buf[0].page,
                 max(e.last_page for e in buf), buf_section)
        buf, buf_tokens = [], 0

    def push(el: Element, n: int) -> None:
        nonlocal buf_tokens, buf_section
        if not buf:
            buf_section = section_path()
        buf.append(el)
        buf_tokens += n

    for el in elements:
        n = count_tokens(el.text)
        if el.type == "heading":
            # A new section closes the current chunk, unless that chunk is too small to stand
            # alone (e.g. just a heading) - then it rolls forward into the new section.
            if buf_tokens >= cfg.min_chunk_tokens:
                flush()
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.text))
            push(el, n)
        elif el.type == "table":
            # Tables are standalone chunks. A small pending buffer (typically the heading /
            # caption introducing the table) is kept with the table instead of orphaned.
            intro, intro_pages = "", (el.page, el.last_page)
            if buf and buf_tokens < cfg.min_chunk_tokens:
                intro = "\n\n".join(e.text for e in buf) + "\n\n"
                intro_pages = (buf[0].page, max(el.last_page, *(e.last_page for e in buf)))
                buf, buf_tokens = [], 0
            flush()
            for i, piece in enumerate(split_table(el.text, cfg.max_table_tokens)):
                text = (intro + piece) if i == 0 else piece
                emit(text, intro_pages[0] if i == 0 else el.page, intro_pages[1], section_path())
        elif n > cfg.chunk_size:
            # A single oversized paragraph: the only case where we cut mid-text. As with tables,
            # a small pending buffer (its heading) goes with the first window, not on its own.
            intro, first_page = "", el.page
            if buf and buf_tokens < cfg.min_chunk_tokens:
                intro = "\n\n".join(e.text for e in buf) + "\n\n"
                first_page = buf[0].page
                buf, buf_tokens = [], 0
            flush()
            windows = _token_windows([(el.text, el.page)], cfg.chunk_size, 0)
            for i, (text, *_) in enumerate(windows):
                emit(intro + text if i == 0 else text, first_page if i == 0 else el.page,
                     el.last_page, section_path())
        else:
            if buf_tokens + n > cfg.chunk_size:
                flush()
            push(el, n)
    flush()

    doc = elements[0].doc_name
    return [
        Chunk(doc, i, d.text, d.page_start, d.page_end, count_tokens(d.text),
              {"section": d.section} if d.section else {})
        for i, d in enumerate(drafts)
    ]


# ---------------------------------------------------------------- entry point


def context_header(doc_meta: dict, section: str | None) -> str:
    parts = [str(doc_meta[k]) for k in ("company", "doc_type", "doc_period") if doc_meta.get(k)]
    if section:
        parts.append(section)
    return f"[{' | '.join(parts)}]" if parts else ""


def chunk_document(elements: list[Element], cfg: ChunkingConfig,
                   doc_meta: dict | None = None) -> list[Chunk]:
    if not elements:
        return []
    if cfg.strategy == "fixed_tokens":
        chunks = chunk_fixed_tokens(elements, cfg.chunk_size, cfg.chunk_overlap)
    elif cfg.strategy == "structure":
        chunks = chunk_structure(elements, cfg)
    else:
        raise ValueError(f"unknown chunking strategy: {cfg.strategy}")

    doc_meta = doc_meta or {}
    for c in chunks:
        c.metadata = {**doc_meta, **c.metadata}
        if cfg.context_header:
            header = context_header(doc_meta, c.metadata.get("section"))
            if header:
                c.text = f"{header}\n\n{c.text}"
                c.n_tokens = count_tokens(c.text)
    return chunks
