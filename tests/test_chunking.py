import pytest

from ragforreal.config import ChunkingConfig
from ragforreal.ingest.chunking import chunk_document, chunk_fixed_tokens, count_tokens
from ragforreal.ingest.parse import Element


def _pages(n: int, words_per_page: int = 300) -> list[Element]:
    return [
        Element(doc_name="ACME_2023_10K", page=i,
             text=" ".join(f"p{i}w{j}" for j in range(words_per_page)))
        for i in range(n)
    ]


def test_chunks_respect_size_and_cover_all_pages():
    pages = _pages(5)
    chunks = chunk_fixed_tokens(pages, size=200, overlap=20)
    assert all(c.n_tokens <= 200 for c in chunks)
    assert chunks[0].page_start == 0
    assert chunks[-1].page_end == 4
    covered = {p for c in chunks for p in range(c.page_start, c.page_end + 1)}
    assert covered == set(range(5))


def test_page_spans_are_monotonic_and_ids_unique():
    chunks = chunk_fixed_tokens(_pages(8), size=128, overlap=16)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.page_start <= b.page_start
        assert a.page_start <= a.page_end


def test_overlap_repeats_tokens_between_neighbours():
    chunks = chunk_fixed_tokens(_pages(3), size=100, overlap=30)
    tail = chunks[0].text.split()[-5:]
    assert all(w in chunks[1].text for w in tail)


def test_short_document_is_single_chunk():
    pages = [Element(doc_name="X", page=0, text="Revenue was $10 million.")]
    chunks = chunk_fixed_tokens(pages, size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Revenue was $10 million."
    assert chunks[0].n_tokens == count_tokens("Revenue was $10 million.\n\n")


def test_empty_document():
    assert chunk_document([], ChunkingConfig()) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------- structure strategy

STRUCT = ChunkingConfig(strategy="structure", chunk_size=60, max_table_tokens=80,
                        min_chunk_tokens=10)


def _el(type_, text, page, level=0, page_end=None):
    return Element("ACME_2023_10K", page, text, type_, page_end, level)


def _para(i, words=12):
    return " ".join(f"s{i}w{j}" for j in range(words))


def test_structure_splits_on_sections_and_tracks_path():
    els = [
        _el("heading", "Item 7. MD&A", 0, level=0),
        _el("text", _para(1), 0),
        _el("heading", "Liquidity", 1, level=1),
        _el("text", _para(2), 1),
        _el("heading", "Item 8. Financial Statements", 2, level=0),
        _el("text", _para(3), 2),
    ]
    chunks = chunk_document(els, STRUCT)
    assert [c.metadata["section"] for c in chunks] == [
        "Item 7. MD&A", "Item 7. MD&A > Liquidity", "Item 8. Financial Statements"]
    assert [(c.page_start, c.page_end) for c in chunks] == [(0, 0), (1, 1), (2, 2)]
    assert chunks[1].text.startswith("Liquidity")


def test_structure_never_splits_a_paragraph_that_fits():
    els = [_el("text", _para(i, words=15), i) for i in range(6)]
    chunks = chunk_document(els, STRUCT)
    assert len(chunks) > 1
    for i in range(6):
        assert sum(_para(i, words=15) in c.text for c in chunks) == 1


def test_table_is_standalone_and_keeps_its_intro_heading():
    table = "| Item | 2023 |\n|---|---|\n| Revenue | 10 |\n| Capex | (2) |"
    els = [
        _el("heading", "Item 8. Financial Statements", 4, level=0),
        _el("text", _para(1, words=30), 4),
        _el("heading", "Cash Flows", 5, level=1),
        _el("table", table, 5, page_end=6),
        _el("text", _para(2), 6),
    ]
    chunks = chunk_document(els, STRUCT)
    tbl = next(c for c in chunks if "| Capex |" in c.text)
    assert tbl.text.startswith("Cash Flows\n\n| Item |")
    assert (tbl.page_start, tbl.page_end) == (5, 6)
    assert tbl.metadata["section"] == "Item 8. Financial Statements > Cash Flows"
    assert _para(2) not in tbl.text


def test_big_table_split_by_rows_with_header_repeated():
    from ragforreal.ingest.chunking import split_table

    rows = "\n".join(f"| line item {i} | {i * 100} |" for i in range(40))
    md = f"| Item | 2023 |\n|---|---|\n{rows}"
    pieces = split_table(md, max_tokens=80)
    assert len(pieces) > 1
    assert all(p.startswith("| Item | 2023 |\n|---|---|\n") for p in pieces)
    assert all(count_tokens(p) <= 80 for p in pieces)
    body = [ln for p in pieces for ln in p.split("\n")[2:]]
    assert body == rows.split("\n")  # every row exactly once, in order


def test_oversized_paragraph_is_windowed():
    chunks = chunk_document([_el("text", _para(1, words=200), 3)], STRUCT)
    assert len(chunks) > 1
    assert all(c.n_tokens <= STRUCT.chunk_size for c in chunks)


def test_context_header_prepended_and_counted():
    cfg = ChunkingConfig(strategy="structure", chunk_size=60, min_chunk_tokens=10,
                         context_header=True)
    meta = {"company": "3M", "doc_type": "10k", "doc_period": 2018}
    els = [_el("heading", "Item 8. Financial Statements", 0), _el("text", _para(1), 0)]
    [chunk] = chunk_document(els, cfg, meta)
    assert chunk.text.startswith("[3M | 10k | 2018 | Item 8. Financial Statements]\n\n")
    assert chunk.n_tokens == count_tokens(chunk.text)
    assert chunk.metadata["company"] == "3M"

    fixed = chunk_document([_el("text", "x", 0)], ChunkingConfig(context_header=True), meta)
    assert fixed[0].text == "[3M | 10k | 2018]\n\nx"


def test_compact_markdown_table():
    from ragforreal.ingest.parse import compact_markdown_table

    md = ("Cash flows\n\n| (Millions)      | 2018    |\n|-----------------|---------|\n"
          "| Purchases of PP&E | (1,577) |")
    assert compact_markdown_table(md) == (
        "Cash flows\n\n| (Millions) | 2018 |\n| --- | --- |\n| Purchases of PP&E | (1,577) |")


def test_heading_stays_with_oversized_paragraph():
    els = [_el("heading", "Product Liability Litigation", 2, level=1),
           _el("text", _para(1, words=200), 2)]
    chunks = chunk_document(els, STRUCT)
    assert chunks[0].text.startswith("Product Liability Litigation\n\n")
    assert not any(c.text == "Product Liability Litigation" for c in chunks)
