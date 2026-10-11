"""Runs where docling-core is installed (CI installs it; skipped otherwise)."""

import pytest

from ragforreal.ingest.docling_elements import docling_to_elements, heading_level

doc_mod = pytest.importorskip("docling_core.types.doc")


def _prov(page_no: int):
    from docling_core.types.doc import BoundingBox, ProvenanceItem

    return ProvenanceItem(page_no=page_no, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _table_data():
    from docling_core.types.doc import TableCell, TableData

    cells = [
        TableCell(text=t, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                  start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=r == 0)
        for r, row in enumerate([["(Millions)", "2018"], ["Purchases of PP&E", "(1,577)"]])
        for c, t in enumerate(row)
    ]
    return TableData(num_rows=2, num_cols=2, table_cells=cells)


def test_docling_to_elements():
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    doc = DoclingDocument(name="3M_2018_10K")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Table of Contents", prov=_prov(1))
    doc.add_heading("Item 8. Financial Statements", level=1, prov=_prov(1))
    doc.add_heading("Consolidated Statement of Cash Flows", level=1, prov=_prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="Years ended December 31", prov=_prov(1))
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Cash flows", prov=_prov(2))
    table = doc.add_table(data=_table_data(), caption=caption, prov=_prov(2))
    table.prov.append(_prov(3))
    doc.add_list_item("First bullet", prov=_prov(3))

    els = docling_to_elements(doc)
    assert [e["type"] for e in els] == ["heading", "heading", "text", "table", "list_item"]
    assert els[0] == {"page": 0, "type": "heading", "text": "Item 8. Financial Statements",
                      "level": 0}
    assert els[1]["level"] == 1
    tbl = els[3]
    assert (tbl["page"], tbl["page_end"]) == (1, 2)
    assert tbl["text"].startswith("Cash flows\n\n|")
    assert "Purchases of PP&E" in tbl["text"] and "(1,577)" in tbl["text"]
    assert all("Table of Contents" not in e["text"] for e in els)  # page header dropped


@pytest.mark.parametrize("text,level,expected", [
    ("Item 7. Management's Discussion", 2, 0),
    ("PART II", 1, 0),
    ("ITEM 1A. RISK FACTORS", 1, 0),
    ("Liquidity and Capital Resources", 2, 2),
    ("Items in transit", 1, 1),
])
def test_heading_level(text, level, expected):
    assert heading_level(text, level) == expected
