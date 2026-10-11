"""DoclingDocument -> our flat element list. Runs inside the Modal container (where docling is
installed); only needs ``docling_core``, imported lazily so the local package stays light."""

from __future__ import annotations

import re
from typing import Any

# SEC filings are organised as "PART II" / "Item 7. Management's Discussion...". Docling tags
# most headings as level 1, so we promote these to level 0 to anchor the section path.
SEC_TOP_HEADING = re.compile(r"^\s*(part\s+[ivx]+\b|item\s+\d+[a-z]?\b)", re.I)

SKIP_LABELS = {"page_header", "page_footer", "picture", "chart", "caption", "document_index"}


def heading_level(text: str, docling_level: int) -> int:
    return 0 if SEC_TOP_HEADING.match(text) else max(1, docling_level)


def docling_to_elements(doc: Any) -> list[dict]:
    from docling_core.types.doc.document import (
        ListItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    out: list[dict] = []
    for item, _depth in doc.iterate_items():
        prov = getattr(item, "prov", None)
        label = getattr(getattr(item, "label", None), "value", "")
        if not prov or label in SKIP_LABELS:
            continue
        pages = [p.page_no - 1 for p in prov]  # docling pages are 1-based
        base = {"page": min(pages)}
        if max(pages) != min(pages):
            base["page_end"] = max(pages)

        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc).strip()  # includes the caption, if any
            if md:
                out.append({**base, "type": "table", "text": md})
        elif isinstance(item, SectionHeaderItem | TitleItem):
            text = item.text.strip()
            if text:
                level = getattr(item, "level", 1) if isinstance(item, SectionHeaderItem) else 1
                out.append({**base, "type": "heading", "text": text,
                            "level": heading_level(text, level)})
        elif isinstance(item, ListItem):
            if item.text.strip():
                out.append({**base, "type": "list_item", "text": item.text.strip()})
        elif isinstance(item, TextItem) and item.text.strip():
            out.append({**base, "type": "text", "text": item.text.strip()})
    return out
