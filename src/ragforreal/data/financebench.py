"""FinanceBench: 150 expert-written Q&A pairs over public SEC filings, with gold evidence pages.

Source: https://github.com/patronus-ai/financebench (CC-BY-NC-4.0).

The gold ``evidence`` (document + page) lets us score *retrieval* without an LLM judge, which
makes retrieval experiments cheap, fast and deterministic.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.progress import track

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOC_INFO_FILE = "financebench_document_information.jsonl"

# The dev corpus is the N documents with the most questions: small enough to iterate on in
# minutes, still spread across many companies so retrieval has real distractors.
DEV_DOC_COUNT = 20


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # 0-based page index, as in FinanceBench's ``evidence_page_num``
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    doc_name: str
    company: str
    question: str
    answer: str
    question_type: str
    question_reasoning: str | None
    evidence: tuple[Evidence, ...]


def _raw_dir(data_dir: Path) -> Path:
    return data_dir / "financebench"


def fetch_metadata(data_dir: Path) -> None:
    raw = _raw_dir(data_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name in (QUESTIONS_FILE, DOC_INFO_FILE):
        resp = httpx.get(f"{RAW_BASE}/data/{name}", follow_redirects=True, timeout=60)
        resp.raise_for_status()
        (raw / name).write_bytes(resp.content)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_all_questions(data_dir: Path) -> list[Question]:
    rows = _read_jsonl(_raw_dir(data_dir) / QUESTIONS_FILE)
    return [
        Question(
            id=r["financebench_id"],
            doc_name=r["doc_name"],
            company=r["company"],
            question=r["question"],
            answer=r["answer"],
            question_type=r["question_type"],
            question_reasoning=r.get("question_reasoning"),
            evidence=tuple(
                Evidence(doc_name=e["doc_name"], page=int(e["evidence_page_num"]),
                         text=e["evidence_text"])
                for e in r["evidence"]
            ),
        )
        for r in rows
    ]


def corpus_doc_names(data_dir: Path, corpus: str) -> list[str]:
    questions = load_all_questions(data_dir)
    counts = Counter(q.doc_name for q in questions)
    if corpus == "full":
        return sorted(counts)
    # Sort by question count desc, then name, so the subset is deterministic.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted(name for name, _ in ranked[:DEV_DOC_COUNT])


def load_questions(data_dir: Path, corpus: str) -> list[Question]:
    """Questions whose gold evidence lies entirely inside the chosen corpus."""
    docs = set(corpus_doc_names(data_dir, corpus))
    return [
        q for q in load_all_questions(data_dir)
        if q.doc_name in docs and all(e.doc_name in docs for e in q.evidence)
    ]


def pdf_dir(data_dir: Path) -> Path:
    return _raw_dir(data_dir) / "pdfs"


def fetch_pdfs(data_dir: Path, corpus: str) -> list[Path]:
    out = pdf_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name in track(corpus_doc_names(data_dir, corpus), description="Downloading PDFs"):
            path = out / f"{name}.pdf"
            if not path.exists():
                resp = client.get(f"{RAW_BASE}/pdfs/{name}.pdf")
                resp.raise_for_status()
                path.write_bytes(resp.content)
            paths.append(path)
    return paths
