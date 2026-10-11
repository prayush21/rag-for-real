"""Metadata filters extracted from the question.

FinanceBench questions name their company ("AMEX", "JnJ", "Pepsico", "3M"), so a rule-based
matcher over company names + common aliases finds it without an LLM call. A question naming no
company ("What drove the reduction in SG&A expense in FY2023?") gets no filter - an LLM could
not recover the company from that text either.
"""

from __future__ import annotations

import re

# Extra ways questions refer to a company. The full company name (and the name without a
# corporate suffix) always matches; list only what that misses. Matching is case-insensitive.
ALIASES: dict[str, list[str]] = {
    "American Express": [r"amex"],
    "Johnson & Johnson": [r"jnj", r"j\s*&\s*j", r"johnson\s+and\s+johnson"],
    "MGM Resorts": [r"mgm"],
    "CVS Health": [r"cvs"],
    "Activision Blizzard": [r"activision"],
    "Ulta Beauty": [r"ulta"],
    "Coca-Cola": [r"coca\s+cola", r"coke"],
    "JPMorgan": [r"jp\s*morgan", r"jpm"],
    "Kraft Heinz": [r"kraft"],
    "Lockheed Martin": [r"lockheed"],
    "American Water Works": [r"american\s+water"],
    "Foot Locker": [r"footlocker"],
}
# Names that are also common words: match only with the original capitalisation.
CASE_SENSITIVE = {"Block"}
_SUFFIXES = re.compile(r"\s+(corporation|corp|inc|company|co)\.?$", re.I)


def _patterns(company: str) -> list[str]:
    names = {company, _SUFFIXES.sub("", company)}
    pats = [r"\s+".join(map(re.escape, n.split())) for n in names]
    if company in CASE_SENSITIVE:
        pats = [f"(?-i:{p})" for p in pats]
    return pats + ALIASES.get(company, [])


def companies_in(question: str, companies: set[str]) -> set[str]:
    found = set()
    for company in companies:
        for pat in _patterns(company):
            # Word boundaries, plus possessives ("AMEX's", "American Express'").
            if re.search(rf"(?<![\w&]){pat}(?![\w&])", question, re.I):
                found.add(company)
                break
    return found


def filter_docs(question: str, doc_companies: dict[str, str]) -> list[str] | None:
    """Doc names to search, or None to search everything."""
    found = companies_in(question, set(doc_companies.values()))
    if not found:
        return None
    return sorted(d for d, c in doc_companies.items() if c in found)
