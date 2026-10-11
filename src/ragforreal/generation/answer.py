"""Grounded answer generation with numbered citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ragforreal.config import GenerationConfig
from ragforreal.store.pgvector import RetrievedChunk

SYSTEM_PROMPT = """You are a financial analyst answering questions about SEC filings.

Rules:
- Answer ONLY from the numbered context passages. Do not use outside knowledge.
- Cite every fact with the passage number in square brackets, e.g. [2].
- Show the numbers you used for any calculation, then state the final answer clearly.
- If the passages do not contain enough information, say "Insufficient information in the \
provided context." and explain what is missing. Never guess."""

_RETRYABLE = (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
              litellm.InternalServerError, litellm.ServiceUnavailableError)


@dataclass
class Answer:
    text: str
    citations: list[int]
    sources: list[RetrievedChunk]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        pages = (f"page {c.page_start + 1}" if c.page_start == c.page_end
                 else f"pages {c.page_start + 1}-{c.page_end + 1}")
        blocks.append(f"[{i}] {c.doc_name} ({pages})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


@retry(retry=retry_if_exception_type(_RETRYABLE),
       wait=wait_exponential_jitter(multiplier=2, max=60), stop=stop_after_attempt(6), reraise=True)
def _complete(model: str, messages: list[dict], temperature: float | None, max_tokens: int):
    extra = {} if temperature is None else {"temperature": temperature}
    return litellm.completion(model=model, messages=messages, max_tokens=max_tokens, **extra)


def generate_answer(question: str, chunks: list[RetrievedChunk], cfg: GenerationConfig) -> Answer:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{format_context(chunks)}\n\n"
                                    f"Question: {question}"},
    ]
    resp = _complete(cfg.model, messages, cfg.temperature, cfg.max_tokens)
    text = resp.choices[0].message.content or ""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(chunks)})
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:  # unknown pricing for this model
        cost = 0.0
    return Answer(
        text=text,
        citations=cited,
        sources=chunks,
        model=cfg.model,
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        cost_usd=cost,
    )
