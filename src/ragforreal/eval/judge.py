"""LLM-as-judge for answer correctness against the FinanceBench gold answer.

Mirrors the FinanceBench paper's grading: an answer is correct if it matches the gold answer
in substance (small rounding / unit-format differences allowed), a refusal is scored
separately from a wrong answer - in finance, "I don't know" is far cheaper than a wrong number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

JUDGE_PROMPT = """You are grading an AI system's answer to a financial question against a \
gold answer written by an expert.

Question: {question}

Gold answer: {gold}

System answer: {candidate}

Grade the system answer:
- "correct": it reaches the same conclusion / number as the gold answer. Allow small rounding \
differences (within ~1%) and different but equivalent units or formats.
- "incorrect": it gives a different number or conclusion, or contradicts the gold answer.
- "refused": it declines to answer or says the information is insufficient.

Respond with JSON only:
{{"verdict": "correct" | "incorrect" | "refused", "reason": "<one sentence>"}}"""

_RETRYABLE = (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
              litellm.InternalServerError, litellm.ServiceUnavailableError)


@dataclass
class Verdict:
    verdict: str
    reason: str
    cost_usd: float = 0.0


@retry(retry=retry_if_exception_type(_RETRYABLE),
       wait=wait_exponential_jitter(multiplier=2, max=60), stop=stop_after_attempt(6), reraise=True)
def judge_answer(question: str, gold: str, candidate: str, model: str) -> Verdict:
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, gold=gold, candidate=candidate)}],
        max_tokens=8000,  # room for the judge's own reasoning before the JSON verdict
    )
    raw = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).lower()
    if verdict not in {"correct", "incorrect", "refused"}:
        # Malformed JSON (e.g. an unescaped quote inside "reason") still usually carries a
        # clean verdict field - read it directly rather than discarding the grade.
        field = re.search(r'"verdict"\s*:\s*"(correct|incorrect|refused)"', raw, re.I)
        verdict = field.group(1).lower() if field else "unparseable"
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:
        cost = 0.0
    return Verdict(verdict=verdict, reason=str(data.get("reason", raw[:200])), cost_usd=cost)
