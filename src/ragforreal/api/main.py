"""HTTP API. Run with: uvicorn ragforreal.api.main:app --reload"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ragforreal.config import ROOT, ExperimentConfig, get_settings
from ragforreal.generation.answer import generate_answer
from ragforreal.retrieval.retriever import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg_path = Path(os.environ.get("RAG_CONFIG", ROOT / "configs/p1_baseline.yaml"))
    state["cfg"] = ExperimentConfig.load(cfg_path)
    state["retriever"] = Retriever(state["cfg"], get_settings())
    yield
    state.clear()


app = FastAPI(title="rag-for-real", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    ref: int
    doc_name: str
    page_start: int  # 1-based for humans
    page_end: int
    score: float
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    experiment: str
    model: str
    cost_usd: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "experiment": state["cfg"].name if "cfg" in state else None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    cfg = state["cfg"]
    try:
        # Retrieval and the LLM call are blocking I/O; keep them off the event loop.
        chunks = await run_in_threadpool(state["retriever"].retrieve, req.question, req.top_k)
        ans = await run_in_threadpool(generate_answer, req.question, chunks, cfg.generation)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}") from e
    return QueryResponse(
        answer=ans.text,
        sources=[
            Source(ref=i, doc_name=c.doc_name, page_start=c.page_start + 1,
                   page_end=c.page_end + 1, score=round(c.score, 4), cited=i in ans.citations)
            for i, c in enumerate(chunks, 1)
        ],
        experiment=cfg.name,
        model=ans.model,
        cost_usd=ans.cost_usd,
    )
