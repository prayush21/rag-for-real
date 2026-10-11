"""Two kinds of configuration, kept deliberately separate:

* ``Settings`` - environment / secrets (.env): where things live and how to reach them.
* ``ExperimentConfig`` - a YAML file in ``configs/``: *what* pipeline to run. Every eval
  result records the full experiment config so any number in ``results/`` can be reproduced.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = ""
    corpus: Literal["dev", "full"] = "dev"
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"
    cache_dir: Path = ROOT / ".cache"

    # Client-side throttle for the embedding API. Voyage without a card on file allows
    # ~3 RPM / 10K TPM; with a card the limits are far higher. 0 disables a limit.
    embed_rpm: int = 0
    embed_tpm: int = 0


class ChunkingConfig(BaseModel):
    # fixed_tokens: sliding token window, blind to structure (Phase 1 baseline).
    # structure: split on headings, never break a paragraph, keep tables whole (Phase 2).
    strategy: Literal["fixed_tokens", "structure"] = "fixed_tokens"
    chunk_size: int = 512
    chunk_overlap: int = 64  # fixed_tokens only
    max_table_tokens: int = 1024  # structure: larger tables split by rows, header repeated
    min_chunk_tokens: int = 64  # structure: smaller sections merge into the next one
    # Prepend "[company | doc type | period | section path]" to every chunk before embedding.
    context_header: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info) -> int:
        if v >= info.data.get("chunk_size", 0):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "voyage/voyage-4"  # any LiteLLM embedding model string
    dimensions: int = 1024
    batch_size: int = 64
    # Asymmetric models (Voyage, Cohere, ...) embed queries and documents differently.
    query_input_type: str | None = "query"
    document_input_type: str | None = "document"


class IngestConfig(BaseModel):
    parser: Literal["pypdf", "docling"] = "pypdf"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class RetrievalConfig(BaseModel):
    """Retrieval as composable stages: candidates -> fusion -> rerank -> top_k.

    ``strategy`` picks the candidate generators: dense (vector), sparse (Postgres full-text) or
    hybrid (both, fused with Reciprocal Rank Fusion). With a reranker, ``candidate_k``
    candidates are re-scored by a cross-encoder; without one, candidates are simply cut to k.
    """

    strategy: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 10
    # Keyword ranker: Postgres full-text (ts_rank_cd, no IDF) or in-process BM25.
    sparse: Literal["postgres", "bm25"] = "postgres"
    candidate_k: int = 50  # per generator, before fusion / reranking
    rrf_k: int = 60  # RRF damping constant (Cormack et al. 2009)
    reranker: str | None = None  # LiteLLM rerank model, e.g. "voyage/rerank-2.5"
    # Restrict search to filings of the companies named in the question (rule-based).
    company_filter: bool = False


class GenerationConfig(BaseModel):
    model: str = "anthropic/claude-haiku-5-5"  # any LiteLLM chat model string
    # None = provider default. Newer reasoning models (Claude Haiku 5.5+, Gemini 3+) reject or
    # deprecate non-default sampling settings, so only send one when explicitly configured.
    temperature: float | None = None
    # Thinking models spend part of this on reasoning; too low a cap truncates the answer.
    max_tokens: int = 16000


class EvalConfig(BaseModel):
    ks: list[int] = [1, 3, 5, 10, 20]
    judge_model: str = "anthropic/claude-haiku-5-5"


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def index_name(self, corpus: str) -> str:
        """Name of the vector index this config produces.

        Derived from a hash of everything that changes the stored vectors (parser, chunking,
        embedding model) plus the corpus. Two experiments that differ only in retrieval or
        generation settings share an index; changing chunking yields a new one. This is
        index versioning: old indexes stay queryable for side-by-side comparison.
        """
        # exclude_defaults: adding a new option (with a default) later must not rename
        # every existing index.
        payload = json.dumps(self.ingest.model_dump(exclude_defaults=True), sort_keys=True)
        payload += corpus
        digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
        model_slug = re.sub(r"[^a-z0-9]+", "_", self.ingest.embedding.model.lower()).strip("_")
        return f"{corpus}_{self.ingest.parser}_{model_slug}_{digest}"


def get_settings() -> Settings:
    # Provider SDKs (LiteLLM) read API keys from the process environment, not from Settings.
    # Real environment variables win over .env.
    load_dotenv(ROOT / ".env", override=False)
    return Settings()
