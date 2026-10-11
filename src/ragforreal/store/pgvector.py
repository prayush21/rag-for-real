"""Postgres + pgvector store with versioned indexes.

Each index (one per distinct parser/chunking/embedding config) gets its own table, and a
``rag_indexes`` registry records the exact config that built it. Old indexes stay queryable,
so experiments can be compared side by side and a bad re-index is rolled back by pointing at
the previous name rather than by re-embedding.
"""

from __future__ import annotations

import atexit
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ragforreal.ingest.chunking import Chunk

REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS rag_indexes (
    name        TEXT PRIMARY KEY,
    corpus      TEXT NOT NULL,
    config      JSONB NOT NULL,
    dimensions  INT NOT NULL,
    n_chunks    INT,
    status      TEXT NOT NULL DEFAULT 'building',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


@dataclass
class RetrievedChunk:
    id: str
    doc_name: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    metadata: dict
    score: float


def _table(index_name: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]{1,55}", index_name):
        raise ValueError(f"unsafe index name: {index_name!r}")
    return f"chunks_{index_name}"


class PgVectorStore:
    """Holds a small connection pool. Opening a TLS connection to a hosted Postgres costs
    hundreds of milliseconds - more than the vector search itself - so connections are reused
    across queries instead of opened per call."""

    def __init__(self, database_url: str, max_connections: int = 4):
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set - copy .env.example to .env and fill it in")
        # The extension must exist before pooled connections can register the vector type.
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(REGISTRY_DDL)
        self.pool = ConnectionPool(
            database_url, min_size=1, max_size=max_connections, open=True,
            kwargs={"autocommit": True}, configure=register_vector,
            check=ConnectionPool.check_connection,  # serverless Postgres drops idle connections
        )
        atexit.register(self.close)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with self.pool.connection() as conn:
            yield conn

    # ---------- write path ----------

    def create_index(self, name: str, corpus: str, config: dict, dimensions: int) -> None:
        table = _table(name)
        with self.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"""CREATE TABLE {table} (
                    id          TEXT PRIMARY KEY,
                    doc_name    TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    page_start  INT NOT NULL,
                    page_end    INT NOT NULL,
                    n_tokens    INT NOT NULL,
                    text        TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}',
                    embedding   vector({dimensions}) NOT NULL
                )"""
            )
            conn.execute(
                """INSERT INTO rag_indexes (name, corpus, config, dimensions)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config,
                     status = 'building', n_chunks = NULL, updated_at = now()""",
                (name, corpus, json.dumps(config), dimensions),
            )

    def insert_chunks(self, name: str, chunks: list[Chunk], vectors: np.ndarray,
                      batch_size: int = 500) -> None:
        table = _table(name)
        rows = [
            (c.id, c.doc_name, c.chunk_index, c.page_start, c.page_end, c.n_tokens, c.text,
             json.dumps(c.metadata), v)
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.connect() as conn, conn.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                cur.executemany(
                    f"""INSERT INTO {table} (id, doc_name, chunk_index, page_start, page_end,
                        n_tokens, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    rows[i : i + batch_size],
                )

    def finalize_index(self, name: str) -> None:
        """Build the ANN index after bulk load (much faster than maintaining it per insert)."""
        table = _table(name)
        with self.connect() as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc ON {table} (doc_name)")
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            conn.execute(
                "UPDATE rag_indexes SET status='ready', n_chunks=%s, updated_at=now() "
                "WHERE name=%s",
                (n, name),
            )

    # ---------- read path ----------

    def index_status(self, name: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM rag_indexes WHERE name=%s", (name,)).fetchone()
        return row[0] if row else None

    def list_indexes(self) -> list[dict]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT name, corpus, status, n_chunks, dimensions, created_at, config "
                "FROM rag_indexes ORDER BY created_at"
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def dense_search(self, name: str, query_vec: np.ndarray, k: int,
                     doc_names: list[str] | None = None, ef_search: int = 100
                     ) -> list[RetrievedChunk]:
        table = _table(name)
        where, params = "", [query_vec]
        if doc_names:
            where, params = "WHERE doc_name = ANY(%s)", [query_vec, doc_names]
        with self.connect() as conn:
            # Raise HNSW's candidate list above the default (40) so top-k recall holds for k=20.
            conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
            rows = conn.execute(
                f"""SELECT id, doc_name, chunk_index, page_start, page_end, text, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM {table} {where}
                    ORDER BY embedding <=> %s LIMIT %s""",
                [*params, query_vec, k],
            ).fetchall()
        return [RetrievedChunk(*r) for r in rows]
