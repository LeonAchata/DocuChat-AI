"""PostgreSQL store: pgvector HNSW for dense vectors, ``sparsevec`` HNSW for learned sparse
vectors and a generated ``tsvector`` column (GIN) for full-text search - one database for
metadata and all three retrieval signals."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from docuchat.index.store import Scored
from docuchat.index.text import STOPWORDS, WORD_RE
from docuchat.models import Chunk, Document, SparseVector

SPARSE_DIM = 250_000  # covers BERT (30k) and XLM-R (250k) vocabularies
CHUNK_COLUMNS = (
    "id, document_id, idx, text, search_text, section, page_start, page_end, token_count, embedding"
)


def schema_sql(dim: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        filename TEXT NOT NULL,
        content_hash TEXT NOT NULL UNIQUE,
        pages INT,
        chunk_count INT NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        idx INT NOT NULL,
        text TEXT NOT NULL,
        search_text TEXT NOT NULL,
        section TEXT,
        page_start INT,
        page_end INT,
        token_count INT NOT NULL DEFAULT 0,
        embedding vector({dim}) NOT NULL,
        sparse sparsevec({SPARSE_DIM}),
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED
    );
    CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, idx);
    CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS chunks_sparse_hnsw ON chunks
        USING hnsw (sparse sparsevec_ip_ops);
    """


class PostgresStore:
    def __init__(self, url: str, *, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(schema_sql(dim))

        def configure(conn: Any) -> None:
            register_vector(conn)
            conn.row_factory = dict_row

        self._pool = ConnectionPool(url, min_size=1, max_size=8, configure=configure, open=True)

    def close(self) -> None:
        self._pool.close()

    # ---- documents -------------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, title, filename, content_hash, pages, chunk_count,
                                       status, error, created_at)
                VALUES (%(id)s, %(title)s, %(filename)s, %(content_hash)s, %(pages)s,
                        %(chunk_count)s, %(status)s, %(error)s, %(created_at)s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title, pages = EXCLUDED.pages,
                    chunk_count = EXCLUDED.chunk_count, status = EXCLUDED.status,
                    error = EXCLUDED.error
                """,
                doc.model_dump(),
            )

    def get_document(self, document_id: str) -> Document | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
        return Document.model_validate(row) if row else None

    def find_by_hash(self, content_hash: str) -> Document | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = %s", (content_hash,)
            ).fetchone()
        return Document.model_validate(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [Document.model_validate(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        from pgvector import SparseVector as PgSparse

        rows = [
            (
                c.id,
                c.document_id,
                c.index,
                c.text,
                c.search_text,
                c.section,
                c.page_start,
                c.page_end,
                c.token_count,
                c.dense,
                PgSparse(dict(zip(c.sparse.indices, c.sparse.values, strict=True)), SPARSE_DIM)
                if c.sparse and c.sparse.indices
                else None,
            )
            for c in chunks
        ]
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, document_id, idx, text, search_text, section,
                                    page_start, page_end, token_count, embedding, sparse)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

    # ---- search ----------------------------------------------------------------------------

    def dense_search(
        self, vector: np.ndarray, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        return self._search(
            "1 - (embedding <=> %(q)s)",
            "embedding <=> %(q)s",
            {"q": vector.astype(np.float32)},
            k,
            document_ids,
        )

    def sparse_search(
        self, vector: SparseVector, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        from pgvector import SparseVector as PgSparse

        if not vector.indices:
            return []
        q = PgSparse(dict(zip(vector.indices, vector.values, strict=True)), SPARSE_DIM)
        return self._search(
            "-(sparse <#> %(q)s)",
            "sparse <#> %(q)s",
            {"q": q},
            k,
            document_ids,
            where="sparse IS NOT NULL",
        )

    def lexical_search(
        self, query: str, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        words = [w for w in WORD_RE.findall(query.lower()) if w not in STOPWORDS]
        if not words:
            return []
        # OR semantics (like BM25) instead of websearch_to_tsquery's AND.
        tsquery = " | ".join(dict.fromkeys(words))
        return self._search(
            "ts_rank_cd(tsv, to_tsquery('english', %(q)s), 32)",
            "ts_rank_cd(tsv, to_tsquery('english', %(q)s), 32) DESC",
            {"q": tsquery},
            k,
            document_ids,
            where="tsv @@ to_tsquery('english', %(q)s)",
        )

    def neighbors(self, document_id: str, indices: Sequence[int]) -> list[Chunk]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {CHUNK_COLUMNS} FROM chunks "
                "WHERE document_id = %s AND idx = ANY(%s) ORDER BY idx",
                (document_id, list(indices)),
            ).fetchall()
        return [_chunk(r) for r in rows]

    def _search(
        self,
        score_sql: str,
        order_sql: str,
        params: dict[str, Any],
        k: int,
        document_ids: Sequence[str] | None,
        where: str = "TRUE",
    ) -> Scored:
        sql = f"""
            SELECT {CHUNK_COLUMNS}, {score_sql} AS score
            FROM chunks
            WHERE {where} AND (%(docs)s::text[] IS NULL OR document_id = ANY(%(docs)s))
            ORDER BY {order_sql}
            LIMIT %(k)s
        """
        with self._pool.connection() as conn, conn.transaction():
            # Keep recall high when a document filter prunes HNSW candidates.
            conn.execute("SET LOCAL hnsw.ef_search = 100")
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(
                sql, {**params, "docs": list(document_ids) if document_ids else None, "k": k}
            ).fetchall()
        return [(_chunk(r), float(r["score"])) for r in rows]


def _chunk(row: dict[str, Any]) -> Chunk:
    chunk = Chunk(
        id=row["id"],
        document_id=row["document_id"],
        index=row["idx"],
        text=row["text"],
        search_text=row["search_text"],
        section=row["section"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        token_count=row["token_count"],
    )
    embedding = row.get("embedding")
    if embedding is not None:
        # pgvector >= 0.4 returns a Vector wrapper; older versions return a numpy array.
        values = embedding.to_numpy() if hasattr(embedding, "to_numpy") else embedding
        chunk.dense = np.asarray(values, dtype=np.float32)
    return chunk
