"""In-process store: exact dense search with numpy, Okapi BM25 and sparse dot products over
inverted indexes. Persists to a directory (JSON + .npy). Good up to ~100k chunks; use the
Postgres store beyond that or when several API workers share one index."""

import json
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from docuchat.index.store import Scored
from docuchat.index.text import terms
from docuchat.models import Chunk, Document, SparseVector

BM25_K1 = 1.2
BM25_B = 0.75


class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._docs: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._dirty = True
        if path and (path / "documents.json").exists():
            self._load()

    # ---- documents -------------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        with self._lock:
            self._docs[doc.id] = doc
            self._save_documents()

    def get_document(self, document_id: str) -> Document | None:
        return self._docs.get(document_id)

    def find_by_hash(self, content_hash: str) -> Document | None:
        return next((d for d in self._docs.values() if d.content_hash == content_hash), None)

    def list_documents(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._docs.pop(document_id, None)
            self._chunks = [c for c in self._chunks if c.document_id != document_id]
            self._dirty = True
            self._save()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._chunks.extend(chunks)
            self._dirty = True
            self._save()

    # ---- search ----------------------------------------------------------------------------

    def dense_search(
        self, vector: np.ndarray, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        with self._lock:
            self._build()
            if self._dense is None or not len(self._chunks):
                return []
            scores = self._dense @ vector.astype(np.float32)
            return self._top(scores, k, document_ids)

    def lexical_search(
        self, query: str, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        with self._lock:
            self._build()
            n = len(self._chunks)
            if not n:
                return []
            scores = np.zeros(n, dtype=np.float32)
            for term in set(terms(query)):
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
                for i, tf in postings:
                    norm = 1 - BM25_B + BM25_B * self._lengths[i] / self._avg_len
                    scores[i] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
            return self._top(scores, k, document_ids, positive_only=True)

    def sparse_search(
        self, vector: SparseVector, k: int, document_ids: Sequence[str] | None = None
    ) -> Scored:
        with self._lock:
            self._build()
            if not self._chunks:
                return []
            scores = np.zeros(len(self._chunks), dtype=np.float32)
            for index, weight in zip(vector.indices, vector.values, strict=True):
                for i, w in self._sparse_postings.get(index, ()):
                    scores[i] += weight * w
            return self._top(scores, k, document_ids, positive_only=True)

    def neighbors(self, document_id: str, indices: Sequence[int]) -> list[Chunk]:
        wanted = set(indices)
        return sorted(
            (c for c in self._chunks if c.document_id == document_id and c.index in wanted),
            key=lambda c: c.index,
        )

    # ---- internals -------------------------------------------------------------------------

    def _top(
        self,
        scores: np.ndarray,
        k: int,
        document_ids: Sequence[str] | None,
        positive_only: bool = False,
    ) -> Scored:
        if document_ids is not None:
            allowed = set(document_ids)
            mask = np.array([c.document_id in allowed for c in self._chunks])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[: k * 2 if positive_only else k]
        results = [(self._chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]
        if positive_only:
            results = [(c, s) for c, s in results if s > 0]
        return results[:k]

    def _build(self) -> None:
        if not self._dirty:
            return
        vectors = [c.dense for c in self._chunks if c.dense is not None]
        self._dense = np.stack(vectors) if len(vectors) == len(self._chunks) and vectors else None
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._sparse_postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._lengths: list[int] = []
        for i, chunk in enumerate(self._chunks):
            counts = Counter(terms(chunk.search_text))
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((i, tf))
            if chunk.sparse is not None:
                for idx, w in zip(chunk.sparse.indices, chunk.sparse.values, strict=True):
                    self._sparse_postings[idx].append((i, w))
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        self._dirty = False

    def _save_documents(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        payload = [d.model_dump(mode="json") for d in self._docs.values()]
        _atomic_write(self.path / "documents.json", json.dumps(payload))

    def _save(self) -> None:
        if not self.path:
            return
        self._save_documents()
        rows = []
        for c in self._chunks:
            row = c.model_dump(mode="json")
            row["sparse"] = c.sparse.model_dump() if c.sparse else None
            rows.append(row)
        _atomic_write(self.path / "chunks.json", json.dumps(rows))
        if self._chunks and all(c.dense is not None for c in self._chunks):
            np.save(self.path / "dense.npy", np.stack([c.dense for c in self._chunks]))  # type: ignore[misc]

    def _load(self) -> None:
        assert self.path is not None
        docs = json.loads((self.path / "documents.json").read_text(encoding="utf-8"))
        self._docs = {d["id"]: Document.model_validate(d) for d in docs}
        chunks_file = self.path / "chunks.json"
        if not chunks_file.exists():
            return
        rows = json.loads(chunks_file.read_text(encoding="utf-8"))
        dense_file = self.path / "dense.npy"
        dense = np.load(dense_file) if dense_file.exists() else None
        for i, row in enumerate(rows):
            sparse = row.pop("sparse", None)
            chunk = Chunk.model_validate(row)
            chunk.sparse = SparseVector.model_validate(sparse) if sparse else None
            chunk.dense = dense[i] if dense is not None and i < len(dense) else None
            self._chunks.append(chunk)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
