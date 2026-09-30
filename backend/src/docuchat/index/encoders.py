"""Embedding and reranking models behind small protocols.

Production implementations run locally as ONNX through fastembed (no GPU, no torch, no API
key). Hashing implementations are deterministic, dependency-free stand-ins for tests."""

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docuchat.index.text import terms
from docuchat.models import SparseVector

BATCH_SIZE = 32


class DenseEncoder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SparseEncoder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...
    def embed_query(self, text: str) -> SparseVector: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query as a probability in [0, 1]."""
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized: np.ndarray = (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)
    return normalized


# ---- fastembed (production) ------------------------------------------------------------------


class FastEmbedDense:
    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts, batch_size=BATCH_SIZE))
        return _normalize(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(next(iter(self._model.query_embed(text)))))


class FastEmbedSparse:
    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        from fastembed import SparseTextEmbedding

        self.name = model
        self._model = SparseTextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e) for e in self._model.embed(texts, batch_size=BATCH_SIZE)]

    def embed_query(self, text: str) -> SparseVector:
        return _to_sparse(next(iter(self._model.query_embed(text))))


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices], values=[float(v) for v in embedding.values]
    )


class FastEmbedReranker:
    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model, cache_dir=str(cache_dir) if cache_dir else None)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self._model.rerank(query, texts, batch_size=BATCH_SIZE)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


# ---- hashing (tests / offline) ---------------------------------------------------------------


def _bucket(term: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big") % dim


class HashingDense:
    """Bag-of-stems hashed into a fixed-size vector. Lexical, but shaped like an embedding."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for term, count in Counter(terms(text)).items():
            vec[_bucket(term, self.dim)] += 1 + math.log(count)
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))


class HashingSparse:
    def __init__(self, vocab: int = 30_000) -> None:
        self.name = "hashing-sparse"
        self.vocab = vocab

    def embed_query(self, text: str) -> SparseVector:
        counts = Counter(_bucket(t, self.vocab) for t in terms(text))
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[float(counts[i]) for i in indices])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.embed_query(t) for t in texts]


class OverlapReranker:
    """Fraction of query terms present in the text: a transparent stand-in for tests."""

    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        wanted = set(terms(query))
        if not wanted:
            return [0.0 for _ in texts]
        return [len(wanted & set(terms(t))) / len(wanted) for t in texts]
