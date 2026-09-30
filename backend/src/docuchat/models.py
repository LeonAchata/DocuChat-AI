"""Core data types shared by ingestion, indexing, retrieval and generation."""

from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

DocStatus = Literal["processing", "ready", "failed"]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]

    def dot(self, other: "SparseVector") -> float:
        mine = dict(zip(self.indices, self.values, strict=True))
        return float(
            sum(mine.get(i, 0.0) * v for i, v in zip(other.indices, other.values, strict=True))
        )


class Document(BaseModel):
    id: str
    title: str
    filename: str
    content_hash: str
    pages: int | None = None
    chunk_count: int = 0
    status: DocStatus = "processing"
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """A retrievable unit. ``text`` is what we show and cite; ``search_text`` is what we index
    (text plus title/section breadcrumbs and, optionally, an LLM-written context)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    document_id: str
    index: int
    text: str
    search_text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    dense: np.ndarray | None = Field(default=None, exclude=True, repr=False)
    sparse: SparseVector | None = Field(default=None, exclude=True, repr=False)


class Hit(BaseModel):
    """A chunk with the scores it collected along the retrieval pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: Chunk
    score: float = 0.0
    ranks: dict[str, int] = Field(default_factory=dict)
    rerank_score: float | None = None


class Passage(BaseModel):
    """Contiguous chunks from one document, as sent to the LLM for grounding."""

    id: int
    document_id: str
    document_title: str
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    chunk_ids: list[str]
    relevance: float | None = None

    @property
    def location(self) -> str:
        parts = []
        if self.page_start:
            same = self.page_end in (None, self.page_start)
            parts.append(
                f"p. {self.page_start}" if same else f"pp. {self.page_start}-{self.page_end}"
            )
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)
