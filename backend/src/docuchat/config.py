"""Runtime configuration from environment variables (prefix ``DOCUCHAT_``) or ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUCHAT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Storage ------------------------------------------------------------------------------
    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://docuchat:docuchat@localhost:5432/docuchat"
    data_dir: Path = Path("data")

    # --- Models (local, ONNX via fastembed) ---------------------------------------------------
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str | None = Field(
        default="prithivida/Splade_PP_en_v1",
        description="Learned sparse encoder (SPLADE). Set empty to disable.",
    )
    reranker_model: str | None = Field(
        default="Xenova/ms-marco-MiniLM-L-12-v2",
        description="Cross-encoder reranker. Set empty to disable reranking.",
    )

    # --- Chunking -----------------------------------------------------------------------------
    chunk_tokens: int = Field(default=320, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=48, ge=0, le=512)
    contextualize: bool = Field(
        default=False,
        description="Anthropic-style contextual retrieval: an LLM writes a short context for "
        "every chunk before indexing. Better recall, extra ingestion cost.",
    )

    # --- Retrieval ----------------------------------------------------------------------------
    candidates_per_retriever: int = Field(default=30, ge=1, le=500)
    rerank_candidates: int = Field(default=40, ge=1, le=200)
    top_k: int = Field(default=6, ge=1, le=30)
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_lexical: float = 1.0
    weight_sparse: float = 1.0
    mmr_lambda: float = Field(default=0.75, ge=0, le=1, description="1 = relevance only.")
    window: int = Field(default=1, ge=0, le=3, description="Neighbour chunks added per hit.")
    min_relevance: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Reranker probability below which evidence counts as insufficient.",
    )
    query_expansion: bool = Field(default=True, description="Multi-query + HyDE via the LLM.")
    max_retrieval_rounds: int = Field(default=2, ge=1, le=3)

    # --- LLM ----------------------------------------------------------------------------------
    model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    context_model: str = Field(
        default="claude-opus-5-5", description="Model used for contextual chunk headers."
    )
    refusal_fallbacks: bool = True

    # --- API ----------------------------------------------------------------------------------
    max_upload_mb: int = 25
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "threads.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
