"""Builds the application object graph from settings and exposes a streaming chat API."""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from docuchat.agent import RagDeps, RagState, build_graph
from docuchat.config import Settings
from docuchat.generation.claude import AnswerLLM, ClaudeLLM
from docuchat.index.encoders import (
    DenseEncoder,
    FastEmbedDense,
    FastEmbedReranker,
    FastEmbedSparse,
    Reranker,
    SparseEncoder,
)
from docuchat.index.memory import MemoryStore
from docuchat.index.store import Store
from docuchat.ingest.pipeline import Ingestor
from docuchat.retrieval.retriever import HybridRetriever

log = logging.getLogger(__name__)
_UNSET: Any = object()


@dataclass
class Components:
    settings: Settings
    store: Store
    retriever: HybridRetriever
    ingestor: Ingestor
    llm: AnswerLLM


def build_components(
    settings: Settings,
    *,
    llm: AnswerLLM | None = None,
    dense: DenseEncoder | None = None,
    sparse: SparseEncoder | None = _UNSET,
    reranker: Reranker | None = _UNSET,
    store: Store | None = None,
) -> Components:
    cache = settings.data_dir / "models"
    dense = dense or FastEmbedDense(settings.dense_model, cache)
    if sparse is _UNSET:
        sparse = FastEmbedSparse(settings.sparse_model, cache) if settings.sparse_model else None
    if reranker is _UNSET:
        reranker = (
            FastEmbedReranker(settings.reranker_model, cache) if settings.reranker_model else None
        )
    if store is None:
        if settings.store == "postgres":
            from docuchat.index.postgres import PostgresStore

            dim = int(dense.embed_query("dimension probe").shape[0])
            store = PostgresStore(settings.database_url, dim=dim)
        else:
            store = MemoryStore(settings.data_dir / "index")

    claude: ClaudeLLM | None = None
    if llm is None:
        claude = ClaudeLLM(
            model=settings.model,
            effort=settings.effort,
            refusal_fallbacks=settings.refusal_fallbacks,
        )
        llm = claude
    contextualizer = None
    if settings.contextualize:
        context_llm = ClaudeLLM(
            model=settings.context_model, refusal_fallbacks=settings.refusal_fallbacks
        )
        contextualizer = context_llm.contextualize

    retriever = HybridRetriever(
        store=store, dense=dense, sparse=sparse, reranker=reranker, settings=settings
    )
    ingestor = Ingestor(
        store=store, dense=dense, sparse=sparse, settings=settings, contextualizer=contextualizer
    )
    return Components(settings, store, retriever, ingestor, llm)


class DocuChat:
    def __init__(
        self, components: Components, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> None:
        self.c = components
        self.graph: CompiledStateGraph[Any, Any, Any, Any] = build_graph(
            RagDeps(
                store=components.store,
                retriever=components.retriever,
                llm=components.llm,
                settings=components.settings,
            ),
            checkpointer,
        )

    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _input(message: str, document_ids: list[str] | None) -> RagState:
        return {"messages": [{"role": "user", "content": message}], "document_ids": document_ids}

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def stream(
        self, message: str, thread_id: str, document_ids: list[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream(
            self._input(message, document_ids), self._config(thread_id), stream_mode="custom"
        )

    async def astream(
        self, message: str, thread_id: str, document_ids: list[str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self.graph.astream(
            self._input(message, document_ids), self._config(thread_id), stream_mode="custom"
        ):
            yield chunk

    def ask(self, message: str, document_ids: list[str] | None = None) -> dict[str, Any]:
        state = self.graph.invoke(
            self._input(message, document_ids), self._config(self.new_thread())
        )
        return dict(state)
