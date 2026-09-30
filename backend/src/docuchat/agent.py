"""Conversational RAG as a LangGraph state machine (a corrective-RAG loop):

    start -> plan --(small talk)--> respond
                  --> retrieve -> grade --(weak evidence, rounds left)--> reformulate -> retrieve
                                        --(weak evidence, no rounds)----> abstain
                                        --(good evidence)---------------> generate

``generate`` streams tokens and citations through LangGraph's custom stream channel.
"""

import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from docuchat.config import Settings
from docuchat.generation.claude import AnswerLLM, LLMError, Turn
from docuchat.index.store import Store
from docuchat.models import Passage
from docuchat.retrieval.retriever import HybridRetriever

log = logging.getLogger(__name__)

Status = Literal["running", "answered", "no_evidence", "chitchat", "failed"]


class Source(TypedDict):
    id: int
    document_id: str
    title: str
    location: str
    text: str
    relevance: float | None


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sources: NotRequired[list[Source]]
    citations: NotRequired[list[dict[str, Any]]]


class RagState(TypedDict, total=False):
    messages: Annotated[list[Message], operator.add]
    question: str
    document_ids: list[str] | None
    queries: list[str]
    hypothetical: str | None
    passages: list[dict[str, Any]]
    confidence: float | None
    round: int
    status: Status
    reply: str | None


@dataclass
class RagDeps:
    store: Store
    retriever: HybridRetriever
    llm: AnswerLLM
    settings: Settings


def build_graph(
    deps: RagDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    s = deps.settings

    def start(state: RagState) -> RagState:
        return {
            "question": state["messages"][-1]["content"],
            "queries": [],
            "hypothetical": None,
            "passages": [],
            "confidence": None,
            "round": 0,
            "status": "running",
            "reply": None,
        }

    def plan(state: RagState) -> RagState:
        if not any(d.status == "ready" for d in deps.store.list_documents()):
            return {"status": "chitchat", "reply": "Upload a document first, then ask about it."}
        question = state["question"]
        if not s.query_expansion:
            return {"queries": [question]}
        try:
            result = deps.llm.plan_query(_history(state), question)
        except LLMError as exc:
            log.warning("Query planning failed, searching the raw question: %s", exc)
            return {"queries": [question]}
        if not result.needs_retrieval and result.reply:
            return {"status": "chitchat", "reply": result.reply}
        standalone = result.standalone_question or question
        queries = [standalone, *result.search_queries[:3]]
        get_stream_writer()(
            {
                "event": "plan",
                "question": standalone,
                "queries": queries,
                "hypothetical": result.hypothetical_answer,
            }
        )
        return {
            "question": standalone,
            "queries": queries,
            "hypothetical": result.hypothetical_answer,
        }

    def retrieve(state: RagState) -> RagState:
        result = deps.retriever.search(
            state["queries"],
            rerank_query=state["question"],
            hypothetical=state.get("hypothetical"),
            document_ids=state.get("document_ids"),
        )
        get_stream_writer()(
            {
                "event": "retrieval",
                "round": state.get("round", 0) + 1,
                "queries": result.queries,
                "candidates": result.candidates,
                "confidence": result.confidence,
                "timings_ms": result.timings_ms,
                "passages": [_source(p) for p in result.passages],
            }
        )
        return {
            "passages": [p.model_dump() for p in result.passages],
            "confidence": result.confidence,
            "round": state.get("round", 0) + 1,
        }

    def reformulate(state: RagState) -> RagState:
        try:
            result = deps.llm.reformulate(state["question"], state["queries"])
            queries = [state["question"], *result.search_queries[:3]]
        except LLMError:
            queries = [state["question"]]
        get_stream_writer()({"event": "reformulate", "queries": queries})
        return {"queries": queries, "hypothetical": None}

    def generate(state: RagState) -> RagState:
        write = get_stream_writer()
        passages = [Passage.model_validate(p) for p in state["passages"]]
        blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        try:
            for event in deps.llm.stream_answer(_history(state), state["question"], passages):
                if event.kind == "text":
                    while len(blocks) <= event.block:
                        blocks.append("")
                    blocks[event.block] += event.text
                    write({"event": "delta", "block": event.block, "text": event.text})
                elif event.kind == "citation" and event.passage is not None:
                    cite = {"block": event.block, "passage": event.passage, "text": event.text}
                    citations.append(cite)
                    write({"event": "citation", **cite})
                elif event.kind == "stop" and event.stop_reason == "refusal":
                    raise LLMError("The model declined to answer this request.")
        except LLMError as exc:
            return _finish("failed", f"Sorry, I couldn't generate an answer: {exc}", passages)
        return _finish("answered", "".join(blocks), passages, citations, blocks)

    def abstain(state: RagState) -> RagState:
        passages = [Passage.model_validate(p) for p in state["passages"]][:3]
        text = (
            "I couldn't find information about this in your documents. "
            "The closest passages I found are listed below; try rephrasing or uploading a "
            "document that covers the topic."
        )
        return _finish("no_evidence", text, passages)

    def respond(state: RagState) -> RagState:
        return _finish(state.get("status", "chitchat"), state.get("reply") or "", [])

    def after_plan(state: RagState) -> str:
        return "respond" if state.get("status") == "chitchat" else "retrieve"

    def grade(state: RagState) -> str:
        confidence = state.get("confidence")
        if not state.get("passages"):
            weak = True
        elif confidence is None:  # no reranker configured: trust retrieval
            weak = False
        else:
            weak = confidence < s.min_relevance
        if not weak:
            return "generate"
        return "reformulate" if state.get("round", 0) < s.max_retrieval_rounds else "abstain"

    graph = StateGraph(RagState)
    for name, fn in [
        ("start", start),
        ("plan", plan),
        ("retrieve", retrieve),
        ("reformulate", reformulate),
        ("generate", generate),
        ("abstain", abstain),
        ("respond", respond),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "start")
    graph.add_edge("start", "plan")
    graph.add_conditional_edges("plan", after_plan, ["respond", "retrieve"])
    graph.add_conditional_edges("retrieve", grade, ["generate", "reformulate", "abstain"])
    graph.add_edge("reformulate", "retrieve")
    for terminal in ("generate", "abstain", "respond"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer)


def _history(state: RagState) -> list[Turn]:
    return [Turn(role=m["role"], content=m["content"]) for m in state["messages"][:-1]]


def _source(p: Passage) -> Source:
    return {
        "id": p.id,
        "document_id": p.document_id,
        "title": p.document_title,
        "location": p.location,
        "text": p.text,
        "relevance": p.relevance,
    }


def _finish(
    status: Status,
    text: str,
    passages: list[Passage],
    citations: list[dict[str, Any]] | None = None,
    blocks: list[str] | None = None,
) -> RagState:
    message: Message = {
        "role": "assistant",
        "content": text,
        "sources": [_source(p) for p in passages],
        "citations": citations or [],
    }
    get_stream_writer()(
        {
            "event": "answer",
            "status": status,
            "content": text,
            "blocks": blocks or [text],
            "sources": message["sources"],
            "citations": message["citations"],
        }
    )
    return {"status": status, "messages": [message]}
