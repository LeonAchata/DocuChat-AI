"""Everything that talks to Claude: query transformation, contextual chunk headers and
grounded answers with native citations."""

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

import anthropic
import pydantic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam, BetaTextBlockParam
from pydantic import BaseModel, Field

from docuchat.config import Effort
from docuchat.models import Passage

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
MAX_CONTEXT_DOC_CHARS = 400_000  # ~100k tokens; longer documents are truncated for context


class LLMError(RuntimeError):
    pass


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryPlan(BaseModel):
    needs_retrieval: bool = Field(
        description="False only for greetings/small talk or questions about the assistant itself."
    )
    standalone_question: str = Field(
        description="The latest question rewritten to be self-contained using the conversation."
    )
    search_queries: list[str] = Field(
        description="2-3 diverse search queries: keyword-style reformulations, synonyms, or "
        "sub-questions for multi-part questions. Do not repeat the question verbatim."
    )
    hypothetical_answer: str | None = Field(
        default=None,
        description="A short plausible passage (2-3 sentences) that would answer the question, "
        "written as it might appear in a document. Used for HyDE retrieval.",
    )
    reply: str | None = Field(
        default=None, description="Direct reply when needs_retrieval is false; otherwise null."
    )


class Reformulation(BaseModel):
    search_queries: list[str] = Field(
        description="2-3 new search queries that approach the question differently from the "
        "ones that failed (broader terms, synonyms, related concepts)."
    )


@dataclass
class AnswerEvent:
    kind: Literal["text", "citation", "stop"]
    block: int = 0
    text: str = ""
    passage: int | None = None
    stop_reason: str | None = None


class AnswerLLM(Protocol):
    def plan_query(self, history: list[Turn], question: str) -> QueryPlan: ...
    def reformulate(self, question: str, failed: list[str]) -> Reformulation: ...
    def stream_answer(
        self, history: list[Turn], question: str, passages: list[Passage]
    ) -> Iterator[AnswerEvent]: ...


PLAN_SYSTEM = """\
You prepare searches over a user's private document collection (a hybrid keyword + semantic
index). Given the conversation and the latest message, produce a retrieval plan. Keep the user's
language for the standalone question; search queries may mix in English technical terms when
documents are likely in English."""

REFORMULATE_SYSTEM = """\
A search over the user's documents found nothing relevant. Propose different search queries that
could surface the answer: broaden, use synonyms, split compound questions, or search for the
underlying concept."""

ANSWER_SYSTEM = """\
You are a careful research assistant answering questions about the user's documents.

- Use only the provided documents. Cite the specific passages that support each claim.
- If the documents do not contain the answer, say so plainly and briefly state what they do
  cover. Never fill gaps with outside knowledge.
- If sources disagree, point out the disagreement.
- Be direct: lead with the answer, then supporting detail. Use short paragraphs or lists.
- Answer in the language of the question."""

CONTEXT_PROMPT = """\
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short, succinct context (1-2 sentences) that situates this chunk within the overall
document, to improve search retrieval of the chunk. Answer only with the context."""


class ClaudeLLM:
    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        refusal_fallbacks: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort: Effort = effort
        self.fallbacks = refusal_fallbacks and model in FALLBACK_MODELS
        self.client = client or anthropic.Anthropic()

    # ---- structured helpers ----------------------------------------------------------------

    def plan_query(self, history: list[Turn], question: str) -> QueryPlan:
        convo = "\n".join(f"{t.role}: {t.content}" for t in history[-8:])
        prompt = f"Conversation so far:\n{convo or '(none)'}\n\nLatest message:\n{question}"
        return self._structured(PLAN_SYSTEM, prompt, QueryPlan, effort="low")

    def reformulate(self, question: str, failed: list[str]) -> Reformulation:
        prompt = f"Question: {question}\nQueries that found nothing: {failed}"
        return self._structured(REFORMULATE_SYSTEM, prompt, Reformulation, effort="low")

    def _structured(self, system: str, prompt: str, schema: type[T], *, effort: Effort) -> T:
        output_config: BetaOutputConfigParam = {
            "effort": effort,
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        response = self._call(
            lambda: self.client.beta.messages.create(
                model=self.model,
                max_tokens=8_000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                **self._fallback_kwargs(),
            )
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"Model stopped early ({response.stop_reason}).")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate_json(text)
        except pydantic.ValidationError as exc:
            raise LLMError("The model did not return the expected structure.") from exc

    # ---- answers with citations ------------------------------------------------------------

    def stream_answer(
        self, history: list[Turn], question: str, passages: list[Passage]
    ) -> Iterator[AnswerEvent]:
        documents: list[dict[str, Any]] = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": p.text},
                "title": p.document_title[:200],
                "context": p.location or None,
                "citations": {"enabled": True},
            }
            for p in passages
        ]
        for doc in documents:
            if doc["context"] is None:
                del doc["context"]
        recent = history[-8:]
        while recent and recent[0].role != "user":  # the API expects a user turn first
            recent = recent[1:]
        messages: list[BetaMessageParam] = [{"role": t.role, "content": t.content} for t in recent]
        messages.append(
            {"role": "user", "content": [*documents, {"type": "text", "text": question}]}  # type: ignore[list-item]
        )
        system: list[BetaTextBlockParam] = [
            {"type": "text", "text": ANSWER_SYSTEM, "cache_control": {"type": "ephemeral"}}
        ]
        block = -1
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=16_000,
                system=system,
                messages=messages,
                output_config={"effort": self.effort},
                **self._fallback_kwargs(),
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start" and event.content_block.type == "text":
                        block += 1
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield AnswerEvent("text", block=max(block, 0), text=delta.text)
                        elif delta.type == "citations_delta":
                            citation = delta.citation
                            yield AnswerEvent(
                                "citation",
                                block=max(block, 0),
                                text=getattr(citation, "cited_text", ""),
                                passage=getattr(citation, "document_index", None),
                            )
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(_describe(exc)) from exc
        yield AnswerEvent("stop", stop_reason=final.stop_reason)

    # ---- contextual retrieval --------------------------------------------------------------

    def contextualize(self, document: str, chunks: list[str], workers: int = 4) -> list[str]:
        """Anthropic's Contextual Retrieval: situate every chunk within its document. The
        document is sent once per chunk but cached, so repeated reads are cheap."""
        doc_block: BetaTextBlockParam = {
            "type": "text",
            "text": f"<document>\n{document[:MAX_CONTEXT_DOC_CHARS]}\n</document>",
            "cache_control": {"type": "ephemeral"},
        }

        def one(chunk: str) -> str:
            try:
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=4_000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                doc_block,
                                {"type": "text", "text": CONTEXT_PROMPT.format(chunk=chunk)},
                            ],
                        }
                    ],
                    output_config={"effort": "low"},
                    **self._fallback_kwargs(),
                )
            except anthropic.APIError as exc:
                log.warning("Contextualization failed for a chunk: %s", exc)
                return ""
            if response.stop_reason == "refusal":
                return ""
            return "".join(b.text for b in response.content if b.type == "text").strip()

        if not chunks:
            return []
        first = one(chunks[0])  # warms the cache before the parallel requests
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [first, *pool.map(one, chunks[1:])]

    # ---- plumbing --------------------------------------------------------------------------

    def _fallback_kwargs(self) -> dict[str, Any]:
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.fallbacks else {}

    def _call(self, fn: Any) -> Any:
        try:
            return fn()
        except anthropic.APIError as exc:
            raise LLMError(_describe(exc)) from exc


def _describe(exc: anthropic.APIError) -> str:
    if isinstance(exc, anthropic.RateLimitError):
        return "The model is rate limited; try again shortly."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Model request failed ({exc.status_code}): {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the model API."
    return str(exc)
