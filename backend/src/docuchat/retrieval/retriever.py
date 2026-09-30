"""Hybrid retrieval pipeline.

queries (+ HyDE) --> dense | BM25/full-text | SPLADE   (per query variant)
                 --> weighted reciprocal rank fusion
                 --> cross-encoder rerank
                 --> MMR diversification
                 --> neighbour-window expansion --> passages
"""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from docuchat.config import Settings
from docuchat.index.encoders import DenseEncoder, Reranker, SparseEncoder
from docuchat.index.store import Scored, Store
from docuchat.models import Chunk, Hit, Passage
from docuchat.retrieval.fusion import maximal_marginal_relevance, reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    queries: list[str]
    candidates: int
    hits: list[Hit]
    passages: list[Passage]
    confidence: float | None
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class HybridRetriever:
    store: Store
    dense: DenseEncoder
    settings: Settings
    sparse: SparseEncoder | None = None
    reranker: Reranker | None = None

    def rank(
        self,
        queries: Sequence[str],
        *,
        rerank_query: str | None = None,
        hypothetical: str | None = None,
        document_ids: Sequence[str] | None = None,
        use: Sequence[str] = ("dense", "lexical", "sparse"),
        rerank: bool = True,
        timings: dict[str, int] | None = None,
    ) -> list[Hit]:
        """Full ranked candidate list: fused, with the head reranked by the cross-encoder."""
        s = self.settings
        timings = timings if timings is not None else {}
        k = s.candidates_per_retriever

        clock = time.perf_counter()
        ranked: dict[str, Scored] = {}
        for i, query in enumerate(queries):
            if "dense" in use:
                ranked[f"dense:{i}"] = self.store.dense_search(
                    self.dense.embed_query(query), k, document_ids
                )
            if "lexical" in use:
                ranked[f"lexical:{i}"] = self.store.lexical_search(query, k, document_ids)
            if "sparse" in use and self.sparse is not None:
                ranked[f"sparse:{i}"] = self.store.sparse_search(
                    self.sparse.embed_query(query), k, document_ids
                )
        if hypothetical and "dense" in use:
            # HyDE: a hypothetical answer lives in "document space", so embed it as a document.
            vector = self.dense.embed_documents([hypothetical])[0]
            ranked["dense:hyde"] = self.store.dense_search(vector, k, document_ids)
        timings["search"] = _ms(clock)

        weights = {"dense": s.weight_dense, "lexical": s.weight_lexical, "sparse": s.weight_sparse}
        fused = reciprocal_rank_fusion(ranked, weights, k=s.rrf_k)
        if not (rerank and self.reranker is not None and fused):
            return fused

        clock = time.perf_counter()
        head, tail = fused[: s.rerank_candidates], fused[s.rerank_candidates :]
        scores = self.reranker.score(
            rerank_query or queries[0], [h.chunk.search_text for h in head]
        )
        for hit, score in zip(head, scores, strict=True):
            hit.rerank_score = score
        head.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        timings["rerank"] = _ms(clock)
        return head + tail

    def search(
        self,
        queries: Sequence[str],
        *,
        rerank_query: str | None = None,
        hypothetical: str | None = None,
        document_ids: Sequence[str] | None = None,
        use: Sequence[str] = ("dense", "lexical", "sparse"),
        rerank: bool = True,
    ) -> RetrievalResult:
        """Retrieve, diversify with MMR and expand hits into passages for generation."""
        s = self.settings
        timings: dict[str, int] = {}
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        ranked = self.rank(
            queries,
            rerank_query=rerank_query,
            hypothetical=hypothetical,
            document_ids=document_ids,
            use=use,
            rerank=rerank,
            timings=timings,
        )
        confidence = ranked[0].rerank_score if ranked else None
        pool = ranked[: s.top_k * 3]
        hits = maximal_marginal_relevance(pool, top_k=s.top_k, lambda_=s.mmr_lambda)
        clock = time.perf_counter()
        passages = self.build_passages(hits)
        timings["expand"] = _ms(clock)
        return RetrievalResult(
            queries=queries,
            candidates=len(ranked),
            hits=hits,
            passages=passages,
            confidence=confidence,
            timings_ms=timings,
        )

    def build_passages(self, hits: list[Hit]) -> list[Passage]:
        """Expand each hit with up to ``window`` neighbours from the same section, then merge
        overlapping or adjacent spans of the same document into one passage."""
        w = self.settings.window
        spans: dict[str, list[_Span]] = defaultdict(list)
        order: list[str] = []
        for hit in hits:
            c = hit.chunk
            if c.document_id not in spans:
                order.append(c.document_id)
            around = self.store.neighbors(
                c.document_id, range(max(c.index - w, 0), c.index + w + 1)
            )
            same = {n.index for n in around if n.section == c.section} | {c.index}
            start = end = c.index
            while start - 1 in same:
                start -= 1
            while end + 1 in same:
                end += 1
            spans[c.document_id].append(_Span(start, end, hit.rerank_score, c.section))

        passages: list[Passage] = []
        for document_id in order:
            doc = self.store.get_document(document_id)
            title = doc.title if doc else "Untitled"
            for span in _merge_spans(spans[document_id]):
                chunks = self.store.neighbors(document_id, range(span.start, span.end + 1))
                if chunks:
                    passages.append(_passage(len(passages), document_id, title, chunks, span))
        passages.sort(key=lambda p: p.relevance or 0.0, reverse=True)
        for i, p in enumerate(passages):
            p.id = i
        return passages


@dataclass
class _Span:
    start: int
    end: int
    relevance: float | None
    section: str | None  # section of the most relevant hit in the span


def _merge_spans(spans: list[_Span]) -> list[_Span]:
    merged: list[_Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        prev = merged[-1] if merged else None
        if prev is None or span.start > prev.end + 1 or span.section != prev.section:
            merged.append(span)
            continue
        prev.end = max(prev.end, span.end)
        if (span.relevance or 0.0) > (prev.relevance or 0.0):
            prev.relevance, prev.section = span.relevance, span.section
    return merged


def _passage(pid: int, document_id: str, title: str, chunks: list[Chunk], span: _Span) -> Passage:
    text = chunks[0].text
    for chunk in chunks[1:]:
        text = _join_overlapping(text, chunk.text)
    pages = [p for c in chunks for p in (c.page_start, c.page_end) if p is not None]
    return Passage(
        id=pid,
        document_id=document_id,
        document_title=title,
        text=text,
        section=span.section,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        chunk_ids=[c.id for c in chunks],
        relevance=span.relevance,
    )


def _join_overlapping(left: str, right: str, max_overlap: int = 2000) -> str:
    """Concatenate consecutive chunks, dropping the sentence overlap they share."""
    for n in range(min(len(left), len(right), max_overlap), 20, -1):
        if left.endswith(right[:n]):
            return left + right[n:]
    return f"{left}\n\n{right}"


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
