from pathlib import Path

import numpy as np
import pytest

from docuchat.config import Settings
from docuchat.evals import _metrics
from docuchat.index.encoders import HashingDense, HashingSparse
from docuchat.index.memory import MemoryStore
from docuchat.index.store import Store
from docuchat.models import Chunk, Document, Hit
from docuchat.retrieval.fusion import maximal_marginal_relevance, reciprocal_rank_fusion
from docuchat.retrieval.retriever import _join_overlapping
from docuchat.service import Components
from tests.conftest import postgres_url


def chunk(cid: str, text: str, doc: str = "d1", index: int = 0) -> Chunk:
    c = Chunk(id=cid, document_id=doc, index=index, text=text, search_text=text)
    c.dense = HashingDense().embed_query(text)
    c.sparse = HashingSparse().embed_query(text)
    return c


CORPUS = [
    chunk("a", "The cat sat on the mat.", "d1", 0),
    chunk("b", "Dogs are loyal companions and love long walks.", "d1", 1),
    chunk("c", "Quantum computers use qubits to perform calculations.", "d2", 0),
    chunk("d", "A kitten is a young cat that loves to play.", "d2", 1),
]


def make_store(kind: str, tmp_path: Path) -> Store:
    if kind == "memory":
        return MemoryStore(tmp_path / "idx")
    from docuchat.index.postgres import PostgresStore

    pg = PostgresStore(postgres_url() or "", dim=256)
    for d in pg.list_documents():
        pg.delete_document(d.id)
    return pg


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    if request.param == "postgres" and not postgres_url():
        pytest.skip("DOCUCHAT_TEST_DATABASE_URL not set")
    s = make_store(request.param, tmp_path)
    for doc_id in ("d1", "d2"):
        s.upsert_document(
            Document(
                id=doc_id,
                title=doc_id,
                filename=f"{doc_id}.md",
                content_hash=doc_id,
                status="ready",
            )
        )
    s.add_chunks([c.model_copy() for c in CORPUS])
    return s


def test_store_contract(store: Store) -> None:
    dense = store.dense_search(HashingDense().embed_query("young cat kitten"), 2)
    assert dense[0][0].id == "d"
    lexical = store.lexical_search("qubits calculations", 3)
    assert [c.id for c, _ in lexical][:1] == ["c"]
    sparse = store.sparse_search(HashingSparse().embed_query("loyal dogs"), 2)
    assert sparse[0][0].id == "b"
    filtered = store.lexical_search("cat", 5, document_ids=["d1"])
    assert {c.document_id for c, _ in filtered} == {"d1"}
    assert [c.id for c in store.neighbors("d2", [0, 1, 7])] == ["c", "d"]
    assert store.find_by_hash("d2") is not None
    store.delete_document("d2")
    assert store.lexical_search("qubits", 3) == []
    assert store.get_document("d2") is None


def test_memory_store_persists(tmp_path: Path) -> None:
    s = MemoryStore(tmp_path / "idx")
    s.upsert_document(Document(id="d1", title="t", filename="f", content_hash="h", status="ready"))
    s.add_chunks([CORPUS[0], CORPUS[1]])
    reloaded = MemoryStore(tmp_path / "idx")
    assert reloaded.get_document("d1") is not None
    hit = reloaded.dense_search(HashingDense().embed_query("cat mat"), 1)[0][0]
    assert hit.id == "a" and hit.dense is not None
    assert reloaded.sparse_search(HashingSparse().embed_query("dogs"), 1)[0][0].id == "b"


def test_rrf_rewards_agreement_and_applies_weights() -> None:
    a, b, c = CORPUS[0], CORPUS[1], CORPUS[2]
    ranked = {"dense:0": [(a, 0.9), (b, 0.8)], "lexical:0": [(b, 12.0), (c, 3.0)]}
    fused = reciprocal_rank_fusion(ranked, {"dense": 1.0, "lexical": 1.0}, k=60)
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]
    assert fused[0].ranks == {"dense:0": 2, "lexical:0": 1}
    heavy = reciprocal_rank_fusion(ranked, {"dense": 1.0, "lexical": 0.0}, k=60)
    assert heavy[0].chunk.id == "a"


def test_mmr_skips_near_duplicates() -> None:
    dup = chunk("a2", "The cat sat on the mat.")
    hits = [
        Hit(chunk=CORPUS[0], rerank_score=0.9),
        Hit(chunk=dup, rerank_score=0.89),
        Hit(chunk=CORPUS[2], rerank_score=0.5),
    ]
    picked = maximal_marginal_relevance(hits, top_k=2, lambda_=0.5)
    assert [h.chunk.id for h in picked] == ["a", "c"]
    assert [h.chunk.id for h in maximal_marginal_relevance(hits, top_k=2, lambda_=1.0)] == [
        "a",
        "a2",
    ]


def test_join_overlapping_chunks() -> None:
    left = "First sentence here. Second sentence is shared across chunks."
    right = "Second sentence is shared across chunks. Third sentence follows."
    assert _join_overlapping(left, right) == (
        "First sentence here. Second sentence is shared across chunks. Third sentence follows."
    )
    assert _join_overlapping("abc", "xyz") == "abc\n\nxyz"


def test_hybrid_search_end_to_end(components: Components) -> None:
    result = components.retriever.search(
        ["How many vacation days do employees get?"], rerank_query="vacation days per year"
    )
    top = result.passages[0]
    assert top.document_title == "Employee Handbook"
    assert "25 days of paid vacation" in top.text
    assert top.section and "Vacation policy" in top.section
    assert result.confidence is not None and result.confidence > 0.5
    assert {"search", "rerank", "expand"} <= set(result.timings_ms)
    # neighbouring chunks are merged into one passage, without duplicated overlap
    assert top.text.count("25 days of paid vacation") == 1
    # window expansion never crosses into another section
    assert "stipend" not in top.text


def test_document_filter(components: Components) -> None:
    security = next(
        d for d in components.store.list_documents() if d.title == "Security Guidelines"
    )
    result = components.retriever.search(["vacation days"], document_ids=[security.id])
    assert all(p.document_id == security.id for p in result.passages)


def test_retrieval_metrics() -> None:
    metrics = _metrics(["x", "a", "y", "b"], {"a": 1, "b": 1})
    assert metrics["mrr@10"] == 0.5
    assert metrics["recall@10"] == 1.0
    ideal = 1 + 1 / np.log2(3)
    assert metrics["ndcg@10"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)


def test_settings_defaults_are_sane() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.rerank_candidates >= s.top_k and s.candidates_per_retriever >= s.top_k
