"""Retrieval benchmark on BEIR datasets: runs each retrieval configuration over the same
index and reports nDCG@10, MRR@10, Recall@10 and Recall@100.

The index is built once per dataset and model and cached under ``data/beir``."""

import csv
import io
import json
import math
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from docuchat.config import Settings
from docuchat.index.encoders import FastEmbedDense, FastEmbedReranker, FastEmbedSparse
from docuchat.index.memory import MemoryStore
from docuchat.models import Chunk, SparseVector
from docuchat.retrieval.retriever import HybridRetriever

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Config:
    name: str
    use: tuple[str, ...]
    rerank: bool = False


CONFIGS = [
    Config("BM25", ("lexical",)),
    Config("Dense", ("dense",)),
    Config("SPLADE", ("sparse",)),
    Config("Hybrid: BM25 + dense", ("lexical", "dense")),
    Config("Hybrid: BM25 + dense + SPLADE", ("lexical", "dense", "sparse")),
    Config("Hybrid + cross-encoder rerank", ("lexical", "dense", "sparse"), rerank=True),
]


def run_beir(
    settings: Settings,
    *,
    dataset: str = "scifact",
    max_queries: int | None = None,
    output: Path | None = None,
    console: Console,
) -> list[dict[str, object]]:
    root = _download(dataset, settings.data_dir / "beir")
    corpus = _read_jsonl(root / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in _read_jsonl(root / "queries.jsonl")}
    qrels = _read_qrels(root / "qrels" / "test.tsv")
    query_ids = [q for q in qrels if q in queries][:max_queries]

    cache = settings.data_dir / "models"
    dense = FastEmbedDense(settings.dense_model, cache)
    sparse = FastEmbedSparse(settings.sparse_model, cache) if settings.sparse_model else None
    reranker = (
        FastEmbedReranker(settings.reranker_model, cache) if settings.reranker_model else None
    )

    store = MemoryStore()
    chunks = [
        Chunk(
            id=d["_id"],
            document_id=d["_id"],
            index=0,
            text=d["text"],
            search_text=f"{d.get('title', '')}\n{d['text']}".strip(),
        )
        for d in corpus
    ]
    _embed_cached(chunks, dense, sparse, root.parent / f"{dataset}-cache", console)
    store.add_chunks(chunks)

    eval_settings = settings.model_copy(
        update={"candidates_per_retriever": 100, "rerank_candidates": 50}
    )
    retriever = HybridRetriever(
        store=store, dense=dense, sparse=sparse, reranker=reranker, settings=eval_settings
    )

    rows: list[dict[str, object]] = []
    for config in CONFIGS:
        if ("sparse" in config.use and sparse is None) or (config.rerank and reranker is None):
            continue
        metrics: dict[str, list[float]] = defaultdict(list)
        started = time.perf_counter()
        for qid in track(query_ids, description=config.name, console=console):
            ranking = [
                h.chunk.document_id
                for h in retriever.rank([queries[qid]], use=config.use, rerank=config.rerank)
            ]
            for name, value in _metrics(ranking, qrels[qid]).items():
                metrics[name].append(value)
        elapsed = (time.perf_counter() - started) / max(len(query_ids), 1)
        rows.append(
            {
                "config": config.name,
                **{k: round(float(np.mean(v)), 4) for k, v in metrics.items()},
                "ms_per_query": round(elapsed * 1000),
            }
        )

    _print(rows, dataset, len(query_ids), settings, console)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "queries": len(query_ids),
                    "documents": len(corpus),
                    "dense_model": settings.dense_model,
                    "sparse_model": settings.sparse_model,
                    "reranker_model": settings.reranker_model,
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return rows


def _metrics(ranking: list[str], relevant: dict[str, int]) -> dict[str, float]:
    dcg = sum(
        (2 ** relevant.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(ranking[:10])
    )
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    rr = next((1 / (i + 1) for i, doc in enumerate(ranking[:10]) if doc in relevant), 0.0)
    positives = [d for d, g in relevant.items() if g > 0]
    return {
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "mrr@10": rr,
        "recall@10": len(set(ranking[:10]) & set(positives)) / len(positives),
        "recall@100": len(set(ranking[:100]) & set(positives)) / len(positives),
    }


def _embed_cached(
    chunks: list[Chunk],
    dense: FastEmbedDense,
    sparse: FastEmbedSparse | None,
    cache_dir: Path,
    console: Console,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dense_file = cache_dir / f"dense-{_slug(dense.name)}.npy"
    texts = [c.search_text for c in chunks]
    if dense_file.exists():
        vectors = np.load(dense_file)
    else:
        console.print(f"Embedding {len(texts)} documents with {dense.name}…")
        vectors = dense.embed_documents(texts)
        np.save(dense_file, vectors)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.dense = vector

    if sparse is None:
        return
    sparse_file = cache_dir / f"sparse-{_slug(sparse.name)}.json"
    if sparse_file.exists():
        encoded = [SparseVector.model_validate(v) for v in json.loads(sparse_file.read_text())]
    else:
        console.print(f"Encoding {len(texts)} documents with {sparse.name}…")
        encoded = sparse.embed_documents(texts)
        sparse_file.write_text(json.dumps([v.model_dump() for v in encoded]))
    for chunk, vec in zip(chunks, encoded, strict=True):
        chunk.sparse = vec


def _download(name: str, root: Path) -> Path:
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    response = httpx.get(BEIR_URL.format(name=name), timeout=300, follow_redirects=True)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(root)
    return target


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return qrels


def _slug(name: str) -> str:
    return name.replace("/", "__")


def _print(
    rows: list[dict[str, object]], dataset: str, n: int, settings: Settings, console: Console
) -> None:
    columns = ("config", "ndcg@10", "mrr@10", "recall@10", "recall@100", "ms_per_query")
    table = Table(title=f"BEIR {dataset} - {n} test queries")
    for column in columns:
        table.add_column(column, justify="left" if column == "config" else "right")
    for row in rows:
        table.add_row(*(str(row[c]) for c in columns))
    console.print(table)
    console.print(
        f"[dim]dense={settings.dense_model} sparse={settings.sparse_model} "
        f"reranker={settings.reranker_model}[/]"
    )
