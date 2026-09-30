"""Ingestion: file -> blocks -> chunks -> (optional LLM context) -> dense + sparse vectors -> store.

Documents are content-addressed (SHA-256), so re-uploading the same file is a no-op."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docuchat.config import Settings
from docuchat.index.encoders import DenseEncoder, SparseEncoder
from docuchat.index.store import Store
from docuchat.ingest.chunker import RawChunk, chunk_blocks
from docuchat.ingest.parsers import parse_file
from docuchat.models import Chunk, Document

log = logging.getLogger(__name__)

Contextualizer = Callable[[str, list[str]], list[str]]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def search_text(title: str, raw: RawChunk, context: str = "") -> str:
    """What gets embedded and keyword-indexed: breadcrumbs + optional context + the chunk."""
    header = title if not raw.section else f"{title} > {raw.section}"
    parts = [header, context.strip(), raw.text]
    return "\n".join(p for p in parts if p)


@dataclass
class Ingestor:
    store: Store
    dense: DenseEncoder
    settings: Settings
    sparse: SparseEncoder | None = None
    contextualizer: Contextualizer | None = None

    def register(self, path: Path, *, filename: str | None = None) -> tuple[Document, bool]:
        """Create (or find) the document record. Returns (document, needs_processing)."""
        digest = file_hash(path)
        existing = self.store.find_by_hash(digest)
        if existing and existing.status != "failed":
            return existing, False
        if existing:
            self.store.delete_document(existing.id)
        doc = Document(
            id=digest[:24],
            title=Path(filename or path.name).stem,
            filename=filename or path.name,
            content_hash=digest,
        )
        self.store.upsert_document(doc)
        return doc, True

    def process(self, doc: Document, path: Path) -> Document:
        """Parse, chunk, embed and index. Records failures on the document instead of raising."""
        try:
            parsed = parse_file(path)
            if not parsed.blocks:
                raise ValueError("No extractable text (scanned PDFs need OCR first).")
            raws = chunk_blocks(
                parsed.blocks,
                max_tokens=self.settings.chunk_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            contexts = [""] * len(raws)
            if self.contextualizer is not None:
                full_text = "\n\n".join(b.text for b in parsed.blocks)
                contexts = self.contextualizer(full_text, [r.text for r in raws])
            doc.title = parsed.title or doc.title
            chunks = [
                Chunk(
                    id=f"{doc.id}:{i}",
                    document_id=doc.id,
                    index=i,
                    text=raw.text,
                    search_text=search_text(doc.title, raw, ctx),
                    section=raw.section,
                    page_start=raw.page_start,
                    page_end=raw.page_end,
                    token_count=raw.tokens,
                )
                for i, (raw, ctx) in enumerate(zip(raws, contexts, strict=True))
            ]
            self.embed(chunks)
            self.store.add_chunks(chunks)
            doc.pages, doc.chunk_count, doc.status, doc.error = (
                parsed.pages,
                len(chunks),
                "ready",
                None,
            )
            log.info("Indexed %s: %d chunks", doc.filename, len(chunks))
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc.filename)
            doc.status, doc.error = "failed", str(exc)[:500]
        self.store.upsert_document(doc)
        return doc

    def ingest(self, path: Path, *, filename: str | None = None) -> Document:
        doc, needed = self.register(path, filename=filename)
        return self.process(doc, path) if needed else doc

    def embed(self, chunks: list[Chunk]) -> None:
        texts = [c.search_text for c in chunks]
        if not texts:
            return
        for chunk, vector in zip(chunks, self.dense.embed_documents(texts), strict=True):
            chunk.dense = vector
        if self.sparse is not None:
            for chunk, sparse in zip(chunks, self.sparse.embed_documents(texts), strict=True):
                chunk.sparse = sparse
