"""Structure-aware chunking.

Chunks never span two sections, respect a token budget, split oversized paragraphs at sentence
boundaries and carry a small sentence-level overlap. Each chunk remembers its heading path
("Methods > Data collection") and page range, which are used for retrieval and for citations.
"""

import re
from dataclasses import dataclass, field

from docuchat.ingest.parsers import Block

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def count_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate; close to BPE counts for English prose."""
    return len(_TOKEN_RE.findall(text))


@dataclass
class RawChunk:
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    tokens: int


@dataclass
class _Piece:
    text: str
    page: int | None
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.text)


def chunk_blocks(blocks: list[Block], *, max_tokens: int, overlap_tokens: int) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    headings: list[tuple[int, str]] = []
    current: list[_Piece] = []

    def section_path() -> str | None:
        return " > ".join(text for _, text in headings) or None

    def emit(carry_overlap: bool) -> None:
        nonlocal current
        if not current:
            return
        pages = [p.page for p in current if p.page is not None]
        chunks.append(
            RawChunk(
                text=" ".join(p.text for p in current).strip(),
                section=section_path(),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                tokens=sum(p.tokens for p in current),
            )
        )
        current = _tail(current, overlap_tokens) if carry_overlap else []

    for block in blocks:
        if block.heading_level is not None:
            emit(carry_overlap=False)  # a new section starts a fresh chunk
            headings = [(lvl, t) for lvl, t in headings if lvl < block.heading_level]
            headings.append((block.heading_level, block.text))
            continue
        for piece in _split(block, max_tokens):
            if current and sum(p.tokens for p in current) + piece.tokens > max_tokens:
                emit(carry_overlap=True)
                # the overlap alone must leave room for the new piece
                while current and sum(p.tokens for p in current) + piece.tokens > max_tokens:
                    current.pop(0)
            current.append(piece)
    emit(carry_overlap=False)
    return chunks


def _split(block: Block, max_tokens: int) -> list[_Piece]:
    """A paragraph as one piece, or its sentences (hard-wrapped if still too long)."""
    whole = _Piece(block.text, block.page)
    if whole.tokens <= max_tokens:
        return [whole]
    pieces: list[_Piece] = []
    for sentence in _SENTENCE_RE.split(block.text):
        piece = _Piece(sentence, block.page)
        if piece.tokens <= max_tokens:
            pieces.append(piece)
            continue
        words = sentence.split()
        step = max(1, int(max_tokens * 0.7))
        pieces += [
            _Piece(" ".join(words[i : i + step]), block.page) for i in range(0, len(words), step)
        ]
    return pieces


def _tail(pieces: list[_Piece], budget: int) -> list[_Piece]:
    """Trailing sentences of the previous chunk, up to ``budget`` tokens, for overlap."""
    if budget <= 0:
        return []
    last = pieces[-1]
    sentences = [_Piece(s, last.page) for s in _SENTENCE_RE.split(last.text)]
    tail: list[_Piece] = []
    used = 0
    for sentence in reversed(sentences):
        if used + sentence.tokens > budget:
            break
        tail.insert(0, sentence)
        used += sentence.tokens
    return tail
