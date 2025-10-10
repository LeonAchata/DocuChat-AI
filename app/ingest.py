
import os
from typing import List, Optional
from app.models import Document, Chunk
from app.config import settings

# PDF loader: usa pdfplumber
import pdfplumber

def load_pdf(path: str) -> str:
	"""Carga el texto completo de un PDF."""
	with pdfplumber.open(path) as pdf:
		text = "\n".join(page.extract_text() or "" for page in pdf.pages)
	return text

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
	"""Divide el texto en chunks solapados."""
	chunk_size = chunk_size or settings.CHUNK_SIZE
	overlap = overlap or settings.CHUNK_OVERLAP
	words = text.split()
	chunks = []
	i = 0
	while i < len(words):
		chunk = " ".join(words[i:i+chunk_size])
		chunks.append(chunk)
		i += chunk_size - overlap
	return chunks


# --- Vectorización con bge-m3 ---
from app.embeddings import get_bge_model

def vectorize_chunks(chunks: List[str]) -> List[List[float]]:
	"""Obtiene los embeddings de cada chunk usando bge-m3."""
	# bge-m3 requiere normalización a norma 1 para pgvector/cosine
	model, device = get_bge_model()
	embeddings = model.encode(chunks, show_progress_bar=True, device=device, normalize_embeddings=True)
	return embeddings.tolist()

def ingest_pdf(path: str, source: str = "pdf", title: Optional[str] = None, uri: Optional[str] = None) -> List[Chunk]:
	"""Carga PDF, lo divide en chunks, vectoriza y prepara objetos Chunk."""
	text = load_pdf(path)
	chunk_texts = chunk_text(text)
	doc = Document(source=source, uri=uri or path, title=title)

	# Vectorizar usando singleton
	embeddings = vectorize_chunks(chunk_texts)

	chunks = []
	for idx, (chunk_str, emb) in enumerate(zip(chunk_texts, embeddings)):
		chunk = Chunk(
			document_id=None,  # Se asigna tras insertar el documento
			chunk_index=idx,
			content=chunk_str,
			content_tokens=len(chunk_str.split()),
			embedding=emb,
			metadata={}
		)
		chunks.append(chunk)
	return doc, chunks
