
import psycopg
from psycopg.rows import dict_row
from typing import Any, List, Optional
from app.models import Document, Chunk
from app.config import settings

def get_connection():
	"""Devuelve una conexión a la base de datos usando settings centralizado."""
	return psycopg.connect(
		host=settings.POSTGRES_HOST,
		port=settings.POSTGRES_PORT,
		dbname=settings.POSTGRES_DB,
		user=settings.POSTGRES_USER,
		password=settings.POSTGRES_PASSWORD,
		autocommit=True,
		row_factory=dict_row
	)

def insert_document(doc: Document) -> Any:
	"""Inserta un documento y retorna el id."""
	with get_connection() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO public.documents (source, uri, title, metadata)
				VALUES (%s, %s, %s, %s)
				RETURNING id;
				""",
				(doc.source, doc.uri, doc.title, doc.metadata)
			)
			result = cur.fetchone()
			return result["id"] if result else None

def insert_chunk(chunk: Chunk) -> Any:
	"""Inserta un chunk con embedding."""
	with get_connection() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO public.chunks (
					document_id, chunk_index, content, content_tokens, embedding, metadata
				) VALUES (%s, %s, %s, %s, %s, %s)
				RETURNING id;
				""",
				(
					chunk.document_id,
					chunk.chunk_index,
					chunk.content,
					chunk.content_tokens,
					chunk.embedding,
					chunk.metadata
				)
			)
			result = cur.fetchone()
			return result["id"] if result else None

def get_document_by_id(doc_id: str) -> Optional[dict]:
	"""Obtiene un documento por id."""
	with get_connection() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT * FROM public.documents WHERE id = %s;", (doc_id,))
			return cur.fetchone()

def get_chunks_by_document(doc_id: str) -> List[dict]:
	"""Obtiene todos los chunks de un documento."""
	with get_connection() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT * FROM public.chunks WHERE document_id = %s ORDER BY chunk_index;", (doc_id,))
			return cur.fetchall()
