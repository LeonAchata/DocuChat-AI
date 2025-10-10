

import os
from typing import List
from app.models import Query, RetrievedChunk
from app.db import get_connection
from app.embeddings import get_bge_model

def retrieve_chunks(query: Query) -> List[RetrievedChunk]:
    """
    Recupera chunks relevantes usando búsqueda híbrida (vector + full-text).
    """
    model, device = get_bge_model()
    q_embedding = model.encode([query.question], device=device, normalize_embeddings=True)[0]
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Vector search (top 30)
            cur.execute(
                """
                SELECT id, document_id, chunk_index, content, metadata,
                       1 - (embedding <=> %s::vector) AS vec_score
                FROM public.chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 30;
                """,
                (q_embedding, q_embedding)
            )
            vec_results = cur.fetchall()

            # Full-text search (top 30)
            cur.execute(
                """
                SELECT id, document_id, chunk_index, content, metadata,
                       ts_rank(tsv, plainto_tsquery('simple', %s)) AS fts_score
                FROM public.chunks
                WHERE tsv @@ plainto_tsquery('simple', %s)
                ORDER BY fts_score DESC
                LIMIT 30;
                """,
                (query.question, query.question)
            )
            fts_results = cur.fetchall()

    # Fusionar resultados (simple: unir y rankear por score combinado)
    chunk_map = {}
    for r in vec_results:
        chunk_map[r['id']] = RetrievedChunk(
            id=r['id'],
            content=r['content'],
            score=r['vec_score'],
            document_id=r['document_id'],
            chunk_index=r['chunk_index'],
            metadata=r['metadata']
        )
    for r in fts_results:
        if r['id'] in chunk_map:
            # Combina score (ajusta pesos si quieres)
            chunk_map[r['id']].score = 0.7 * chunk_map[r['id']].score + 0.3 * min(r['fts_score'], 1)
        else:
            chunk_map[r['id']] = RetrievedChunk(
                id=r['id'],
                content=r['content'],
                score=min(r['fts_score'], 1),
                document_id=r['document_id'],
                chunk_index=r['chunk_index'],
                metadata=r['metadata']
            )
    # Ordena por score combinado y devuelve top_k
    results = sorted(chunk_map.values(), key=lambda x: x.score, reverse=True)
    return results[:query.top_k]
