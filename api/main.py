
import sys
import os
# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.models import Query, Answer
from app.graph import build_rag_graph
from app.ingest import ingest_pdf
from app.db import insert_document, insert_chunk
import logging
import os
import tempfile

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG PDF Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compilar el grafo al inicio (una sola vez)
logger.info("Compilando grafo RAG...")
rag_graph = build_rag_graph()
logger.info("Grafo RAG compilado correctamente.")

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    context: list

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), title: str = None):
    """Endpoint para subir y indexar un PDF."""
    try:
        logger.info(f"Recibido archivo: {file.filename}")
        
        # Guardar temporalmente el PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Procesar e indexar
        doc, chunks = ingest_pdf(tmp_path, title=title or file.filename)
        doc_id = insert_document(doc)
        logger.info(f"Documento insertado con id: {doc_id}")
        
        for chunk in chunks:
            chunk.document_id = doc_id
            insert_chunk(chunk)
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        logger.info(f"PDF indexado correctamente: {len(chunks)} chunks")
        return {"status": "success", "document_id": str(doc_id), "chunks_count": len(chunks)}
    
    except Exception as e:
        logger.error(f"Error en /ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Endpoint para hacer preguntas sobre los PDFs indexados."""
    try:
        logger.info(f"Recibida pregunta: {request.question}")
        
        # Ejecutar el grafo RAG compilado
        state = {"query": Query(question=request.question, top_k=request.top_k)}
        result = rag_graph.invoke(state)
        
        if not result.get("retrieved_chunks"):
            logger.warning("No se encontraron chunks relevantes.")
            return AskResponse(answer="No se encontró contexto relevante para la pregunta.", context=[])
        
        answer = result.get("answer", "No se pudo generar respuesta.")
        context = [c.content for c in result.get("retrieved_chunks", [])]
        
        logger.info("Respuesta generada correctamente.")
        return AskResponse(answer=answer, context=context)
    
    except Exception as e:
        logger.error(f"Error en /ask: {e}")
        return AskResponse(answer="Ocurrió un error procesando la pregunta.", context=[])

@app.get("/status")
async def status():
    """Endpoint para verificar el estado del sistema."""
    from app.db import get_connection
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM public.documents;")
                doc_count = cur.fetchone()["count"]
                cur.execute("SELECT COUNT(*) as count FROM public.chunks;")
                chunk_count = cur.fetchone()["count"]
        return {
            "status": "ok",
            "documents": doc_count,
            "chunks": chunk_count
        }
    except Exception as e:
        logger.error(f"Error en /status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
