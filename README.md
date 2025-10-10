# RAG PDF Chatbot - README Completo

Sistema extremo a extremo para preguntas y respuestas sobre PDFs usando RAG, bge-m3, LangGraph, Groq API y pgvector.

## 🎯 ¿Qué hace?
- Permite subir PDFs y vectorizarlos automáticamente desde el frontend.
- Usa **bge-m3** para embeddings (con GPU/CPU fallback automático).
- Almacena vectores en **PostgreSQL con pgvector**.
- Recupera contexto relevante usando **búsqueda híbrida** (vectorial + full-text).
- Genera respuestas con **Groq API** (LLaMA 3.1, rápido y gratis).
- Frontend simple con **Streamlit**.
- Orquestación con **LangGraph** (retrieval → generation).

## 🏗️ Estructura
```
RAG1/
├── app/              # Lógica principal
│   ├── config.py     # Configuración centralizada (BaseSettings)
│   ├── embeddings.py # Singleton para bge-m3
│   ├── llm.py        # Cliente Groq API
│   ├── ingest.py     # Carga y chunking de PDFs
│   ├── retrieval.py  # Búsqueda híbrida
│   ├── qa.py         # Pipeline QA
│   ├── graph.py      # Grafo LangGraph
│   ├── db.py         # Conexión PostgreSQL
│   └── models.py     # Modelos Pydantic
├── api/              # API REST (FastAPI)
│   └── main.py       # Endpoints: /ingest, /ask, /status
├── frontend/         # Streamlit chatbot
│   └── app.py
├── scripts/          # CLI para ingestión manual
│   └── ingest_pdf.py
├── migrations/       # Scripts SQL
│   └── 001_init.sql
├── data/             # PDFs de ejemplo
├── requirements.txt  # Dependencias
├── .env.example      # Template de configuración
└── README.md
```

## 📦 Instalación

### 1. Clonar e instalar dependencias
```powershell
pip install -r requirements.txt
```

### 2. Configurar PostgreSQL
Ejecuta el script SQL para crear las tablas e índices:
```powershell
psql -U postgres -d RAG -f migrations/001_init.sql
```

### 3. Configurar variables de entorno
Copia `.env.example` a `.env` y completa:
```bash
# Base de datos
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=RAG
POSTGRES_USER=postgres
POSTGRES_PASSWORD=tu_password

# Groq API (obtén gratis en https://console.groq.com)
GROQ_API_KEY=tu_api_key_aqui

# Modelos (opcional, usa valores por defecto)
BGE_MODEL_NAME=BAAI/bge-m3
LLM_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

## 🚀 Ejecución

### 1. Iniciar la API
```powershell
uvicorn api.main:app --reload
```

### 2. Iniciar el frontend
```powershell
streamlit run frontend/app.py
```

### 3. Usar el chatbot
1. Sube un PDF desde el frontend.
2. Haz clic en "Indexar PDF" (se vectoriza automáticamente).
3. Haz preguntas sobre el contenido del PDF.

## 🔧 Características técnicas

### Singleton para bge-m3
- El modelo se carga **una sola vez** al inicio.
- Se reutiliza para ingestión y queries.
- Fallback automático GPU → CPU.

### LangGraph con estado tipado
- Estado tipado con `TypedDict` (no dicts genéricos).
- Grafo compilado al inicio (`workflow.compile()`).
- Nodos: retrieval → generation.

### Groq API
- Sin descargas de modelos pesados.
- Respuestas en <1 segundo.
- Gratis para uso personal.

### Búsqueda híbrida
- Combina búsqueda vectorial (pgvector) + full-text (tsvector).
- Fusión de scores con pesos ajustables.

### Endpoints API
- `POST /ingest`: Subir y vectorizar PDF.
- `POST /ask`: Hacer preguntas.
- `GET /status`: Ver documentos y chunks indexados.

## 📝 Uso manual (CLI)
Si prefieres indexar desde terminal:
```powershell
python scripts/ingest_pdf.py data/tu_pdf.pdf "Título opcional"
```

## 🎨 Personalización

### Cambiar el modelo LLM
Edita en `.env`:
```bash
LLM_MODEL=llama-3.1-70b-versatile  # Modelo más potente
```

### Ajustar chunking
Edita en `.env`:
```bash
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### Agregar nodos al grafo
Edita `app/graph.py` y agrega nodos como rewriter o reranker:
```python
workflow.add_node("rerank", rerank_node)
workflow.add_edge("retrieval", "rerank")
workflow.add_edge("rerank", "generation")
```

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY no está configurada"
Obtén una API key gratis en https://console.groq.com y agrégala al `.env`.

### Error: "Import psycopg could not be resolved"
Instala las dependencias:
```powershell
pip install -r requirements.txt
```

### Error: "Could not connect to database"
Verifica que PostgreSQL esté corriendo y los datos en `.env` sean correctos.

### bge-m3 se descarga muy lento
Solo se descarga la primera vez (~2GB). Después se cachea en `~/.cache/huggingface/`.

## 📊 Stack tecnológico
- **Backend**: Python 3.10+, FastAPI
- **RAG**: LangChain, LangGraph
- **Embeddings**: bge-m3 (BAAI)
- **LLM**: Groq API (LLaMA 3.1)
- **Base de datos**: PostgreSQL 15+ con pgvector
- **Frontend**: Streamlit
- **Validación**: Pydantic BaseModel y BaseSettings

## 🎓 Autor
Proyecto generado para demostrar habilidades en RAG, embeddings, bases vectoriales y LangGraph.
