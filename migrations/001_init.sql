-- 1) Habilitar extensión unaccent (si no está creada)
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 2) Tabla documentos fuente
CREATE TABLE IF NOT EXISTS public.documents (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source      TEXT,
  uri         TEXT,
  title       TEXT,
  metadata    JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Índice único para evitar duplicados
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source_uri
  ON public.documents (source, uri);

-- 3) Tabla chunks (trozos indexables)
CREATE TABLE IF NOT EXISTS public.chunks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  chunk_index   INT NOT NULL,
  content       TEXT NOT NULL,
  content_tokens INT,
  embedding     vector(1024),
  tsv           tsvector,
  metadata      JSONB DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4) Función trigger para actualizar tsvector con unaccent y to_tsvector
CREATE OR REPLACE FUNCTION update_chunks_tsv()
RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', unaccent(coalesce(NEW.content, '')));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- 5) Trigger que ejecuta la función antes de insertar o actualizar
CREATE TRIGGER trg_chunks_tsv
BEFORE INSERT OR UPDATE ON public.chunks
FOR EACH ROW
EXECUTE FUNCTION update_chunks_tsv();

-- 6) Índices para búsqueda

-- Índice GIN para búsqueda full-text sobre tsvector
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
  ON public.chunks USING GIN (tsv);

-- Índice vectorial usando HNSW (ajusta según versión pgvector)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON public.chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Índices para filtros habituales
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
  ON public.chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_created_at
  ON public.chunks (created_at);
