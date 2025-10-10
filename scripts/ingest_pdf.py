
"""
Script CLI para cargar un PDF, vectorizarlo y guardar en la base de datos.
Uso:
	python scripts/ingest_pdf.py ruta/al/archivo.pdf "Titulo opcional"
"""
import sys
from app.ingest import ingest_pdf
from app.db import insert_document, insert_chunk

def main():
	if len(sys.argv) < 2:
		print("Uso: python scripts/ingest_pdf.py ruta/al/archivo.pdf [titulo]")
		sys.exit(1)
	pdf_path = sys.argv[1]
	title = sys.argv[2] if len(sys.argv) > 2 else None

	doc, chunks = ingest_pdf(pdf_path, title=title)
	doc_id = insert_document(doc)
	print(f"Documento insertado con id: {doc_id}")

	for chunk in chunks:
		chunk.document_id = doc_id
		chunk_id = insert_chunk(chunk)
		print(f"Chunk {chunk.chunk_index} insertado con id: {chunk_id}")

if __name__ == "__main__":
	main()
