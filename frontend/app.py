# Frontend simple con Streamlit para cargar PDF y hacer preguntas
import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("🤖 RAG PDF Chatbot")

# URL de la API
API_URL = "http://localhost:8000"

# Verificar estado del sistema
try:
    status = requests.get(f"{API_URL}/status").json()
    st.sidebar.success(f"✅ Sistema conectado")
    st.sidebar.metric("Documentos indexados", status["documents"])
    st.sidebar.metric("Chunks en BD", status["chunks"])
except:
    st.sidebar.error("❌ API no disponible")

st.markdown("---")

# Subida de PDF
st.subheader("📄 Subir y vectorizar PDF")
pdf_file = st.file_uploader("Sube un PDF", type=["pdf"])

if pdf_file:
    title = st.text_input("Título del documento (opcional)", pdf_file.name)
    if st.button("Indexar PDF"):
        with st.spinner("Vectorizando y guardando en la base de datos..."):
            try:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                data = {"title": title}
                response = requests.post(f"{API_URL}/ingest", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ PDF indexado: {result['chunks_count']} chunks creados")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error al indexar: {str(e)}")

st.markdown("---")

# Chatbot para preguntas
st.subheader("💬 Haz una pregunta sobre los PDFs indexados")
question = st.text_input("Pregunta:")
top_k = st.slider("Número de chunks de contexto", 1, 10, 5)

if st.button("Preguntar") and question:
    with st.spinner("Consultando..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": top_k}
            )
            if response.status_code == 200:
                data = response.json()
                st.markdown(f"**🤖 Respuesta:**")
                st.info(data['answer'])
                
                if data["context"]:
                    with st.expander("📚 Ver contexto usado"):
                        for idx, chunk in enumerate(data["context"], 1):
                            st.markdown(f"**Chunk {idx}:**")
                            st.text(chunk)
                            st.markdown("---")
            else:
                st.error("Error al consultar la API")
        except Exception as e:
            st.error(f"Error: {str(e)}")
