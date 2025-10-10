# Cliente LLM usando Groq API (rápido y gratis)
# Singleton para reutilizar la instancia

from groq import Groq
from typing import Optional
from app.config import settings

class GroqLLMSingleton:
    _instance: Optional[Groq] = None
    
    @classmethod
    def get_client(cls):
        """Devuelve la instancia única del cliente Groq."""
        if cls._instance is None:
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY no está configurada en .env")
            print("Inicializando cliente Groq...")
            cls._instance = Groq(api_key=settings.GROQ_API_KEY)
            print("Cliente Groq inicializado correctamente.")
        return cls._instance

# Instancia global
_groq_singleton = GroqLLMSingleton()

def get_groq_client():
    """Obtiene el cliente Groq (singleton)."""
    return _groq_singleton.get_client()

def generate_answer(context: str, question: str) -> str:
    """Genera respuesta usando Groq API."""
    client = get_groq_client()
    prompt = f"""Responde la pregunta usando SOLO el contexto proporcionado. Si no encuentras la respuesta en el contexto, di "No tengo suficiente información para responder".

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"
