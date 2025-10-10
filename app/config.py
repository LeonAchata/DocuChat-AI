# Configuración centralizada usando pydantic_settings.BaseSettings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base de datos
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    
    # Modelos
    BGE_MODEL_NAME: str = "BAAI/bge-m3"
    
    # LLM API (Groq es gratis y rápido)
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.1-8b-instant"  # Groq model
    
    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
