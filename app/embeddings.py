# Singleton para el modelo bge-m3
# Se carga una sola vez y se reutiliza en toda la aplicación

from sentence_transformers import SentenceTransformer
import torch
from typing import Optional
from app.config import settings

class BGEModelSingleton:
    _instance: Optional[SentenceTransformer] = None
    _device: Optional[str] = None
    
    @classmethod
    def get_model(cls):
        """Devuelve la instancia única del modelo bge-m3."""
        if cls._instance is None:
            cls._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Cargando modelo bge-m3 en {cls._device}...")
            cls._instance = SentenceTransformer(settings.BGE_MODEL_NAME, device=cls._device)
            print("Modelo bge-m3 cargado correctamente.")
        return cls._instance, cls._device

# Instancia global
_bge_singleton = BGEModelSingleton()

def get_bge_model():
    """Obtiene el modelo bge-m3 (singleton)."""
    return _bge_singleton.get_model()
