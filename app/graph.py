
# Grafo de orquestación del pipeline RAG usando LangGraph
# Nodos: retrieval y generación con estado tipado

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from app.models import Query, RetrievedChunk
from app.retrieval import retrieve_chunks
from app.llm import generate_answer

# Estado del grafo (tipado con TypedDict)
class RAGState(TypedDict):
    query: Query
    retrieved_chunks: List[RetrievedChunk]
    answer: str

def retrieval_node(state: RAGState) -> RAGState:
    """Nodo de retrieval: recupera chunks relevantes."""
    retrieved = retrieve_chunks(state["query"])
    state["retrieved_chunks"] = retrieved
    return state

def generation_node(state: RAGState) -> RAGState:
    """Nodo de generación: genera respuesta usando los chunks recuperados."""
    context_chunks = state["retrieved_chunks"]
    context_text = "\n".join([c.content for c in context_chunks])
    answer_text = generate_answer(context_text, state["query"].question)
    state["answer"] = answer_text
    return state

def build_rag_graph():
    """Construye el grafo RAG con nodos de retrieval y generación."""
    workflow = StateGraph(RAGState)
    
    # Agregar nodos
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    
    # Definir edges
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", END)
    
    # Compilar y retornar
    return workflow.compile()
