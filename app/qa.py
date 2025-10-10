
from app.models import Query, Answer
from app.retrieval import retrieve_chunks
from app.llm import generate_answer
from typing import List

def answer_query(query: Query) -> Answer:
    """
    Pipeline QA: recupera chunks y genera respuesta usando LLM.
    """
    context_chunks = retrieve_chunks(query)
    context_text = "\n".join([c.content for c in context_chunks])

    # Usa Groq API para generación rápida
    answer_text = generate_answer(context_text, query.question)

    return Answer(
        question=query.question,
        answer=answer_text,
        context=context_chunks,
        citations=None
    )
