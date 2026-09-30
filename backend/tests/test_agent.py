from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from docuchat.config import Settings
from docuchat.generation.claude import LLMError, QueryPlan
from docuchat.service import Components, DocuChat
from tests.conftest import FakeLLM, make_components


def run(chat: DocuChat, question: str, thread: str = "t1") -> list[dict[str, Any]]:
    return list(chat.stream(question, thread))


def with_llm(components: Components, llm: FakeLLM) -> DocuChat:
    components.llm = llm
    return DocuChat(components, checkpointer=InMemorySaver())


def test_happy_path_streams_plan_retrieval_tokens_and_citations(components: Components) -> None:
    llm = FakeLLM(
        [
            QueryPlan(
                needs_retrieval=True,
                standalone_question="How many vacation days do employees get?",
                search_queries=["annual paid vacation days", "vacation policy"],
                hypothetical_answer="Employees receive 25 days of paid vacation per year.",
            )
        ]
    )
    events = run(with_llm(components, llm), "vacation days?")
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["plan", "retrieval"]
    assert kinds.count("delta") == 2 and "citation" in kinds and kinds[-1] == "answer"

    retrieval = events[1]
    assert retrieval["queries"][0] == "How many vacation days do employees get?"
    assert "Vacation policy" in retrieval["passages"][0]["location"]

    answer = events[-1]
    assert answer["status"] == "answered"
    assert answer["content"] == "Employees get 25 days of vacation."
    assert answer["citations"] == [{"block": 0, "passage": 0, "text": "25 days"}]
    # the model was grounded on the reranked passages
    question, passages = llm.answered[0]
    assert question == "How many vacation days do employees get?"
    assert "25 days of paid vacation" in passages[0].text


def test_small_talk_skips_retrieval(components: Components) -> None:
    llm = FakeLLM(
        [
            QueryPlan(
                needs_retrieval=False, standalone_question="hi", search_queries=[], reply="Hello!"
            )
        ]
    )
    events = run(with_llm(components, llm), "hi")
    assert [e["event"] for e in events] == ["answer"]
    assert events[0]["status"] == "chitchat" and events[0]["content"] == "Hello!"
    assert llm.answered == []


def test_weak_evidence_reformulates_then_abstains(components: Components) -> None:
    llm = FakeLLM()
    events = run(with_llm(components, llm), "What is the capital of Mongolia?")
    kinds = [e["event"] for e in events]
    assert kinds == ["plan", "retrieval", "reformulate", "retrieval", "answer"]
    assert events[-1]["status"] == "no_evidence"
    assert llm.answered == []  # never asked the model to answer without evidence
    assert llm.reformulations == [
        ["What is the capital of Mongolia?", "What is the capital of Mongolia?"]
    ]


def test_planning_failure_falls_back_to_raw_question(components: Components) -> None:
    llm = FakeLLM([LLMError("rate limited")])
    events = run(with_llm(components, llm), "remote work stipend")
    assert events[-1]["status"] == "answered"
    assert events[0]["event"] == "retrieval" and events[0]["queries"] == ["remote work stipend"]


def test_empty_library_asks_for_upload(settings: Settings) -> None:
    chat = DocuChat(make_components(settings), checkpointer=InMemorySaver())
    events = run(chat, "anything")
    assert events[-1]["content"].startswith("Upload a document first")


def test_history_is_kept_per_thread(components: Components) -> None:
    chat = with_llm(components, FakeLLM())
    run(chat, "vacation days", "t1")
    run(chat, "remote work", "t1")
    state = chat.graph.get_state({"configurable": {"thread_id": "t1"}}).values
    roles = [m["role"] for m in state["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert state["messages"][1]["sources"][0]["title"] == "Employee Handbook"


def test_without_reranker_there_is_no_gate(settings: Settings, components: Components) -> None:
    components.retriever.reranker = None
    events = run(with_llm(components, FakeLLM()), "What is the capital of Mongolia?")
    assert events[-1]["status"] == "answered"
