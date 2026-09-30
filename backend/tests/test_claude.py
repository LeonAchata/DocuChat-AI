"""The Claude adapter against a local fake Messages API (JSON and SSE)."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import anthropic
import pytest

from docuchat.generation.claude import ClaudeLLM, LLMError, QueryPlan, Turn
from docuchat.models import Passage


class FakeAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.json: dict[str, Any] | None = None
        self.sse: list[dict[str, Any]] | None = None


@pytest.fixture
def api() -> Iterator[tuple[FakeAPI, str]]:
    state = FakeAPI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append({"headers": dict(self.headers), "json": body})
            if body.get("stream"):
                payload = "".join(
                    f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in state.sse or []
                ).encode()
                content_type = "text/event-stream"
            else:
                payload = json.dumps(state.json).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def llm(url: str, **kwargs: Any) -> ClaudeLLM:
    client = anthropic.Anthropic(api_key="test", base_url=url, max_retries=0)
    return ClaudeLLM(client=client, **kwargs)


def message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


PASSAGES = [
    Passage(
        id=0,
        document_id="d1",
        document_title="Employee Handbook",
        text="Full-time employees accrue 25 days of paid vacation per year.",
        section="Vacation policy",
        page_start=3,
        page_end=3,
        chunk_ids=["d1:0"],
    )
]


def test_plan_query_uses_structured_output(api: tuple[FakeAPI, str]) -> None:
    state, url = api
    state.json = message(
        json.dumps(
            {
                "needs_retrieval": True,
                "standalone_question": "Vacation days per year?",
                "search_queries": ["annual leave"],
                "hypothetical_answer": "Employees get 25 days.",
            }
        )
    )
    plan = llm(url).plan_query([Turn(role="user", content="hi")], "and vacation?")
    assert isinstance(plan, QueryPlan) and plan.search_queries == ["annual leave"]
    sent = state.requests[0]["json"]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] == "low"
    assert sent["fallbacks"] == "default"


def test_refusal_raises(api: tuple[FakeAPI, str]) -> None:
    state, url = api
    state.json = message("", stop_reason="refusal")
    with pytest.raises(LLMError):
        llm(url).plan_query([], "x")


def test_stream_answer_sends_documents_and_maps_citations(api: tuple[FakeAPI, str]) -> None:
    state, url = api
    start = message("")
    start["content"], start["stop_reason"] = [], None
    state.sse = [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "You get 25 days"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "25 days of paid vacation",
                    "document_index": 0,
                    "document_title": "Employee Handbook",
                    "start_char_index": 27,
                    "end_char_index": 51,
                },
            },
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": " per year."},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    history = [
        Turn(role="assistant", content="Hi!"),
        Turn(role="user", content="q1"),
        Turn(role="assistant", content="a1"),
    ]
    events = list(llm(url).stream_answer(history, "How much vacation?", PASSAGES))

    assert [(e.kind, e.block) for e in events] == [
        ("text", 0),
        ("citation", 0),
        ("text", 1),
        ("stop", 0),
    ]
    assert events[1].passage == 0 and events[1].text == "25 days of paid vacation"
    assert events[-1].stop_reason == "end_turn"

    sent = state.requests[0]["json"]
    assert sent["messages"][0]["role"] == "user"  # leading assistant turn dropped
    doc = sent["messages"][-1]["content"][0]
    assert doc["type"] == "document" and doc["citations"] == {"enabled": True}
    assert doc["title"] == "Employee Handbook" and doc["context"] == "p. 3 · Vacation policy"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "output_config" in sent and "format" not in sent["output_config"]
