import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docuchat import api as api_module
from docuchat.config import Settings
from tests.conftest import HANDBOOK, make_components


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)
    monkeypatch.setattr(api_module, "build_components", lambda s: make_components(s))
    return TestClient(api_module.create_app())


def test_upload_ingest_chat_delete(client: TestClient, tmp_path: Path) -> None:
    with client:
        upload = client.post(
            "/api/documents",
            files=[("files", ("handbook.md", HANDBOOK.encode(), "text/markdown"))],
        )
        assert upload.status_code == 202
        doc_id = upload.json()[0]["id"]

        docs = client.get("/api/documents").json()  # background task already ran
        assert docs[0]["status"] == "ready" and docs[0]["chunk_count"] > 0

        duplicate = client.post(
            "/api/documents", files=[("files", ("copy.md", HANDBOOK.encode(), "text/markdown"))]
        )
        assert duplicate.json()[0]["id"] == doc_id

        chat = client.post("/api/chat", json={"message": "How many vacation days?"})
        events = parse_sse(chat.text)
        names = [n for n, _ in events]
        assert names[0] == "thread" and names[-1] == "done"
        assert "retrieval" in names and "citation" in names
        answer = next(d for n, d in events if n == "answer")
        assert answer["sources"][0]["title"] == "Employee Handbook"

        thread = events[0][1]["thread_id"]
        assert len(client.get(f"/api/threads/{thread}").json()["messages"]) == 2

        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.get("/api/documents").json() == []
        assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_rejects_unsupported_and_oversized(client: TestClient, settings: Settings) -> None:
    with client:
        bad = client.post(
            "/api/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]
        )
        assert bad.status_code == 415
        settings.max_upload_mb = 0
        big = client.post("/api/documents", files=[("files", ("a.txt", b"x" * 10, "text/plain"))])
        assert big.status_code == 413


def test_chat_validates_thread_id(client: TestClient) -> None:
    with client:
        response = client.post("/api/chat", json={"message": "hi", "thread_id": "../etc"})
    assert response.status_code == 422
