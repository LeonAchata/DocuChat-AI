"""HTTP API: document management and a Server-Sent-Events chat endpoint."""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from docuchat.config import get_settings
from docuchat.ingest.parsers import SUPPORTED_EXTENSIONS
from docuchat.models import Document
from docuchat.service import DocuChat, build_components

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    document_ids: list[str] | None = Field(default=None, max_length=100)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
        app.state.chat = DocuChat(build_components(settings), checkpointer=saver)
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DocuChat", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    def chat_app(request: Request) -> DocuChat:
        return request.app.state.chat  # type: ignore[no-any-return]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/documents")
    def list_documents(request: Request) -> list[Document]:
        return chat_app(request).c.store.list_documents()

    @app.post("/api/documents", status_code=202)
    async def upload(
        files: list[UploadFile], request: Request, background: BackgroundTasks
    ) -> list[Document]:
        docuchat = chat_app(request)
        limit = settings.max_upload_mb * 1024 * 1024
        accepted: list[Document] = []
        for upload in files:
            name = Path(upload.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    415, f"{name}: supported types are {sorted(SUPPORTED_EXTENSIONS)}"
                )
            path = await _save_upload(upload, limit)
            doc, needed = docuchat.c.ingestor.register(path, filename=name)
            if needed:
                background.add_task(_process, docuchat, doc, path)
            else:
                path.unlink(missing_ok=True)
            accepted.append(doc)
        return accepted

    @app.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, request: Request) -> None:
        store = chat_app(request).c.store
        if store.get_document(document_id) is None:
            raise HTTPException(404, "Document not found")
        store.delete_document(document_id)

    @app.get("/api/threads/{thread_id}")
    async def thread(thread_id: str, request: Request) -> dict[str, Any]:
        snapshot = await chat_app(request).graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot or not snapshot.values:
            raise HTTPException(404, "Thread not found")
        return {"thread_id": thread_id, "messages": snapshot.values.get("messages", [])}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        docuchat = chat_app(request)
        thread_id = body.thread_id or docuchat.new_thread()

        async def events() -> AsyncIterator[str]:
            yield _sse("thread", {"thread_id": thread_id})
            try:
                async for event in docuchat.astream(body.message, thread_id, body.document_ids):
                    name = event.pop("event", "message")
                    yield _sse(name, event)
            except Exception:
                log.exception("Chat turn failed")
                yield _sse("error", {"message": "Internal error while answering."})
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _save_upload(upload: UploadFile, limit: int) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="docuchat-")
    os.close(fd)
    tmp = Path(name)
    size = 0
    with tmp.open("wb") as out:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"{upload.filename}: file exceeds {limit // 2**20} MB")
            out.write(chunk)
    return tmp


async def _process(docuchat: DocuChat, doc: Document, path: Path) -> None:
    try:
        await asyncio.to_thread(docuchat.c.ingestor.process, doc, path)
    finally:
        path.unlink(missing_ok=True)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


app = create_app()
