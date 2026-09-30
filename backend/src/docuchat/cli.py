"""Command-line interface: ``docuchat ingest | documents | ask | chat | serve | eval``."""

import logging
from pathlib import Path
from typing import Any

import typer
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from docuchat.config import get_settings
from docuchat.ingest.parsers import SUPPORTED_EXTENSIONS
from docuchat.service import DocuChat, build_components

app = typer.Typer(add_completion=False, help="Chat with your documents.")
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)


@app.command()
def ingest(paths: list[Path] = typer.Argument(..., exists=True)) -> None:
    """Index files or folders."""
    components = build_components(get_settings())
    files = [
        f
        for p in paths
        for f in ([p] if p.is_file() else sorted(p.rglob("*")))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    for file in files:
        doc = components.ingestor.ingest(file)
        colour = {"ready": "green", "failed": "red"}.get(doc.status, "yellow")
        detail = f"{doc.chunk_count} chunks" if doc.status == "ready" else doc.error
        console.print(f"[{colour}]{doc.status:>7}[/] {file.name} [dim]({detail})[/]")


@app.command()
def documents() -> None:
    """List indexed documents."""
    for doc in build_components(get_settings()).store.list_documents():
        console.print(f"{doc.id}  {doc.status:<10} {doc.chunk_count:>5} chunks  {doc.title}")


@app.command()
def ask(question: str) -> None:
    """Answer one question with sources."""
    chat = DocuChat(build_components(get_settings()), checkpointer=InMemorySaver())
    _run_turn(chat, question, chat.new_thread())


@app.command("chat")
def chat_cmd() -> None:
    """Interactive multi-turn session."""
    chat = DocuChat(build_components(get_settings()), checkpointer=InMemorySaver())
    thread = chat.new_thread()
    console.print("[dim]Ask about your documents. Ctrl+C to exit.[/]")
    while True:
        try:
            question = console.input("\n[bold cyan]you >[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question:
            _run_turn(chat, question, thread)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("docuchat.api:app", host=host, port=port, reload=reload)


@app.command("download-models")
def download_models() -> None:
    """Fetch the configured ONNX models into the local cache."""
    from docuchat.index.encoders import FastEmbedDense, FastEmbedReranker, FastEmbedSparse

    s = get_settings()
    cache = s.data_dir / "models"
    FastEmbedDense(s.dense_model, cache)
    if s.sparse_model:
        FastEmbedSparse(s.sparse_model, cache)
    if s.reranker_model:
        FastEmbedReranker(s.reranker_model, cache)
    console.print(f"Models cached in {cache}")


@app.command("eval")
def run_eval(
    dataset: str = typer.Option("scifact", help="BEIR dataset name."),
    max_queries: int | None = typer.Option(None, help="Only the first N test queries."),
    output: Path = typer.Option(Path("evals/results/scifact.json")),
) -> None:
    """Benchmark retrieval configurations on a BEIR dataset (no LLM calls)."""
    from docuchat.evals import run_beir

    run_beir(
        get_settings(), dataset=dataset, max_queries=max_queries, output=output, console=console
    )


def _run_turn(chat: DocuChat, question: str, thread: str) -> None:
    final: dict[str, Any] | None = None
    streamed = False
    for event in chat.stream(question, thread):
        kind = event.get("event")
        if kind == "plan":
            console.print(f"[dim]queries → {' | '.join(event['queries'])}[/]")
        elif kind == "retrieval":
            conf = event["confidence"]
            conf_text = f", top relevance {conf:.2f}" if conf is not None else ""
            console.print(
                f"[dim]retrieved {event['candidates']} candidates → "
                f"{len(event['passages'])} passages{conf_text}[/]"
            )
        elif kind == "reformulate":
            console.print(
                f"[yellow]weak evidence, retrying with: {' | '.join(event['queries'])}[/]"
            )
        elif kind == "delta":
            console.print(event["text"], end="")
            streamed = True
        elif kind == "answer":
            final = event
    if streamed:
        console.print()
    if not final:
        return
    if not streamed:
        console.print(Panel(Markdown(final["content"]), border_style="yellow"))
    cited = sorted({c["passage"] for c in final.get("citations", [])})
    sources = final.get("sources", [])
    for source in sources:
        if cited and source["id"] not in cited:
            continue
        console.print(
            f"  [bold][{source['id'] + 1}][/] {source['title']} [dim]{source['location']}[/]"
        )
