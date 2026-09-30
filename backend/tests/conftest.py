import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from docuchat.config import Settings
from docuchat.generation.claude import AnswerEvent, QueryPlan, Reformulation, Turn
from docuchat.index.encoders import HashingDense, HashingSparse, OverlapReranker
from docuchat.index.memory import MemoryStore
from docuchat.index.store import Store
from docuchat.models import Passage
from docuchat.service import Components, build_components

HANDBOOK = """# Employee Handbook

## Vacation policy

Full-time employees accrue 25 days of paid vacation per year. Unused vacation days can be
carried over to the next year, up to a maximum of 5 days.

Vacation requests must be submitted in the HR portal at least two weeks in advance.

## Remote work

Employees may work remotely up to three days per week with their manager's approval.
The company provides a monthly stipend of 50 euros for home office internet costs.

## Expenses

Travel expenses are reimbursed within 30 days after submitting receipts in the finance system.
Meals during business travel are covered up to 40 euros per day.
"""

SECURITY = """# Security Guidelines

## Passwords

Passwords must be at least 14 characters long and rotated every 180 days.
Multi-factor authentication is mandatory for all production systems.

## Incident response

Report suspected security incidents to the security team within one hour of discovery.
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        chunk_tokens=64,
        chunk_overlap_tokens=10,
        top_k=4,
        min_relevance=0.3,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text(HANDBOOK, encoding="utf-8")
    (folder / "security.md").write_text(SECURITY, encoding="utf-8")
    return folder


class FakeLLM:
    """Scripted stand-in for Claude. Cites the first passage it is given."""

    def __init__(self, plans: list[QueryPlan | Exception] | None = None) -> None:
        self.plans = list(plans or [])
        self.reformulations: list[list[str]] = []
        self.answered: list[tuple[str, list[Passage]]] = []

    def plan_query(self, history: list[Turn], question: str) -> QueryPlan:
        if self.plans:
            plan = self.plans.pop(0)
            if isinstance(plan, Exception):
                raise plan
            return plan
        return QueryPlan(
            needs_retrieval=True,
            standalone_question=question,
            search_queries=[question],
            hypothetical_answer=None,
        )

    def reformulate(self, question: str, failed: list[str]) -> Reformulation:
        self.reformulations.append(failed)
        return Reformulation(search_queries=["something else entirely"])

    def stream_answer(
        self, history: list[Turn], question: str, passages: list[Passage]
    ) -> Iterator[AnswerEvent]:
        self.answered.append((question, passages))
        yield AnswerEvent("text", block=0, text="Employees get ")
        yield AnswerEvent("text", block=0, text="25 days of vacation.")
        yield AnswerEvent("citation", block=0, text="25 days", passage=0)
        yield AnswerEvent("stop", stop_reason="end_turn")


def make_components(
    settings: Settings, llm: Any = None, store: Store | None = None, reranker: Any = "default"
) -> Components:
    return build_components(
        settings,
        llm=llm or FakeLLM(),
        dense=HashingDense(),
        sparse=HashingSparse(),
        reranker=OverlapReranker() if reranker == "default" else reranker,
        store=store or MemoryStore(settings.data_dir / "index"),
    )


@pytest.fixture
def components(settings: Settings, docs_dir: Path) -> Components:
    comps = make_components(settings)
    for file in sorted(docs_dir.iterdir()):
        comps.ingestor.ingest(file)
    return comps


def postgres_url() -> str | None:
    return os.environ.get("DOCUCHAT_TEST_DATABASE_URL")
