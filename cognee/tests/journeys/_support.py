"""Shared helpers for the journey tests: corpus loading, result scoring, store snapshots."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

MODE = os.getenv("COGNEE_JOURNEY_MODE", "mock").strip().lower()
IS_MOCK = MODE == "mock"
CORPUS_DIR = Path(__file__).parent / "golden_corpus"


# ---------------------------------------------------------------------------
# Golden corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    knowledge_graph: dict
    summary: dict


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    doc: str
    expected_any: tuple[str, ...]
    forbidden: tuple[str, ...]


def load_documents() -> list[Document]:
    raw = json.loads((CORPUS_DIR / "documents.json").read_text())
    return [
        Document(
            id=d["id"],
            title=d["title"],
            text=d["text"],
            knowledge_graph=d["knowledge_graph"],
            summary=d["summary"],
        )
        for d in raw["documents"]
    ]


def load_questions() -> list[Question]:
    raw = json.loads((CORPUS_DIR / "questions.json").read_text())
    return [
        Question(
            id=q["id"],
            question=q["question"],
            doc=q["doc"],
            expected_any=tuple(t.lower() for t in q["expected_any"]),
            forbidden=tuple(t.lower() for t in q.get("forbidden", [])),
        )
        for q in raw["questions"]
    ]


def mock_graphs(documents: Iterable[Document]) -> dict[str, dict]:
    """Title -> replay entry, the shape ``mock_ai.MockLLM`` expects."""
    return {
        d.title: {"knowledge_graph": d.knowledge_graph, "summary": d.summary} for d in documents
    }


def document_by_id(documents: Iterable[Document], doc_id: str) -> Document:
    for d in documents:
        if d.id == doc_id:
            return d
    raise KeyError(doc_id)


# ---------------------------------------------------------------------------
# Result scoring
# ---------------------------------------------------------------------------

# Fields under which search/recall results carry human-readable content:
# chunk / summary / graph-entry ``text``, completion strings, session Q&A
# ``answer``, and the ``search_result`` payload of a ``SearchResult``. Every
# other field (ids, timestamps, dataset envelope, the echoed ``question``) is
# deliberately not scored: a gold token like ``41`` must never match the hex of
# a UUID or the digits of a timestamp, and a fact must not pass because the
# question mentioned it.
_CONTENT_FIELDS = ("text", "completion", "answer", "summary", "search_result")


def content_text(results: Any) -> str:
    """The retrieved content only, lowercased: chunk texts, summaries, completions.

    Handles every shape the public API returns: ``recall()`` items
    (``SearchResultItem`` with ``.text``, session ``QAEntry`` with ``.answer``),
    ``SearchResult`` objects (``.search_result``), the plain dict form
    ``search()`` returns when a ``session_id`` is given, and bare strings or
    lists of any of those.
    """
    parts: list[str] = []

    def _collect(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, (list, tuple, set)):
            for sub in item:
                _collect(sub)
        elif isinstance(item, dict):
            for key in _CONTENT_FIELDS:
                if key in item:
                    _collect(item[key])
        else:
            for key in _CONTENT_FIELDS:
                value = getattr(item, key, None)
                if value is not None and not callable(value):
                    _collect(value)

    _collect(results)
    return "\n".join(parts).lower()


def contains_token(text: str, token: str) -> bool:
    """Whole-token match: ``41`` must not match inside ``1941`` or a UUID."""
    return re.search(rf"(?<![0-9a-z]){re.escape(token)}(?![0-9a-z])", text) is not None


def answered(question: Question, text: str) -> bool:
    return any(contains_token(text, token) for token in question.expected_any)


def forbidden_hits(question: Question, text: str) -> list[str]:
    return [token for token in question.forbidden if contains_token(text, token)]


@dataclass
class Scorecard:
    label: str
    total: int = 0
    hits: int = 0
    misses: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)

    def record(self, question: Question, text: str, check_forbidden: bool) -> None:
        """Score ``text`` (retrieved content) for the expected fact and, when asked,
        for facts leaked from other documents."""
        self.total += 1
        if answered(question, text):
            self.hits += 1
        else:
            self.misses.append(
                f"{question.id}: {question.question!r} -> none of {question.expected_any}"
            )
        if check_forbidden:
            leaked = forbidden_hits(question, text)
            if leaked:
                self.leaks.append(f"{question.id}: leaked {leaked}")

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def report(self) -> str:
        lines = [f"{self.label}: {self.hits}/{self.total} answered ({self.rate:.0%})"]
        lines += [f"  MISS {m}" for m in self.misses]
        lines += [f"  LEAK {leak}" for leak in self.leaks]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Store snapshots
# ---------------------------------------------------------------------------


@dataclass
class StoreSnapshot:
    data_rows: int
    graph_nodes: int
    graph_edges: int
    vector_rows: dict[str, int] | None  # collection -> rows; None when the adapter can't report

    @property
    def vector_total(self) -> int | None:
        return None if self.vector_rows is None else sum(self.vector_rows.values())

    def __str__(self) -> str:
        return (
            f"StoreSnapshot(data_rows={self.data_rows}, graph_nodes={self.graph_nodes}, "
            f"graph_edges={self.graph_edges}, vector_total={self.vector_total})"
        )


def as_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


async def snapshot_dataset(dataset_id: Any, user) -> StoreSnapshot:
    """Count what the three stores hold for one dataset.

    Runs inside the dataset's database context so per-dataset isolation
    (ENABLE_BACKEND_ACCESS_CONTROL) resolves to the right graph and vector DB.
    """
    import cognee
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    dataset_id = as_uuid(dataset_id)
    data_rows = len(await cognee.datasets.list_data(dataset_id, user))

    async with set_database_global_context_variables(dataset_id, user.id):
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        vector_rows = await _vector_row_counts()

    return StoreSnapshot(
        data_rows=data_rows,
        graph_nodes=len(nodes),
        graph_edges=len(edges),
        vector_rows=vector_rows,
    )


async def _lance_connection():
    """The LanceDB connection of the current vector engine, or None on other adapters.

    Row counts and payload dumps read LanceDB tables directly; other adapters
    (pgvector, community ones) do not expose an equivalent, so callers treat
    None as "cannot report" rather than as an empty store.
    """
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.config import get_vectordb_config

    if get_vectordb_config().vector_db_provider.lower() != "lancedb":
        return None
    engine = await get_vector_engine_async()
    return await engine.get_connection()


async def _vector_row_counts() -> dict[str, int] | None:
    connection = await _lance_connection()
    if connection is None:
        return None
    counts: dict[str, int] = {}
    for name in await connection.table_names():
        table = await connection.open_table(name)
        counts[name] = int(await table.count_rows())
    return counts


async def vector_texts(dataset_id: Any, user, collection: str = "DocumentChunk_text") -> list[str]:
    """All ``text`` payloads in a vector collection for one dataset (lowercased).

    Empty when the collection does not exist; None-safe on non-LanceDB adapters
    (returns an empty list, so callers must pair it with a store snapshot).
    """
    from cognee.context_global_variables import set_database_global_context_variables

    dataset_id = as_uuid(dataset_id)

    async with set_database_global_context_variables(dataset_id, user.id):
        connection = await _lance_connection()
        if connection is None or collection not in await connection.table_names():
            return []
        table = await connection.open_table(collection)
        rows = await table.to_arrow()
        payloads = rows.column("payload").to_pylist() if "payload" in rows.column_names else []

    texts = []
    for payload in payloads:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                texts.append(payload.lower())
                continue
        if isinstance(payload, dict) and isinstance(payload.get("text"), str):
            texts.append(payload["text"].lower())
    return texts


async def graph_node_names(dataset_id: Any, user) -> list[str]:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    dataset_id = as_uuid(dataset_id)

    async with set_database_global_context_variables(dataset_id, user.id):
        graph_engine = await get_graph_engine()
        nodes, _ = await graph_engine.get_graph_data()
    names = []
    for node in nodes:
        props = node[1] if isinstance(node, (list, tuple)) and len(node) > 1 else {}
        if isinstance(props, dict):
            for key in ("name", "text", "label"):
                value = props.get(key)
                if isinstance(value, str):
                    names.append(value.lower())
    return names


async def dataset_id_by_name(name: str, user) -> UUID | None:
    import cognee

    for dataset in await cognee.datasets.list_datasets(user):
        if dataset.name == name:
            return dataset.id
    return None


def stored_text(row) -> str:
    """The processed text of a ``Data`` row, read from where ingestion stored it.

    Text passed to ``remember()`` is written to a file under the data root and
    ``raw_data_location`` points at it; that is the only place the original
    content survives, since ``Data.name`` is a content hash for inline text.
    """
    location = getattr(row, "raw_data_location", None)
    if not isinstance(location, str):
        return ""
    path = Path(location.removeprefix("file://"))
    if not path.is_file():
        return ""
    return path.read_text(errors="ignore").lower()
