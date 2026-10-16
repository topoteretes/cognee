"""store_code_docs: writes what the graph lacks, passes the whole batch on."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.tasks.code_docs import CodeDocument, code_document_id, store_code_docs
from cognee.tasks.code_docs.store_code_docs import store_new_code_docs
from cognee.tasks.summarization.models import TextSummary

store_module = importlib.import_module("cognee.tasks.code_docs.store_code_docs")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

REPO = "demo_repo"
FILE = "app/db.py"


def _document(symbol_count=2, node_set=None):
    return CodeDocument(
        id=code_document_id(REPO, FILE),
        name=FILE,
        raw_data_location=f"/repos/{REPO}/{FILE}",
        external_metadata=json.dumps(
            {"code_doc": {"repo": REPO, "file": FILE, "language": "python"}}
        ),
        repo=REPO,
        file_path=FILE,
        language="python",
        symbol_count=symbol_count,
        source_node_set=node_set,
    )


def _chunk(document, docstring, index, node_set=None):
    text = f"{FILE}:{index + 1}\napp/db.f{index} (function)\ndef f{index}():\n\n{docstring}"
    content_hash = chunk_content_hash(text)
    return DocumentChunk(
        id=content_chunk_id(document.id, content_hash, 0),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=index,
        cut_type="code_docstring",
        content_hash=content_hash,
        chunker_id="code_docs",
        is_part_of=document,
        contains=[],
        document_id=str(document.id),
        document_name=FILE,
        external_metadata=json.dumps({"code_doc": {"repo": REPO, "file": FILE, "line": index + 1}}),
        source_node_set=node_set,
    )


def _stored(point, **overrides):
    """A stored node the way get_nodes answers it: core columns plus a JSON properties blob."""
    properties = {
        "symbol_count": getattr(point, "symbol_count", None),
        "source_node_set": point.source_node_set,
        **overrides,
    }
    return {
        "id": str(point.id),
        "name": getattr(point, "name", None),
        "type": type(point).__name__,
        "properties": json.dumps(properties),
    }


@pytest.fixture
def engine(monkeypatch):
    """The graph engine; ``get_nodes`` answers what a test says is stored (nothing by default)."""
    engine = AsyncMock()
    engine.get_nodes.return_value = []
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


@pytest.fixture
def writes(monkeypatch):
    """What reaches add_data_points, call by call."""
    written = []

    async def fake_add_data_points(data_points, ctx=None, graph_only=False):
        written.append(SimpleNamespace(points=list(data_points), ctx=ctx, graph_only=graph_only))
        return data_points

    monkeypatch.setattr(store_module, "add_data_points", fake_add_data_points)
    return written


@pytest.mark.asyncio
async def test_first_run_writes_every_point_documents_first(engine, writes):
    document = _document()
    first, second = _chunk(document, "First.", 0), _chunk(document, "Second.", 1)
    batch = [first, document, second]

    result = await store_code_docs(batch)

    assert result is batch
    engine.get_nodes.assert_awaited_once_with([str(first.id), str(document.id), str(second.id)])
    assert [write.points for write in writes] == [[document, first, second]]
    assert writes[0].ctx is None and writes[0].graph_only is False


@pytest.mark.asyncio
async def test_unchanged_rerun_writes_nothing(engine, writes):
    document = _document()
    chunks = [_chunk(document, "First.", 0), _chunk(document, "Second.", 1)]
    engine.get_nodes.return_value = [_stored(document)] + [_stored(chunk) for chunk in chunks]

    counts = await store_new_code_docs([document, *chunks])

    assert writes == []
    assert (counts.documents, counts.chunks, counts.extractions, counts.skipped) == (0, 0, 0, 3)
    assert counts.written == 0
    # The task form passes the batch through all the same.
    assert await store_code_docs([document, *chunks]) == [document, *chunks]


@pytest.mark.asyncio
async def test_changed_docstring_writes_the_new_chunk_and_its_document(engine, writes):
    document = _document()
    kept, old = _chunk(document, "Kept.", 0), _chunk(document, "Old text.", 1)
    new = _chunk(document, "New text.", 1)
    engine.get_nodes.return_value = [_stored(document), _stored(kept), _stored(old)]

    counts = await store_new_code_docs([document, kept, new])

    assert [write.points for write in writes] == [[document, new]]
    assert (counts.documents, counts.chunks, counts.skipped) == (1, 1, 1)


@pytest.mark.asyncio
async def test_removed_symbol_rewrites_the_document_alone(engine, writes):
    document = _document(symbol_count=1)
    chunk = _chunk(document, "Kept.", 0)
    # The stored document still counts the symbol whose chunk the sweep drops.
    engine.get_nodes.return_value = [_stored(document, symbol_count=2), _stored(chunk)]

    counts = await store_new_code_docs([document, chunk])

    assert [write.points for write in writes] == [[document]]
    assert (counts.documents, counts.chunks, counts.skipped) == (1, 0, 1)


@pytest.mark.asyncio
async def test_retagged_points_are_rewritten(engine, writes):
    document = _document(node_set="code, team-a")
    chunk = _chunk(document, "Kept.", 0, node_set="code, team-a")
    engine.get_nodes.return_value = [
        _stored(document, source_node_set="code"),
        _stored(chunk, source_node_set="code"),
    ]

    counts = await store_new_code_docs([document, chunk])

    assert [write.points for write in writes] == [[document, chunk]]
    assert (counts.documents, counts.chunks, counts.skipped) == (1, 1, 0)


@pytest.mark.asyncio
async def test_document_read_from_source_for_the_first_time_is_rewritten(engine, writes):
    """Enola's docs stood in while the source was unreadable; now the file was read.
    (A stored row without doc_source -- stored before the field, or an adapter
    answering the core columns only -- is not called changed on it: every other
    test here stores without it.)"""
    document = _document()
    chunk = _chunk(document, "Kept.", 0)
    engine.get_nodes.return_value = [_stored(document, doc_source="enola"), _stored(chunk)]

    counts = await store_new_code_docs([document, chunk])

    assert document.doc_source == "source"
    assert [write.points for write in writes] == [[document]]
    assert (counts.documents, counts.chunks, counts.skipped) == (1, 0, 1)


@pytest.mark.asyncio
async def test_chunk_whose_position_moved_is_rewritten(engine, writes):
    """Same text, same id, another chunk_index: a symbol above it gained a docstring
    without moving it. The stored node is refreshed so two chunks of one document
    do not share an index; a stored row that carries no chunk_index (an adapter
    answering the core columns alone) is not called changed on it."""
    document = _document()
    kept, moved, bare = (
        _chunk(document, "Kept.", 0),
        _chunk(document, "Moved.", 2),
        _chunk(document, "Bare.", 3),
    )
    engine.get_nodes.return_value = [
        _stored(document),
        _stored(kept, chunk_index=0),
        _stored(moved, chunk_index=1),
        _stored(bare),
    ]

    counts = await store_new_code_docs([document, kept, moved, bare])

    assert [write.points for write in writes] == [[document, moved]]
    assert (counts.documents, counts.chunks, counts.skipped) == (1, 1, 2)


@pytest.mark.asyncio
async def test_extraction_output_is_always_written(engine, writes):
    document = _document()
    summarized, extracted = _chunk(document, "Summarized.", 0), _chunk(document, "Extracted.", 1)
    engine.get_nodes.return_value = [
        _stored(document),
        _stored(summarized),
        _stored(extracted),
    ]
    summary = TextSummary(text="Summary.", made_from=summarized)
    entity = Entity(
        name="db", description="The database.", is_a=EntityType(name="x", description="x")
    )
    extracted.contains = [(Edge(relationship_type="contains"), entity)]

    # Entity mode: cognify_code_doc_chunks returned summaries and chunks, no
    # documents. Both stored chunks carry new extraction output.
    counts = await store_new_code_docs([summary, extracted])

    assert [write.points for write in writes] == [[summary, extracted]]
    # The summarized chunk is stored already (not a new chunk); the extracted
    # one is rewritten for its entities; their document is written with them.
    assert (counts.documents, counts.chunks, counts.extractions, counts.skipped) == (1, 1, 1, 0)


@pytest.mark.asyncio
async def test_context_and_graph_only_reach_add_data_points(engine, writes):
    document = _document()
    chunk = _chunk(document, "Text.", 0)
    ctx = SimpleNamespace(dataset=SimpleNamespace(id="ds-1"))

    await store_code_docs([document, chunk], ctx=ctx, graph_only=True)

    assert writes[0].ctx is ctx and writes[0].graph_only is True


@pytest.mark.asyncio
async def test_id_lookups_are_batched(engine, writes, monkeypatch):
    monkeypatch.setattr(store_module, "_SWEEP_CHUNK_SIZE", 2)
    document = _document()
    chunks = [_chunk(document, f"Text {index}.", index) for index in range(2)]

    await store_code_docs([document, *chunks])

    assert [len(call.args[0]) for call in engine.get_nodes.await_args_list] == [2, 1]


@pytest.mark.asyncio
async def test_empty_batch_touches_nothing(engine, writes):
    assert await store_code_docs([]) == []
    engine.get_nodes.assert_not_awaited()
    assert writes == []
