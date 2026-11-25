"""add_code_doc_edges: describes edges by id, idempotent re-runs and a repo-scoped sweep."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.tasks.code_docs import (
    DESCRIBES_RELATIONSHIP,
    CodeDocument,
    add_code_doc_edges,
    code_document_id,
)
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.summarization.models import TextSummary

add_code_doc_edges_module = importlib.import_module("cognee.tasks.code_docs.add_code_doc_edges")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
vector_module = importlib.import_module("cognee.infrastructure.databases.vector")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

REPO = "demo_repo"
FILE = "app/db.py"
SYMBOL = "app/db.Database"
SYMBOL_ID = str(fact_node_id(REPO, "symbol", SYMBOL))
MODULE_ID = str(fact_node_id(REPO, "module", "app"))
READ_TYPES = ["CodeDocument", "DocumentChunk", "TextSummary", "CodeSymbol", "CodeModule"]

CODE_NODES = [
    (SYMBOL_ID, {"type": "CodeSymbol", "name": SYMBOL, "repo": REPO}),
    (MODULE_ID, {"type": "CodeModule", "name": "app", "repo": REPO}),
]


def _document(file_path=FILE, repo=REPO):
    return CodeDocument(
        id=code_document_id(repo, file_path),
        name=file_path,
        raw_data_location=f"/repos/{repo}/{file_path}",
        external_metadata=json.dumps(
            {"code_doc": {"repo": repo, "file": file_path, "language": "python"}}
        ),
        repo=repo,
        file_path=file_path,
        language="python",
        symbol_count=1,
    )


def _chunk(document, text, chunk_index, metadata):
    content_hash = chunk_content_hash(text)
    return DocumentChunk(
        id=content_chunk_id(document.id, content_hash, 0),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=chunk_index,
        cut_type="code_docstring",
        content_hash=content_hash,
        chunker_id="code_docs",
        is_part_of=document,
        contains=[],
        document_id=str(document.id),
        document_name=document.file_path,
        external_metadata=json.dumps({"code_doc": metadata}),
    )


def _symbol_chunk(document, docstring="Open the database.", symbol_id=SYMBOL_ID, line=12):
    text = f"{FILE}:{line}\n{SYMBOL} (class)\nclass Database:\n\n{docstring}"
    metadata = {
        "repo": REPO,
        "file": FILE,
        "line": line,
        "end_line": 40,
        "symbol": SYMBOL,
        "symbol_id": symbol_id,
        "symbol_kind": "class",
        "language": "python",
        "doc_source": "source",
    }
    return _chunk(document, text, 1, metadata)


def _module_chunk(document, module_id=MODULE_ID):
    metadata = {
        "repo": REPO,
        "file": FILE,
        "line": 1,
        "end_line": None,
        "symbol": "app/db",
        "symbol_id": None,
        "symbol_kind": "module",
        "language": "python",
        "doc_source": "source",
        "module_id": module_id,
    }
    return _chunk(document, f"{FILE}:1\napp/db (module)\n\nDatabase helpers.", 0, metadata)


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = (list(nodes), list(edges))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


def _edge_keys(engine):
    return [(edge[0], edge[1], edge[2]) for edge in engine.add_edges.await_args.args[0]]


def _deleted(engine):
    return sorted(
        node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
    )


class _FakeVectorEngine:
    """Records the sweep's vector deletes; every collection exists unless ``collections`` says."""

    def __init__(self):
        self.collections = None
        self.deleted = []

    async def has_collection(self, name):
        return self.collections is None or name in self.collections

    async def delete_data_points(self, collection, ids):
        self.deleted.append((collection, list(ids)))


@pytest.fixture(autouse=True)
def vectors(monkeypatch):
    """The vector engine the sweep drops embeddings through (no LanceDB in unit tests)."""
    fake = _FakeVectorEngine()
    monkeypatch.setattr(vector_module, "get_vector_engine_async", AsyncMock(return_value=fake))
    return fake


@pytest.fixture
def invalidations(monkeypatch):
    """Record every code-retriever snapshot cache invalidation (its kwargs)."""
    calls = []
    monkeypatch.setattr(
        code_retriever_module,
        "invalidate_code_graph_snapshot_cache",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


@pytest.mark.asyncio
async def test_writes_describes_edges_in_the_code_graph_shape(monkeypatch, invalidations):
    document = _document()
    symbol_chunk = _symbol_chunk(document)
    module_chunk = _module_chunk(document)
    engine = _mock_engine(monkeypatch, CODE_NODES)

    result = await add_code_doc_edges([document, module_chunk, symbol_chunk])

    assert result == [document, module_chunk, symbol_chunk]
    engine.get_filtered_graph_data.assert_awaited_once_with([{"type": READ_TYPES}])
    # Chunk -> code node, in batch order; a module docstring links to its CodeModule.
    assert _edge_keys(engine) == [
        (str(module_chunk.id), MODULE_ID, "describes"),
        (str(symbol_chunk.id), SYMBOL_ID, "describes"),
    ]
    properties = engine.add_edges.await_args.args[0][1][3]
    assert properties["source_node_id"] == str(symbol_chunk.id)
    assert properties["target_node_id"] == SYMBOL_ID
    assert properties["relationship_name"] == DESCRIBES_RELATIONSHIP == "describes"
    assert properties["edge_text"] == "describes"
    assert properties["doc_source"] == "source"
    assert properties["chunk_index"] == 1
    assert "updated_at" in properties
    # An empty graph has nothing stale; the snapshot cache is invalidated regardless.
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == [{}]


@pytest.mark.asyncio
async def test_rerun_adds_no_duplicate_edges(monkeypatch, invalidations, vectors):
    document = _document()
    chunk = _symbol_chunk(document)
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO, "name": FILE}),
        (str(chunk.id), {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(chunk.id), SYMBOL_ID, "describes", {}),
        (str(chunk.id), str(document.id), "is_part_of", {}),
    ]
    engine = _mock_engine(monkeypatch, nodes, edges)

    await add_code_doc_edges([document, chunk])

    engine.add_edges.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []


@pytest.mark.asyncio
async def test_sweep_deletes_only_the_stale_docs_and_chunks_of_the_batch_repo(
    monkeypatch, invalidations, vectors
):
    document = _document()
    kept_chunk = _symbol_chunk(document)
    changed_chunk_id = str(uuid4())  # the symbol's previous docstring text
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    stale_chunk_id = str(uuid4())
    other_repo_document_id = str(code_document_id("other_repo", "lib/x.py"))
    other_repo_chunk_id = str(uuid4())
    text_document_id = str(uuid4())  # a TextDocument: not a read type, never ours
    text_chunk_id = str(uuid4())
    stale_summary_id, kept_summary_id, mixed_summary_id = (str(uuid4()) for _ in range(3))

    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO, "name": FILE}),
        (str(kept_chunk.id), {"type": "DocumentChunk"}),
        # Properties nested as JSON text, the way some adapters answer.
        (changed_chunk_id, {"name": None, "properties": json.dumps({"type": "DocumentChunk"})}),
        (stale_document_id, {"type": "CodeDocument", "repo": REPO, "name": "app/legacy.py"}),
        (stale_chunk_id, {"type": "DocumentChunk"}),
        (other_repo_document_id, {"type": "CodeDocument", "repo": "other_repo"}),
        (other_repo_chunk_id, {"type": "DocumentChunk"}),
        (text_chunk_id, {"type": "DocumentChunk"}),
        (stale_summary_id, {"type": "TextSummary"}),
        (kept_summary_id, {"type": "TextSummary"}),
        (mixed_summary_id, {"type": "TextSummary"}),
    ]
    edges = [
        (str(kept_chunk.id), str(document.id), "is_part_of", {}),
        (changed_chunk_id, str(document.id), "is_part_of", {}),
        (stale_chunk_id, stale_document_id, "is_part_of", {}),
        (other_repo_chunk_id, other_repo_document_id, "is_part_of", {}),
        (text_chunk_id, text_document_id, "is_part_of", {}),
        (stale_summary_id, stale_chunk_id, "made_from", {}),
        (kept_summary_id, str(kept_chunk.id), "made_from", {}),
        (mixed_summary_id, changed_chunk_id, "made_from", {}),
        (mixed_summary_id, str(kept_chunk.id), "made_from", {}),
        (str(kept_chunk.id), SYMBOL_ID, "describes", {}),
        (changed_chunk_id, SYMBOL_ID, "describes", {}),
    ]
    engine = _mock_engine(monkeypatch, nodes, edges)

    await add_code_doc_edges([document, kept_chunk])

    engine.add_edges.assert_not_awaited()
    # The document the batch no longer produces goes with its chunk and its
    # summary; the kept document loses only the chunk the batch lacks. Other
    # repos, non-code documents and a summary that still has a source stay.
    assert _deleted(engine) == sorted(
        [stale_document_id, stale_chunk_id, stale_summary_id, changed_chunk_id]
    )
    # The same ids leave the collections add_data_points embedded them into,
    # so CHUNKS and RAG_COMPLETION stop returning the replaced text.
    assert vectors.deleted == [
        ("CodeDocument_name", [stale_document_id]),
        ("DocumentChunk_text", sorted([stale_chunk_id, changed_chunk_id])),
        ("TextSummary_text", [stale_summary_id]),
    ]


@pytest.mark.asyncio
async def test_sweep_deletes_in_chunks(monkeypatch, invalidations, vectors):
    monkeypatch.setattr(add_code_doc_edges_module, "_SWEEP_CHUNK_SIZE", 2)
    document = _document()
    chunk = _symbol_chunk(document)
    stale_ids = [str(code_document_id(REPO, f"app/old_{index}.py")) for index in range(3)]
    nodes = CODE_NODES + [
        (node_id, {"type": "CodeDocument", "repo": REPO}) for node_id in stale_ids
    ]
    engine = _mock_engine(monkeypatch, nodes)

    await add_code_doc_edges([document, chunk])

    assert [len(call.args[0]) for call in engine.delete_nodes.await_args_list] == [2, 1]
    assert _deleted(engine) == sorted(stale_ids)
    assert [(collection, len(ids)) for collection, ids in vectors.deleted] == [
        ("CodeDocument_name", 2),
        ("CodeDocument_name", 1),
    ]


@pytest.mark.asyncio
async def test_sweep_skips_vector_collections_that_were_never_written(
    monkeypatch, invalidations, vectors
):
    # A graph_only run (add_data_points without a vector engine) has no
    # collections: the graph nodes go, the vector engine is left alone.
    vectors.collections = set()
    document = _document()
    chunk = _symbol_chunk(document)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [(stale_document_id, {"type": "CodeDocument", "repo": REPO})]
    engine = _mock_engine(monkeypatch, nodes)

    await add_code_doc_edges([document, chunk])

    assert _deleted(engine) == [stale_document_id]
    assert vectors.deleted == []


@pytest.mark.asyncio
async def test_sweep_can_be_disabled(monkeypatch, invalidations, vectors):
    document = _document()
    chunk = _symbol_chunk(document)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [(stale_document_id, {"type": "CodeDocument", "repo": REPO})]
    engine = _mock_engine(monkeypatch, nodes)

    await add_code_doc_edges([document, chunk], sweep=False)

    engine.add_edges.assert_awaited_once()
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []


@pytest.mark.asyncio
async def test_empty_and_foreign_batches_touch_nothing(monkeypatch, invalidations):
    engine = _mock_engine(monkeypatch, CODE_NODES)
    plain_document = TextDocument(name="notes.txt", raw_data_location="/x", external_metadata=None)
    plain_chunk = DocumentChunk(
        text="plain",
        chunk_size=1,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=plain_document,
    )

    assert await add_code_doc_edges([]) == []
    assert await add_code_doc_edges([plain_document, plain_chunk]) == [plain_document, plain_chunk]

    engine.get_filtered_graph_data.assert_not_awaited()
    engine.add_edges.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == []


@pytest.mark.asyncio
async def test_chunks_whose_code_node_is_missing_are_not_linked(monkeypatch, invalidations):
    document = _document()
    orphan = _symbol_chunk(document, symbol_id=str(fact_node_id(REPO, "symbol", "app/db.Gone")))
    # extract_code_docs found no CodeModule for the file's directory.
    module_chunk = _module_chunk(document, module_id=None)
    engine = _mock_engine(monkeypatch, CODE_NODES[:1])

    await add_code_doc_edges([document, orphan, module_chunk])

    engine.add_edges.assert_not_awaited()

    # With the CodeModule present, the module docstring resolves the way the
    # code graph names modules: the file's directory within the repo.
    engine = _mock_engine(monkeypatch, CODE_NODES)
    await add_code_doc_edges([document, module_chunk])
    assert _edge_keys(engine) == [(str(module_chunk.id), MODULE_ID, "describes")]


@pytest.mark.asyncio
async def test_summaries_carry_the_chunks_in_entity_mode(monkeypatch, invalidations):
    document = _document()
    chunk = _symbol_chunk(document)
    summary = TextSummary(text="Opens the database.", made_from=chunk)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [(stale_document_id, {"type": "CodeDocument", "repo": REPO})]
    engine = _mock_engine(monkeypatch, nodes)

    await add_code_doc_edges([summary])

    assert _edge_keys(engine) == [(str(chunk.id), SYMBOL_ID, "describes")]
    # The document under the chunk counts as produced; the other one is swept.
    assert _deleted(engine) == [stale_document_id]


@pytest.mark.asyncio
async def test_invalidates_the_dataset_snapshot_even_when_the_write_fails(
    monkeypatch, invalidations
):
    document = _document()
    chunk = _symbol_chunk(document)
    engine = _mock_engine(monkeypatch, CODE_NODES)
    engine.add_edges.side_effect = RuntimeError("graph down")
    ctx = SimpleNamespace(dataset=SimpleNamespace(id="ds-1"), user=None, data_item={})

    with pytest.raises(RuntimeError):
        await add_code_doc_edges([document, chunk], ctx=ctx)

    assert invalidations == [{"dataset_id": "ds-1"}]


@pytest.mark.asyncio
async def test_rollback_ledger_rows_only_for_a_persisted_data_item(monkeypatch, invalidations):
    document = _document()
    chunk = _symbol_chunk(document)
    upsert_edges = AsyncMock()
    monkeypatch.setattr("cognee.modules.graph.methods.upsert_edges", upsert_edges)
    user = SimpleNamespace(id="u1", tenant_id=None)
    dataset = SimpleNamespace(id="ds-1")

    # A memify run: its data item is the `{}` seed, which has no ledger row.
    _mock_engine(monkeypatch, CODE_NODES)
    await add_code_doc_edges(
        [document, chunk],
        ctx=SimpleNamespace(user=user, dataset=dataset, data_item={}, pipeline_run_id="run-1"),
    )
    upsert_edges.assert_not_awaited()

    _mock_engine(monkeypatch, CODE_NODES)
    await add_code_doc_edges(
        [document, chunk],
        ctx=SimpleNamespace(
            user=user,
            dataset=dataset,
            data_item=SimpleNamespace(id="data-1"),
            pipeline_run_id="run-1",
        ),
    )
    kwargs = upsert_edges.await_args.kwargs
    assert (kwargs["dataset_id"], kwargs["data_id"], kwargs["pipeline_run_id"]) == (
        "ds-1",
        "data-1",
        "run-1",
    )
    assert [(e[0], e[1], e[2]) for e in upsert_edges.await_args.args[0]] == [
        (str(chunk.id), SYMBOL_ID, "describes")
    ]
