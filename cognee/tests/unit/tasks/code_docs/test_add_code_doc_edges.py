"""add_code_doc_edges: describes edges by id, idempotent re-runs and a scoped sweep."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    NodeDeleteData,
    make_chunk_source_ref_key,
    make_source_ref_key,
)
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.tasks.code_docs import (
    DESCRIBES_RELATIONSHIP,
    CodeDocument,
    add_code_doc_edges,
    code_document_id,
)
from cognee.tasks.code_docs.add_code_doc_edges import link_code_docs
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.summarization.models import TextSummary

add_code_doc_edges_module = importlib.import_module("cognee.tasks.code_docs.add_code_doc_edges")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
vector_module = importlib.import_module("cognee.infrastructure.databases.vector")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

REPO = "demo_repo"
OTHER_REPO = "other_repo"
FILE = "app/db.py"
SYMBOL = "app/db.Database"
SYMBOL_ID = str(fact_node_id(REPO, "symbol", SYMBOL))
MODULE_ID = str(fact_node_id(REPO, "module", "app"))
READ_TYPES = [
    "CodeDocument",
    "DocumentChunk",
    "TextSummary",
    "CodeSymbol",
    "CodeModule",
    "CodeRepository",
]
ENTITY_READ_TYPES = ["DocumentChunk", "Entity", "EntityType"]

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


def _stored_document(repo, file_path):
    return (str(code_document_id(repo, file_path)), {"type": "CodeDocument", "repo": repo})


def _mock_engine(monkeypatch, nodes=(), edges=(), entity_graph=None):
    """The graph engine: the main read answers (nodes, edges), the entity read ``entity_graph``."""
    engine = AsyncMock()

    async def read(filters):
        if filters == [{"type": ENTITY_READ_TYPES}]:
            return entity_graph if entity_graph is not None else ([], [])
        return list(nodes), list(edges)

    engine.get_filtered_graph_data.side_effect = read
    # No stored provenance and no entity neighbours unless a test says so.
    engine.get_node_delete_data.return_value = {}
    engine.get_connections.return_value = []
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


def _edge_keys(engine):
    return [(edge[0], edge[1], edge[2]) for edge in engine.add_edges.await_args.args[0]]


def _deleted(engine):
    return sorted(
        node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
    )


def _reads(engine):
    return [call.args[0] for call in engine.get_filtered_graph_data.await_args_list]


def _delete_data(node_id, source_ref_keys):
    return NodeDeleteData(
        node_id=node_id,
        node_type="DocumentChunk",
        indexed_fields=["text"],
        node_properties={},
        source_ref_keys=source_ref_keys,
        source_dataset_ids=[],
        source_run_ids=[],
        source_run_refs=[],
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
    engine.get_node_delete_data.assert_not_awaited()
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
async def test_sweep_deletes_only_the_stale_docs_and_chunks_in_scope(
    monkeypatch, invalidations, vectors
):
    document = _document()
    kept_chunk = _symbol_chunk(document)
    changed_chunk_id = str(uuid4())  # the symbol's previous docstring text
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    stale_chunk_id = str(uuid4())
    other_repo_document_id = str(code_document_id(OTHER_REPO, "lib/x.py"))
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
        (other_repo_document_id, {"type": "CodeDocument", "repo": OTHER_REPO}),
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

    # The run was filtered to this repository, so the sweep is too.
    await add_code_doc_edges([document, kept_chunk], repo=REPO)

    engine.add_edges.assert_not_awaited()
    # The document the batch no longer produces goes with its chunk and its
    # summary; the kept document loses only the chunk the batch lacks. The
    # other repository, non-code documents and a summary that still has a
    # source stay.
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
    # The stale chunks' provenance was consulted before anything was deleted.
    engine.get_node_delete_data.assert_awaited_once_with(sorted([changed_chunk_id, stale_chunk_id]))


@pytest.mark.asyncio
async def test_sweep_without_a_filter_covers_every_repository_with_documents(
    monkeypatch, invalidations, vectors
):
    document = _document()
    chunk = _symbol_chunk(document)
    other_document_id, other_props = _stored_document(OTHER_REPO, "lib/x.py")
    other_chunk_id = str(uuid4())
    # No CodeRepository node for other_repo (forgotten or renamed): its docs remain.
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(chunk.id), {"type": "DocumentChunk"}),
        (other_document_id, other_props),
        (other_chunk_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(chunk.id), str(document.id), "is_part_of", {}),
        (other_chunk_id, other_document_id, "is_part_of", {}),
    ]
    engine = _mock_engine(monkeypatch, nodes, edges)

    await add_code_doc_edges([document, chunk])

    assert _deleted(engine) == sorted([other_document_id, other_chunk_id])
    assert vectors.deleted == [
        ("CodeDocument_name", [other_document_id]),
        ("DocumentChunk_text", [other_chunk_id]),
    ]


@pytest.mark.asyncio
async def test_sweep_with_a_filter_covers_that_repository_even_from_an_empty_batch(
    monkeypatch, invalidations, vectors
):
    document = _document()
    chunk = _symbol_chunk(document)
    other_document_id, other_props = _stored_document(OTHER_REPO, "lib/x.py")
    other_chunk_id = str(uuid4())
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(chunk.id), {"type": "DocumentChunk"}),
        (other_document_id, other_props),
        (other_chunk_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(chunk.id), str(document.id), "is_part_of", {}),
        (other_chunk_id, other_document_id, "is_part_of", {}),
    ]

    # The filtered repository derived nothing (forgotten, or its docstrings
    # vanished): its docs go, the other repository's stay.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=OTHER_REPO)
    assert _deleted(engine) == sorted([other_document_id, other_chunk_id])

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=REPO)
    assert _deleted(engine) == sorted([str(document.id), str(chunk.id)])

    # Unfiltered and empty: no repository has current docs.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([])
    assert _deleted(engine) == sorted(
        [str(document.id), str(chunk.id), other_document_id, other_chunk_id]
    )
    assert len(invalidations) == 3


def _repository_node(repo, path):
    return (
        str(fact_node_id(repo, "repository", repo)),
        {"type": "CodeRepository", "name": repo, "path": str(path)},
    )


@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_whose_source_it_could_not_read(
    monkeypatch, invalidations, vectors, tmp_path
):
    """The extraction leaves out a repository whose directory is not readable
    when its stored docs were read from source (a lost clone, an earlier run's
    staging directory), so its absence from the batch says nothing about its
    docs: the sweep leaves it out by the same rule. A repository whose stored
    docs are Enola's, or whose CodeRepository node is gone, is swept; the
    caller's repo_path (the directory a route just indexed) is what makes the
    filtered repository readable again."""
    forgotten = "forgotten_repo"
    kept_document_id = str(code_document_id(REPO, FILE))
    kept_chunk_id = str(uuid4())
    enola_document_id = str(code_document_id(OTHER_REPO, "lib/x.py"))
    enola_chunk_id = str(uuid4())
    forgotten_document_id = str(code_document_id(forgotten, "pkg/y.py"))
    forgotten_chunk_id = str(uuid4())
    nodes = [
        _repository_node(REPO, tmp_path / "gone"),
        _repository_node(OTHER_REPO, tmp_path / "also-gone"),
        (kept_document_id, {"type": "CodeDocument", "repo": REPO, "doc_source": "source"}),
        (kept_chunk_id, {"type": "DocumentChunk"}),
        (enola_document_id, {"type": "CodeDocument", "repo": OTHER_REPO, "doc_source": "enola"}),
        (enola_chunk_id, {"type": "DocumentChunk"}),
        # No CodeRepository node: nothing to read, swept outright.
        (
            forgotten_document_id,
            {"type": "CodeDocument", "repo": forgotten, "doc_source": "source"},
        ),
        (forgotten_chunk_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (kept_chunk_id, kept_document_id, "is_part_of", {}),
        (enola_chunk_id, enola_document_id, "is_part_of", {}),
        (forgotten_chunk_id, forgotten_document_id, "is_part_of", {}),
    ]

    # Unfiltered and empty: of the two repositories without a readable
    # source, the one holding full docstrings stays and the one holding
    # Enola's goes; so does the forgotten one.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([])
    assert _deleted(engine) == sorted(
        [enola_document_id, enola_chunk_id, forgotten_document_id, forgotten_chunk_id]
    )
    assert vectors.deleted == [
        ("CodeDocument_name", sorted([enola_document_id, forgotten_document_id])),
        ("DocumentChunk_text", sorted([enola_chunk_id, forgotten_chunk_id])),
    ]

    # Filtered to the unreadable repository: nothing.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=REPO)
    engine.delete_nodes.assert_not_awaited()

    # The caller's directory is what this run read: the stale docs go.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=REPO, repo_path=tmp_path)
    assert _deleted(engine) == sorted([kept_document_id, kept_chunk_id])

    # A document stored before doc_source existed counts as read from source.
    legacy = [
        (node_id, {"type": "CodeDocument", "repo": REPO})
        if node_id == kept_document_id
        else (node_id, props)
        for node_id, props in nodes
    ]
    engine = _mock_engine(monkeypatch, legacy, edges)
    await add_code_doc_edges([], repo=REPO)
    engine.delete_nodes.assert_not_awaited()


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
async def test_foreign_batch_touches_nothing_while_an_empty_one_sweeps(monkeypatch, invalidations):
    engine = _mock_engine(monkeypatch, CODE_NODES)
    plain_document = TextDocument(name="notes.txt", raw_data_location="/x", external_metadata=None)
    plain_chunk = DocumentChunk(
        text="plain",
        chunk_size=1,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=plain_document,
    )

    # Composed into another pipeline: no code docs in the batch, no read.
    assert await add_code_doc_edges([plain_document, plain_chunk]) == [plain_document, plain_chunk]
    engine.get_filtered_graph_data.assert_not_awaited()
    engine.add_edges.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == []

    # A code docs run that derived nothing still reads and sweeps (nothing
    # stale here) and invalidates the snapshot.
    assert await add_code_doc_edges([]) == []
    engine.get_filtered_graph_data.assert_awaited_once_with([{"type": READ_TYPES}])
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == [{}]


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


def _entity_graph(kept_id, stale_id, entities, types, contains, is_a):
    """A chunk/Entity/EntityType read: ``contains`` (chunk, entity) and ``is_a`` (entity, type)."""
    nodes = [(kept_id, {"type": "DocumentChunk"}), (stale_id, {"type": "DocumentChunk"})]
    nodes += [(entity_id, {"type": "Entity"}) for entity_id in entities]
    nodes += [(type_id, {"type": "EntityType"}) for type_id in types]
    edges = [(chunk_id, entity_id, "contains", {}) for chunk_id, entity_id in contains]
    edges += [(entity_id, type_id, "is_a", {}) for entity_id, type_id in is_a]
    return nodes, edges


@pytest.mark.asyncio
async def test_entity_sweep_removes_the_entities_only_stale_chunks_held(
    monkeypatch, invalidations, vectors
):
    document = _document()
    kept = _symbol_chunk(document)
    stale_id, summary_id = str(uuid4()), str(uuid4())
    shared, old, old_too = (str(uuid4()) for _ in range(3))  # Entity nodes
    shared_type, old_type = str(uuid4()), str(uuid4())  # EntityType nodes
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (stale_id, {"type": "DocumentChunk"}),
        (summary_id, {"type": "TextSummary"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (stale_id, str(document.id), "is_part_of", {}),
        (summary_id, stale_id, "made_from", {}),
        (str(kept.id), SYMBOL_ID, "describes", {}),
    ]
    entity_graph = _entity_graph(
        str(kept.id),
        stale_id,
        entities=[shared, old, old_too],
        types=[shared_type, old_type],
        # ``shared`` is also the kept chunk's; ``old`` and ``old_too`` are the
        # stale chunk's alone. ``old_too`` is of the shared type.
        contains=[(stale_id, shared), (str(kept.id), shared), (stale_id, old), (stale_id, old_too)],
        is_a=[(shared, shared_type), (old, old_type), (old_too, shared_type)],
    )
    entity_graph[1].append((old, shared, "relates_to", {}))
    engine = _mock_engine(monkeypatch, nodes, edges, entity_graph)
    engine.get_connections.return_value = [
        ({"id": stale_id, "type": "DocumentChunk"}, {"relationship_name": "is_part_of"}, {}),
        (
            {"id": stale_id, "type": "DocumentChunk"},
            {"relationship_name": "contains", "properties": json.dumps({"edge_text": None})},
            {"id": old, "name": "old", "type": "Entity"},
        ),
    ]

    counts = await link_code_docs([document, kept])

    # The stale chunk was asked for its neighbours; an entity among them paid
    # the entity read.
    engine.get_connections.assert_awaited_once_with(stale_id)
    assert _reads(engine) == [[{"type": READ_TYPES}], [{"type": ENTITY_READ_TYPES}]]
    # The stale chunk, its summary, the entities only it held and the type only
    # they were go; the shared entity and the shared type stay.
    assert _deleted(engine) == sorted([stale_id, summary_id, old, old_too, old_type])
    assert vectors.deleted == [
        ("DocumentChunk_text", [stale_id]),
        ("TextSummary_text", [summary_id]),
        ("Entity_name", sorted([old, old_too])),
        ("EntityType_name", [old_type]),
    ]
    assert (counts.swept.chunks, counts.swept.summaries, counts.swept.entities) == (1, 1, 3)
    assert counts.swept.owned == 0 and counts.swept.total == 5


@pytest.mark.asyncio
async def test_entity_sweep_skips_the_entity_read_without_entity_neighbours(
    monkeypatch, invalidations, vectors
):
    document = _document()
    kept = _symbol_chunk(document)
    stale_id = str(uuid4())
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (stale_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (stale_id, str(document.id), "is_part_of", {}),
    ]
    engine = _mock_engine(monkeypatch, nodes, edges)
    # Embed-only: the stale chunk hangs off its document and its symbol only.
    engine.get_connections.return_value = [
        ({"id": stale_id, "type": "DocumentChunk"}, {"relationship_name": "is_part_of"}, {}),
        ({"id": stale_id, "type": "DocumentChunk"}, {"relationship_name": "describes"}, {}),
    ]

    await add_code_doc_edges([document, kept])

    assert _reads(engine) == [[{"type": READ_TYPES}]]
    assert _deleted(engine) == [stale_id]
    assert vectors.deleted == [("DocumentChunk_text", [stale_id])]


@pytest.mark.asyncio
async def test_entity_sweep_reads_the_graph_directly_above_the_lookup_limit(
    monkeypatch, invalidations, vectors
):
    monkeypatch.setattr(add_code_doc_edges_module, "_NEIGHBOR_LOOKUP_LIMIT", 0)
    document = _document()
    kept = _symbol_chunk(document)
    stale_id, old, old_type = (str(uuid4()) for _ in range(3))
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (stale_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (stale_id, str(document.id), "is_part_of", {}),
    ]
    entity_graph = _entity_graph(
        str(kept.id), stale_id, [old], [old_type], [(stale_id, old)], [(old, old_type)]
    )
    engine = _mock_engine(monkeypatch, nodes, edges, entity_graph)

    await add_code_doc_edges([document, kept])

    engine.get_connections.assert_not_awaited()
    assert _reads(engine) == [[{"type": READ_TYPES}], [{"type": ENTITY_READ_TYPES}]]
    assert _deleted(engine) == sorted([stale_id, old, old_type])


@pytest.mark.asyncio
async def test_owned_stale_chunks_are_retired_through_the_provenance_planner(
    monkeypatch, invalidations, vectors
):
    dataset_id, data_id = uuid4(), uuid4()
    document = _document()
    kept = _symbol_chunk(document)
    owned_id, unowned_id = str(uuid4()), str(uuid4())
    owned_summary_id, unowned_summary_id, owned_entity_id = (str(uuid4()) for _ in range(3))
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (owned_id, {"type": "DocumentChunk"}),
        (unowned_id, {"type": "DocumentChunk"}),
        (owned_summary_id, {"type": "TextSummary"}),
        (unowned_summary_id, {"type": "TextSummary"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (owned_id, str(document.id), "is_part_of", {}),
        (unowned_id, str(document.id), "is_part_of", {}),
        (owned_summary_id, owned_id, "made_from", {}),
        (unowned_summary_id, unowned_id, "made_from", {}),
    ]
    engine = _mock_engine(monkeypatch, nodes, edges)
    # The owned chunk was stored under a pipeline context with a data item: it
    # carries its own chunk-scoped ref (next to the document-scoped one). The
    # other chunk came from a memify run and carries none.
    engine.get_node_delete_data.return_value = {
        owned_id: _delete_data(
            owned_id,
            [
                make_source_ref_key(dataset_id, data_id),
                make_chunk_source_ref_key(dataset_id, data_id, UUID(owned_id)),
            ],
        ),
        unowned_id: _delete_data(unowned_id, []),
    }
    planner = AsyncMock(
        return_value=SimpleNamespace(deleted_node_ids=[owned_id, owned_summary_id, owned_entity_id])
    )
    monkeypatch.setattr(add_code_doc_edges_module, "delete_chunks_incremental", planner)

    counts = await link_code_docs([document, kept])

    # One planner pass per (dataset, data item), keyed the way the chunk was stored.
    planner.assert_awaited_once_with([owned_id], str(dataset_id), str(data_id))
    # The planner's chunk and its summary are not deleted by hand (nor their
    # vectors); the unowned chunk and its summary are.
    assert _deleted(engine) == sorted([unowned_id, unowned_summary_id])
    assert vectors.deleted == [
        ("DocumentChunk_text", [unowned_id]),
        ("TextSummary_text", [unowned_summary_id]),
    ]
    assert (counts.swept.chunks, counts.swept.summaries, counts.swept.owned) == (2, 1, 2)
    assert counts.swept.total == 5


@pytest.mark.asyncio
async def test_sweep_falls_back_to_the_manual_path_without_usable_provenance(
    monkeypatch, invalidations, vectors
):
    document = _document()
    kept = _symbol_chunk(document)
    stale_id = str(uuid4())
    nodes = CODE_NODES + [
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (stale_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (stale_id, str(document.id), "is_part_of", {}),
    ]
    planner = AsyncMock()
    monkeypatch.setattr(add_code_doc_edges_module, "delete_chunks_incremental", planner)

    # A backend without graph provenance.
    engine = _mock_engine(monkeypatch, nodes, edges)
    engine.get_node_delete_data.side_effect = UnsupportedProvenanceCapability()
    await add_code_doc_edges([document, kept])
    assert _deleted(engine) == [stale_id]

    # A document-scoped ref only, or another chunk's ref: not this chunk's own.
    engine = _mock_engine(monkeypatch, nodes, edges)
    engine.get_node_delete_data.return_value = {
        stale_id: _delete_data(
            stale_id,
            [
                make_source_ref_key(uuid4(), uuid4()),
                make_chunk_source_ref_key(uuid4(), uuid4(), uuid4()),
                "not-a-source-ref",
            ],
        )
    }
    await add_code_doc_edges([document, kept])
    assert _deleted(engine) == [stale_id]

    planner.assert_not_awaited()
    assert vectors.deleted == [("DocumentChunk_text", [stale_id])] * 2


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
    # The ledger's endpoint columns are UUIDs (the graph engine was handed strings).
    assert [(e[0], e[1], e[2]) for e in upsert_edges.await_args.args[0]] == [
        (chunk.id, UUID(SYMBOL_ID), "describes")
    ]
    assert all(type(e[0]) is UUID and type(e[1]) is UUID for e in upsert_edges.await_args.args[0])
