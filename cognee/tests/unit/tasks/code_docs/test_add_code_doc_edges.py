"""add_code_doc_edges: describes edges by id, idempotent re-runs and a scoped sweep.

Only the extraction's own batch (CodeDocsBatch, which records the snapshot each
repository's docs were read from) is swept against; a plain list is linked and
never swept. The sweep reads no snapshot: whether a repository's docs could be
read is the extraction's decision (test_extract_code_docs), so the graph and
vector engines are all these tests need, and both are mocked.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
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
from cognee.tasks.code_docs.add_code_doc_edges import _sweep_scope, link_code_docs
from cognee.tasks.code_docs.models import CodeDocsBatch
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.summarization.models import TextSummary
from cognee.tests.utils.enola_snapshot import enola_fact_id

add_code_doc_edges_module = importlib.import_module("cognee.tasks.code_docs.add_code_doc_edges")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
vector_module = importlib.import_module("cognee.infrastructure.databases.vector")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

REPO = "demo_repo"
OTHER_REPO = "other_repo"
FORGOTTEN_REPO = "forgotten_repo"
FILE = "app/db.py"
# A module doc is per package: its ``file`` is the package directory.
PACKAGE = "app"
SYMBOL = "app/db.Database"
SYMBOL_ID = str(fact_node_id(REPO, "symbol", SYMBOL))
MODULE_ID = str(fact_node_id(REPO, "module", PACKAGE))
# The fact ids docs.jsonl lines carry, stored as the code nodes' enola_id.
SYMBOL_ENOLA_ID = enola_fact_id(REPO, "symbol", SYMBOL, FILE)
MODULE_ENOLA_ID = enola_fact_id(REPO, "module", PACKAGE, PACKAGE)
# The snapshot the code graph was built from (CodeRepository.last_snapshot_id):
# what extract_code_docs records for a repository whose docs it read.
SNAPSHOT_ID = "snapshot-1"
READ_TYPES = [
    "CodeDocument",
    "DocumentChunk",
    "TextSummary",
    "CodeSymbol",
    "CodeModule",
    # Whether each repository still has its node, and which snapshot its graph holds.
    "CodeRepository",
]
ENTITY_READ_TYPES = ["DocumentChunk", "Entity", "EntityType"]

CODE_NODES = [
    (SYMBOL_ID, {"type": "CodeSymbol", "name": SYMBOL, "repo": REPO, "enola_id": SYMBOL_ENOLA_ID}),
    (MODULE_ID, {"type": "CodeModule", "name": PACKAGE, "repo": REPO, "enola_id": MODULE_ENOLA_ID}),
]


def _document(file_path=FILE, repo=REPO, symbol_count=1):
    return CodeDocument(
        id=code_document_id(repo, file_path),
        name=file_path,
        raw_data_location=file_path,
        external_metadata=json.dumps(
            {"code_doc": {"repo": repo, "file": file_path, "language": "python"}}
        ),
        repo=repo,
        file_path=file_path,
        language="python",
        symbol_count=symbol_count,
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
    text = f"{FILE}:{line}\n{SYMBOL} (class)\n\n{docstring}"
    metadata = {
        "repo": REPO,
        "file": FILE,
        "line": line,
        "end_line": 40,
        "symbol": SYMBOL,
        "symbol_id": symbol_id,
        "symbol_kind": "class",
        "language": "python",
        "enola_id": SYMBOL_ENOLA_ID,
    }
    return _chunk(document, text, 1, metadata)


def _module_chunk(document, module_id=MODULE_ID):
    """A module doc in ``document``: no line, the package named, its CodeModule as module_id."""
    metadata = {
        "repo": REPO,
        "file": document.file_path,
        "line": None,
        "end_line": None,
        "symbol": PACKAGE,
        "symbol_id": None,
        "symbol_kind": "module",
        "language": "python",
        "enola_id": MODULE_ENOLA_ID,
        "module_id": module_id,
    }
    text = f"{document.file_path}\n{PACKAGE} (module)\n\nDatabase helpers."
    return _chunk(document, text, 0, metadata)


def _stored_document(repo, file_path):
    return (str(code_document_id(repo, file_path)), {"type": "CodeDocument", "repo": repo})


def _stored_docs(repo, file_path):
    """A stored CodeDocument of ``repo`` with one chunk: (nodes, edges, [document id, chunk id])."""
    document_id, properties = _stored_document(repo, file_path)
    chunk_id = str(uuid4())
    nodes = [(document_id, properties), (chunk_id, {"type": "DocumentChunk"})]
    return nodes, [(chunk_id, document_id, "is_part_of", {})], [document_id, chunk_id]


def _as_stored(document, *chunks):
    """The graph rows of a batch's document and chunks once the store wrote them: (nodes, edges)."""
    nodes = [(str(document.id), {"type": "CodeDocument", "repo": document.repo})]
    nodes += [(str(chunk.id), {"type": "DocumentChunk"}) for chunk in chunks]
    edges = [(str(chunk.id), str(document.id), "is_part_of", {}) for chunk in chunks]
    return nodes, edges


def _repository_node(repo, snapshot_id=SNAPSHOT_ID):
    """A stored CodeRepository whose code graph holds ``snapshot_id`` (its last_snapshot_id)."""
    properties = {"type": "CodeRepository", "name": repo, "last_snapshot_id": snapshot_id}
    return str(fact_node_id(repo, "repository", repo)), properties


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


def _kept_warnings(logger):
    """The sweep's "kept as stored" warnings: (the repositories named, the reason given)."""
    return [
        (call.args[1], call.args[2])
        for call in logger.warning.call_args_list
        if "kept as stored" in call.args[0]
    ]


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


@pytest.fixture
def link_logger(monkeypatch):
    """The link task's logger, recorded."""
    logger = MagicMock()
    monkeypatch.setattr(add_code_doc_edges_module, "logger", logger)
    return logger


@pytest.mark.asyncio
async def test_writes_describes_edges_in_the_code_graph_shape(monkeypatch, invalidations):
    document = _document()
    package = _document(PACKAGE, symbol_count=0)
    symbol_chunk = _symbol_chunk(document)
    module_chunk = _module_chunk(package)
    stored_nodes, stored_edges = _as_stored(document, symbol_chunk)
    package_nodes, package_edges = _as_stored(package, module_chunk)
    engine = _mock_engine(
        monkeypatch, CODE_NODES + stored_nodes + package_nodes, stored_edges + package_edges
    )

    result = await add_code_doc_edges([document, package, module_chunk, symbol_chunk])

    assert result == [document, package, module_chunk, symbol_chunk]
    engine.get_filtered_graph_data.assert_awaited_once_with([{"type": READ_TYPES}])
    # Chunk -> code node, in batch order; a module doc links to its CodeModule.
    assert _edge_keys(engine) == [
        (str(module_chunk.id), MODULE_ID, "describes"),
        (str(symbol_chunk.id), SYMBOL_ID, "describes"),
    ]
    properties = engine.add_edges.await_args.args[0][1][3]
    # The props build_code_graph_edges writes plus the chunk's index, nothing
    # else: Enola is the only source of a doc, so none is recorded.
    assert set(properties) == {
        "chunk_index",
        "source_node_id",
        "target_node_id",
        "relationship_name",
        "edge_text",
        "updated_at",
    }
    assert properties["source_node_id"] == str(symbol_chunk.id)
    assert properties["target_node_id"] == SYMBOL_ID
    assert properties["relationship_name"] == DESCRIBES_RELATIONSHIP == "describes"
    assert properties["edge_text"] == "describes"
    assert properties["chunk_index"] == 1
    # A plain list is linked, never swept; the snapshot cache is invalidated regardless.
    engine.delete_nodes.assert_not_awaited()
    engine.get_node_delete_data.assert_not_awaited()
    assert invalidations == [{}]


@pytest.mark.asyncio
async def test_rerun_adds_no_duplicate_edges(monkeypatch, invalidations, vectors, link_logger):
    """An unchanged re-run yields the ids the graph holds: the edge diff adds
    nothing and the sweep, which runs against the extraction's batch, finds
    nothing stale."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes]
    edges = [(str(chunk.id), SYMBOL_ID, "describes", {}), *stored_edges]
    engine = _mock_engine(monkeypatch, nodes, edges)

    await add_code_doc_edges(CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID}))

    engine.add_edges.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []
    assert _kept_warnings(link_logger) == []


@pytest.mark.asyncio
async def test_a_plain_list_is_linked_and_never_swept(
    monkeypatch, invalidations, vectors, link_logger
):
    """A plain list from any caller but the extraction carries no record of
    what was read, so it says nothing about the docs it lacks: its describes
    edges are written and nothing is deleted, filtered or not -- not even a
    forgotten repository's docs, which an unfiltered extraction batch sweeps.
    An empty one has nothing to link and does not even read the graph."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    legacy_nodes, legacy_edges, legacy_ids = _stored_docs(REPO, "app/legacy.py")
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes, *legacy_nodes, *forgotten_nodes]
    edges = stored_edges + legacy_edges + forgotten_edges

    for repo in (None, REPO):
        engine = _mock_engine(monkeypatch, nodes, edges)
        counts = await link_code_docs([document, chunk], repo=repo)
        assert _edge_keys(engine) == [(str(chunk.id), SYMBOL_ID, "describes")]
        assert (counts.edges, counts.swept.total) == (1, 0)
        engine.get_node_delete_data.assert_not_awaited()
        engine.delete_nodes.assert_not_awaited()

    for repo in (None, REPO):
        engine = _mock_engine(monkeypatch, nodes, edges)
        assert await add_code_doc_edges([], repo=repo) == []
        engine.get_filtered_graph_data.assert_not_awaited()
        engine.delete_nodes.assert_not_awaited()
    # Nothing was read or written for the empty lists, so nothing was invalidated.
    assert invalidations == [{}, {}]
    assert vectors.deleted == []
    assert _kept_warnings(link_logger) == []

    # The same points as the extraction's batch: the legacy and forgotten docs go.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID}))
    assert _deleted(engine) == sorted(legacy_ids + forgotten_ids)


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
        _repository_node(REPO),
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
    await add_code_doc_edges(
        CodeDocsBatch.of([document, kept_chunk], {REPO: SNAPSHOT_ID}), repo=REPO
    )

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
        ("DocumentChunk_text", sorted([stale_chunk_id, changed_chunk_id])),
        ("TextSummary_text", [stale_summary_id]),
        ("CodeDocument_name", [stale_document_id]),
    ]
    # The stale chunks' provenance was consulted before anything was deleted.
    engine.get_node_delete_data.assert_awaited_once_with(sorted([changed_chunk_id, stale_chunk_id]))


@pytest.mark.asyncio
async def test_sweep_covers_a_repository_read_from_the_snapshot_its_graph_holds(
    monkeypatch, invalidations, vectors, link_logger
):
    """The extraction read the repository's docs from the snapshot its code
    graph still holds, and the store wrote them: the batch is what the run
    derived, so the stored docs it lacks are stale, with a filter or without."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    legacy_nodes, legacy_edges, legacy_ids = _stored_docs(REPO, "app/legacy.py")
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes, *legacy_nodes]
    edges = stored_edges + legacy_edges
    batch = CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID})

    for repo in (REPO, None):
        engine = _mock_engine(monkeypatch, nodes, edges)
        await add_code_doc_edges(batch, repo=repo)
        assert _deleted(engine) == sorted(legacy_ids)
    assert _kept_warnings(link_logger) == []


@pytest.mark.asyncio
async def test_a_forgotten_repository_is_swept_unless_the_run_is_filtered_to_another(
    monkeypatch, invalidations, vectors, link_logger
):
    """A repository whose CodeRepository node is gone (forgotten, or renamed)
    has no snapshot left to read and no current docs. An unfiltered run covers
    every repository with stored docs, so its docs go, from the graph and the
    vectors; a run filtered to another repository covers that one alone."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes, *forgotten_nodes]
    edges = stored_edges + forgotten_edges
    batch = CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID})

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(batch)
    assert _deleted(engine) == sorted(forgotten_ids)

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(batch, repo=REPO)
    engine.delete_nodes.assert_not_awaited()

    assert vectors.deleted == [
        ("DocumentChunk_text", forgotten_ids[1:]),
        ("CodeDocument_name", forgotten_ids[:1]),
    ]
    assert _kept_warnings(link_logger) == []


@pytest.mark.asyncio
async def test_sweep_with_a_filter_covers_that_repository_even_from_an_empty_batch(
    monkeypatch, invalidations, vectors
):
    repo_nodes, repo_edges, repo_ids = _stored_docs(REPO, FILE)
    other_nodes, other_edges, other_ids = _stored_docs(OTHER_REPO, "lib/x.py")
    # demo_repo's graph holds the snapshot the runs read; other_repo has no
    # CodeRepository node (forgotten, or renamed).
    nodes = [_repository_node(REPO), *repo_nodes, *other_nodes]
    edges = repo_edges + other_edges

    # Filtered to other_repo, the extraction found no repository of that name
    # to read: its docs go, demo_repo's stay.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {}), repo=OTHER_REPO)
    assert _deleted(engine) == sorted(other_ids)

    # Filtered to demo_repo, whose snapshot documents nothing any more (a
    # vouched, empty docs.jsonl, read): its docs go, the other's stay.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {REPO: SNAPSHOT_ID}), repo=REPO)
    assert _deleted(engine) == sorted(repo_ids)

    # Unfiltered and empty: no repository has current docs.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {REPO: SNAPSHOT_ID}))
    assert _deleted(engine) == sorted(repo_ids + other_ids)
    assert len(invalidations) == 3


def test_sweep_scope_is_the_filtered_repository_or_every_one_with_documents(link_logger):
    """The scope is read off the stored CodeDocuments, a forgotten repository's
    included, not off the CodeRepository nodes; a filter is the scope by
    itself."""
    nodes = dict(
        [
            _repository_node(REPO),
            # A node without stored docs: outside an unfiltered scope.
            _repository_node(OTHER_REPO),
            _stored_document(REPO, FILE),
            _stored_document(FORGOTTEN_REPO, "pkg/y.py"),
            (str(uuid4()), {"type": "CodeDocument", "repo": None}),
            (str(uuid4()), {"type": "DocumentChunk", "repo": "not_a_document"}),
        ]
    )
    read_snapshots = {REPO: SNAPSHOT_ID, OTHER_REPO: SNAPSHOT_ID}

    assert _sweep_scope(None, nodes, read_snapshots, {}) == {REPO, FORGOTTEN_REPO}
    assert _sweep_scope(REPO, nodes, read_snapshots, {}) == {REPO}
    assert _sweep_scope(OTHER_REPO, nodes, read_snapshots, {}) == {OTHER_REPO}
    assert _sweep_scope("new_repo", nodes, read_snapshots, {}) == {"new_repo"}
    assert _kept_warnings(link_logger) == []


def test_sweep_scope_keeps_each_repository_the_batch_does_not_account_for(link_logger):
    """Of the repositories in scope that still have their CodeRepository node,
    kept out are one the extraction did not read, one whose graph holds
    another snapshot than the one read, and one with batch ids the graph read
    lacks, each kind named in a warning of its own. A repository without a
    node has no snapshot to compare and stays in."""
    stored_chunk_id, unstored_chunk_id = str(uuid4()), str(uuid4())
    names = ("read", "not_read", "moved", "unstored")
    nodes = dict(
        [
            _repository_node("read"),
            _repository_node("not_read"),
            _repository_node("moved", snapshot_id="snapshot-2"),
            _repository_node("unstored"),
            *(_stored_document(name, FILE) for name in (*names, FORGOTTEN_REPO)),
            (stored_chunk_id, {"type": "DocumentChunk"}),
        ]
    )
    read_snapshots = {name: SNAPSHOT_ID for name in ("read", "moved", "unstored")}
    batch_ids = {"read": {stored_chunk_id}, "unstored": {unstored_chunk_id}}

    assert _sweep_scope(None, nodes, read_snapshots, batch_ids) == {"read", FORGOTTEN_REPO}
    reasons = dict(_kept_warnings(link_logger))
    assert set(reasons) == {"'not_read'", "'moved'", "'unstored'"}
    assert "did not read" in reasons["'not_read'"]
    assert "moved to another snapshot" in reasons["'moved'"]
    assert "not stored" in reasons["'unstored'"]


@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_the_extraction_did_not_read(
    monkeypatch, invalidations, vectors, link_logger
):
    """The extraction derives nothing for a repository whose snapshot docs it
    cannot read (a lost clone, a gone staging directory, a snapshot another
    enola run wrote over) and leaves it out of its record, so the batch says
    nothing about its stored docs: they are kept, and a warning names the
    repository. In the same run a repository it read loses the docs it no
    longer has, and a forgotten one (no CodeRepository node) goes."""
    kept_nodes, kept_edges, _ = _stored_docs(REPO, FILE)
    read_nodes, read_edges, read_ids = _stored_docs(OTHER_REPO, "lib/x.py")
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = [
        _repository_node(REPO),
        _repository_node(OTHER_REPO),
        *kept_nodes,
        *read_nodes,
        *forgotten_nodes,
    ]
    edges = kept_edges + read_edges + forgotten_edges

    # other_repo was read (it documents nothing any more); demo_repo was not.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {OTHER_REPO: SNAPSHOT_ID}))
    assert _deleted(engine) == sorted(read_ids + forgotten_ids)

    # Filtered to it: nothing is swept at all.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {}), repo=REPO)
    engine.delete_nodes.assert_not_awaited()

    assert vectors.deleted == [
        ("DocumentChunk_text", sorted([read_ids[1], forgotten_ids[1]])),
        ("CodeDocument_name", sorted([read_ids[0], forgotten_ids[0]])),
    ]
    warnings = _kept_warnings(link_logger)
    assert [named for named, _why in warnings] == [repr(REPO)] * 2
    assert all("did not read" in why for _named, why in warnings)


@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_whose_graph_moved_to_another_snapshot(
    monkeypatch, invalidations, vectors, link_logger
):
    """Another run loaded a newer snapshot of the repository after this run's
    extraction read the older one, and stored the newer snapshot's docs, which
    this batch lacks. Sweeping against the batch would delete them: the
    stored docs are kept, with a warning, until a run reads the snapshot the
    graph holds."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    newer_nodes, newer_edges, newer_ids = _stored_docs(REPO, "app/newer.py")
    nodes = CODE_NODES + [*stored_nodes, *newer_nodes]
    edges = stored_edges + newer_edges
    batch = CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID})

    engine = _mock_engine(monkeypatch, [_repository_node(REPO, "snapshot-2"), *nodes], edges)
    await add_code_doc_edges(batch, repo=REPO)
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []
    warnings = _kept_warnings(link_logger)
    assert [named for named, _why in warnings] == [repr(REPO)]
    assert "moved to another snapshot" in warnings[0][1]

    # Against a graph that still holds the snapshot read, the same batch sweeps them.
    engine = _mock_engine(monkeypatch, [_repository_node(REPO), *nodes], edges)
    await add_code_doc_edges(batch, repo=REPO)
    assert _deleted(engine) == sorted(newer_ids)


@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_whose_batch_the_graph_does_not_hold(
    monkeypatch, invalidations, vectors, link_logger
):
    """The link task runs after the store. A batch whose docs the graph read
    does not hold (linked without being stored, or after a store that failed
    part of the way) would have the sweep replace the stored docs with
    nothing: they are kept, with a warning, until the batch is stored."""
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    legacy_nodes, legacy_edges, legacy_ids = _stored_docs(REPO, "app/legacy.py")
    batch = CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID})

    # The batch's document is in the graph, its chunk is not.
    nodes = CODE_NODES + [_repository_node(REPO), stored_nodes[0], *legacy_nodes]
    engine = _mock_engine(monkeypatch, nodes, legacy_edges)
    await add_code_doc_edges(batch)
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []
    warnings = _kept_warnings(link_logger)
    assert [named for named, _why in warnings] == [repr(REPO)]
    assert "not stored" in warnings[0][1]

    # Once the chunk is stored, the same batch sweeps the legacy docs.
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes, *legacy_nodes]
    engine = _mock_engine(monkeypatch, nodes, stored_edges + legacy_edges)
    await add_code_doc_edges(batch)
    assert _deleted(engine) == sorted(legacy_ids)


@pytest.mark.asyncio
async def test_sweep_deletes_the_vectors_first_and_the_documents_last(
    monkeypatch, invalidations, vectors
):
    """Vectors first: a sweep that fails part of the way leaves graph nodes
    without vectors rather than vectors without their nodes, which retrieval
    would keep serving. In the graph, each node goes before the node the next
    sweep finds it through: a summary (found through made_from) before its
    chunk, chunks before their documents (found through is_part_of), and the
    documents last."""
    monkeypatch.setattr(add_code_doc_edges_module, "_SWEEP_CHUNK_SIZE", 2)
    document = _document()
    kept = _symbol_chunk(document)
    changed_id, summary_id = str(uuid4()), str(uuid4())
    stored_nodes, stored_edges = _as_stored(document, kept)
    first_nodes, first_edges, first_ids = _stored_docs(REPO, "app/first.py")
    second_nodes, second_edges, second_ids = _stored_docs(REPO, "app/second.py")
    nodes = CODE_NODES + [
        _repository_node(REPO),
        *stored_nodes,
        (changed_id, {"type": "DocumentChunk"}),
        (summary_id, {"type": "TextSummary"}),
        *first_nodes,
        *second_nodes,
    ]
    edges = [
        *stored_edges,
        (changed_id, str(document.id), "is_part_of", {}),
        (summary_id, changed_id, "made_from", {}),
        *first_edges,
        *second_edges,
    ]
    events = []

    async def delete_vectors(collection, ids):
        events.append((collection, list(ids)))

    monkeypatch.setattr(vectors, "delete_data_points", delete_vectors)
    engine = _mock_engine(monkeypatch, nodes, edges)
    engine.delete_nodes.side_effect = lambda ids: events.append(("graph", list(ids)))

    await add_code_doc_edges(CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID}))

    # Three stale chunks, a summary and two stale documents, two ids a call:
    # every vector delete comes before the first graph delete.
    assert [target for target, _ids in events] == [
        "DocumentChunk_text",
        "DocumentChunk_text",
        "TextSummary_text",
        "CodeDocument_name",
        "graph",
        "graph",
        "graph",
    ]
    stale_documents = sorted([first_ids[0], second_ids[0]])
    assert events[-1] == ("graph", stale_documents)
    graph_order = [node_id for _target, ids in events[4:] for node_id in ids]
    assert sorted(graph_order) == sorted(
        [changed_id, first_ids[1], second_ids[1], summary_id, *stale_documents]
    )
    # The summary goes first, before the chunk it was made from.
    assert graph_order[0] == summary_id
    assert graph_order.index(summary_id) < graph_order.index(changed_id)


@pytest.mark.asyncio
async def test_sweep_deletes_in_chunks(monkeypatch, invalidations, vectors):
    monkeypatch.setattr(add_code_doc_edges_module, "_SWEEP_CHUNK_SIZE", 2)
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    stale_ids = [str(code_document_id(REPO, f"app/old_{index}.py")) for index in range(3)]
    nodes = CODE_NODES + [_repository_node(REPO), *stored_nodes]
    nodes += [(node_id, {"type": "CodeDocument", "repo": REPO}) for node_id in stale_ids]
    engine = _mock_engine(monkeypatch, nodes, stored_edges)

    await add_code_doc_edges(CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID}))

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
    stored_nodes, stored_edges = _as_stored(document, chunk)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [
        _repository_node(REPO),
        *stored_nodes,
        (stale_document_id, {"type": "CodeDocument", "repo": REPO}),
    ]
    engine = _mock_engine(monkeypatch, nodes, stored_edges)

    await add_code_doc_edges(CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID}))

    assert _deleted(engine) == [stale_document_id]
    assert vectors.deleted == []


@pytest.mark.asyncio
async def test_sweep_can_be_disabled(monkeypatch, invalidations, vectors):
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [
        _repository_node(REPO),
        *stored_nodes,
        (stale_document_id, {"type": "CodeDocument", "repo": REPO}),
    ]
    batch = CodeDocsBatch.of([document, chunk], {REPO: SNAPSHOT_ID})
    engine = _mock_engine(monkeypatch, nodes, stored_edges)

    await add_code_doc_edges(batch, sweep=False)

    engine.add_edges.assert_awaited_once()
    engine.delete_nodes.assert_not_awaited()
    assert vectors.deleted == []

    # The same batch with the sweep on removes the stale document.
    engine = _mock_engine(monkeypatch, nodes, stored_edges)
    await add_code_doc_edges(batch)
    assert _deleted(engine) == [stale_document_id]


@pytest.mark.asyncio
async def test_foreign_batch_touches_nothing_while_an_empty_extraction_batch_sweeps(
    monkeypatch, invalidations
):
    engine = _mock_engine(monkeypatch, CODE_NODES)
    plain_document = TextDocument(name="notes.txt", raw_data_location="/x", external_metadata=None)
    plain_chunk = DocumentChunk(
        text="plain",
        chunk_size=1,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=plain_document,
    )
    foreign = [plain_document, plain_chunk]

    # Composed into another pipeline: no code docs in the batch, no read,
    # whether or not the batch carries an extraction record.
    assert await add_code_doc_edges(foreign) == foreign
    assert await add_code_doc_edges(CodeDocsBatch.of(foreign, {REPO: SNAPSHOT_ID})) == foreign
    engine.get_filtered_graph_data.assert_not_awaited()
    engine.add_edges.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == []

    # A code docs run that derived nothing still reads and sweeps (nothing
    # stale here) and invalidates the snapshot.
    empty = CodeDocsBatch.of([], {REPO: SNAPSHOT_ID})
    assert await add_code_doc_edges(empty) is empty
    engine.get_filtered_graph_data.assert_awaited_once_with([{"type": READ_TYPES}])
    engine.delete_nodes.assert_not_awaited()
    assert invalidations == [{}]


@pytest.mark.asyncio
async def test_chunks_whose_code_node_is_missing_are_not_linked(monkeypatch, invalidations):
    document = _document()
    package = _document(PACKAGE, symbol_count=0)
    orphan = _symbol_chunk(document, symbol_id=str(fact_node_id(REPO, "symbol", "app/db.Gone")))
    module_chunk = _module_chunk(package)
    stored_nodes, _ = _as_stored(document, orphan)
    package_nodes, _ = _as_stored(package, module_chunk)
    # The graph holds neither the symbol the chunk names nor the CodeModule.
    engine = _mock_engine(monkeypatch, CODE_NODES[:1] + stored_nodes + package_nodes)

    counts = await link_code_docs([document, orphan, package, module_chunk])

    engine.add_edges.assert_not_awaited()
    assert (counts.edges, counts.present, counts.unresolved, counts.unstored) == (0, 0, 2, 0)


@pytest.mark.asyncio
async def test_chunks_the_graph_does_not_hold_are_not_linked(monkeypatch, invalidations):
    """A chunk is linked only once it is stored: an edge from a chunk the graph
    lacks would be dropped by the engine and still counted as written. Such a
    chunk is counted as unstored and named in a warning instead."""
    document = _document()
    chunk = _symbol_chunk(document)
    engine = _mock_engine(monkeypatch, CODE_NODES)
    logger = MagicMock()
    monkeypatch.setattr(add_code_doc_edges_module, "logger", logger)

    counts = await link_code_docs([document, chunk])

    engine.add_edges.assert_not_awaited()
    assert (counts.edges, counts.present, counts.unresolved, counts.unstored) == (0, 0, 0, 1)
    assert any("not in the graph" in call.args[0] for call in logger.warning.call_args_list)


@pytest.mark.asyncio
async def test_a_module_doc_without_its_module_id_is_not_linked(monkeypatch, invalidations):
    """The target is the node extract_code_docs joined the docs.jsonl line to,
    never a module derived from a path: a chunk without its module_id stays
    unlinked although the graph holds the CodeModule of its file's directory."""
    document = _document("app/__init__.py", symbol_count=0)
    module_chunk = _module_chunk(document, module_id=None)
    stored_nodes, stored_edges = _as_stored(document, module_chunk)
    engine = _mock_engine(monkeypatch, CODE_NODES + stored_nodes, stored_edges)

    counts = await link_code_docs([document, module_chunk])

    engine.add_edges.assert_not_awaited()
    assert (counts.edges, counts.unresolved) == (0, 1)


@pytest.mark.asyncio
async def test_summaries_carry_the_chunks_in_entity_mode(monkeypatch, invalidations):
    document = _document()
    chunk = _symbol_chunk(document)
    summary = TextSummary(text="Opens the database.", made_from=chunk)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    stale_document_id = str(code_document_id(REPO, "app/legacy.py"))
    nodes = CODE_NODES + [
        _repository_node(REPO),
        *stored_nodes,
        (stale_document_id, {"type": "CodeDocument", "repo": REPO}),
    ]
    engine = _mock_engine(monkeypatch, nodes, stored_edges)

    await add_code_doc_edges(CodeDocsBatch.of([summary], {REPO: SNAPSHOT_ID}))

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
        _repository_node(REPO),
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

    counts = await link_code_docs(CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID}))

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
        _repository_node(REPO),
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

    await add_code_doc_edges(CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID}))

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
        _repository_node(REPO),
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

    await add_code_doc_edges(CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID}))

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
        _repository_node(REPO),
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

    counts = await link_code_docs(CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID}))

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
        _repository_node(REPO),
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(kept.id), {"type": "DocumentChunk"}),
        (stale_id, {"type": "DocumentChunk"}),
    ]
    edges = [
        (str(kept.id), str(document.id), "is_part_of", {}),
        (stale_id, str(document.id), "is_part_of", {}),
    ]
    batch = CodeDocsBatch.of([document, kept], {REPO: SNAPSHOT_ID})
    planner = AsyncMock()
    monkeypatch.setattr(add_code_doc_edges_module, "delete_chunks_incremental", planner)

    # A backend without graph provenance.
    engine = _mock_engine(monkeypatch, nodes, edges)
    engine.get_node_delete_data.side_effect = UnsupportedProvenanceCapability()
    await add_code_doc_edges(batch)
    assert _deleted(engine) == [stale_id]

    # An adapter bug is not a missing capability: it raises, nothing is swept.
    engine = _mock_engine(monkeypatch, nodes, edges)
    engine.get_node_delete_data.side_effect = AttributeError("adapter bug")
    with pytest.raises(AttributeError, match="adapter bug"):
        await add_code_doc_edges(batch)
    assert _deleted(engine) == []

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
    await add_code_doc_edges(batch)
    assert _deleted(engine) == [stale_id]

    planner.assert_not_awaited()
    assert vectors.deleted == [("DocumentChunk_text", [stale_id])] * 2


@pytest.mark.asyncio
async def test_invalidates_the_dataset_snapshot_even_when_the_write_fails(
    monkeypatch, invalidations
):
    document = _document()
    chunk = _symbol_chunk(document)
    stored_nodes, stored_edges = _as_stored(document, chunk)
    engine = _mock_engine(monkeypatch, CODE_NODES + stored_nodes, stored_edges)
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
    stored_nodes, stored_edges = _as_stored(document, chunk)

    # A memify run: its data item is the `{}` seed, which has no ledger row.
    _mock_engine(monkeypatch, CODE_NODES + stored_nodes, stored_edges)
    await add_code_doc_edges(
        [document, chunk],
        ctx=SimpleNamespace(user=user, dataset=dataset, data_item={}, pipeline_run_id="run-1"),
    )
    upsert_edges.assert_not_awaited()

    _mock_engine(monkeypatch, CODE_NODES + stored_nodes, stored_edges)
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
