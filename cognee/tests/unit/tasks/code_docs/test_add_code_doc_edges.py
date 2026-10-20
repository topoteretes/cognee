"""add_code_doc_edges: describes edges by id, idempotent re-runs and a scoped sweep.

The sweep leaves alone a repository whose snapshot docs the run could not read;
those cases run against real snapshot directories (write_enola_snapshot).
"""

import importlib
import json
import threading
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
from cognee.tasks.code_docs.extract_code_docs import snapshot_docs
from cognee.tasks.code_docs.models import CodeDocsBatch
from cognee.tasks.code_graph.enola import EnolaDocsUnavailable
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.summarization.models import TextSummary
from cognee.tests.utils.enola_snapshot import (
    doc_line,
    enola_fact_id,
    sha256_label,
    write_enola_snapshot,
)

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
# The snapshot the code graph was built from (CodeRepository.last_snapshot_id).
SNAPSHOT_ID = "snapshot-1"
DOC_LINE = doc_line(REPO, "symbol", SYMBOL, FILE, "Open the database.", line=12)
READ_TYPES = [
    "CodeDocument",
    "DocumentChunk",
    "TextSummary",
    "CodeSymbol",
    "CodeModule",
    # The sweep reads each repository's path and last_snapshot_id.
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


def _repository_node(repo, path, snapshot_id=SNAPSHOT_ID):
    """A stored CodeRepository: where enola wrote its snapshot, and the one its graph holds.

    ``path`` or ``snapshot_id`` None leaves that property out.
    """
    properties = {"type": "CodeRepository", "name": repo}
    if path is not None:
        properties["path"] = str(path)
    if snapshot_id is not None:
        properties["last_snapshot_id"] = snapshot_id
    return str(fact_node_id(repo, "repository", repo)), properties


def _write_snapshot(directory, **overrides):
    """A snapshot as enola 0.4.32 leaves it: the graph's id, a receipt vouching for docs.jsonl."""
    return write_enola_snapshot(
        directory, **{"docs": [DOC_LINE], "snapshot_id": SNAPSHOT_ID, **overrides}
    )


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
    """What each of the sweep's "kept as stored" warnings named."""
    return [
        call.args[1] for call in logger.warning.call_args_list if "kept as stored" in call.args[0]
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
    engine = _mock_engine(monkeypatch, CODE_NODES)

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
    monkeypatch, invalidations, vectors, tmp_path
):
    _write_snapshot(tmp_path / ".enola")
    document = _document()
    chunk = _symbol_chunk(document)
    other_document_id, other_props = _stored_document(OTHER_REPO, "lib/x.py")
    other_chunk_id = str(uuid4())
    # demo_repo's graph holds the snapshot the run read. other_repo has no
    # CodeRepository node (forgotten or renamed): its docs remain.
    nodes = CODE_NODES + [
        _repository_node(REPO, tmp_path),
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
    monkeypatch, invalidations, vectors, tmp_path
):
    # demo_repo's snapshot documents nothing any more: a vouched, empty docs.jsonl.
    _write_snapshot(tmp_path / ".enola", docs=[])
    document = _document()
    chunk = _symbol_chunk(document)
    other_document_id, other_props = _stored_document(OTHER_REPO, "lib/x.py")
    other_chunk_id = str(uuid4())
    nodes = CODE_NODES + [
        _repository_node(REPO, tmp_path),
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


@pytest.mark.asyncio
async def test_sweep_scope_is_the_filtered_repository_or_every_one_with_documents(
    tmp_path, link_logger
):
    """The scope is read off the stored CodeDocuments, a forgotten repository's
    included, not off the CodeRepository nodes; a filter is the scope by
    itself. A repository with a node stays in only while its snapshot docs are
    available."""
    _write_snapshot(tmp_path / ".enola")
    nodes = dict(
        [
            _repository_node(REPO, tmp_path),
            # A node without stored docs: outside an unfiltered scope, never checked.
            _repository_node(OTHER_REPO, tmp_path / "gone"),
            _stored_document(REPO, FILE),
            _stored_document(FORGOTTEN_REPO, "pkg/y.py"),
            (str(uuid4()), {"type": "CodeDocument", "repo": None}),
            (str(uuid4()), {"type": "DocumentChunk", "repo": "not_a_document"}),
        ]
    )

    assert await _sweep_scope(None, nodes) == {REPO, FORGOTTEN_REPO}
    assert await _sweep_scope(REPO, nodes) == {REPO}
    assert await _sweep_scope("new_repo", nodes) == {"new_repo"}
    assert _kept_warnings(link_logger) == []

    # Filtered to the repository whose snapshot is gone: nothing in scope.
    assert await _sweep_scope(OTHER_REPO, nodes) == set()
    assert _kept_warnings(link_logger) == [repr(OTHER_REPO)]


@pytest.mark.asyncio
async def test_sweep_covers_a_repository_whose_snapshot_docs_are_available(
    monkeypatch, invalidations, vectors, tmp_path, link_logger
):
    """The receipt vouches for docs.jsonl and the snapshot id is the one the
    code graph holds: the batch is what the run derived from those docs, so
    the stored docs it lacks are stale."""
    _write_snapshot(tmp_path / ".enola")
    document = _document()
    chunk = _symbol_chunk(document)
    legacy_nodes, legacy_edges, legacy_ids = _stored_docs(REPO, "app/legacy.py")
    nodes = CODE_NODES + [
        _repository_node(REPO, tmp_path),
        (str(document.id), {"type": "CodeDocument", "repo": REPO}),
        (str(chunk.id), {"type": "DocumentChunk"}),
        *legacy_nodes,
    ]
    edges = [(str(chunk.id), str(document.id), "is_part_of", {}), *legacy_edges]

    for repo in (REPO, None):
        engine = _mock_engine(monkeypatch, nodes, edges)
        await add_code_doc_edges([document, chunk], repo=repo)
        assert _deleted(engine) == sorted(legacy_ids)

    # A vouched, empty docs.jsonl says the repository documents nothing any
    # more (not that enola did not run): every stored doc of it goes.
    _write_snapshot(tmp_path / ".enola", docs=[])
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=REPO)
    assert _deleted(engine) == sorted([str(document.id), str(chunk.id), *legacy_ids])
    assert _kept_warnings(link_logger) == []


@pytest.mark.parametrize(
    "snapshot, repository",
    [
        # enola before 0.4.32, or a configuration without the docs renderer.
        pytest.param({"docs": None}, {}, id="no-docs-file"),
        # A docs.jsonl an earlier run left: this run's receipt does not list it.
        pytest.param({"vouch_docs": False}, {}, id="not-vouched"),
        # A later enola run wrote over the snapshot the code graph was built from.
        pytest.param({"snapshot_id": "snapshot-2"}, {}, id="other-snapshot"),
        # The node records no snapshot id to match, or no directory to read.
        pytest.param({}, {"snapshot_id": None}, id="no-stored-snapshot-id"),
        pytest.param({}, {"path": None}, id="no-path"),
        # A lost clone, or an earlier run's staging directory.
        pytest.param(None, {}, id="no-snapshot-directory"),
    ],
)
@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_whose_snapshot_docs_are_unavailable(
    monkeypatch, invalidations, vectors, tmp_path, link_logger, snapshot, repository
):
    """The extraction derived nothing for a repository whose snapshot docs it
    could not read, so the batch says nothing about its stored docs: the sweep
    leaves them alone and names the repository. A forgotten repository (no
    CodeRepository node, so no snapshot to read) goes in the same run."""
    if snapshot is not None:
        _write_snapshot(tmp_path / ".enola", **snapshot)
    kept_nodes, kept_edges, _ = _stored_docs(REPO, FILE)
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = [
        _repository_node(REPO, **{"path": tmp_path, **repository}),
        *kept_nodes,
        *forgotten_nodes,
    ]
    edges = kept_edges + forgotten_edges

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([])
    assert _deleted(engine) == sorted(forgotten_ids)
    assert vectors.deleted == [
        ("CodeDocument_name", forgotten_ids[:1]),
        ("DocumentChunk_text", forgotten_ids[1:]),
    ]

    # Filtered to it: nothing is swept at all.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], repo=REPO)
    engine.delete_nodes.assert_not_awaited()
    assert _kept_warnings(link_logger) == [repr(REPO)] * 2


@pytest.mark.asyncio
async def test_sweep_reads_the_callers_snapshot_dir_instead_of_the_stored_path(
    monkeypatch, invalidations, vectors, tmp_path, link_logger
):
    """The routes pass the snapshot they just loaded: once an unchanged
    snapshot is skipped, a node's path can name an earlier run's staging
    directory, long gone. The caller's directory then decides for every
    repository in scope, so one whose graph holds another snapshot is kept --
    even though its own <path>/.enola would vouch for it."""
    staged = _write_snapshot(tmp_path / "staging" / ".enola")
    _write_snapshot(tmp_path / "clone" / ".enola", snapshot_id="snapshot-2")
    repo_nodes, repo_edges, repo_ids = _stored_docs(REPO, FILE)
    other_nodes, other_edges, other_ids = _stored_docs(OTHER_REPO, "lib/x.py")
    nodes = [
        _repository_node(REPO, tmp_path / "gone"),
        _repository_node(OTHER_REPO, tmp_path / "clone", snapshot_id="snapshot-2"),
        *repo_nodes,
        *other_nodes,
    ]
    edges = repo_edges + other_edges

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([], snapshot_dir=staged)
    assert _deleted(engine) == sorted(repo_ids)
    assert _kept_warnings(link_logger) == [repr(OTHER_REPO)]

    # Without it, each node's own <path>/.enola decides: the other way round.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges([])
    assert _deleted(engine) == sorted(other_ids)
    assert _kept_warnings(link_logger)[1:] == [repr(REPO)]

    # link_code_docs takes it the same way, as a string too.
    engine = _mock_engine(monkeypatch, nodes, edges)
    counts = await link_code_docs([], repo=REPO, snapshot_dir=str(staged))
    assert _deleted(engine) == sorted(repo_ids)
    assert (counts.swept.documents, counts.swept.chunks) == (1, 1)


@pytest.mark.asyncio
async def test_sweep_checks_a_snapshot_off_the_event_loop_without_parsing_it(
    monkeypatch, invalidations, vectors, tmp_path
):
    """The check reads and hashes docs.jsonl -- megabytes on a large
    repository -- so it runs in a worker thread, and leaves the lines unparsed.
    A repository without a node has no snapshot to ask about."""
    _write_snapshot(tmp_path / ".enola")
    repo_nodes, repo_edges, repo_ids = _stored_docs(REPO, FILE)
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = [_repository_node(REPO, tmp_path), *repo_nodes, *forgotten_nodes]
    calls = []

    def recording_snapshot_docs(repository, snapshot_dir=None, **kwargs):
        calls.append((repository["name"], snapshot_dir, kwargs, threading.get_ident()))
        return snapshot_docs(repository, snapshot_dir, **kwargs)

    monkeypatch.setattr(add_code_doc_edges_module, "snapshot_docs", recording_snapshot_docs)
    engine = _mock_engine(monkeypatch, nodes, repo_edges + forgotten_edges)

    await add_code_doc_edges([])

    assert _deleted(engine) == sorted(repo_ids + forgotten_ids)
    assert [call[:3] for call in calls] == [(REPO, None, {"parse": False})]
    assert calls[0][3] != threading.get_ident()


@pytest.mark.asyncio
async def test_sweep_keeps_a_repository_whose_vouched_docs_are_not_utf8(
    monkeypatch, invalidations, vectors, tmp_path
):
    """Same rule both ways: a docs.jsonl the extraction cannot decode derives
    nothing for the repository, even when the receipt lists its bytes, so the
    sweep must keep the repository's stored docs as well."""
    directory = _write_snapshot(tmp_path / ".enola")
    content = json.dumps({**DOC_LINE, "doc": "Open the café."}, ensure_ascii=False)
    content = content.encode("latin-1") + b"\n"
    (directory / "docs.jsonl").write_bytes(content)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    receipt["output_hashes"]["docs.jsonl"] = sha256_label(content)
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    repository_id, repository = _repository_node(REPO, tmp_path)
    # What the extraction reads for it: nothing.
    with pytest.raises(EnolaDocsUnavailable, match="not UTF-8"):
        snapshot_docs(repository)
    stored_nodes, stored_edges, _ = _stored_docs(REPO, FILE)
    engine = _mock_engine(monkeypatch, [(repository_id, repository), *stored_nodes], stored_edges)

    await add_code_doc_edges([], repo=REPO)

    engine.delete_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_follows_the_extractions_read_set_not_the_snapshots_on_disk(
    monkeypatch, invalidations, vectors, tmp_path, link_logger
):
    """A batch from extract_code_docs says which repositories the extraction
    read (CodeDocsBatch.read_repos), and the sweep follows that decision rather
    than looking at the snapshots again: a repository the extraction left out
    is kept even though its snapshot has become readable since (another run
    wrote it while this one was under way), one it read is swept even though
    its snapshot is gone by now, and a forgotten one goes as before. No
    snapshot is checked."""
    _write_snapshot(tmp_path / "late" / ".enola")
    late_nodes, late_edges, _ = _stored_docs(REPO, FILE)
    read_nodes, read_edges, read_ids = _stored_docs(OTHER_REPO, "lib/x.py")
    forgotten_nodes, forgotten_edges, forgotten_ids = _stored_docs(FORGOTTEN_REPO, "pkg/y.py")
    nodes = [
        _repository_node(REPO, tmp_path / "late"),
        _repository_node(OTHER_REPO, tmp_path / "gone"),
        *late_nodes,
        *read_nodes,
        *forgotten_nodes,
    ]
    edges = late_edges + read_edges + forgotten_edges
    checks = []
    monkeypatch.setattr(
        add_code_doc_edges_module, "snapshot_docs", lambda *args, **kwargs: checks.append(args)
    )

    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], {OTHER_REPO}))

    assert _deleted(engine) == sorted(read_ids + forgotten_ids)
    assert _kept_warnings(link_logger) == [repr(REPO)]
    assert checks == []

    # Filtered to a repository the extraction did not read: nothing goes.
    engine = _mock_engine(monkeypatch, nodes, edges)
    await add_code_doc_edges(CodeDocsBatch.of([], ()), repo=REPO)
    engine.delete_nodes.assert_not_awaited()
    assert checks == []


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
    package = _document(PACKAGE, symbol_count=0)
    orphan = _symbol_chunk(document, symbol_id=str(fact_node_id(REPO, "symbol", "app/db.Gone")))
    module_chunk = _module_chunk(package)
    # The graph holds neither the symbol the chunk names nor the CodeModule.
    engine = _mock_engine(monkeypatch, CODE_NODES[:1])

    counts = await link_code_docs([document, orphan, package, module_chunk])

    engine.add_edges.assert_not_awaited()
    assert (counts.edges, counts.present, counts.unresolved) == (0, 0, 2)


@pytest.mark.asyncio
async def test_a_module_doc_without_its_module_id_is_not_linked(monkeypatch, invalidations):
    """The target is the node extract_code_docs joined the docs.jsonl line to,
    never a module derived from a path: a chunk without its module_id stays
    unlinked although the graph holds the CodeModule of its file's directory."""
    document = _document("app/__init__.py", symbol_count=0)
    module_chunk = _module_chunk(document, module_id=None)
    engine = _mock_engine(monkeypatch, CODE_NODES)

    counts = await link_code_docs([document, module_chunk])

    engine.add_edges.assert_not_awaited()
    assert (counts.edges, counts.unresolved) == (0, 1)


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
