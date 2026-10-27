"""The code docs chain against a real Ladybug graph: store, link, re-run, change, remove.

The mocked tests next to this one assert the calls each task makes. This one
runs the enola code graph tasks over a snapshot of a small Python repository
and then the code docs chain -- extract, store (graph only), link and sweep --
against a real engine, and reads the stored graph back, so the stored shapes,
the edge targets, the idempotent re-run (no write, no edge, no sweep), the
sweep of a forgotten repository, the docs kept while a repository's snapshot
docs are not available or while the batch a sweep runs against is not the
graph's (another run loaded a newer snapshot first, or the batch was never
stored), and entity mode's cleanup (with and without stored provenance) are
checked end to end.

A repository here is a directory that holds only the snapshot enola 0.4.32
writes into its ``.enola``: facts.jsonl (a documented fact carries its
comment's first paragraph), docs.jsonl (the whole comment, under the fact's
id, which is the ``enola_id`` of its code node) and a receipt that vouches for
docs.jsonl. The chain reads no source file. It is entered two ways. The
memify path (``_sync_code_docs``) calls the tasks directly, in the pipeline's
order -- ``extract_code_docs``, ``cognify_code_doc_chunks`` in entity mode,
``store_code_docs``, ``add_code_doc_edges`` -- each handed the batch the one
before returned, and without a snapshot directory, so each repository's docs
are read from ``<CodeRepository.path>/.enola``. The route path goes through
``code_docs_tail``: ``_route`` runs the code graph tasks and then the tail per
repository label, handed the snapshot just loaded, as add_code_graph_docs
does, and a few tests call the tail directly, as a route or a staged code
file runs it. The route-shaped tests at the end run it under a pipeline
context per repository and then delete the repository through the
graph-provenance path ``forget(data_id=...)`` takes. Two tests call the tasks
directly out of the pipeline's order: a memify run that stores and sweeps
after a route run loaded a newer snapshot, and a batch linked without being
stored.
"""

import importlib
import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    make_chunk_source_ref_key,
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.infrastructure.databases.provenance.markers import mark_graph_provenance_if_empty
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.external_metadata import decode_external_metadata
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge as GraphEdge
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node
from cognee.modules.graph.methods.try_delete_data_by_graph_provenance import (
    try_delete_data_by_graph_provenance,
)
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.code_docs import (
    add_code_doc_edges,
    code_docs_tail,
    code_document_id,
    cognify_code_doc_chunks,
    extract_code_docs,
    store_code_docs,
)
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    code_graph_repo_labels,
    extract_code_graph,
    fact_node_id,
)
from cognee.tasks.summarization.models import TextSummary
from cognee.tests.utils.enola_snapshot import doc_line, enola_fact_id, write_enola_snapshot

extract_module = importlib.import_module("cognee.tasks.code_docs.extract_code_docs")
store_module = importlib.import_module("cognee.tasks.code_docs.store_code_docs")
cognify_module = importlib.import_module("cognee.tasks.code_docs.cognify_code_doc_chunks")
link_module = importlib.import_module("cognee.tasks.code_docs.add_code_doc_edges")
incremental_module = importlib.import_module(
    "cognee.modules.graph.methods.delete_chunks_incremental"
)
provenance_delete_module = importlib.import_module(
    "cognee.modules.graph.methods.try_delete_data_by_graph_provenance"
)
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
vector_module = importlib.import_module("cognee.infrastructure.databases.vector")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

SHOP = "shop"
BILLING = "billing"
CODE_TYPES = {"CodeRepository", "CodeModule", "CodeSymbol"}


def _module(repo, name):
    """A package's module fact: enola names and files it by the package directory."""
    return {
        "id": enola_fact_id(repo, "module", name, name),
        "kind": "module",
        "name": name,
        "file": name,
        "repo": repo,
        "props": {"language": "python"},
    }


def _symbol(repo, name, file, line, kind="function", calls=()):
    return {
        "id": enola_fact_id(repo, "symbol", name, file),
        "kind": "symbol",
        "name": name,
        "file": file,
        "line": line,
        "repo": repo,
        "props": {"language": "python", "symbol_kind": kind},
        "relations": [{"kind": "calls", "target": target} for target in calls],
    }


# Enola facts of the two repositories, each with its fact id: the code node's
# enola_id, the key a docs.jsonl line names. The shop is a root package with
# app.py, and an orders package with checkout.py.
SHOP_FACTS = [
    _module(SHOP, "."),
    _module(SHOP, "orders"),
    _symbol(SHOP, "app.main", "app.py", 4, calls=["app.serve"]),
    _symbol(SHOP, "app.serve", "app.py", 9),
    _symbol(SHOP, "orders/checkout.Order", "orders/checkout.py", 4, kind="class"),
    _symbol(SHOP, "orders/checkout.Order.total", "orders/checkout.py", 7, kind="method"),
    _symbol(
        SHOP,
        "orders/checkout.place_order",
        "orders/checkout.py",
        12,
        calls=["orders/checkout.Order.total"],
    ),
]
BILLING_FACTS = [_module(BILLING, "."), _symbol(BILLING, "invoices.issue", "invoices.py", 1)]

# The whole comment of each documented fact, by fact name: what docs.jsonl
# holds (facts.jsonl holds its first paragraph). enola documents a package,
# not a file, so the shop's two module docs are its __init__.py docstrings, of
# the root and of orders/; a module docstring of app.py or checkout.py gets no
# line. app.serve and billing's package are undocumented.
SHOP_DOCS = {
    ".": "The shop application.",
    "orders": "Order checkout.",
    "app.main": "Start the shop's HTTP server.",
    "orders/checkout.Order": "An order.",
    "orders/checkout.Order.total": "Sum of the lines.\n\nTaxes are not included.",
    "orders/checkout.place_order": "Place an order.",
}
BILLING_DOCS = {"invoices.issue": "Issue an invoice for an order."}
# main's docstring edited in its first paragraph: its chunk's text and id
# change, and since facts.jsonl changes with it, so does the snapshot id.
MAIN_EDITED = {**SHOP_DOCS, "app.main": "Start the shop's HTTP server and register its routes."}
# Order.total's docstring edited past its first paragraph: docs.jsonl and its
# hash in the receipt change; facts.jsonl and the snapshot id do not.
TOTAL_EDITED = {
    **SHOP_DOCS,
    "orders/checkout.Order.total": "Sum of the lines.\n\nTaxes are included.",
}

# Chunk header (its first line: file:line; for a module doc the package
# directory, "<repo>/" for the package at the repository root, which enola
# names ".") -> the code node its describes edge must reach.
EXPECTED_TARGETS = {
    "shop/": (SHOP, "module", "."),
    "app.py:4": (SHOP, "symbol", "app.main"),
    "orders": (SHOP, "module", "orders"),
    "orders/checkout.py:4": (SHOP, "symbol", "orders/checkout.Order"),
    "orders/checkout.py:7": (SHOP, "symbol", "orders/checkout.Order.total"),
    "orders/checkout.py:12": (SHOP, "symbol", "orders/checkout.place_order"),
    "invoices.py:1": (BILLING, "symbol", "invoices.issue"),
}

# What a docs run that derives, stores, links and sweeps nothing reports.
NOTHING = {"documents": 0, "chunks": 0, "new": 0, "edges": 0, "swept": 0}

# The entities the fake extractor derives: "shop" from every chunk, one more
# from main's docstring that changes with its text.
HTTP_SERVER, ROUTES, SHOP_ENTITY = (
    str(Entity.id_for(name)) for name in ("http server", "routes", "shop")
)
SERVICE, PROJECT = (str(EntityType.id_for(name)) for name in ("service", "project"))


def _snapshot_dir(repo_dir):
    """Where enola writes a repository's snapshot, and where the memify path reads it."""
    return repo_dir / ".enola"


def _write_snapshot(directory, facts, docs, snapshot_id):
    """Write the snapshot enola writes for ``facts``, ``docs`` ({fact name: whole comment}) documenting them.

    A documented fact carries its comment's first paragraph as its ``doc``
    prop in facts.jsonl; docs.jsonl holds the whole comment under the fact's
    id (a module's line has no ``line``), sorted by id as enola sorts it, and
    the receipt vouches for it with its hash. A fact the snapshot lacks has
    no line, whatever ``docs`` says.
    """
    documented, lines = [], []
    for fact in facts:
        doc = docs.get(fact["name"])
        if doc is not None:
            fact = {**fact, "props": {**fact["props"], "doc": doc.split("\n\n", 1)[0]}}
            lines.append(
                doc_line(
                    fact["repo"],
                    fact["kind"],
                    fact["name"],
                    fact["file"],
                    doc,
                    line=fact.get("line"),
                    fact_id=fact["id"],
                )
            )
        documented.append(fact)
    lines.sort(key=lambda line: (line["id"], line.get("line") or 0, line["doc"]))
    return write_enola_snapshot(directory, documented, docs=lines, snapshot_id=snapshot_id)


class _FakeVectors:
    """Where the sweep (and the provenance planner) drop embeddings.

    The chain here stores graph only (no embedding provider in unit tests), so
    the fake reports every collection present to make the deletes observable.
    """

    def __init__(self):
        self.deleted = []

    async def has_collection(self, name):
        return True

    async def delete_data_points(self, collection, ids):
        self.deleted.append((collection, list(ids)))

    async def remove_belongs_to_set_tags(self, tags):
        return None


@pytest.fixture
def vectors(monkeypatch):
    fake = _FakeVectors()
    monkeypatch.setattr(vector_module, "get_vector_engine_async", AsyncMock(return_value=fake))
    return fake


@pytest.fixture
def engine(tmp_path, monkeypatch, vectors):
    """A real Ladybug graph that every task of both chains reads and writes."""
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    engine = LadybugAdapter(db_path=str(tmp_path / "graph.lbug"))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    # No embedding engine in unit tests: count words, fixed budget.
    monkeypatch.setattr(extract_module, "_token_counter", lambda: lambda text: len(text.split()))
    monkeypatch.setattr(extract_module, "_default_chunk_budget", AsyncMock(return_value=256))
    return engine


@pytest.fixture
def writes(monkeypatch):
    """The ids store_code_docs hands add_data_points, call by call."""
    real_add_data_points = store_module.add_data_points
    written = []

    async def spy(data_points, *args, **kwargs):
        written.append({point.id for point in data_points})
        return await real_add_data_points(data_points, *args, **kwargs)

    monkeypatch.setattr(store_module, "add_data_points", spy)
    return written


@pytest.fixture
def repos(tmp_path):
    """The two repositories' directories; once indexed, each holds its snapshot in .enola."""
    return SimpleNamespace(shop=tmp_path / SHOP, billing=tmp_path / BILLING)


def _entity(name):
    kind = "project" if name == "shop" else "service"
    return Entity(
        name=name,
        description=f"The {name}.",
        is_a=EntityType(name=kind, description=f"A {kind}."),
    )


@pytest.fixture
def extractor(monkeypatch):
    """extract_graph_and_summarize stand-in: deterministic entities per chunk, a summary each.

    Every chunk contains "shop"; main's docstring adds "http server", its
    edited text "routes" instead. Records the headers of each call's chunks.
    """
    calls = []

    async def extract(data_chunks, graph_model, ctx=None):
        calls.append([chunk.text.split("\n", 1)[0] for chunk in data_chunks])
        summaries = []
        for chunk in data_chunks:
            names = ["shop"]
            if "register its routes" in chunk.text:
                names.append("routes")
            elif "HTTP server" in chunk.text:
                names.append("http server")
            chunk.contains = [(Edge(relationship_type="contains"), _entity(name)) for name in names]
            summaries.append(TextSummary(text=f"Summary of {chunk.text}", made_from=chunk))
        return summaries

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", extract)
    return calls


async def _sync_code_graph(repo_dir, ctx=None):
    """The enola code graph tasks over <repo_dir>/.enola, in get_code_graph_tasks order.

    Returns what they pass on: the mapped data points, none when the graph
    already holds that snapshot (the load is skipped).
    """
    snapshot = _snapshot_dir(repo_dir)
    data_points = await extract_code_graph(repo_path=str(repo_dir), snapshot_dir=snapshot)
    state = await add_code_graph_data_points(data_points, ctx=ctx)
    return await add_code_graph_edges(
        state, repo_path=str(repo_dir), snapshot_dir=snapshot, ctx=ctx
    )


async def _index(repo_dir, facts, docs, snapshot_id, ctx=None):
    """Index a repository: enola writes its snapshot into <repo_dir>/.enola, the code graph loads it."""
    _write_snapshot(_snapshot_dir(repo_dir), facts, docs, snapshot_id)
    return await _sync_code_graph(repo_dir, ctx=ctx)


async def _sync_both_code_graphs(repos):
    await _index(repos.shop, SHOP_FACTS, SHOP_DOCS, "s1")
    await _index(repos.billing, BILLING_FACTS, BILLING_DOCS, "b1")


async def _sync_code_docs(repo=None, ctx=None, extract_entities=False):
    """The memify path: extract, (cognify,) store what is new (graph only), link and sweep.

    No snapshot directory, as the memify pipeline runs the tasks: each
    repository's docs are read from <CodeRepository.path>/.enola. Each task
    is handed what the one before returned, as the pipeline hands it on, so
    the extraction's record of what it read (CodeDocsBatch) reaches the sweep.
    """
    data_points = await extract_code_docs(repo=repo, ctx=ctx)
    if extract_entities:
        data_points = await cognify_code_doc_chunks(data_points, ctx=ctx)
    stored = await store_code_docs(data_points, ctx=ctx, graph_only=True)
    assert stored is data_points
    await add_code_doc_edges(data_points, ctx=ctx, repo=repo)
    return data_points


async def _route(repo_dir, ctx=None, extract_entities=False):
    """The route path over <repo_dir>/.enola: the code graph tasks, then the docs tail.

    One code_docs_tail per repository label the run covered, handed the
    snapshot the run loaded: what add_code_graph_docs does (graph only here,
    as unit tests have no embedding engine). Returns {label: counts}.
    """
    snapshot = _snapshot_dir(repo_dir)
    state = await _sync_code_graph(repo_dir, ctx=ctx)
    counts = {}
    for label in code_graph_repo_labels(state, repo_dir, snapshot):
        counts[label] = await code_docs_tail(
            label,
            ctx=ctx,
            extract_entities=extract_entities,
            graph_only=True,
            snapshot_dir=snapshot,
        )
    return counts


async def _graph(engine):
    """The stored graph: {node id: properties} and (source, target, name, properties) edges."""
    nodes, edges = await engine.get_graph_data()
    return dict(nodes), edges


async def _stored(engine):
    """The stored graph with every node and edge property, in a shape two reads compare by."""
    nodes, edges = await engine.get_graph_data()
    return dict(nodes), sorted(
        (source, target, name, json.dumps(properties, sort_keys=True, default=str))
        for source, target, name, properties in edges
    )


def _code_graph(nodes, edges):
    code_nodes = {node_id: props for node_id, props in nodes.items() if props["type"] in CODE_TYPES}
    code_edges = sorted(
        (source, target, name) for source, target, name, _ in edges if source in code_nodes
    )
    return code_nodes, code_edges


def _edge_keys(edges):
    return {(source, target, name) for source, target, name, _ in edges}


def _describes(edges):
    return {(source, target) for source, target, name, _ in edges if name == "describes"}


def _chunks_by_header(data_points):
    """The run's chunks by their first line, unwrapped from entity mode's summaries."""
    chunks = (
        point if isinstance(point, DocumentChunk) else getattr(point, "made_from", None)
        for point in data_points
    )
    return {chunk.text.split("\n", 1)[0]: chunk for chunk in chunks if chunk is not None}


def _summary_of(edges, chunk_id):
    return next(
        source for source, target, name, _ in edges if name == "made_from" and target == chunk_id
    )


def _stored_chunk(nodes, header):
    """The id of the stored DocumentChunk whose text starts with ``header``."""
    return next(
        node_id
        for node_id, props in nodes.items()
        if props["type"] == "DocumentChunk"
        and (props.get("text") or "").split("\n", 1)[0] == header
    )


def _repository_nodes(nodes, repo):
    """One repository's stored (code nodes, CodeDocuments, chunks), by id."""
    code = {
        node_id
        for node_id, props in nodes.items()
        if props["type"] in CODE_TYPES and repo in (props.get("repo"), props.get("name"))
    }
    documents = {
        node_id
        for node_id, props in nodes.items()
        if props["type"] == "CodeDocument" and props.get("repo") == repo
    }
    chunks = {
        node_id
        for node_id, props in nodes.items()
        if props["type"] == "DocumentChunk" and props.get("document_id") in documents
    }
    return code, documents, chunks


def _deleted_by_collection(vectors):
    collections = {}
    for collection, ids in vectors.deleted:
        collections.setdefault(collection, []).extend(ids)
    return collections


def _route_ctx(dataset):
    """A code route's context: its data item is the repository's own Data row."""
    return PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        dataset=dataset,
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )


def _unified(engine, vectors, monkeypatch):
    """The unified engine the provenance planner and forget's delete path read through.

    The relational rollback ledger the routes also write is not under test.
    """
    unified = UnifiedStoreEngine(
        graph_engine=engine,
        vector_engine=vectors,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
    monkeypatch.setattr(incremental_module, "get_unified_engine", AsyncMock(return_value=unified))
    monkeypatch.setattr(
        provenance_delete_module, "get_unified_engine", AsyncMock(return_value=unified)
    )
    monkeypatch.setattr("cognee.modules.graph.methods.upsert_edges", AsyncMock())
    return unified


async def _forget(dataset, ctx):
    """What forget(data_id=...) does on a provenance-marked graph: delete the item's refs."""
    result = await try_delete_data_by_graph_provenance(dataset.id, ctx.data_item.id)
    assert result is not None, "the graph-provenance delete path did not handle the item"
    return {str(node_id) for node_id in result.deleted_node_ids}


def _data_ids(delete_data, node_id):
    """The data items a stored node's source refs name."""
    return {str(parse_source_ref_key(ref).data_id) for ref in delete_data[node_id].source_ref_keys}


@pytest.mark.asyncio
async def test_stores_documents_and_chunks_linked_to_their_code_nodes(engine, repos):
    await _sync_both_code_graphs(repos)
    code_graph_before = _code_graph(*await _graph(engine))

    data_points = await _sync_code_docs()

    nodes, edges = await _graph(engine)
    documents = [point for point in data_points if isinstance(point, CodeDocument)]
    by_header = _chunks_by_header(data_points)
    assert len(documents) + len(by_header) == len(data_points)

    # One CodeDocument per documented file -- for a module doc, its package
    # directory -- repositories by name, files by path; a module doc is not a
    # documented symbol. The package at the repository root keeps enola's "."
    # as its file, and so its id, but is named after its repository.
    assert [
        (document.repo, document.file_path, document.name, document.symbol_count)
        for document in documents
    ] == [
        (BILLING, "invoices.py", "invoices.py", 1),
        (SHOP, ".", "shop/", 0),
        (SHOP, "app.py", "app.py", 1),
        (SHOP, "orders", "orders", 0),
        (SHOP, "orders/checkout.py", "orders/checkout.py", 3),
    ]
    for document in documents:
        assert document.id == code_document_id(document.repo, document.file_path)
        stored = nodes[str(document.id)]
        assert (
            stored["type"],
            stored["name"],
            stored["repo"],
            stored["file_path"],
            stored["raw_data_location"],
        ) == ("CodeDocument", document.name, document.repo, document.file_path, document.file_path)

    # Plain DocumentChunks: their type names the collection every retriever
    # searches; each is stored with its text under its content-derived id and
    # hangs off its document.
    assert set(by_header) == set(EXPECTED_TARGETS)
    for chunk in by_header.values():
        assert type(chunk) is DocumentChunk
        assert type(chunk).vector_collection("text") == "DocumentChunk_text"
        stored = nodes[str(chunk.id)]
        assert (stored["type"], stored["text"]) == ("DocumentChunk", chunk.text)
        assert (str(chunk.id), chunk.document_id, "is_part_of") in _edge_keys(edges)
    assert by_header["shop/"].text == "shop/\nshop (module)\n\nThe shop application."
    assert by_header["shop/"].document_name == "shop/"
    assert by_header["orders"].text == "orders\norders (module)\n\nOrder checkout."
    assert by_header["app.py:4"].text == (
        "app.py:4\napp.main (function)\n\nStart the shop's HTTP server."
    )
    # The whole comment, as docs.jsonl holds it; the code node keeps the first
    # paragraph facts.jsonl carries.
    assert by_header["orders/checkout.py:7"].text == (
        "orders/checkout.py:7\n"
        "orders/checkout.Order.total (method)\n"
        "\n"
        "Sum of the lines.\n"
        "\n"
        "Taxes are not included."
    )
    total_id = str(fact_node_id(SHOP, "symbol", "orders/checkout.Order.total"))
    assert nodes[total_id]["doc"] == "Sum of the lines."
    # Within a document the docstrings go by line, whatever order docs.jsonl has.
    assert {header: chunk.chunk_index for header, chunk in by_header.items()} == {
        "shop/": 0,
        "app.py:4": 0,
        "orders": 0,
        "orders/checkout.py:4": 0,
        "orders/checkout.py:7": 1,
        "orders/checkout.py:12": 2,
        "invoices.py:1": 0,
    }

    # Each chunk was joined to its code node by the docs line's id, which is the
    # stored node's enola_id, and names that node.
    fact_ids = {
        (fact["repo"], fact["kind"], fact["name"]): fact["id"]
        for fact in SHOP_FACTS + BILLING_FACTS
    }
    for header, target in EXPECTED_TARGETS.items():
        code_doc = decode_external_metadata(by_header[header].external_metadata)["code_doc"]
        node_id = str(fact_node_id(*target))
        assert code_doc["enola_id"] == nodes[node_id]["enola_id"] == fact_ids[target]
        assert code_doc["module_id" if target[1] == "module" else "symbol_id"] == node_id

    # Exactly one describes edge per chunk, to the code node with that id.
    describes = [(s, t, props) for s, t, name, props in edges if name == "describes"]
    targets = {source: target for source, target, _ in describes}
    assert len(describes) == len(targets) == len(by_header)
    assert {header: targets[str(chunk.id)] for header, chunk in by_header.items()} == {
        header: str(fact_node_id(*target)) for header, target in EXPECTED_TARGETS.items()
    }
    main_edge = next(props for s, _t, props in describes if s == str(by_header["app.py:4"].id))
    assert (main_edge["relationship_name"], main_edge["chunk_index"]) == ("describes", 0)
    assert "doc_source" not in main_edge

    # The code graph's own nodes (hashes included) and edges are untouched.
    assert _code_graph(nodes, edges) == code_graph_before


@pytest.mark.asyncio
async def test_resolve_edges_to_text_renders_the_chunk_from_its_file_line_header(engine, repos):
    await _sync_both_code_graphs(repos)
    data_points = await _sync_code_docs()
    nodes, edges = await _graph(engine)

    chunk = _chunks_by_header(data_points)["app.py:4"]
    symbol_id = str(fact_node_id(SHOP, "symbol", "app.main"))
    edge_properties = next(
        props
        for s, t, name, props in edges
        if (s, t, name) == (str(chunk.id), symbol_id, "describes")
    )

    text = await resolve_edges_to_text(
        [
            GraphEdge(
                Node(str(chunk.id), dict(nodes[str(chunk.id)])),
                Node(symbol_id, dict(nodes[symbol_id])),
                attributes=dict(edge_properties),
            )
        ]
    )

    assert (
        "__node_content_start__\n"
        "app.py:4\napp.main (function)\n\nStart the shop's HTTP server.\n"
        "__node_content_end__"
    ) in text
    assert "--[describes]--> app.main" in text


@pytest.mark.asyncio
async def test_rerun_change_and_removal_touch_exactly_their_chunks(
    engine, repos, vectors, writes, monkeypatch
):
    """The memify path over a repository indexed again after each edit: enola rewrites
    its .enola and the code graph tasks load it before the docs run."""
    await _sync_both_code_graphs(repos)
    first = await _sync_code_docs()
    nodes_1, edges_1 = await _graph(engine)
    assert vectors.deleted == []
    # The first run stores every point, in one write.
    assert writes == [{point.id for point in first}]

    written, deleted = [], []
    real_add_edges, real_delete_nodes = engine.add_edges, engine.delete_nodes

    async def spy_add_edges(edges, *args, **kwargs):
        written.extend((str(e[0]), str(e[1]), e[2]) for e in edges if e[2] == "describes")
        await real_add_edges(edges, *args, **kwargs)

    async def spy_delete_nodes(node_ids):
        deleted.extend(node_ids)
        await real_delete_nodes(node_ids)

    monkeypatch.setattr(engine, "add_edges", spy_add_edges)
    monkeypatch.setattr(engine, "delete_nodes", spy_delete_nodes)

    # The snapshot unchanged: the same ids, nothing stored, no new edge,
    # nothing swept.
    writes.clear()
    second = await _sync_code_docs()
    assert [point.id for point in second] == [point.id for point in first]
    assert (writes, written, deleted, vectors.deleted) == ([], [], [], [])
    nodes_2, edges_2 = await _graph(engine)
    assert set(nodes_2) == set(nodes_1)
    assert _edge_keys(edges_2) == _edge_keys(edges_1)

    # A changed docstring: enola rewrites main's docs.jsonl line and the
    # receipt's hash of it (and, the first paragraph being in facts.jsonl too,
    # the snapshot id); the code graph loads that snapshot and deletes nothing.
    # The docs run replaces main's chunk (new id, new text, same position) and
    # sweeps the old one, every other chunk kept; only the new chunk and its
    # document are stored. The run covers the shop only, so billing is not in
    # the batch and must stay untouched.
    old_main = _chunks_by_header(second)["app.py:4"]
    main_id = str(fact_node_id(SHOP, "symbol", "app.main"))
    await _index(repos.shop, SHOP_FACTS, MAIN_EDITED, "s2")
    assert deleted == []

    third = await _sync_code_docs(repo=SHOP)

    new_main = _chunks_by_header(third)["app.py:4"]
    assert new_main.id != old_main.id
    assert new_main.text.endswith("Start the shop's HTTP server and register its routes.")
    assert new_main.chunk_index == old_main.chunk_index == 0
    assert writes == [{new_main.id, code_document_id(SHOP, "app.py")}]
    assert written == [(str(new_main.id), main_id, "describes")]
    assert deleted == [str(old_main.id)]
    # The old chunk leaves DocumentChunk_text too, so CHUNKS stops returning it.
    assert vectors.deleted == [("DocumentChunk_text", [str(old_main.id)])]
    nodes_3, edges_3 = await _graph(engine)
    assert set(nodes_3) == (set(nodes_2) - {str(old_main.id)}) | {str(new_main.id)}
    assert nodes_3[str(new_main.id)]["text"] == new_main.text
    assert _describes(edges_3) == (_describes(edges_2) - {(str(old_main.id), main_id)}) | {
        (str(new_main.id), main_id)
    }

    # A symbol deleted from the code: enola writes neither its fact nor its
    # docs.jsonl line. The code graph's own sweep drops its CodeSymbol (and
    # with it the describes edge), the docs run sweeps its chunk and rewrites
    # its document (one symbol fewer), nothing else.
    writes.clear()
    written.clear()
    deleted.clear()
    vectors.deleted.clear()
    facts = [fact for fact in SHOP_FACTS if fact["name"] != "orders/checkout.place_order"]
    await _index(repos.shop, facts, MAIN_EDITED, "s3")
    docs_lines = (_snapshot_dir(repos.shop) / "docs.jsonl").read_text(encoding="utf-8")
    assert "orders/checkout.place_order" not in docs_lines
    place_order_id = str(fact_node_id(SHOP, "symbol", "orders/checkout.place_order"))
    assert deleted == [place_order_id]
    deleted.clear()
    old_place_order = _chunks_by_header(third)["orders/checkout.py:12"]

    fourth = await _sync_code_docs()

    assert writes == [{code_document_id(SHOP, "orders/checkout.py")}]
    assert (written, deleted) == ([], [str(old_place_order.id)])
    assert vectors.deleted == [("DocumentChunk_text", [str(old_place_order.id)])]
    assert {point.id for point in fourth} == (
        {point.id for point in second} - {old_main.id, old_place_order.id}
    ) | {new_main.id}
    nodes_4, edges_4 = await _graph(engine)
    assert set(nodes_4) == set(nodes_3) - {str(old_place_order.id), place_order_id}
    assert nodes_4[str(code_document_id(SHOP, "orders/checkout.py"))]["symbol_count"] == 2
    assert _describes(edges_4) == _describes(edges_3) - {(str(old_place_order.id), place_order_id)}
    # Billing, never in a changed run, is exactly as the first run left it.
    invoice = _chunks_by_header(first)["invoices.py:1"]
    assert nodes_4[str(invoice.id)]["text"] == invoice.text
    assert nodes_4[str(code_document_id(BILLING, "invoices.py"))]["type"] == "CodeDocument"
    assert (str(invoice.id), str(fact_node_id(BILLING, "symbol", "invoices.issue"))) in _describes(
        edges_4
    )


@pytest.mark.asyncio
async def test_doc_edit_past_the_first_paragraph_replaces_exactly_its_chunk(
    engine, repos, vectors, writes
):
    """enola keeps a comment's first paragraph in facts.jsonl and the whole comment
    in docs.jsonl, so an edit past the first paragraph rewrites docs.jsonl and its
    hash in the receipt and leaves facts.jsonl and the snapshot id as they were.
    On the route path the code graph tasks then skip the load (the graph holds
    that snapshot), and the docs tail, which runs anyway, reads the new docs.jsonl
    (vouched for, of the graph's snapshot) and replaces exactly that chunk."""
    snapshot = _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, SHOP_DOCS, "s1")
    assert await _route(repos.shop) == {
        SHOP: {"documents": 4, "chunks": 6, "new": 6, "edges": 6, "swept": 0}
    }
    nodes_1, edges_1 = await _graph(engine)
    old_total = _stored_chunk(nodes_1, "orders/checkout.py:7")
    facts_1 = (snapshot / "facts.jsonl").read_bytes()
    receipt_1 = json.loads((snapshot / "receipt.json").read_text(encoding="utf-8"))

    _write_snapshot(snapshot, SHOP_FACTS, TOTAL_EDITED, "s1")

    receipt_2 = json.loads((snapshot / "receipt.json").read_text(encoding="utf-8"))
    assert (snapshot / "facts.jsonl").read_bytes() == facts_1
    assert receipt_2["snapshot_id"] == receipt_1["snapshot_id"]
    assert receipt_2["output_hashes"]["docs.jsonl"] != receipt_1["output_hashes"]["docs.jsonl"]
    writes.clear()

    assert await _route(repos.shop) == {
        SHOP: {"documents": 4, "chunks": 6, "new": 1, "edges": 1, "swept": 1}
    }

    nodes_2, edges_2 = await _graph(engine)
    # The load was skipped: the code graph, its repository node's snapshot
    # stamp included, is exactly as it was.
    assert _code_graph(nodes_2, edges_2) == _code_graph(nodes_1, edges_1)
    new_total = _stored_chunk(nodes_2, "orders/checkout.py:7")
    assert nodes_2[new_total]["text"] == (
        "orders/checkout.py:7\n"
        "orders/checkout.Order.total (method)\n"
        "\n"
        "Sum of the lines.\n"
        "\n"
        "Taxes are included."
    )
    assert nodes_2[new_total]["chunk_index"] == nodes_1[old_total]["chunk_index"] == 1
    assert set(nodes_2) == (set(nodes_1) - {old_total}) | {new_total}
    total_id = str(fact_node_id(SHOP, "symbol", "orders/checkout.Order.total"))
    assert _describes(edges_2) == (_describes(edges_1) - {(old_total, total_id)}) | {
        (new_total, total_id)
    }
    assert writes == [{UUID(new_total), code_document_id(SHOP, "orders/checkout.py")}]
    assert vectors.deleted == [("DocumentChunk_text", [old_total])]


@pytest.mark.asyncio
async def test_staged_rerun_of_an_unchanged_file_keeps_its_docstrings(
    engine, vectors, writes, tmp_path
):
    """The per-file CODE route stages the file in a new temporary directory every
    run and enola writes its snapshot there; the snapshot id is content-derived,
    so the second run of an unchanged item skips the code graph load
    (extract_code_graph returns [], the repository node is not rewritten) while
    the stored CodeRepository.path -- the first run's staging directory, its
    snapshot with it -- is gone from disk. The tail is handed the snapshot in the
    current staging directory (what add_code_graph_docs passes, for the labels
    code_graph_repo_labels reads off it), so the second run derives the same
    chunk from its docs.jsonl and stores, embeds and sweeps nothing."""
    label = f"app.py_{uuid4()}"
    facts = [
        _module(label, "."),
        _symbol(label, "app.main", "app.py", 4, calls=["app.serve"]),
        _symbol(label, "app.serve", "app.py", 9),
    ]
    # A lone staged file is in no package, so it has no module doc.
    docs = {"app.main": SHOP_DOCS["app.main"]}
    runs = []
    for staging in ("first", "second"):
        staging_root = tmp_path / staging
        repo_dir = staging_root / label
        snapshot = _write_snapshot(_snapshot_dir(repo_dir), facts, docs, "sha256:same-content")
        data_points = await extract_code_graph(repo_path=str(repo_dir), snapshot_dir=snapshot)
        state = await add_code_graph_data_points(data_points, graph_only=True)
        await add_code_graph_edges(state, repo_path=str(repo_dir), snapshot_dir=snapshot)
        assert code_graph_repo_labels(state, repo_dir, snapshot) == [label]
        counts = await code_docs_tail(label, graph_only=True, snapshot_dir=snapshot)
        nodes, edges = await _graph(engine)
        runs.append((len(data_points), counts, nodes, edges))
        # The route's TemporaryDirectory is gone once its block exits.
        shutil.rmtree(staging_root)

    (mapped_1, counts_1, nodes_1, edges_1), (mapped_2, counts_2, nodes_2, edges_2) = runs
    repository = str(fact_node_id(label, "repository", label))
    # The second run skipped the unchanged snapshot and left the first run's
    # path on the repository node, a directory that no longer exists.
    assert mapped_1 > 0 and mapped_2 == 0
    assert nodes_2[repository]["path"] == nodes_1[repository]["path"]
    assert not os.path.isdir(nodes_2[repository]["path"])
    # Without the route's snapshot (the memify path reads the stored path's)
    # nothing could be read.
    unread = await extract_code_docs(repo=label)
    assert unread == [] and unread.read_snapshots == {}

    assert counts_1 == {"documents": 1, "chunks": 1, "new": 1, "edges": 1, "swept": 0}
    assert counts_2 == {"documents": 1, "chunks": 1, "new": 0, "edges": 0, "swept": 0}
    assert set(nodes_2) == set(nodes_1) and _edge_keys(edges_2) == _edge_keys(edges_1)
    chunks = {props["text"] for props in nodes_2.values() if props["type"] == "DocumentChunk"}
    assert chunks == {"app.py:4\napp.main (function)\n\nStart the shop's HTTP server."}
    assert len(writes) == 1 and vectors.deleted == []


@pytest.mark.asyncio
async def test_run_without_the_snapshot_docs_keeps_the_stored_docs(engine, repos, vectors, writes):
    """A repository whose snapshot docs are not available is left out of the run
    and of the sweep, so its stored docs are never traded for nothing, nor for
    the docs of other facts: the clone moved away (lost on a restart, on another
    host), a docs.jsonl its receipt does not vouch for (bytes this enola run did
    not write), a snapshot other than the one the code graph holds (enola ran
    again, the graph was not reloaded). Each time every stored node and edge stays
    as it was, nothing is written and no vector is deleted -- on the memify path,
    unfiltered and filtered, and on the route path, the tail handed that snapshot
    directory -- while billing, still readable, is derived as usual. The run after
    the snapshot is back changes nothing either: the stored docs are the ones it
    derives."""
    await _sync_both_code_graphs(repos)
    first = await _sync_code_docs()
    stored = await _stored(engine)
    assert len(_chunks_by_header(first)) == len(EXPECTED_TARGETS) and len(writes) == 1

    snapshot = _snapshot_dir(repos.shop)
    original = shutil.copytree(snapshot, repos.shop.parent / "shop-snapshot")
    # What a later enola run writes once main's docstring is edited.
    later = _write_snapshot(repos.shop.parent / "later", SHOP_FACTS, MAIN_EDITED, "s2")

    def move_the_clone_away():
        shutil.move(repos.shop, repos.shop.parent / "shop-moved")

    def replace_the_docs_the_receipt_vouches_for():
        # The receipt still hashes the docs.jsonl of snapshot s1.
        shutil.copyfile(later / "docs.jsonl", snapshot / "docs.jsonl")

    def write_another_snapshot_over_it():
        # Its receipt vouches for its docs, but the code graph holds s1.
        shutil.rmtree(snapshot)
        shutil.copytree(later, snapshot)

    for make_unavailable in (
        move_the_clone_away,
        replace_the_docs_the_receipt_vouches_for,
        write_another_snapshot_over_it,
    ):
        case = make_unavailable.__name__
        make_unavailable()
        for repo in (None, SHOP):
            run = await _sync_code_docs(repo=repo)
            documents = {point.repo for point in run if isinstance(point, CodeDocument)}
            assert documents == ({BILLING} if repo is None else set()), case
            # The run's record leaves the shop out, so its sweep keeps the shop's docs.
            assert run.read_snapshots == ({BILLING: "b1"} if repo is None else {}), case
        assert await code_docs_tail(SHOP, graph_only=True, snapshot_dir=snapshot) == NOTHING, case
        assert await _stored(engine) == stored, case
        assert (len(writes), vectors.deleted) == (1, []), case

        # The snapshot back: the run derives the stored docs and changes nothing.
        shutil.rmtree(snapshot, ignore_errors=True)
        shutil.copytree(original, snapshot)
        again = await _sync_code_docs()
        assert [point.id for point in again] == [point.id for point in first], case
        assert await _stored(engine) == stored, case
        assert (len(writes), vectors.deleted) == (1, []), case


@pytest.mark.asyncio
async def test_run_overtaken_by_a_newer_snapshot_keeps_that_snapshots_docs(
    engine, repos, vectors, writes
):
    """A memify run reads the shop's docs of s1; before it stores and sweeps, a
    route run loads s2 (main's docstring edited), stores s2's docs and sweeps s1's
    main chunk. The memify run's batch is s1's: its store writes s1's main chunk
    back (that id is no longer stored), and a sweep against it would delete s2's
    main chunk, leaving the docs of a snapshot the code graph no longer holds.
    Its sweep keeps the shop instead -- read, every doc of it stored, but its
    graph moved on from the snapshot read -- and the next run over s2 sweeps what
    it wrote back, leaving exactly s2's docs."""
    _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, SHOP_DOCS, "s1")
    await _route(repos.shop)
    old_main = _stored_chunk((await _graph(engine))[0], "app.py:4")
    main_id = str(fact_node_id(SHOP, "symbol", "app.main"))

    # The memify run reads s1's docs from the stored path...
    overtaken = await extract_code_docs()
    assert overtaken.read_snapshots == {SHOP: "s1"}

    # ...a route run loads s2, stores its docs and sweeps s1's main chunk...
    _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, MAIN_EDITED, "s2")
    assert await _route(repos.shop) == {
        SHOP: {"documents": 4, "chunks": 6, "new": 1, "edges": 1, "swept": 1}
    }
    nodes_2, edges_2 = await _graph(engine)
    new_main = _stored_chunk(nodes_2, "app.py:4")
    assert nodes_2[new_main]["text"].endswith("and register its routes.")
    assert old_main not in nodes_2
    writes.clear()
    vectors.deleted.clear()

    # ...and only then does the memify run store, and then link and sweep. Its
    # docs are all stored and the shop was read: what keeps the shop out of its
    # sweep is the snapshot the shop's graph holds by now.
    await store_code_docs(overtaken, graph_only=True)
    assert writes == [{UUID(old_main), code_document_id(SHOP, "app.py")}]
    nodes, _edges = await _graph(engine)
    assert {str(point.id) for point in overtaken} <= set(nodes)
    assert nodes[str(fact_node_id(SHOP, "repository", SHOP))]["last_snapshot_id"] == "s2"

    await add_code_doc_edges(overtaken)

    # Linked, not swept: s2's docs all stay, next to the chunk written back.
    nodes, edges = await _graph(engine)
    assert set(nodes) == set(nodes_2) | {old_main}
    assert _describes(edges) == _describes(edges_2) | {(old_main, main_id)}
    assert vectors.deleted == []

    # The next run over s2 (the code graph load skipped, the tail run anyway).
    assert await _route(repos.shop) == {
        SHOP: {"documents": 4, "chunks": 6, "new": 0, "edges": 0, "swept": 1}
    }
    nodes, edges = await _graph(engine)
    assert set(nodes) == set(nodes_2) and _edge_keys(edges) == _edge_keys(edges_2)
    assert vectors.deleted == [("DocumentChunk_text", [old_main])]


@pytest.mark.asyncio
async def test_batch_linked_without_being_stored_deletes_nothing(engine, repos, vectors):
    """A task list that hands extract_code_docs' batch straight to add_code_doc_edges,
    store_code_docs left out, after main's docstring changed: the batch's new chunk
    is not in the graph, so a sweep against the batch would delete main's stored
    chunk and put nothing in its place. The sweep keeps the shop -- read, its graph
    still at the snapshot read, but its docs from this batch not stored -- so the
    stored graph stays exactly as it was and no vector is deleted. The chain with
    its store then replaces the chunk as usual."""
    await _sync_both_code_graphs(repos)
    old_main = _chunks_by_header(await _sync_code_docs())["app.py:4"]
    await _index(repos.shop, SHOP_FACTS, MAIN_EDITED, "s2")
    stored = await _stored(engine)

    unstored = await extract_code_docs()
    new_main = _chunks_by_header(unstored)["app.py:4"]
    assert unstored.read_snapshots == {BILLING: "b1", SHOP: "s2"}
    assert str(new_main.id) not in stored[0]

    assert await add_code_doc_edges(unstored) is unstored

    assert await _stored(engine) == stored
    assert vectors.deleted == []

    await _sync_code_docs()
    nodes, _edges = await _graph(engine)
    assert str(new_main.id) in nodes and str(old_main.id) not in nodes
    assert vectors.deleted == [("DocumentChunk_text", [str(old_main.id)])]


@pytest.mark.asyncio
async def test_forgotten_repository_is_swept_by_an_unfiltered_run_only(
    engine, repos, vectors, monkeypatch
):
    await _sync_both_code_graphs(repos)
    first = await _sync_code_docs()
    invoice = _chunks_by_header(first)["invoices.py:1"]
    billing_document_id = str(code_document_id(BILLING, "invoices.py"))
    billing_docs = {billing_document_id, str(invoice.id)}

    # Billing is forgotten: its code graph goes (what forget / a dataset
    # delete does), its docs remain behind, and so does its snapshot on disk.
    nodes, _edges = await _graph(engine)
    billing_code = [
        node_id
        for node_id, props in nodes.items()
        if props["type"] in CODE_TYPES and BILLING in (props.get("repo"), props.get("name"))
    ]
    assert len(billing_code) == 3
    await engine.delete_nodes(billing_code)

    # A run scoped to the shop leaves billing's docs alone...
    await _sync_code_docs(repo=SHOP)
    nodes, _edges = await _graph(engine)
    assert billing_docs <= set(nodes)
    assert vectors.deleted == []

    # ...an unfiltered run covers every repository that has docs: billing's go,
    # from their collections and then from the graph, the document last each
    # time (a sweep cut short leaves nodes the next one finds again, never a
    # vector without its node), and the shop's stay.
    deletes = []
    real_delete_nodes, real_delete_data_points = engine.delete_nodes, vectors.delete_data_points

    async def delete_nodes(node_ids):
        deletes.append(("graph", list(node_ids)))
        await real_delete_nodes(node_ids)

    async def delete_data_points(collection, ids):
        deletes.append((collection, list(ids)))
        await real_delete_data_points(collection, ids)

    monkeypatch.setattr(engine, "delete_nodes", delete_nodes)
    monkeypatch.setattr(vectors, "delete_data_points", delete_data_points)
    await _sync_code_docs()
    nodes, edges = await _graph(engine)
    assert not billing_docs & set(nodes)
    assert deletes == [
        ("DocumentChunk_text", [str(invoice.id)]),
        ("CodeDocument_name", [billing_document_id]),
        ("graph", [str(invoice.id), billing_document_id]),
    ]
    assert {str(point.id) for point in first} - billing_docs <= set(nodes)
    assert len(_describes(edges)) == len(EXPECTED_TARGETS) - 1


@pytest.mark.asyncio
async def test_entity_mode_sweeps_the_entities_only_the_old_chunk_held(
    engine, repos, vectors, writes, extractor
):
    await _sync_both_code_graphs(repos)

    first = await _sync_code_docs(extract_entities=True)

    nodes, edges = await _graph(engine)
    # Every chunk went to the LLM once, in batch order, and has a summary; the
    # entities and their types are stored once each.
    assert extractor == [list(_chunks_by_header(first))]
    assert set(extractor[0]) == set(EXPECTED_TARGETS)
    chunk_ids = {str(chunk.id) for chunk in _chunks_by_header(first).values()}
    assert {target for _s, target, name, _ in edges if name == "made_from"} == chunk_ids
    assert {HTTP_SERVER, SHOP_ENTITY, SERVICE, PROJECT} <= set(nodes) and ROUTES not in nodes
    assert len(writes) == 1

    # Unchanged: nothing reaches the LLM, nothing is stored or swept.
    await _sync_code_docs(extract_entities=True)
    assert len(extractor) == 1 and len(writes) == 1
    assert vectors.deleted == []
    assert set((await _graph(engine))[0]) == set(nodes)

    # main's docstring changes and the shop is indexed again: only the new
    # chunk goes to the LLM. The old chunk, its summary and the entity only it
    # held go; the entity and the types shared with other chunks stay.
    old_main = _chunks_by_header(first)["app.py:4"]
    old_summary = _summary_of(edges, str(old_main.id))
    await _index(repos.shop, SHOP_FACTS, MAIN_EDITED, "s2")

    third = await _sync_code_docs(repo=SHOP, extract_entities=True)

    assert extractor[-1] == ["app.py:4"]
    new_main = _chunks_by_header(third)["app.py:4"]
    nodes, edges = await _graph(engine)
    assert {str(old_main.id), old_summary, HTTP_SERVER}.isdisjoint(nodes)
    assert {str(new_main.id), ROUTES, SHOP_ENTITY, SERVICE, PROJECT} <= set(nodes)
    assert _summary_of(edges, str(new_main.id)) in nodes
    assert (str(new_main.id), ROUTES, "contains") in _edge_keys(edges)
    assert vectors.deleted == [
        ("DocumentChunk_text", [str(old_main.id)]),
        ("TextSummary_text", [old_summary]),
        ("Entity_name", [HTTP_SERVER]),
    ]


@pytest.mark.asyncio
async def test_owned_chunks_are_retired_through_the_provenance_planner(
    engine, repos, vectors, extractor, monkeypatch
):
    # The route path's storage: the graph stores its provenance (a fresh graph
    # is marked on the first write) and the chain runs under a context with a
    # persisted data item, so add_data_points stamps every chunk, summary and
    # entity with the chunk's own source ref.
    assert await mark_graph_provenance_if_empty(engine)
    dataset = SimpleNamespace(id=uuid4())
    ctx = _route_ctx(dataset)
    data_item = ctx.data_item
    _unified(engine, vectors, monkeypatch)
    planner_calls = []
    real_delete_chunks_incremental = link_module.delete_chunks_incremental

    async def spy(chunk_ids, dataset_id, data_id):
        planner_calls.append((list(chunk_ids), dataset_id, data_id))
        return await real_delete_chunks_incremental(chunk_ids, dataset_id, data_id)

    monkeypatch.setattr(link_module, "delete_chunks_incremental", spy)

    await _sync_both_code_graphs(repos)
    first = await _sync_code_docs(ctx=ctx, extract_entities=True)
    old_main = _chunks_by_header(first)["app.py:4"]
    nodes, edges = await _graph(engine)
    old_summary = _summary_of(edges, str(old_main.id))
    # The chunk carries its own chunk-scoped ref, keyed by the context's dataset and data item.
    own_ref = make_chunk_source_ref_key(dataset.id, data_item.id, old_main.id)
    delete_data = await engine.get_node_delete_data([str(old_main.id), old_summary, HTTP_SERVER])
    assert all(own_ref in data.source_ref_keys for data in delete_data.values())

    await _index(repos.shop, SHOP_FACTS, MAIN_EDITED, "s2")
    third = await _sync_code_docs(repo=SHOP, ctx=ctx, extract_entities=True)

    # One planner pass, keyed exactly as the chunk was stored.
    assert planner_calls == [([str(old_main.id)], str(dataset.id), str(data_item.id))]
    new_main = _chunks_by_header(third)["app.py:4"]
    nodes, edges = await _graph(engine)
    assert {str(old_main.id), old_summary, HTTP_SERVER}.isdisjoint(nodes)
    assert {str(new_main.id), ROUTES, SHOP_ENTITY, SERVICE, PROJECT} <= set(nodes)
    # The planner dropped the vectors of what the chunk owned -- the nodes,
    # and the edge vectors of the edges only it stated -- once each; the
    # manual path dropped nothing on top.
    by_collection = {}
    for collection, ids in vectors.deleted:
        by_collection.setdefault(collection, []).append(ids)
    assert by_collection["DocumentChunk_text"] == [[str(old_main.id)]]
    assert by_collection["TextSummary_text"] == [[old_summary]]
    assert by_collection["Entity_name"] == [[HTTP_SERVER]]
    assert set(by_collection) <= {
        "DocumentChunk_text",
        "TextSummary_text",
        "Entity_name",
        "Triplet_text",
        "EdgeType_relationship_name",
    }


@pytest.mark.asyncio
async def test_code_docs_tail_reports_the_run(engine, repos, vectors):
    await _sync_both_code_graphs(repos)

    # Every repository, each read from its stored path (no one snapshot
    # directory holds both).
    counts = await code_docs_tail(None, graph_only=True)
    assert counts == {"documents": 5, "chunks": 7, "new": 7, "edges": 7, "swept": 0}

    # Unchanged: nothing stored, linked or swept.
    assert await code_docs_tail(None, graph_only=True) == {
        "documents": 5,
        "chunks": 7,
        "new": 0,
        "edges": 0,
        "swept": 0,
    }

    # One docstring edited and its repository indexed again; the run scoped to
    # it and handed its snapshot, as a route runs it.
    await _index(repos.shop, SHOP_FACTS, MAIN_EDITED, "s2")
    assert await code_docs_tail(SHOP, graph_only=True, snapshot_dir=_snapshot_dir(repos.shop)) == {
        "documents": 4,
        "chunks": 6,
        "new": 1,
        "edges": 1,
        "swept": 1,
    }
    assert [collection for collection, _ids in vectors.deleted] == ["DocumentChunk_text"]


@pytest.mark.asyncio
async def test_route_chain_forget_removes_the_repository_with_its_docs_and_vectors(
    engine, repos, vectors, monkeypatch
):
    """The code routes' shape, one data item per repository: the enola tasks and then
    code_docs_tail, handed the snapshot they loaded, under the route's ctx.
    Everything the shop's run stored is attributed to the shop's data item -- the
    code nodes under the item's doc-scope ref, each chunk under its own
    chunk-scoped ref -- and the graph-provenance path forget(data_id=...) takes
    (try_delete_data_by_graph_provenance -> delete_by_document) removes the shop's
    code graph, documents, chunks and vectors in one pass; billing's stay, with
    their describes edge."""
    _unified(engine, vectors, monkeypatch)
    dataset = SimpleNamespace(id=uuid4())
    shop_ctx, billing_ctx = _route_ctx(dataset), _route_ctx(dataset)

    _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, SHOP_DOCS, "s1")
    assert await _route(repos.shop, ctx=shop_ctx) == {
        SHOP: {"documents": 4, "chunks": 6, "new": 6, "edges": 6, "swept": 0}
    }
    _write_snapshot(_snapshot_dir(repos.billing), BILLING_FACTS, BILLING_DOCS, "b1")
    assert await _route(repos.billing, ctx=billing_ctx) == {
        BILLING: {"documents": 1, "chunks": 1, "new": 1, "edges": 1, "swept": 0}
    }

    nodes, edges = await _graph(engine)
    shop_code, shop_documents, shop_chunks = _repository_nodes(nodes, SHOP)
    shop_nodes = shop_code | shop_documents | shop_chunks
    billing_nodes = set().union(*_repository_nodes(nodes, BILLING))
    assert (len(shop_code), len(shop_documents), len(shop_chunks), len(billing_nodes)) == (
        8,
        4,
        6,
        5,
    )
    assert set(nodes) == shop_nodes | billing_nodes

    # Attribution: the code nodes carry the item's doc-scope ref, every chunk its
    # own chunk-scoped ref, and no shop node names any other data item.
    delete_data = await engine.get_node_delete_data(sorted(shop_nodes))
    item_ref = make_source_ref_key(dataset.id, shop_ctx.data_item.id)
    assert all(item_ref in delete_data[node_id].source_ref_keys for node_id in shop_code)
    for chunk_id in shop_chunks:
        own_ref = make_chunk_source_ref_key(dataset.id, shop_ctx.data_item.id, UUID(chunk_id))
        assert own_ref in delete_data[chunk_id].source_ref_keys
    assert all(
        _data_ids(delete_data, node_id) == {str(shop_ctx.data_item.id)} for node_id in shop_nodes
    )

    # forget(data_id=<the shop's row>).
    assert await _forget(dataset, shop_ctx) == shop_nodes

    nodes, edges = await _graph(engine)
    assert set(nodes) == billing_nodes
    assert all(source in nodes and target in nodes for source, target, _n, _p in edges)
    invoice = _stored_chunk(nodes, "invoices.py:1")
    assert (invoice, str(fact_node_id(BILLING, "symbol", "invoices.issue"))) in _describes(edges)
    # The shop's embeddings went with its nodes, once each; billing's were not touched.
    deleted = _deleted_by_collection(vectors)
    assert sorted(deleted["DocumentChunk_text"]) == sorted(shop_chunks)
    assert sorted(deleted["CodeDocument_name"]) == sorted(shop_documents)
    assert not set().union(*deleted.values()) & billing_nodes

    # A docs run over the forgotten repository, its snapshot still on disk, has
    # nothing to document and nothing left to sweep.
    assert (
        await code_docs_tail(
            SHOP,
            ctx=_route_ctx(dataset),
            graph_only=True,
            snapshot_dir=_snapshot_dir(repos.shop),
        )
        == NOTHING
    )


@pytest.mark.asyncio
async def test_route_chain_entity_mode_change_then_forget_retire_what_the_item_owned(
    engine, repos, vectors, extractor, monkeypatch
):
    """Entity mode under the routes' ctx. A changed docstring (the repository indexed
    again, its snapshot handed to the tail) sends only its chunk to the LLM and the
    planner retires the old chunk with its summary and the entity only it held.
    forget(data_id=...) of the shop then removes its whole output -- chunks,
    summaries, the entity and type only its chunks held -- while the entity and
    type billing's chunk also holds survive, with the shop's refs stripped."""
    _unified(engine, vectors, monkeypatch)
    dataset = SimpleNamespace(id=uuid4())
    shop_ctx, billing_ctx = _route_ctx(dataset), _route_ctx(dataset)
    _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, SHOP_DOCS, "s1")
    _write_snapshot(_snapshot_dir(repos.billing), BILLING_FACTS, BILLING_DOCS, "b1")

    first = await _route(repos.shop, ctx=shop_ctx, extract_entities=True)
    await _route(repos.billing, ctx=billing_ctx, extract_entities=True)
    assert first == {SHOP: {"documents": 4, "chunks": 6, "new": 6, "edges": 6, "swept": 0}}
    assert [sorted(call) for call in extractor] == [
        sorted(header for header in EXPECTED_TARGETS if header != "invoices.py:1"),
        ["invoices.py:1"],
    ]
    nodes, edges = await _graph(engine)
    assert {HTTP_SERVER, SHOP_ENTITY, SERVICE, PROJECT} <= set(nodes)
    # "shop" came out of both repositories' chunks: it names both data items.
    delete_data = await engine.get_node_delete_data([SHOP_ENTITY, PROJECT, HTTP_SERVER])
    both = {str(shop_ctx.data_item.id), str(billing_ctx.data_item.id)}
    assert _data_ids(delete_data, SHOP_ENTITY) == _data_ids(delete_data, PROJECT) == both
    assert _data_ids(delete_data, HTTP_SERVER) == {str(shop_ctx.data_item.id)}

    # main's docstring changes: one chunk to the LLM; the old chunk, its summary
    # and "http server" are retired through the planner (3 nodes swept with it).
    old_main = _stored_chunk(nodes, "app.py:4")
    old_summary = _summary_of(edges, old_main)
    _write_snapshot(_snapshot_dir(repos.shop), SHOP_FACTS, MAIN_EDITED, "s2")
    counts = await _route(repos.shop, ctx=shop_ctx, extract_entities=True)
    assert extractor[-1] == ["app.py:4"]
    assert counts == {SHOP: {"documents": 4, "chunks": 6, "new": 1, "edges": 1, "swept": 3}}
    nodes, edges = await _graph(engine)
    new_main = _stored_chunk(nodes, "app.py:4")
    assert new_main != old_main
    assert {old_main, old_summary, HTTP_SERVER}.isdisjoint(nodes)
    assert {ROUTES, SHOP_ENTITY, SERVICE, PROJECT} <= set(nodes)
    assert (new_main, ROUTES, "contains") in _edge_keys(edges)

    # forget(data_id=<the shop's row>): "routes" and "service" were held by the
    # shop's chunks alone and go; "shop" and "project", held by billing's chunk
    # too, stay and lose the shop's refs.
    vectors.deleted.clear()
    _shop_code, shop_documents, shop_chunks = _repository_nodes(nodes, SHOP)
    shop_summaries = {_summary_of(edges, chunk_id) for chunk_id in shop_chunks}
    deleted_nodes = await _forget(dataset, shop_ctx)

    nodes, edges = await _graph(engine)
    assert not any(_repository_nodes(nodes, SHOP))
    assert {ROUTES, SERVICE}.isdisjoint(nodes) and {SHOP_ENTITY, PROJECT} <= set(nodes)
    invoice = _stored_chunk(nodes, "invoices.py:1")
    assert set(nodes) == set().union(*_repository_nodes(nodes, BILLING)) | {
        _summary_of(edges, invoice),
        SHOP_ENTITY,
        PROJECT,
    }
    assert all(source in nodes and target in nodes for source, target, _n, _p in edges)
    assert {(invoice, SHOP_ENTITY, "contains"), (SHOP_ENTITY, PROJECT, "is_a")} <= _edge_keys(edges)
    delete_data = await engine.get_node_delete_data([SHOP_ENTITY, PROJECT])
    assert (
        _data_ids(delete_data, SHOP_ENTITY)
        == _data_ids(delete_data, PROJECT)
        == {str(billing_ctx.data_item.id)}
    )
    assert {ROUTES, SERVICE} | shop_chunks | shop_summaries | shop_documents <= deleted_nodes
    deleted = _deleted_by_collection(vectors)
    assert sorted(deleted["DocumentChunk_text"]) == sorted(shop_chunks)
    assert sorted(deleted["TextSummary_text"]) == sorted(shop_summaries)
    assert sorted(deleted["CodeDocument_name"]) == sorted(shop_documents)
    assert (deleted["Entity_name"], deleted["EntityType_name"]) == ([ROUTES], [SERVICE])
