"""The code docs chain against a real Ladybug graph: store, link, re-run, change, remove.

The mocked tests next to this one assert the calls each task makes. This one
runs what the memify pipeline runs -- the enola code graph tasks over a
snapshot of a small Python repository, then ``extract_code_docs``,
``add_data_points(graph_only=True)`` and ``add_code_doc_edges`` -- against a
real engine and reads the stored graph back, so the stored shapes, the edge
targets, the idempotent re-run and the sweep are checked end to end.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.tasks.code_docs import add_code_doc_edges, code_document_id, extract_code_docs
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    extract_code_graph,
    fact_node_id,
)
from cognee.tasks.storage.add_data_points import add_data_points

extract_module = importlib.import_module("cognee.tasks.code_docs.extract_code_docs")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
vector_module = importlib.import_module("cognee.infrastructure.databases.vector")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

SHOP = "shop"
BILLING = "billing"
CODE_TYPES = {"CodeRepository", "CodeModule", "CodeSymbol"}

APP_PY = '''"""The shop application."""


def main():
    """Start the shop's HTTP server."""
    serve()


def serve():
    return None
'''
# main's docstring edited: the chunk's text, and so its id, change.
APP_PY_EDITED = APP_PY.replace(
    "Start the shop's HTTP server.", "Start the shop's HTTP server and register its routes."
)

CHECKOUT_PY = '''"""Order checkout."""


class Order:
    """An order."""

    def total(self) -> int:
        """Sum of the lines."""
        return 0


def place_order(order: Order) -> str:
    """Place an order."""
    return "ok"
'''
# place_order deleted from the code.
CHECKOUT_PY_WITHOUT_PLACE_ORDER = CHECKOUT_PY[: CHECKOUT_PY.index("\n\n\ndef place_order")] + "\n"

INVOICES_PY = '''def issue(order_id: str) -> str:
    """Issue an invoice for an order."""
    return order_id
'''


def _module(repo, name):
    return {"kind": "module", "name": name, "file": name, "repo": repo, "props": {}}


def _symbol(repo, name, file, line, kind="function", calls=()):
    return {
        "kind": "symbol",
        "name": name,
        "file": file,
        "line": line,
        "repo": repo,
        "props": {"language": "python", "symbol_kind": kind},
        "relations": [{"kind": "calls", "target": target} for target in calls],
    }


# Enola-style facts of the two repositories: def lines as the sources above have them.
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

# Chunk header (its first line) -> the code node its describes edge must reach.
EXPECTED_TARGETS = {
    "app.py:1": (SHOP, "module", "."),
    "app.py:4": (SHOP, "symbol", "app.main"),
    "orders/checkout.py:1": (SHOP, "module", "orders"),
    "orders/checkout.py:4": (SHOP, "symbol", "orders/checkout.Order"),
    "orders/checkout.py:7": (SHOP, "symbol", "orders/checkout.Order.total"),
    "orders/checkout.py:12": (SHOP, "symbol", "orders/checkout.place_order"),
    "invoices.py:1": (BILLING, "symbol", "invoices.issue"),
}


def _write_repo(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _write_snapshot(directory, facts, snapshot_id):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    receipt = {"snapshot_id": snapshot_id, "format_version": 1, "fact_count": len(facts)}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory


@pytest.fixture
def vectors(monkeypatch):
    """Where the sweep drops embeddings.

    The chain here stores graph only (no embedding provider in unit tests), so
    the fake reports every collection present to make the deletes observable.
    """
    fake = SimpleNamespace(deleted=[])

    async def has_collection(name):
        return True

    async def delete_data_points(collection, ids):
        fake.deleted.append((collection, list(ids)))

    fake.has_collection, fake.delete_data_points = has_collection, delete_data_points
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
def repos(tmp_path):
    """Two repositories' sources, and where their snapshots go."""
    return SimpleNamespace(
        shop=_write_repo(tmp_path / SHOP, {"app.py": APP_PY, "orders/checkout.py": CHECKOUT_PY}),
        billing=_write_repo(tmp_path / BILLING, {"invoices.py": INVOICES_PY}),
        snapshots=tmp_path / "snapshots",
    )


async def _sync_code_graph(repo_path, snapshot_dir):
    """The enola code graph tasks, in get_code_graph_tasks order."""
    data_points = await extract_code_graph(repo_path=str(repo_path), snapshot_dir=snapshot_dir)
    state = await add_code_graph_data_points(data_points)
    await add_code_graph_edges(state, repo_path=str(repo_path), snapshot_dir=snapshot_dir)


async def _sync_both_code_graphs(repos):
    await _sync_code_graph(repos.shop, _write_snapshot(repos.snapshots / "shop", SHOP_FACTS, "s1"))
    await _sync_code_graph(
        repos.billing, _write_snapshot(repos.snapshots / "billing", BILLING_FACTS, "b1")
    )


async def _sync_code_docs(repo=None):
    """The code docs pipeline's chain: extract, store (graph only), link and sweep."""
    data_points = await extract_code_docs(repo=repo)
    await add_data_points(data_points, graph_only=True)
    await add_code_doc_edges(data_points)
    return data_points


async def _graph(engine):
    """The stored graph: {node id: properties} and (source, target, name, properties) edges."""
    nodes, edges = await engine.get_graph_data()
    return dict(nodes), edges


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
    chunks = [point for point in data_points if isinstance(point, DocumentChunk)]
    return {chunk.text.split("\n", 1)[0]: chunk for chunk in chunks}


@pytest.mark.asyncio
async def test_stores_documents_and_chunks_linked_to_their_code_nodes(engine, repos):
    await _sync_both_code_graphs(repos)
    code_graph_before = _code_graph(*await _graph(engine))

    data_points = await _sync_code_docs()

    nodes, edges = await _graph(engine)
    documents = [point for point in data_points if isinstance(point, CodeDocument)]
    by_header = _chunks_by_header(data_points)
    assert len(documents) + len(by_header) == len(data_points)

    # One CodeDocument per documented file, repositories by name, files by path.
    assert [(document.repo, document.name) for document in documents] == [
        (BILLING, "invoices.py"),
        (SHOP, "app.py"),
        (SHOP, "orders/checkout.py"),
    ]
    for document in documents:
        assert document.id == code_document_id(document.repo, document.name)
        stored = nodes[str(document.id)]
        assert (stored["type"], stored["name"], stored["repo"]) == (
            "CodeDocument",
            document.name,
            document.repo,
        )

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
    assert by_header["app.py:1"].text == "app.py:1\napp (module)\n\nThe shop application."
    assert by_header["app.py:4"].text == (
        "app.py:4\napp.main (function)\ndef main():\n\nStart the shop's HTTP server."
    )
    assert by_header["orders/checkout.py:7"].text == (
        "orders/checkout.py:7\n"
        "orders/checkout.Order.total (method)\n"
        "def total(self) -> int:\n"
        "\n"
        "Sum of the lines."
    )

    # Exactly one describes edge per chunk, to the code node with that id.
    describes = [(s, t, props) for s, t, name, props in edges if name == "describes"]
    targets = {source: target for source, target, _ in describes}
    assert len(describes) == len(targets) == len(by_header)
    assert {header: targets[str(chunk.id)] for header, chunk in by_header.items()} == {
        header: str(fact_node_id(*target)) for header, target in EXPECTED_TARGETS.items()
    }
    main_edge = next(props for s, _t, props in describes if s == str(by_header["app.py:4"].id))
    assert (main_edge["relationship_name"], main_edge["doc_source"], main_edge["chunk_index"]) == (
        "describes",
        "source",
        1,
    )

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
            Edge(
                Node(str(chunk.id), dict(nodes[str(chunk.id)])),
                Node(symbol_id, dict(nodes[symbol_id])),
                attributes=dict(edge_properties),
            )
        ]
    )

    assert (
        "__node_content_start__\n"
        "app.py:4\napp.main (function)\ndef main():\n\nStart the shop's HTTP server.\n"
        "__node_content_end__"
    ) in text
    assert "--[describes]--> app.main" in text


@pytest.mark.asyncio
async def test_rerun_change_and_removal_touch_exactly_their_chunks(
    engine, repos, vectors, monkeypatch
):
    await _sync_both_code_graphs(repos)
    first = await _sync_code_docs()
    nodes_1, edges_1 = await _graph(engine)
    assert vectors.deleted == []

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

    # Unchanged sources: the same ids, no new edge, nothing swept.
    second = await _sync_code_docs()
    assert [point.id for point in second] == [point.id for point in first]
    assert (written, deleted, vectors.deleted) == ([], [], [])
    nodes_2, edges_2 = await _graph(engine)
    assert set(nodes_2) == set(nodes_1)
    assert _edge_keys(edges_2) == _edge_keys(edges_1)

    # A changed docstring: its chunk is replaced (new id, new text, same
    # position), the old one swept, every other chunk kept. The run covers
    # the shop only, so billing is not in the batch and must stay untouched.
    (repos.shop / "app.py").write_text(APP_PY_EDITED, encoding="utf-8")
    old_main = _chunks_by_header(second)["app.py:4"]
    main_id = str(fact_node_id(SHOP, "symbol", "app.main"))

    third = await _sync_code_docs(repo=SHOP)

    new_main = _chunks_by_header(third)["app.py:4"]
    assert new_main.id != old_main.id
    assert new_main.text.endswith("Start the shop's HTTP server and register its routes.")
    assert new_main.chunk_index == old_main.chunk_index == 1
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

    # A symbol deleted from the code: the code graph's own sweep drops its
    # CodeSymbol (and with it the describes edge), the docs run sweeps its chunk.
    written.clear()
    deleted.clear()
    vectors.deleted.clear()
    (repos.shop / "orders/checkout.py").write_text(
        CHECKOUT_PY_WITHOUT_PLACE_ORDER, encoding="utf-8"
    )
    facts = [fact for fact in SHOP_FACTS if fact["name"] != "orders/checkout.place_order"]
    await _sync_code_graph(repos.shop, _write_snapshot(repos.snapshots / "shop-2", facts, "s2"))
    place_order_id = str(fact_node_id(SHOP, "symbol", "orders/checkout.place_order"))
    assert deleted == [place_order_id]
    deleted.clear()
    old_place_order = _chunks_by_header(third)["orders/checkout.py:12"]

    fourth = await _sync_code_docs()

    assert (written, deleted) == ([], [str(old_place_order.id)])
    assert vectors.deleted == [("DocumentChunk_text", [str(old_place_order.id)])]
    assert {point.id for point in fourth} == (
        {point.id for point in second} - {old_main.id, old_place_order.id}
    ) | {new_main.id}
    nodes_4, edges_4 = await _graph(engine)
    assert set(nodes_4) == set(nodes_3) - {str(old_place_order.id), place_order_id}
    assert _describes(edges_4) == _describes(edges_3) - {(str(old_place_order.id), place_order_id)}
    # Billing, never in a changed run, is exactly as the first run left it.
    invoice = _chunks_by_header(first)["invoices.py:1"]
    assert nodes_4[str(invoice.id)]["text"] == invoice.text
    assert nodes_4[str(code_document_id(BILLING, "invoices.py"))]["type"] == "CodeDocument"
    assert (str(invoice.id), str(fact_node_id(BILLING, "symbol", "invoices.issue"))) in _describes(
        edges_4
    )
