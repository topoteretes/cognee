"""extract_code_docs over a mocked code graph and real enola snapshot directories.

The graph engine is mocked: its one filtered read returns the CodeRepository,
CodeSymbol and CodeModule nodes a test builds. The docs come from snapshot
directories written by write_enola_snapshot -- docs.jsonl and the
receipt.json that vouches for it -- in ``<path>/.enola`` of the repository
node, or wherever ``snapshot_dir`` points. Covered: chunk text, fields, ids
and order; chunk metadata and CodeDocument fields; node sets; how a
docs.jsonl line joins the graph (by id, kind and repository); which snapshot
is read, and every case in which a repository's snapshot docs are
unavailable; splitting a long doc; the per-repository worker thread.
"""

import asyncio
import importlib
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.code_docs import code_document_id, extract_code_docs
from cognee.tasks.code_docs.extract_code_docs import _chunk_texts
from cognee.tasks.code_docs.models import CodeDocsBatch, CodeDocument
from cognee.tasks.code_graph.enola import DOCS_ARTIFACT
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.documents.classify_documents import update_node_set
from cognee.tests.utils.enola_snapshot import (
    doc_line,
    enola_fact_id,
    sha256_label,
    write_enola_snapshot,
)

extract_module = importlib.import_module("cognee.tasks.code_docs.extract_code_docs")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

REPO = "shop"
OTHER = "other"
BUDGET = 64
SNAPSHOT_ID = "snapshot-1"

SYMBOL_METADATA_KEYS = {
    "repo",
    "file",
    "line",
    "end_line",
    "symbol",
    "symbol_id",
    "symbol_kind",
    "language",
    "enola_id",
}

PLACE_ORDER_DOC = (
    "Place an order.\n"
    "\n"
    "The order is priced, then stored; the caller gets its id.\n"
    "\n"
    "Example:\n"
    "    place_order(order, notify=False)"
)
ORDERS_DOC = "Order handling.\n\nPlacing, pricing and the totals of orders."
OTHER_DOCS = [doc_line(OTHER, "symbol", "lib.f", "lib.py", "Other f.", line=1)]


def _words(text: str) -> int:
    return len(text.split())


@pytest.fixture(autouse=True)
def _offline_token_budget(monkeypatch):
    """No embedding engine in unit tests: count words, fixed budget."""
    monkeypatch.setattr(extract_module, "_token_counter", lambda: _words)
    monkeypatch.setattr(extract_module, "_default_chunk_budget", AsyncMock(return_value=BUDGET))


@pytest.fixture
def log(monkeypatch):
    """The task's logger, recorded."""
    logger = MagicMock()
    monkeypatch.setattr(extract_module, "logger", logger)
    return logger


def _mock_engine(monkeypatch, nodes=()):
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = (list(nodes), [])
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


# --- graph nodes --------------------------------------------------------------


def _repository(path, repo=REPO, snapshot_id=SNAPSHOT_ID):
    """A CodeRepository node; ``path`` or ``snapshot_id`` None: the node records none."""
    properties = {"type": "CodeRepository", "name": repo, "kind": "repository"}
    if path is not None:
        properties["path"] = str(path)
    if snapshot_id is not None:
        properties["last_snapshot_id"] = snapshot_id
    return str(fact_node_id(repo, "repository", repo)), properties


def _tag(properties, node_set):
    if node_set:
        properties["belongs_to_set"] = list(node_set)
        properties["source_node_set"] = ", ".join(node_set)


def _symbol(
    name,
    file_path,
    line,
    *,
    kind="function",
    end_line=None,
    language="python",
    node_set=None,
    repo=REPO,
):
    """A CodeSymbol node; its enola_id is the id doc_line gives the same (repo, name, file)."""
    fact_properties = {"symbol_kind": kind}
    if language is not None:
        fact_properties["language"] = language
    properties = {
        "type": "CodeSymbol",
        "name": name,
        "kind": "symbol",
        "repo": repo,
        "file_path": file_path,
        "line": line,
        "end_line": end_line,
        "symbol_kind": kind,
        "enola_id": enola_fact_id(repo, "symbol", name, file_path),
        "fact_properties": fact_properties,
    }
    _tag(properties, node_set)
    return str(fact_node_id(repo, "symbol", name)), properties


def _module(name, directory, *, language="python", node_set=None, repo=REPO):
    """A CodeModule node (a package: ``directory`` is its file)."""
    properties = {
        "type": "CodeModule",
        "name": name,
        "kind": "module",
        "repo": repo,
        "file_path": directory,
        "enola_id": enola_fact_id(repo, "module", name, directory),
        "fact_properties": {"language": language} if language is not None else {},
    }
    _tag(properties, node_set)
    return str(fact_node_id(repo, "module", name)), properties


def _enola_id(node):
    return node[1]["enola_id"]


# --- snapshots ----------------------------------------------------------------


def _snapshot(root, docs):
    """``<root>/.enola`` holding ``docs``, vouched for by a receipt of SNAPSHOT_ID."""
    return write_enola_snapshot(root / ".enola", docs=docs, snapshot_id=SNAPSHOT_ID)


def _shop_nodes(root, node_set=None):
    return [
        _repository(root),
        _module("orders", "orders", node_set=node_set),
        _symbol("app.main", "app.py", 4, end_line=6, node_set=node_set),
        # No docs.jsonl line names it: undocumented.
        _symbol("app.serve", "app.py", 8, end_line=9, node_set=node_set),
        _symbol(
            "orders/checkout.Order",
            "orders/checkout.py",
            4,
            kind="class",
            end_line=11,
            node_set=node_set,
        ),
        _symbol(
            "orders/checkout.Order.total",
            "orders/checkout.py",
            8,
            kind="method",
            end_line=10,
            node_set=node_set,
        ),
        _symbol(
            "orders/checkout.place_order",
            "orders/checkout.py",
            14,
            end_line=21,
            node_set=node_set,
        ),
    ]


def _shop_docs():
    """The shop snapshot's docs.jsonl lines, out of file order: the extraction sorts."""
    return [
        doc_line(
            REPO,
            "symbol",
            "orders/checkout.place_order",
            "orders/checkout.py",
            PLACE_ORDER_DOC,
            line=14,
        ),
        doc_line(
            REPO,
            "symbol",
            "orders/checkout.Order.total",
            "orders/checkout.py",
            "Sum of the lines.",
            line=8,
        ),
        doc_line(REPO, "module", "orders", "orders", ORDERS_DOC),
        doc_line(
            REPO, "symbol", "orders/checkout.Order", "orders/checkout.py", "An order.", line=4
        ),
        doc_line(REPO, "symbol", "app.main", "app.py", "Start the shop's HTTP server.", line=4),
    ]


@pytest.fixture
def shop(tmp_path):
    """The shop repository's directory, holding the snapshot its code graph was built from."""
    root = tmp_path / REPO
    _snapshot(root, _shop_docs())
    return root


def _other_repo(tmp_path, node_set=None):
    """A second repository with its own snapshot: one documented function."""
    root = tmp_path / OTHER
    _snapshot(root, OTHER_DOCS)
    return [_repository(root, OTHER), _symbol("lib.f", "lib.py", 1, repo=OTHER, node_set=node_set)]


# --- results ------------------------------------------------------------------


def _split(data_points):
    documents = [point for point in data_points if isinstance(point, CodeDocument)]
    chunks = [point for point in data_points if isinstance(point, DocumentChunk)]
    assert len(documents) + len(chunks) == len(data_points)
    return documents, chunks


def _metadata(chunk):
    return json.loads(chunk.external_metadata)["code_doc"]


def _chunk_for(chunks, symbol):
    return next(chunk for chunk in chunks if _metadata(chunk)["symbol"] == symbol)


def _warnings(log):
    return [call.args[0] % call.args[1:] for call in log.warning.call_args_list]


def _summary(log, repo):
    """The run's info line for ``repo``: lines read, joined, unmatched, emitted."""
    (message,) = [
        call.args[0] % call.args[1:]
        for call in log.info.call_args_list
        if len(call.args) > 1 and call.args[1] == repo
    ]
    return message


# --- chunk text and order -----------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_text_is_location_and_name_over_the_whole_doc(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    _, chunks = _split(await extract_code_docs())

    assert _chunk_for(chunks, "app.main").text == (
        "app.py:4\napp.main (function)\n\nStart the shop's HTTP server."
    )
    assert _chunk_for(chunks, "orders/checkout.Order.total").text == (
        "orders/checkout.py:8\norders/checkout.Order.total (method)\n\nSum of the lines."
    )
    # Every paragraph of the comment, not only the first one facts.jsonl keeps.
    assert _chunk_for(chunks, "orders/checkout.place_order").text == (
        "orders/checkout.py:14\norders/checkout.place_order (function)\n\n" + PLACE_ORDER_DOC
    )
    # A module (package) doc: headed by its directory, no line.
    assert _chunk_for(chunks, "orders").text == "orders\norders (module)\n\n" + ORDERS_DOC


@pytest.mark.asyncio
async def test_symbol_kind_comes_from_the_node_then_its_fact_properties_then_symbol(
    tmp_path, monkeypatch
):
    root = tmp_path / REPO
    _snapshot(
        root,
        [
            doc_line(REPO, "symbol", "cfg.Config", "cfg.py", "The settings.", line=3),
            doc_line(REPO, "symbol", "cfg.load", "cfg.py", "Read the settings.", line=9),
            doc_line(REPO, "symbol", "cfg.merge", "cfg.py", "Merge two settings.", line=14),
        ],
    )
    load = _symbol("cfg.load", "cfg.py", 9, kind="function")
    load[1]["symbol_kind"] = None  # only the fact's own properties carry it
    _mock_engine(
        monkeypatch,
        [
            _repository(root),
            _symbol("cfg.Config", "cfg.py", 3, kind="class"),
            load,
            _symbol("cfg.merge", "cfg.py", 14, kind=None),
        ],
    )

    _, chunks = _split(await extract_code_docs())

    assert [chunk.text.split("\n\n")[0] for chunk in chunks] == [
        "cfg.py:3\ncfg.Config (class)",
        "cfg.py:9\ncfg.load (function)",
        "cfg.py:14\ncfg.merge (symbol)",
    ]
    assert [_metadata(chunk)["symbol_kind"] for chunk in chunks] == ["class", "function", "symbol"]


@pytest.mark.asyncio
async def test_a_symbol_line_without_a_line_number_is_headed_by_its_file_alone(
    tmp_path, monkeypatch
):
    root = tmp_path / REPO
    _snapshot(root, [doc_line(REPO, "symbol", "app.VERSION", "app.py", "The release.")])
    _mock_engine(
        monkeypatch, [_repository(root), _symbol("app.VERSION", "app.py", None, kind="constant")]
    )

    (chunk,) = _split(await extract_code_docs())[1]

    assert chunk.text == "app.py\napp.VERSION (constant)\n\nThe release."
    assert _metadata(chunk)["line"] is None


@pytest.mark.asyncio
async def test_documents_first_sorted_by_file_then_chunks_in_document_order(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    data_points = await extract_code_docs()
    documents, chunks = _split(data_points)

    assert [type(point) for point in data_points] == [CodeDocument] * 3 + [DocumentChunk] * 5
    assert [document.name for document in documents] == ["app.py", "orders", "orders/checkout.py"]
    assert [
        (chunk.document_name, chunk.chunk_index, _metadata(chunk)["symbol"]) for chunk in chunks
    ] == [
        ("app.py", 0, "app.main"),
        ("orders", 0, "orders"),
        ("orders/checkout.py", 0, "orders/checkout.Order"),
        ("orders/checkout.py", 1, "orders/checkout.Order.total"),
        ("orders/checkout.py", 2, "orders/checkout.place_order"),
    ]
    # app.serve has no docs.jsonl line: no chunk.
    assert all(_metadata(chunk)["symbol"] != "app.serve" for chunk in chunks)


@pytest.mark.asyncio
async def test_docstrings_of_a_document_sort_by_line_then_name_with_the_module_doc_first(
    tmp_path, monkeypatch
):
    """The order inside one document, whatever the docs.jsonl order: a module
    doc has no line and sorts first, symbols follow by (line, name). The two
    docs on line 7 sort the other way by their text."""
    root = tmp_path / REPO
    _snapshot(
        root,
        [
            doc_line(REPO, "symbol", "pkg.zeta", "pkg/core.py", "A late one.", line=20),
            doc_line(REPO, "symbol", "pkg.beta", "pkg/core.py", "A helper.", line=7),
            doc_line(REPO, "module", "pkg", "pkg/core.py", "The package."),
            doc_line(REPO, "symbol", "pkg.alpha", "pkg/core.py", "Yet another helper.", line=7),
        ],
    )
    _mock_engine(
        monkeypatch,
        [
            _repository(root),
            _module("pkg", "pkg/core.py"),
            _symbol("pkg.zeta", "pkg/core.py", 20),
            _symbol("pkg.beta", "pkg/core.py", 7),
            _symbol("pkg.alpha", "pkg/core.py", 7),
        ],
    )

    (document,), chunks = _split(await extract_code_docs())

    assert [(_metadata(chunk)["symbol"], chunk.chunk_index) for chunk in chunks] == [
        ("pkg", 0),
        ("pkg.alpha", 1),
        ("pkg.beta", 2),
        ("pkg.zeta", 3),
    ]
    assert document.symbol_count == 3


@pytest.mark.asyncio
async def test_chunk_fields_and_ids_are_identical_across_runs(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    first = await extract_code_docs()
    second = await extract_code_docs()

    assert [point.id for point in first] == [point.id for point in second]
    documents, chunks = _split(first)
    _, second_chunks = _split(second)
    assert [chunk.text for chunk in chunks] == [chunk.text for chunk in second_chunks]

    by_name = {document.name: document for document in documents}
    for chunk in chunks:
        document = by_name[chunk.document_name]
        assert chunk.is_part_of is document
        assert chunk.document_id == str(document.id)
        assert chunk.content_hash == chunk_content_hash(chunk.text)
        assert chunk.id == content_chunk_id(document.id, chunk.content_hash, 0)
        assert chunk.cut_type == "code_docstring"
        assert chunk.chunker_id == "code_docs"
        assert chunk.max_chunk_tokens == BUDGET
        assert chunk.chunk_size == _words(chunk.text)
        assert chunk.contains == []
        assert chunk.importance_weight == document.importance_weight
        assert chunk.text.startswith(chunk.document_name)
    for document in documents:
        indexes = [chunk.chunk_index for chunk in chunks if chunk.document_name == document.name]
        assert indexes == list(range(len(indexes)))


@pytest.mark.asyncio
async def test_identical_chunk_texts_of_a_document_are_told_apart_by_occurrence(
    tmp_path, monkeypatch
):
    paragraph = "The same words again."
    root = tmp_path / REPO
    _snapshot(
        root,
        [doc_line(REPO, "symbol", "dup.twice", "dup.py", f"{paragraph}\n\n{paragraph}", line=1)],
    )
    _mock_engine(monkeypatch, [_repository(root), _symbol("dup.twice", "dup.py", 1)])

    # Header (3 words) and both paragraphs (4 each) exceed 8: one paragraph
    # per chunk, the same text twice.
    (document,), chunks = _split(await extract_code_docs(max_chunk_tokens=8))

    text = f"dup.py:1\ndup.twice (function)\n\n{paragraph}"
    assert [chunk.text for chunk in chunks] == [text, text]
    content_hash = chunk_content_hash(text)
    assert [chunk.id for chunk in chunks] == [
        content_chunk_id(document.id, content_hash, 0),
        content_chunk_id(document.id, content_hash, 1),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


# --- metadata and documents ---------------------------------------------------


@pytest.mark.asyncio
async def test_external_metadata_of_a_symbol_chunk_and_a_module_chunk(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    documents, chunks = _split(await extract_code_docs())

    total = _metadata(_chunk_for(chunks, "orders/checkout.Order.total"))
    assert set(total) == SYMBOL_METADATA_KEYS
    assert total == {
        "repo": REPO,
        "file": "orders/checkout.py",
        "line": 8,
        "end_line": 10,
        "symbol": "orders/checkout.Order.total",
        "symbol_id": str(fact_node_id(REPO, "symbol", "orders/checkout.Order.total")),
        "symbol_kind": "method",
        "language": "python",
        "enola_id": enola_fact_id(
            REPO, "symbol", "orders/checkout.Order.total", "orders/checkout.py"
        ),
    }

    module = _metadata(_chunk_for(chunks, "orders"))
    assert set(module) == SYMBOL_METADATA_KEYS | {"module_id"}
    assert module == {
        "repo": REPO,
        "file": "orders",
        "line": None,
        "end_line": None,
        "symbol": "orders",
        "symbol_id": None,
        "symbol_kind": "module",
        "language": "python",
        "enola_id": enola_fact_id(REPO, "module", "orders", "orders"),
        "module_id": str(fact_node_id(REPO, "module", "orders")),
    }

    # No doc_source anywhere: Enola is the only source.
    assert "doc_source" not in CodeDocument.model_fields
    assert all("doc_source" not in _metadata(chunk) for chunk in chunks)
    assert all(
        "doc_source" not in json.loads(document.external_metadata)["code_doc"]
        for document in documents
    )


@pytest.mark.asyncio
async def test_code_documents_are_enola_files_counting_symbol_docs_only(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    documents, _ = _split(await extract_code_docs())

    files = ["app.py", "orders", "orders/checkout.py"]
    assert [document.name for document in documents] == files
    for document, file_path in zip(documents, files):
        assert document.id == code_document_id(REPO, file_path)
        # Enola's repo-relative file: the text comes from the snapshot, not
        # from reading the file in the clone.
        assert document.name == document.file_path == document.raw_data_location == file_path
        assert document.repo == REPO
        assert document.language == "python"
        assert document.mime_type == "text/plain"
        assert json.loads(document.external_metadata) == {
            "code_doc": {"repo": REPO, "file": file_path, "language": "python"}
        }
    # The module doc is not a symbol.
    assert [document.symbol_count for document in documents] == [1, 0, 3]


@pytest.mark.asyncio
async def test_language_comes_from_the_nodes_fact_properties(tmp_path, monkeypatch):
    root = tmp_path / REPO
    _snapshot(
        root,
        [
            doc_line(REPO, "symbol", "main.main", "cmd/main.go", "Go entry point.", line=5),
            doc_line(REPO, "symbol", "build.run", "build.sh", "Build it.", line=3),
        ],
    )
    _mock_engine(
        monkeypatch,
        [
            _repository(root),
            _symbol("main.main", "cmd/main.go", 5, language="go"),
            _symbol("build.run", "build.sh", 3, language=None),
        ],
    )

    documents, chunks = _split(await extract_code_docs())

    assert [(document.name, document.language) for document in documents] == [
        ("build.sh", None),
        ("cmd/main.go", "go"),
    ]
    assert [_metadata(chunk)["language"] for chunk in chunks] == [None, "go"]
    assert json.loads(documents[1].external_metadata) == {
        "code_doc": {"repo": REPO, "file": "cmd/main.go", "language": "go"}
    }


# --- node sets ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_sets_are_built_like_classify_documents(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop, node_set=("code", "team-a")))

    documents, chunks = _split(await extract_code_docs())

    reference = TextDocument(
        name="ref",
        raw_data_location="ref.txt",
        mime_type="text/plain",
        external_metadata=json.dumps({"node_set": ["code", "team-a"]}),
    )
    update_node_set(reference)
    expected_ids = [node_set.id for node_set in reference.belongs_to_set]
    assert expected_ids == [NodeSet.id_for("code"), NodeSet.id_for("team-a")]

    for point in documents + chunks:
        assert [node_set.name for node_set in point.belongs_to_set] == ["code", "team-a"]
        assert [node_set.id for node_set in point.belongs_to_set] == expected_ids
        assert point.source_node_set == "code, team-a"
    # One NodeSet object per name across the whole batch.
    assert (
        len({id(node_set) for point in documents + chunks for node_set in point.belongs_to_set})
        == 2
    )


@pytest.mark.asyncio
async def test_a_document_carries_the_union_of_its_nodes_node_sets(tmp_path, monkeypatch):
    root = tmp_path / REPO
    _snapshot(
        root,
        [
            doc_line(REPO, "symbol", "app.main", "app.py", "Main.", line=4),
            doc_line(REPO, "symbol", "app.serve", "app.py", "Serve.", line=8),
            doc_line(REPO, "symbol", "app.stop", "app.py", "Stop.", line=12),
        ],
    )
    _mock_engine(
        monkeypatch,
        [
            _repository(root),
            _symbol("app.main", "app.py", 4, node_set=("code",)),
            _symbol("app.serve", "app.py", 8, node_set=("team-a", "code")),
            _symbol("app.stop", "app.py", 12),
        ],
    )

    (document,), chunks = _split(await extract_code_docs())

    assert [node_set.name for node_set in document.belongs_to_set] == ["code", "team-a"]
    assert document.source_node_set == "code, team-a"
    # Each chunk carries its own node's sets; an untagged node's chunk none.
    assert [[node_set.name for node_set in chunk.belongs_to_set] for chunk in chunks[:2]] == [
        ["code"],
        ["team-a", "code"],
    ]
    assert chunks[2].belongs_to_set is None
    assert [chunk.source_node_set for chunk in chunks] == ["code", "team-a, code", None]


@pytest.mark.asyncio
async def test_untagged_nodes_give_no_node_sets(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    documents, chunks = _split(await extract_code_docs())

    assert all(point.belongs_to_set is None for point in documents + chunks)
    assert all(point.source_node_set is None for point in documents + chunks)


@pytest.mark.asyncio
async def test_node_sets_are_one_object_per_name_across_repositories(shop, tmp_path, monkeypatch):
    _mock_engine(
        monkeypatch,
        _shop_nodes(shop, node_set=("code",)) + _other_repo(tmp_path, node_set=("code",)),
    )

    documents, chunks = _split(await extract_code_docs())

    assert {document.repo for document in documents} == {OTHER, REPO}
    # The NodeSet the first repository built is the one the second reuses.
    assert (
        len({id(node_set) for point in documents + chunks for node_set in point.belongs_to_set})
        == 1
    )


# --- joining docs.jsonl lines to the graph ------------------------------------


@pytest.mark.asyncio
async def test_include_modules_false_leaves_the_module_docs_out(shop, monkeypatch, log):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    documents, chunks = _split(await extract_code_docs(include_modules=False))

    assert [document.name for document in documents] == ["app.py", "orders/checkout.py"]
    assert [_metadata(chunk)["symbol_kind"] for chunk in chunks] == [
        "function",
        "class",
        "method",
        "function",
    ]
    assert "1 module doc(s) left out" in _summary(log, REPO)


@pytest.mark.asyncio
async def test_overloads_sharing_one_id_give_one_chunk_each_for_their_node(tmp_path, monkeypatch):
    root = tmp_path / REPO
    total = _symbol("Pricing.total", "src/Pricing.java", 10, kind="method", language="java")
    _snapshot(
        root,
        [
            doc_line(
                REPO,
                "symbol",
                "Pricing.total",
                "src/Pricing.java",
                "Total with a discount.\n\n@param discount the rate off",
                line=15,
            ),
            doc_line(
                REPO, "symbol", "Pricing.total", "src/Pricing.java", "Total of the lines.", line=10
            ),
        ],
    )
    _mock_engine(monkeypatch, [_repository(root), total])

    (document,), chunks = _split(await extract_code_docs())

    assert [chunk.text for chunk in chunks] == [
        "src/Pricing.java:10\nPricing.total (method)\n\nTotal of the lines.",
        (
            "src/Pricing.java:15\nPricing.total (method)\n\n"
            "Total with a discount.\n\n@param discount the rate off"
        ),
    ]
    assert [_metadata(chunk)["line"] for chunk in chunks] == [10, 15]
    assert {_metadata(chunk)["symbol_id"] for chunk in chunks} == {total[0]}
    assert {_metadata(chunk)["enola_id"] for chunk in chunks} == {_enola_id(total)}
    assert len({chunk.id for chunk in chunks}) == 2
    # Counted per docstring: each overload is a symbol of its own in the source.
    assert document.symbol_count == 2


@pytest.mark.asyncio
async def test_a_line_naming_a_fact_the_graph_does_not_hold_is_skipped_and_counted(
    tmp_path, monkeypatch, log
):
    root = tmp_path / REPO
    _snapshot(
        root,
        [
            doc_line(REPO, "symbol", "app.main", "app.py", "Main.", line=4),
            # No node for it.
            doc_line(REPO, "symbol", "app.removed", "app.py", "Removed since.", line=8),
            # Its node came from a fact without an id: enola_id None, never joined.
            doc_line(REPO, "symbol", "app.legacy", "app.py", "Legacy.", line=12),
            # No CodeModule node for it.
            doc_line(REPO, "module", "app", "app", "The app package."),
        ],
    )
    legacy = _symbol("app.legacy", "app.py", 12)
    legacy[1]["enola_id"] = None
    _mock_engine(monkeypatch, [_repository(root), _symbol("app.main", "app.py", 4), legacy])

    documents, chunks = _split(await extract_code_docs())

    assert [document.name for document in documents] == ["app.py"]
    assert [_metadata(chunk)["symbol"] for chunk in chunks] == ["app.main"]
    summary = _summary(log, REPO)
    assert "4 docs.jsonl line(s)" in summary
    assert "1 symbol and 0 module doc(s) joined" in summary
    assert "3 naming a fact the graph does not hold" in summary
    log.warning.assert_not_called()


@pytest.mark.asyncio
async def test_a_line_whose_kind_does_not_match_its_node_type_is_skipped(
    tmp_path, monkeypatch, log
):
    root = tmp_path / REPO
    symbol = _symbol("orders.place", "orders/api.py", 3)
    module = _module("orders", "orders")
    _snapshot(
        root,
        [
            # A module line naming the CodeSymbol's id, a symbol line the CodeModule's.
            doc_line(
                REPO,
                "module",
                "orders.place",
                "orders/api.py",
                "Not a module.",
                fact_id=_enola_id(symbol),
            ),
            doc_line(
                REPO,
                "symbol",
                "orders",
                "orders",
                "Not a symbol.",
                line=1,
                fact_id=_enola_id(module),
            ),
            doc_line(REPO, "symbol", "orders.place", "orders/api.py", "Place an order.", line=3),
        ],
    )
    _mock_engine(monkeypatch, [_repository(root), symbol, module])

    _, chunks = _split(await extract_code_docs())

    assert [chunk.text for chunk in chunks] == [
        "orders/api.py:3\norders.place (function)\n\nPlace an order."
    ]
    assert "2 naming a fact the graph does not hold" in _summary(log, REPO)


@pytest.mark.asyncio
async def test_a_line_without_a_repository_label_is_joined_by_its_id(tmp_path, monkeypatch):
    root = tmp_path / REPO
    main = _symbol("app.main", "app.py", 4)
    _snapshot(
        root,
        [
            doc_line(
                None, "symbol", "app.main", "app.py", "Unlabelled.", line=4, fact_id=_enola_id(main)
            )
        ],
    )
    _mock_engine(monkeypatch, [_repository(root), main])

    (chunk,) = _split(await extract_code_docs())[1]

    assert chunk.text == "app.py:4\napp.main (function)\n\nUnlabelled."
    assert _metadata(chunk)["repo"] == REPO


@pytest.mark.asyncio
async def test_a_multi_repo_snapshot_gives_each_repository_only_its_own_lines(
    tmp_path, monkeypatch, log
):
    """One enola run over several repositories writes one docs.jsonl, each line
    naming its repository, and the caller passes that directory for all of
    them. A line is read for the repository it names only, even when its id is
    one another repository's graph holds; a repository whose graph was built
    from another snapshot fails the id check and is left out."""
    shop_nodes = [
        _repository(tmp_path / "gone-shop"),
        _symbol("app.main", "app.py", 4),
        _symbol("app.serve", "app.py", 8),
    ]
    other_nodes = [
        _repository(tmp_path / "gone-other", OTHER),
        _symbol("lib.f", "lib.py", 1, repo=OTHER),
    ]
    legacy_nodes = [
        _repository(tmp_path / "gone-legacy", "legacy", snapshot_id="snapshot-0"),
        _symbol("old.g", "old.py", 1, repo="legacy"),
    ]
    workspace = write_enola_snapshot(
        tmp_path / "workspace" / ".enola",
        docs=[
            doc_line(REPO, "symbol", "app.main", "app.py", "Shop main.", line=4),
            doc_line(OTHER, "symbol", "lib.f", "lib.py", "Other f.", line=1),
            doc_line("legacy", "symbol", "old.g", "old.py", "Legacy g.", line=1),
            # Labelled with the other repository, carrying shop's app.serve id.
            doc_line(
                OTHER,
                "symbol",
                "app.serve",
                "app.py",
                "Not the shop's.",
                line=8,
                fact_id=_enola_id(shop_nodes[2]),
            ),
        ],
        snapshot_id=SNAPSHOT_ID,
    )
    _mock_engine(monkeypatch, shop_nodes + other_nodes + legacy_nodes)

    documents, chunks = _split(await extract_code_docs(snapshot_dir=workspace))

    assert [(document.repo, document.name) for document in documents] == [
        (OTHER, "lib.py"),
        (REPO, "app.py"),
    ]
    assert [chunk.text.split("\n\n", 1)[1] for chunk in chunks] == ["Other f.", "Shop main."]
    (warning,) = _warnings(log)
    assert "'legacy'" in warning
    assert "is snapshot-1, the code graph was built from snapshot-0" in warning
    # Another repository's lines are not the shop's to count.
    assert "1 docs.jsonl line(s)" in _summary(log, REPO)
    assert "2 docs.jsonl line(s)" in _summary(log, OTHER)


# --- which snapshot is read ---------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_dir_is_read_in_place_of_the_stored_path(tmp_path, monkeypatch):
    """The routes pass the snapshot they just loaded; ``<path>/.enola`` of the
    repository node is read only without one."""
    root = tmp_path / REPO
    _snapshot(root, [doc_line(REPO, "symbol", "app.main", "app.py", "From the clone.", line=4)])
    staging = _snapshot(
        tmp_path / "staging",
        [doc_line(REPO, "symbol", "app.main", "app.py", "From the staging snapshot.", line=4)],
    )
    main = _symbol("app.main", "app.py", 4)
    _mock_engine(monkeypatch, [_repository(root), main])

    for snapshot_dir in (staging, str(staging)):
        (chunk,) = _split(await extract_code_docs(snapshot_dir=snapshot_dir))[1]
        assert chunk.text.endswith("\n\nFrom the staging snapshot.")
    (chunk,) = _split(await extract_code_docs())[1]
    assert chunk.text.endswith("\n\nFrom the clone.")

    # With it, a stored path gone from disk (an earlier run's staging
    # directory) or no stored path at all is not needed.
    for repository in (_repository(tmp_path / "gone"), _repository(None)):
        _mock_engine(monkeypatch, [repository, main])
        (chunk,) = _split(await extract_code_docs(snapshot_dir=staging))[1]
        assert chunk.text.endswith("\n\nFrom the staging snapshot.")


@pytest.mark.asyncio
async def test_a_snapshot_without_a_receipt_is_matched_by_its_facts_hash(tmp_path, monkeypatch):
    """A historical writer left no receipt: nothing to vouch with, so the docs
    are read, and the snapshot's id is the hash of its facts.jsonl -- the
    value the code graph stamped on the repository node when it loaded it."""
    root = tmp_path / REPO
    main = _symbol("app.main", "app.py", 4)
    directory = write_enola_snapshot(
        root / ".enola",
        facts=[{"id": _enola_id(main), "kind": "symbol", "name": "app.main", "file": "app.py"}],
        docs=[doc_line(REPO, "symbol", "app.main", "app.py", "Main.", line=4)],
        receipt=False,
    )
    facts_hash = sha256_label((directory / "facts.jsonl").read_bytes())
    _mock_engine(monkeypatch, [_repository(root, snapshot_id=facts_hash), main])

    (chunk,) = _split(await extract_code_docs())[1]

    assert chunk.text == "app.py:4\napp.main (function)\n\nMain."


def _no_docs_file(root):
    """A snapshot from enola before 0.4.32, or without the docs renderer."""
    write_enola_snapshot(root / ".enola", docs=None, snapshot_id=SNAPSHOT_ID)
    return _repository(root, OTHER)


def _docs_file_changed_after_the_receipt(root):
    directory = _snapshot(root, OTHER_DOCS)
    with open(directory / DOCS_ARTIFACT, "a", encoding="utf-8") as docs_file:
        docs_file.write(json.dumps(doc_line(OTHER, "symbol", "lib.g", "lib.py", "G.", line=5)))
        docs_file.write("\n")
    return _repository(root, OTHER)


def _receipt_hashing_no_docs_file(root):
    """An earlier run's docs.jsonl next to a run whose receipt does not list it."""
    write_enola_snapshot(
        root / ".enola", docs=OTHER_DOCS, snapshot_id=SNAPSHOT_ID, vouch_docs=False
    )
    return _repository(root, OTHER)


def _receipt_without_hashes_or_the_docs_renderer(root):
    directory = _snapshot(root, OTHER_DOCS)
    (directory / "receipt.json").write_text(
        json.dumps({"format_version": 1, "snapshot_id": SNAPSHOT_ID, "renderers": ["flows"]}),
        encoding="utf-8",
    )
    return _repository(root, OTHER)


def _snapshot_of_another_run(root):
    """Another enola run wrote over the directory since the code graph was loaded."""
    write_enola_snapshot(root / ".enola", docs=OTHER_DOCS, snapshot_id="snapshot-2")
    return _repository(root, OTHER)


def _repository_without_a_snapshot_id(root):
    _snapshot(root, OTHER_DOCS)
    return _repository(root, OTHER, snapshot_id=None)


def _repository_without_a_path(root):
    _snapshot(root, OTHER_DOCS)
    return _repository(None, OTHER)


@pytest.mark.parametrize(
    ("other_repository", "reason"),
    [
        pytest.param(_no_docs_file, "no docs.jsonl in", id="no-docs-file"),
        pytest.param(
            _docs_file_changed_after_the_receipt,
            "does not list these docs.jsonl bytes",
            id="hash-mismatch",
        ),
        pytest.param(
            _receipt_hashing_no_docs_file,
            "does not list these docs.jsonl bytes",
            id="output-hashes-without-docs",
        ),
        pytest.param(
            _receipt_without_hashes_or_the_docs_renderer,
            "does not list these docs.jsonl bytes",
            id="renderers-without-docs",
        ),
        pytest.param(
            _snapshot_of_another_run,
            "is snapshot-2, the code graph was built from snapshot-1",
            id="other-snapshot-id",
        ),
        pytest.param(
            _repository_without_a_snapshot_id, "records no snapshot id", id="no-last-snapshot-id"
        ),
        pytest.param(_repository_without_a_path, "records no source directory", id="no-path"),
    ],
)
@pytest.mark.asyncio
async def test_a_repository_whose_snapshot_docs_are_unavailable_yields_nothing_and_warns(
    other_repository, reason, shop, tmp_path, monkeypatch, log
):
    repository = other_repository(tmp_path / OTHER)
    _mock_engine(
        monkeypatch, _shop_nodes(shop) + [repository, _symbol("lib.f", "lib.py", 1, repo=OTHER)]
    )

    documents, chunks = _split(await extract_code_docs())

    # "other" sorts first; the run still goes on to the shop.
    assert [document.repo for document in documents] == [REPO] * 3
    assert [_metadata(chunk)["repo"] for chunk in chunks] == [REPO] * 5
    (warning,) = _warnings(log)
    assert "'other'" in warning
    assert reason in warning
    assert await extract_code_docs(repo=OTHER) == []


@pytest.mark.asyncio
async def test_a_vouched_empty_docs_file_documents_nothing_without_a_warning(
    tmp_path, monkeypatch, log
):
    """Nothing documented, not docs unavailable: the snapshot was read."""
    root = tmp_path / REPO
    _snapshot(root, [])
    _mock_engine(monkeypatch, [_repository(root), _symbol("app.main", "app.py", 4)])

    assert await extract_code_docs() == []

    log.warning.assert_not_called()
    assert "0 docs.jsonl line(s)" in _summary(log, REPO)


# --- chunk size ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_doc_is_split_on_blank_lines(tmp_path, monkeypatch):
    paragraphs = [f"Paragraph {index} " + " ".join(["word"] * 10) + "." for index in range(4)]
    root = tmp_path / REPO
    _snapshot(
        root, [doc_line(REPO, "symbol", "long.long", "long.py", "\n\n".join(paragraphs), line=1)]
    )
    _mock_engine(monkeypatch, [_repository(root), _symbol("long.long", "long.py", 1)])
    header = "long.py:1\nlong.long (function)\n\n"

    # Within the budget: one chunk, the whole doc.
    (chunk,) = _split(await extract_code_docs())[1]
    assert chunk.text == header + "\n\n".join(paragraphs)

    # Header (3 words) + one paragraph (12 words) fit; two paragraphs do not.
    _, chunks = _split(await extract_code_docs(max_chunk_tokens=20))

    assert len(chunks) == 4
    assert [chunk.text[: len(header)] for chunk in chunks] == [header] * 4
    assert [chunk.text[len(header) :] for chunk in chunks] == paragraphs
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert all(chunk.max_chunk_tokens == 20 for chunk in chunks)
    assert all(chunk.chunk_size <= 20 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 4
    assert {_metadata(chunk)["symbol"] for chunk in chunks} == {"long.long"}

    # With room for two paragraphs per chunk they are packed, never split inside.
    _, chunks = _split(await extract_code_docs(max_chunk_tokens=30))
    assert [chunk.text[len(header) :] for chunk in chunks] == [
        "\n\n".join(paragraphs[:2]),
        "\n\n".join(paragraphs[2:]),
    ]


def test_chunk_texts_splits_inside_a_paragraph_only_when_it_alone_exceeds_the_budget():
    header = "f.py:1\nf (function)"
    short = "Short paragraph."
    long = " ".join(f"w{index}" for index in range(30))

    assert _chunk_texts(header, f"{short}\n\n{long}", 100, _words) == [
        f"{header}\n\n{short}\n\n{long}"
    ]

    # Header: 3 words, so 9 remain. The short paragraph stays whole; the long
    # one is cut into word runs of 9, none of which fits next to it.
    texts = _chunk_texts(header, f"{short}\n\n{long}", 12, _words)
    prefix = f"{header}\n\n"
    assert all(text.startswith(prefix) for text in texts)
    assert texts[0] == f"{prefix}{short}"
    assert " ".join(text[len(prefix) :] for text in texts[1:]) == long
    assert [_words(text) - 3 for text in texts] == [2, 9, 9, 9, 3]


# --- selection and the graph read ---------------------------------------------


@pytest.mark.asyncio
async def test_repo_filter(shop, tmp_path, monkeypatch, log):
    _mock_engine(monkeypatch, _shop_nodes(shop) + _other_repo(tmp_path))

    documents, chunks = _split(await extract_code_docs(repo=OTHER))
    assert [(document.repo, document.name) for document in documents] == [(OTHER, "lib.py")]
    assert [_metadata(chunk)["repo"] for chunk in chunks] == [OTHER]

    data_points = await extract_code_docs()
    documents, _ = _split(data_points)
    assert [(document.repo, document.name) for document in documents] == [
        (OTHER, "lib.py"),
        (REPO, "app.py"),
        (REPO, "orders"),
        (REPO, "orders/checkout.py"),
    ]
    # Every repository's documents, then every repository's chunks.
    assert [type(point) for point in data_points] == [CodeDocument] * 4 + [DocumentChunk] * 6

    assert await extract_code_docs(repo="nope") == []

    # A repository the filter leaves out is not read: its snapshot going
    # missing warns of nothing.
    (shop / ".enola" / DOCS_ARTIFACT).unlink()
    log.reset_mock()
    assert len(_split(await extract_code_docs(repo=OTHER))[1]) == 1
    log.warning.assert_not_called()


@pytest.mark.asyncio
async def test_no_repository_in_the_graph_yields_nothing(shop, monkeypatch):
    _mock_engine(monkeypatch, [])
    assert await extract_code_docs() == []

    # Code nodes and their snapshot, but no CodeRepository node to say which
    # snapshot the graph holds.
    _mock_engine(monkeypatch, _shop_nodes(shop)[1:])
    assert await extract_code_docs() == []


@pytest.mark.asyncio
async def test_the_batch_names_the_repositories_whose_docs_were_read(shop, tmp_path, monkeypatch):
    """The result is a CodeDocsBatch whose read_repos is the extraction's own
    decision, the one the link task's sweep follows: a repository read with
    documented facts, one read with none (a vouched, empty docs.jsonl), and not
    one whose snapshot docs were unavailable. Nothing selected: nothing read."""
    empty = tmp_path / "empty"
    write_enola_snapshot(empty / ".enola", docs=[], snapshot_id=SNAPSHOT_ID)
    nodes = [*_shop_nodes(shop), _repository(empty, "empty"), _repository(tmp_path / "gone", OTHER)]
    _mock_engine(monkeypatch, nodes)

    result = await extract_code_docs()

    assert isinstance(result, CodeDocsBatch)
    assert result.read_repos == frozenset({REPO, "empty"})
    assert {point.repo for point in result if isinstance(point, CodeDocument)} == {REPO}

    _mock_engine(monkeypatch, nodes)
    assert (await extract_code_docs(repo=OTHER)).read_repos == frozenset()
    _mock_engine(monkeypatch, [])
    assert (await extract_code_docs()).read_repos == frozenset()


@pytest.mark.asyncio
async def test_data_argument_is_ignored(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    baseline = [point.id for point in await extract_code_docs()]

    assert baseline
    for data in ([{}], {}, None, ["memory fragment"], "text"):
        assert [point.id for point in await extract_code_docs(data)] == baseline
    assert [point.id for point in await extract_code_docs(data=[{}], ctx=None)] == baseline


@pytest.mark.asyncio
async def test_the_graph_is_read_once_through_the_filtered_query(shop, monkeypatch):
    engine = _mock_engine(monkeypatch, _shop_nodes(shop))

    await extract_code_docs()

    # Stored CodeDocuments are not read: the snapshot alone decides what a
    # repository yields.
    engine.get_filtered_graph_data.assert_awaited_once_with(
        [{"type": ["CodeRepository", "CodeSymbol", "CodeModule"]}]
    )
    # One read and nothing else: the extraction writes nothing.
    assert [name for name, _args, _kwargs in engine.mock_calls] == ["get_filtered_graph_data"]


@pytest.mark.asyncio
async def test_repositories_are_extracted_in_a_worker_thread_one_at_a_time(
    shop, tmp_path, monkeypatch
):
    shop_nodes = _shop_nodes(shop)
    _mock_engine(monkeypatch, shop_nodes + _other_repo(tmp_path))
    entered, release = threading.Event(), threading.Event()
    threads, in_flight, started_beside = [], [], []
    real_extract = extract_module._extract_repository

    def blocking_extract(*args):
        # Holds the extraction until the test lets it go; on the event loop's
        # thread this would freeze the loop and time out below.
        threads.append(threading.current_thread())
        started_beside.append(list(in_flight))
        in_flight.append(args[0])
        entered.set()
        try:
            assert release.wait(5), "the extraction was never released"
            return real_extract(*args)
        finally:
            in_flight.remove(args[0])

    monkeypatch.setattr(extract_module, "_extract_repository", blocking_extract)
    to_thread = AsyncMock(side_effect=asyncio.to_thread)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    task = asyncio.create_task(extract_code_docs())
    loop = asyncio.get_running_loop()
    assert await loop.run_in_executor(None, entered.wait, 5)
    # The extraction thread is parked; the loop still runs other coroutines.
    ticks = 0
    for _ in range(3):
        await asyncio.sleep(0)
        ticks += 1
    assert ticks == 3 and not task.done()
    release.set()
    data_points = await task

    assert threads and all(thread is not threading.main_thread() for thread in threads)
    # One hand-off per repository, in order, none started beside another.
    calls = to_thread.await_args_list
    assert [call.args[:2] for call in calls] == [
        (blocking_extract, OTHER),
        (blocking_extract, REPO),
    ]
    assert started_beside == [[], []]
    # (repo, repository, snapshot_dir, code_nodes, include_modules, budget,
    # count_tokens, known_node_sets)
    _, _, repository, snapshot_dir, code_nodes, include_modules, budget, count_tokens, _ = calls[
        1
    ].args
    assert repository["name"] == REPO and repository["last_snapshot_id"] == SNAPSHOT_ID
    assert snapshot_dir is None
    # The shop's code nodes, by enola id.
    assert code_nodes == {_enola_id(node): node for node in shop_nodes[1:]}
    assert (include_modules, budget, count_tokens) == (True, BUDGET, _words)
    assert [document.repo for document in _split(data_points)[0]] == [OTHER, REPO, REPO, REPO]


@pytest.mark.asyncio
async def test_json_in_string_and_nested_properties_shapes_are_read(shop, monkeypatch):
    """Adapters return a node's properties flat or nested under ``properties``
    (that nesting JSON text on some), with JSON-in-string values and UUID ids."""
    repo_id, repo_properties = _repository(shop)
    _, symbol_properties = _symbol("app.main", "app.py", 4)
    symbol_properties["fact_properties"] = json.dumps(symbol_properties["fact_properties"])
    symbol_properties["belongs_to_set"] = json.dumps([{"name": "code"}])
    symbol_properties["end_line"] = "6"
    _, module_properties = _module("orders", "orders")
    module_properties["fact_properties"] = json.dumps(module_properties["fact_properties"])
    module_properties["source_node_set"] = "code, docs"
    symbol_id = fact_node_id(REPO, "symbol", "app.main")
    module_id = fact_node_id(REPO, "module", "orders")
    _mock_engine(
        monkeypatch,
        [
            (repo_id, {"properties": json.dumps(repo_properties)}),
            (symbol_id, {"properties": symbol_properties}),
            (module_id, module_properties),
        ],
    )

    documents, chunks = _split(await extract_code_docs())

    assert [document.name for document in documents] == ["app.py", "orders"]
    main, orders = chunks
    assert _metadata(main)["symbol_id"] == str(symbol_id)
    assert _metadata(main)["end_line"] == 6
    assert _metadata(main)["language"] == "python"
    assert [node_set.name for node_set in main.belongs_to_set] == ["code"]
    assert _metadata(orders)["module_id"] == str(module_id)
    assert _metadata(orders)["language"] == "python"
    assert [node_set.name for node_set in orders.belongs_to_set] == ["code", "docs"]
