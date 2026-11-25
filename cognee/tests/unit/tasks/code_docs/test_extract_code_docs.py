"""The code docs extraction task over a mocked code graph (cognee/tasks/code_docs)."""

import importlib
import json
from unittest.mock import AsyncMock

import pytest

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.code_docs import DESCRIBES_RELATIONSHIP, code_document_id, extract_code_docs
from cognee.tasks.code_docs.extract_code_docs import _chunk_texts
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.documents.classify_documents import update_node_set

extract_module = importlib.import_module("cognee.tasks.code_docs.extract_code_docs")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

REPO = "shop"
BUDGET = 64

APP_PY = '''"""The shop application."""


def main():
    """Start the shop's HTTP server."""


def serve():
    return None
'''

CHECKOUT_PY = '''import functools


class Order:
    """An order."""

    @functools.cache
    def total(self) -> int:
        """Sum of the lines."""
        return 0


def place_order(order: Order, *, notify: bool = True) -> str:
    """Place an order.

    Second paragraph.
    """
    return "ok"
'''

METADATA_KEYS = {
    "repo",
    "file",
    "line",
    "end_line",
    "symbol",
    "symbol_id",
    "symbol_kind",
    "language",
    "doc_source",
}


def _words(text: str) -> int:
    return len(text.split())


@pytest.fixture(autouse=True)
def _offline_token_budget(monkeypatch):
    """No embedding engine in unit tests: count words, fixed budget."""
    monkeypatch.setattr(extract_module, "_token_counter", lambda: _words)
    monkeypatch.setattr(extract_module, "_default_chunk_budget", AsyncMock(return_value=BUDGET))


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = (list(nodes), list(edges))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


def _repository(path, repo=REPO):
    return (
        str(fact_node_id(repo, "repository", repo)),
        {"type": "CodeRepository", "name": repo, "path": str(path)},
    )


def _module(name, repo=REPO):
    return (
        str(fact_node_id(repo, "module", name)),
        {"type": "CodeModule", "name": name, "repo": repo, "file_path": name, "kind": "module"},
    )


def _symbol(name, file_path, line, *, kind="function", doc=None, node_set=None, repo=REPO):
    properties = {
        "type": "CodeSymbol",
        "name": name,
        "kind": "symbol",
        "repo": repo,
        "file_path": file_path,
        "line": line,
        "symbol_kind": kind,
        "fact_properties": {"language": "python", "symbol_kind": kind},
    }
    if doc is not None:
        properties["doc"] = doc
        properties["fact_properties"]["doc"] = doc
    if node_set:
        properties["belongs_to_set"] = list(node_set)
        properties["source_node_set"] = ", ".join(node_set)
    return str(fact_node_id(repo, "symbol", name)), properties


def _write_repo(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _shop_nodes(root, node_set=("code",), repo=REPO):
    return [
        _repository(root, repo),
        _module(".", repo),
        _module("orders", repo),
        _symbol("app.main", "app.py", 4, node_set=node_set, repo=repo),
        _symbol("app.serve", "app.py", 8, node_set=node_set, repo=repo),
        _symbol("orders/checkout.Order", "orders/checkout.py", 4, kind="class", node_set=node_set),
        # Enola reports the decorator's line for the method.
        _symbol(
            "orders/checkout.Order.total",
            "orders/checkout.py",
            7,
            kind="method",
            node_set=node_set,
            repo=repo,
        ),
        _symbol("orders/checkout.place_order", "orders/checkout.py", 13, node_set=node_set),
    ]


@pytest.fixture
def shop(tmp_path):
    return _write_repo(tmp_path / "shop", {"app.py": APP_PY, "orders/checkout.py": CHECKOUT_PY})


def _split(data_points):
    documents = [point for point in data_points if isinstance(point, CodeDocument)]
    chunks = [point for point in data_points if isinstance(point, DocumentChunk)]
    assert len(documents) + len(chunks) == len(data_points)
    return documents, chunks


def _metadata(chunk):
    return json.loads(chunk.external_metadata)["code_doc"]


def _chunk_for(chunks, symbol):
    return next(chunk for chunk in chunks if _metadata(chunk)["symbol"] == symbol)


@pytest.mark.asyncio
async def test_chunk_text_starts_with_file_line_name_and_signature(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    _, chunks = _split(await extract_code_docs())

    total = _chunk_for(chunks, "orders/checkout.Order.total")
    # The def line from the source, although Enola reported the decorator's.
    assert total.text == (
        "orders/checkout.py:8\n"
        "orders/checkout.Order.total (method)\n"
        "def total(self) -> int:\n"
        "\n"
        "Sum of the lines."
    )
    assert _chunk_for(chunks, "orders/checkout.place_order").text == (
        "orders/checkout.py:13\n"
        "orders/checkout.place_order (function)\n"
        "def place_order(order: Order, *, notify: bool = True) -> str:\n"
        "\n"
        "Place an order.\n"
        "\n"
        "Second paragraph."
    )
    assert _chunk_for(chunks, "orders/checkout.Order").text.startswith(
        "orders/checkout.py:4\norders/checkout.Order (class)\nclass Order:\n\nAn order."
    )
    assert _chunk_for(chunks, "app").text == "app.py:1\napp (module)\n\nThe shop application."


@pytest.mark.asyncio
async def test_documents_first_then_chunks_in_source_order(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    data_points = await extract_code_docs()
    documents, chunks = _split(data_points)

    assert [type(point) for point in data_points[:2]] == [CodeDocument, CodeDocument]
    assert [document.name for document in documents] == ["app.py", "orders/checkout.py"]
    assert [
        (chunk.document_name, chunk.chunk_index, _metadata(chunk)["symbol"]) for chunk in chunks
    ] == [
        ("app.py", 0, "app"),
        ("app.py", 1, "app.main"),
        ("orders/checkout.py", 0, "orders/checkout.Order"),
        ("orders/checkout.py", 1, "orders/checkout.Order.total"),
        ("orders/checkout.py", 2, "orders/checkout.place_order"),
    ]
    # app.serve has no docstring and no Enola doc: skipped.
    assert all(_metadata(chunk)["symbol"] != "app.serve" for chunk in chunks)

    app, checkout = documents
    assert app.id == code_document_id(REPO, "app.py")
    assert app.repo == REPO and app.file_path == "app.py" and app.language == "python"
    assert app.symbol_count == 1  # the module docstring is not a symbol
    assert checkout.symbol_count == 3
    assert app.raw_data_location == str(shop / "app.py")
    assert app.mime_type == "text/plain"
    assert json.loads(app.external_metadata) == {
        "code_doc": {"repo": REPO, "file": "app.py", "language": "python"}
    }


@pytest.mark.asyncio
async def test_chunk_fields_and_deterministic_ids_across_runs(shop, monkeypatch):
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
        assert chunk.text.startswith(f"{chunk.document_name}:")


@pytest.mark.asyncio
async def test_external_metadata_keys_and_values(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    _, chunks = _split(await extract_code_docs())

    total = _metadata(_chunk_for(chunks, "orders/checkout.Order.total"))
    assert set(total) == METADATA_KEYS
    assert total == {
        "repo": REPO,
        "file": "orders/checkout.py",
        "line": 8,
        "end_line": 10,
        "symbol": "orders/checkout.Order.total",
        "symbol_id": str(fact_node_id(REPO, "symbol", "orders/checkout.Order.total")),
        "symbol_kind": "method",
        "language": "python",
        "doc_source": "source",
    }

    module = _metadata(_chunk_for(chunks, "app"))
    assert set(module) == METADATA_KEYS | {"module_id"}
    assert module["symbol_id"] is None
    assert module["symbol_kind"] == "module"
    assert module["line"] == 1
    assert module["module_id"] == str(fact_node_id(REPO, "module", "."))
    assert DESCRIBES_RELATIONSHIP == "describes"


@pytest.mark.asyncio
async def test_module_docstring_links_only_an_existing_module_node(shop, monkeypatch):
    nodes = [node for node in _shop_nodes(shop) if node[1].get("name") != "."]
    _mock_engine(monkeypatch, nodes)

    _, chunks = _split(await extract_code_docs())

    assert _metadata(_chunk_for(chunks, "app"))["module_id"] is None

    _mock_engine(monkeypatch, _shop_nodes(shop))
    documents, chunks = _split(await extract_code_docs(include_modules=False))
    assert all(_metadata(chunk)["symbol_kind"] != "module" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks if chunk.document_name == "app.py"] == [0]
    assert documents[0].symbol_count == 1


@pytest.mark.asyncio
async def test_node_sets_match_classify_documents_ids(shop, monkeypatch):
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
async def test_symbols_without_node_sets_carry_none(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop, node_set=None))

    documents, chunks = _split(await extract_code_docs())

    assert all(point.belongs_to_set is None for point in documents + chunks)
    assert all(point.source_node_set is None for point in documents + chunks)


@pytest.mark.asyncio
async def test_enola_doc_fallback_when_the_source_is_missing(tmp_path, monkeypatch):
    nodes = [
        _repository(tmp_path / "gone"),
        _symbol("app.main", "app.py", 4, doc="Start the shop's HTTP server."),
        _symbol("app.serve", "app.py", 8),
    ]
    _mock_engine(monkeypatch, nodes)

    documents, chunks = _split(await extract_code_docs())

    assert [document.raw_data_location for document in documents] == ["app.py"]
    assert len(chunks) == 1
    assert chunks[0].text == "app.py:4\napp.main (function)\n\nStart the shop's HTTP server."
    metadata = _metadata(chunks[0])
    assert metadata["doc_source"] == "enola"
    assert metadata["line"] == 4 and metadata["end_line"] is None
    assert metadata["symbol_id"] == str(fact_node_id(REPO, "symbol", "app.main"))


@pytest.mark.asyncio
async def test_enola_doc_fallback_for_unreadable_or_foreign_files(tmp_path, monkeypatch):
    root = _write_repo(
        tmp_path / "shop",
        {"broken.py": "def broken(:\n", "main.go": "package main\n"},
    )
    nodes = [
        _repository(root),
        _symbol("broken.thing", "broken.py", 1, doc="Enola read this."),
        _symbol("main.main", "main.go", 3, doc="Go entry point."),
        _symbol("missing.gone", "missing.py", 2, doc="File was deleted."),
        _symbol("missing.silent", "missing.py", 9),
    ]
    nodes[2][1]["fact_properties"]["language"] = "go"
    _mock_engine(monkeypatch, nodes)

    documents, chunks = _split(await extract_code_docs())

    assert [document.name for document in documents] == ["broken.py", "main.go", "missing.py"]
    assert {document.name: document.language for document in documents} == {
        "broken.py": "python",
        "main.go": "go",
        "missing.py": "python",
    }
    assert [_metadata(chunk)["doc_source"] for chunk in chunks] == ["enola"] * 3
    assert (
        _chunk_for(chunks, "main.main").text == "main.go:3\nmain.main (function)\n\nGo entry point."
    )
    # Not read from source: relative location, no signature line.
    assert documents[1].raw_data_location == "main.go"


@pytest.mark.asyncio
async def test_repo_filter(shop, tmp_path, monkeypatch):
    other = _write_repo(tmp_path / "other", {"lib.py": 'def f():\n    """Other f."""\n'})
    nodes = _shop_nodes(shop) + [
        _repository(other, "other"),
        _symbol("lib.f", "lib.py", 1, repo="other"),
    ]
    _mock_engine(monkeypatch, nodes)

    documents, chunks = _split(await extract_code_docs(repo="other"))
    assert [(document.repo, document.name) for document in documents] == [("other", "lib.py")]
    assert [_metadata(chunk)["repo"] for chunk in chunks] == ["other"]

    documents, _ = _split(await extract_code_docs())
    assert [(document.repo, document.name) for document in documents] == [
        ("other", "lib.py"),
        (REPO, "app.py"),
        (REPO, "orders/checkout.py"),
    ]

    assert await extract_code_docs(repo="nope") == []


@pytest.mark.asyncio
async def test_no_code_graph_yields_nothing(monkeypatch):
    _mock_engine(monkeypatch, [])
    assert await extract_code_docs() == []


@pytest.mark.asyncio
async def test_data_argument_is_ignored(shop, monkeypatch):
    _mock_engine(monkeypatch, _shop_nodes(shop))

    baseline = await extract_code_docs()
    for data in ([{}], {}, None, ["memory fragment"], "text"):
        assert [point.id for point in await extract_code_docs(data)] == [p.id for p in baseline]
    assert [point.id for point in await extract_code_docs(data=[{}], ctx=None)] == [
        point.id for point in baseline
    ]


@pytest.mark.asyncio
async def test_long_docstring_is_split_on_blank_lines(tmp_path, monkeypatch):
    paragraphs = [f"Paragraph {index} " + " ".join(["word"] * 10) + "." for index in range(4)]
    source = 'def long():\n    """' + "\n\n    ".join(paragraphs) + '\n    """\n'
    root = _write_repo(tmp_path / "shop", {"long.py": source})
    _mock_engine(monkeypatch, [_repository(root), _symbol("long.long", "long.py", 1)])

    # Header (4 words) + one paragraph (12 words) fit; two paragraphs do not.
    _, chunks = _split(await extract_code_docs(max_chunk_tokens=20))

    assert len(chunks) == 4
    header = "long.py:1\nlong.long (function)\ndef long():\n\n"
    assert [chunk.text[: len(header)] for chunk in chunks] == [header] * 4
    assert [chunk.text[len(header) :] for chunk in chunks] == paragraphs
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert all(chunk.max_chunk_tokens == 20 for chunk in chunks)
    assert all(chunk.chunk_size <= 20 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 4

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


@pytest.mark.asyncio
async def test_json_in_string_properties_and_nested_shapes_are_read(shop, monkeypatch):
    repo_id, repo_props = _repository(shop)
    symbol_id, symbol_props = _symbol("app.main", "app.py", 4, node_set=("code",))
    symbol_props["fact_properties"] = json.dumps(symbol_props["fact_properties"])
    symbol_props["belongs_to_set"] = json.dumps(symbol_props["belongs_to_set"])
    symbol_props["line"] = "4"
    nodes = [
        (repo_id, {"properties": json.dumps(repo_props)}),
        (symbol_id, {"properties": symbol_props}),
    ]
    _mock_engine(monkeypatch, nodes)

    documents, chunks = _split(await extract_code_docs(include_modules=False))

    assert [document.name for document in documents] == ["app.py"]
    assert [node_set.name for node_set in chunks[0].belongs_to_set] == ["code"]
    assert _metadata(chunks[0])["symbol_id"] == symbol_id
    assert chunks[0].text.startswith("app.py:4\napp.main (function)\ndef main():\n\n")


@pytest.mark.asyncio
async def test_stale_line_resolves_by_name_and_overloads_by_their_docstring(tmp_path, monkeypatch):
    source = (
        "from typing import overload\n"
        "\n"
        "\n"
        "def moved():\n"
        '    """Moved here."""\n'
        "\n"
        "\n"
        "def target():\n"
        '    """The target."""\n'
        "\n"
        "\n"
        "@overload\n"
        "def convert(value: int) -> int: ...\n"
        "\n"
        "\n"
        "def convert(value):\n"
        '    """Convert a value."""\n'
    )
    root = _write_repo(tmp_path / "shop", {"mod.py": source})
    nodes = [
        _repository(root),
        # The snapshot's line points at ``moved`` now; the name still resolves.
        _symbol("mod.target", "mod.py", 4),
        # Enola reports the overload stub's line.
        _symbol("mod.convert", "mod.py", 13),
    ]
    _mock_engine(monkeypatch, nodes)

    _, chunks = _split(await extract_code_docs())

    assert (
        _chunk_for(chunks, "mod.target").text
        == "mod.py:8\nmod.target (function)\ndef target():\n\nThe target."
    )
    assert _chunk_for(chunks, "mod.convert").text == (
        "mod.py:16\nmod.convert (function)\ndef convert(value):\n\nConvert a value."
    )


@pytest.mark.asyncio
async def test_stale_line_of_a_symbol_gone_from_the_source_yields_no_chunk(tmp_path, monkeypatch):
    root = _write_repo(tmp_path / "shop", {"mod.py": 'def moved():\n    """Moved here."""\n'})
    nodes = [
        _repository(root),
        # The snapshot's line now holds ``moved`` and neither name is in the file.
        _symbol("mod.gone", "mod.py", 1),
        _symbol("mod.also_gone", "mod.py", 1, doc="What enola had."),
    ]
    _mock_engine(monkeypatch, nodes)

    _, chunks = _split(await extract_code_docs())

    # Not ``moved``'s docstring under another symbol's header; enola's doc when there is one.
    assert [(_metadata(chunk)["symbol"], _metadata(chunk)["doc_source"]) for chunk in chunks] == [
        ("mod.also_gone", "enola")
    ]
    assert chunks[0].text == "mod.py:1\nmod.also_gone (function)\n\nWhat enola had."


@pytest.mark.asyncio
async def test_graph_is_read_through_the_filtered_query(shop, monkeypatch):
    engine = _mock_engine(monkeypatch, _shop_nodes(shop))

    await extract_code_docs()

    engine.get_filtered_graph_data.assert_awaited_once_with(
        [{"type": ["CodeRepository", "CodeSymbol", "CodeModule"]}]
    )
    engine.add_nodes.assert_not_awaited()
    engine.add_edges.assert_not_awaited()
