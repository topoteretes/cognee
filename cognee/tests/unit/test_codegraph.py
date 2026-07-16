"""Unit tests for the code graph pipeline (issue #3668).

Covers the tree-sitter parser output and an offline (no-LLM, no-DB) run of
``ingest_code_graph`` that asserts the assembled graph and its native edges.
"""

import textwrap
from pathlib import Path

import pytest

# Skip the whole module if the codegraph extra (tree-sitter) is not installed.
pytest.importorskip("tree_sitter", reason="codegraph extra (tree-sitter) not installed")
pytest.importorskip(
    "tree_sitter_python", reason="codegraph extra (tree-sitter-python) not installed"
)

from cognee.modules.graph.utils import get_graph_from_model
from cognee.shared.CodeGraphEntities import (
    ClassDefinition,
    CodeFile,
    FunctionDefinition,
    ImportStatement,
)
from cognee.tasks.codegraph import ingest_code_graph, parse_file


SAMPLE = textwrap.dedent(
    '''\
    import os
    from pathlib import Path
    from . import config

    class Greeter:
        """A simple greeter."""

        def greet(self, name: str) -> str:
            return f"Hello, {name}"

    def standalone(x: int) -> int:
        return x * 2
    '''
)


@pytest.fixture()
def py_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE, encoding="utf-8")
    return f


def test_parse_returns_code_file(py_file):
    code_file = parse_file(str(py_file))
    assert isinstance(code_file, CodeFile)
    assert code_file.file_path == str(py_file)
    assert code_file.language == "python"


def test_parse_detects_class(py_file):
    classes = parse_file(str(py_file)).provides_class_definition
    assert any(c.name == "Greeter" for c in classes)


def test_parse_detects_standalone_function(py_file):
    functions = parse_file(str(py_file)).provides_function_definition
    assert any(f.name == "standalone" for f in functions)


def test_parse_qualifies_method_with_class(py_file):
    functions = parse_file(str(py_file)).provides_function_definition
    assert any(f.name == "Greeter.greet" for f in functions)


def test_parse_captures_imports_including_relative(py_file):
    modules = {i.module for i in parse_file(str(py_file)).depends_on}
    # "." is the relative "from . import config" — the module, not the symbol.
    assert {"os", "pathlib", "."} <= modules


def test_stable_ids_across_parses(py_file):
    def ids(code_file):
        return {code_file.id} | {
            n.id
            for n in (
                *code_file.depends_on,
                *code_file.provides_class_definition,
                *code_file.provides_function_definition,
            )
        }

    assert ids(parse_file(str(py_file))) == ids(parse_file(str(py_file)))


def test_missing_file_returns_none():
    assert parse_file("/nonexistent/file.py") is None


@pytest.mark.asyncio
async def test_graph_from_model_emits_native_edges(py_file):
    """The assembled CodeFile yields File->Class/Function/Import edges with no manual wiring."""
    code_file = parse_file(str(py_file))
    _, edges = await get_graph_from_model(
        code_file, added_nodes={}, added_edges={}, visited_properties={}
    )
    relationships = {edge[2] for edge in edges}
    assert {
        "provides_class_definition",
        "provides_function_definition",
        "depends_on",
    } <= relationships


@pytest.mark.asyncio
async def test_ingest_code_graph_stores_parsed_nodes(py_file, monkeypatch):
    """ingest_code_graph parses a tree and persists it via a single add_data_points call."""
    captured = {}

    async def fake_add_data_points(data_points, *args, **kwargs):
        captured["data_points"] = data_points
        return data_points

    monkeypatch.setattr("cognee.tasks.codegraph.pipeline.add_data_points", fake_add_data_points)

    stored = await ingest_code_graph(py_file.parent)

    assert len(captured["data_points"]) == 1
    code_file = captured["data_points"][0]
    assert isinstance(code_file, CodeFile)
    assert any(isinstance(c, ClassDefinition) for c in code_file.provides_class_definition)
    assert any(isinstance(f, FunctionDefinition) for f in code_file.provides_function_definition)
    assert any(isinstance(i, ImportStatement) for i in code_file.depends_on)
    assert code_file.part_of is not None  # directory ingestion attaches a Repository
    assert stored == captured["data_points"]
