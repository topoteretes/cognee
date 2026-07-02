"""Unit tests for the code graph pipeline (issue #3668).

These tests are pure-Python and never call add_data_points or any DB;
they validate only the AST parser output.
"""

import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest

# Stub broken transitive dep so cognee modules can be imported (same fix as
# test_temporal_awareness.py — see that file for the explanation).
if "mistralai" not in sys.modules:
    _spec = importlib.util.spec_from_loader("mistralai", loader=None)
    _m = types.ModuleType("mistralai")
    _m.__spec__ = _spec
    _m.Mistral = object
    sys.modules["mistralai"] = _m

# Skip entire module if tree-sitter extras are not installed
pytest.importorskip("tree_sitter", reason="codegraph extra (tree-sitter) not installed")
pytest.importorskip("tree_sitter_python", reason="codegraph extra (tree-sitter-python) not installed")

from cognee.tasks.codegraph.models import CodeClass, CodeFile, CodeFunction, CodeImport
from cognee.tasks.codegraph.parse import parse_file


@pytest.fixture()
def py_file(tmp_path: Path) -> Path:
    src = textwrap.dedent("""\
        import os
        from pathlib import Path

        class Greeter:
            \"\"\"A simple greeter.\"\"\"

            def greet(self, name: str) -> str:
                \"\"\"Return a greeting.\"\"\"
                return f"Hello, {name}"

        def standalone(x: int) -> int:
            return x * 2
    """)
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    return f


def _by_type(nodes, cls):
    return [n for n in nodes if isinstance(n, cls)]


def test_parse_returns_code_file(py_file):
    nodes, _ = parse_file(str(py_file))
    files = _by_type(nodes, CodeFile)
    assert len(files) == 1
    assert files[0].file_path == str(py_file)


def test_parse_detects_class(py_file):
    nodes, _ = parse_file(str(py_file))
    classes = _by_type(nodes, CodeClass)
    assert any(c.name == "Greeter" for c in classes)


def test_parse_detects_standalone_function(py_file):
    nodes, _ = parse_file(str(py_file))
    fns = _by_type(nodes, CodeFunction)
    assert any(f.name == "standalone" for f in fns)


def test_parse_qualifies_method_with_class(py_file):
    nodes, _ = parse_file(str(py_file))
    fns = _by_type(nodes, CodeFunction)
    assert any(f.name == "Greeter.greet" for f in fns)


def test_parse_detects_imports(py_file):
    nodes, _ = parse_file(str(py_file))
    imports = _by_type(nodes, CodeImport)
    modules = {i.imported_module for i in imports}
    assert "os" in modules
    assert "pathlib" in modules


def test_parse_emits_defines_class_edge(py_file):
    nodes, edges = parse_file(str(py_file))
    file_node = _by_type(nodes, CodeFile)[0]
    class_node = next(c for c in _by_type(nodes, CodeClass) if c.name == "Greeter")
    edge_relations = {(str(e[0]), e[2]) for e in edges}
    assert (str(file_node.id), "defines_class") in edge_relations


def test_parse_emits_has_method_edge(py_file):
    nodes, edges = parse_file(str(py_file))
    class_node = next(c for c in _by_type(nodes, CodeClass) if c.name == "Greeter")
    method_node = next(f for f in _by_type(nodes, CodeFunction) if f.name == "Greeter.greet")
    edge_relations = {(str(e[0]), e[2]) for e in edges}
    assert (str(class_node.id), "has_method") in edge_relations


def test_stable_ids_across_parses(py_file):
    nodes1, _ = parse_file(str(py_file))
    nodes2, _ = parse_file(str(py_file))
    ids1 = {str(n.id) for n in nodes1}
    ids2 = {str(n.id) for n in nodes2}
    assert ids1 == ids2, "Node IDs must be deterministic across runs"


def test_missing_file_returns_empty():
    nodes, edges = parse_file("/nonexistent/file.py")
    assert nodes == []
    assert edges == []


def test_docstring_extracted(py_file):
    nodes, _ = parse_file(str(py_file))
    greeter = next(c for c in _by_type(nodes, CodeClass) if c.name == "Greeter")
    assert greeter.docstring and "greeter" in greeter.docstring.lower()
