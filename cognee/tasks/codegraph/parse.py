"""AST-based code parser for the code graph pipeline.

Parses Python source files using tree-sitter (codegraph extra).
Returns DataPoint instances ready for add_data_points(), with no LLM calls.
"""

from pathlib import Path
from typing import Optional

from .models import CodeClass, CodeFile, CodeFunction, CodeImport

try:
    import tree_sitter_python as _tspy
    from tree_sitter import Language, Parser

    _PY_LANGUAGE = Language(_tspy.language())
    _PY_PARSER = Parser(_PY_LANGUAGE)
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first_child_of_type(node, kind: str):
    for child in node.children:
        if child.type == kind:
            return child
    return None


def _extract_docstring(body_node, src: bytes) -> Optional[str]:
    """Return the docstring from a function/class body if present."""
    if not body_node:
        return None
    for child in body_node.children:
        if child.type == "expression_statement":
            inner = _first_child_of_type(child, "string")
            if inner:
                raw = _text(inner, src).strip("\"' \t\n").strip()
                return raw[:500] if raw else None
    return None


def _make_edge(source_id, target_id, relation: str) -> tuple:
    return (
        source_id,
        target_id,
        relation,
        {
            "relationship_name": relation,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "ontology_valid": False,
        },
    )


def _walk(node, src: bytes, file_path: str, file_node: CodeFile, nodes: list, edges: list, parent_class: Optional[CodeClass] = None) -> None:
    """Recursively walk the AST and collect DataPoints and edges."""
    if node.type == "class_definition":
        name_node = _first_child_of_type(node, "identifier")
        if name_node:
            cls_name = _text(name_node, src)
            body = _first_child_of_type(node, "block")
            cls = CodeClass(
                name=cls_name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_extract_docstring(body, src),
            )
            nodes.append(cls)
            edges.append(_make_edge(file_node.id, cls.id, "defines_class"))
            # recurse into class body looking for methods
            if body:
                for child in body.children:
                    _walk(child, src, file_path, file_node, nodes, edges, parent_class=cls)
        return  # don't fall through to generic children walk

    if node.type == "function_definition":
        name_node = _first_child_of_type(node, "identifier")
        if name_node:
            fn_name = _text(name_node, src)
            # Qualify methods with class name to avoid collisions
            qualified = f"{parent_class.name}.{fn_name}" if parent_class else fn_name
            body = _first_child_of_type(node, "block")
            fn = CodeFunction(
                name=qualified,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_extract_docstring(body, src),
            )
            nodes.append(fn)
            if parent_class:
                edges.append(_make_edge(parent_class.id, fn.id, "has_method"))
            else:
                edges.append(_make_edge(file_node.id, fn.id, "defines_function"))
        return  # no deeper recursion needed for functions

    if node.type in ("import_statement", "import_from_statement"):
        _collect_import(node, src, file_path, file_node, nodes, edges)
        return

    for child in node.children:
        _walk(child, src, file_path, file_node, nodes, edges, parent_class=parent_class)


def _collect_import(node, src: bytes, file_path: str, file_node: CodeFile, nodes: list, edges: list) -> None:
    """Emit CodeImport nodes for import/import-from statements."""
    if node.type == "import_statement":
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                module = _text(_first_child_of_type(child, "dotted_name") or child, src)
                imp = CodeImport(importer_path=file_path, imported_module=module)
                nodes.append(imp)
                edges.append(_make_edge(file_node.id, imp.id, "imports"))
    elif node.type == "import_from_statement":
        # from X import Y  — capture the module (X part)
        for child in node.children:
            if child.type == "dotted_name":
                module = _text(child, src)
                imp = CodeImport(importer_path=file_path, imported_module=module)
                nodes.append(imp)
                edges.append(_make_edge(file_node.id, imp.id, "imports"))
                break  # only capture the module once per from-import


def parse_file(file_path: str) -> tuple[list, list]:
    """Parse a Python source file; return (datapoints, edges).

    If tree-sitter is not installed (codegraph extra missing), returns empty lists.
    """
    if not _HAS_TREE_SITTER:
        return [], []

    try:
        src = Path(file_path).read_bytes()
    except OSError:
        return [], []

    tree = _PY_PARSER.parse(src)
    file_node = CodeFile(file_path=file_path)
    nodes: list = [file_node]
    edges: list = []
    _walk(tree.root_node, src, file_path, file_node, nodes, edges)
    return nodes, edges
