"""AST-based Python parser for the code graph pipeline (issue #3668).

Walks a Python source file with tree-sitter (the ``codegraph`` extra) and builds
cognee's existing code-graph DataPoints from ``cognee.shared.CodeGraphEntities``:
a :class:`CodeFile` wired to its :class:`ImportStatement` /
:class:`ClassDefinition` / :class:`FunctionDefinition` children through the
model's relationship fields. ``add_data_points`` then mints the graph edges
natively — no LLM calls, no hand-rolled edges.

Node ids are derived deterministically from stable identity (file path + symbol
name) via ``DataPoint.id_for``, so re-ingesting the same tree upserts in place
instead of duplicating.
"""

from pathlib import Path
from typing import Optional

from cognee.shared.CodeGraphEntities import (
    ClassDefinition,
    CodeFile,
    FunctionDefinition,
    ImportStatement,
    Repository,
)

try:
    import tree_sitter_python as _tspy
    from tree_sitter import Language, Parser

    _PARSER = Parser(Language(_tspy.language()))
except ImportError:  # codegraph extra not installed
    _PARSER = None


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _child_of_type(node, kind: str):
    for child in node.children:
        if child.type == kind:
            return child
    return None


def _points(node) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return 1-based (row, col) start/end points for a tree-sitter node."""
    return (
        (node.start_point[0] + 1, node.start_point[1]),
        (node.end_point[0] + 1, node.end_point[1]),
    )


def _function(
    node, src: bytes, file_path: str, class_name: Optional[str]
) -> Optional[FunctionDefinition]:
    name_node = _child_of_type(node, "identifier")
    if not name_node:
        return None
    # Qualify methods with their class so names and ids don't collide within a file.
    name = _text(name_node, src)
    qualified = f"{class_name}.{name}" if class_name else name
    start, end = _points(node)
    return FunctionDefinition(
        id=FunctionDefinition.id_for(file_path, qualified),
        name=qualified,
        start_point=start,
        end_point=end,
        source_code=_text(node, src),
        file_path=file_path,
    )


def _class(node, src: bytes, file_path: str) -> Optional[ClassDefinition]:
    name_node = _child_of_type(node, "identifier")
    if not name_node:
        return None
    name = _text(name_node, src)
    start, end = _points(node)
    return ClassDefinition(
        id=ClassDefinition.id_for(file_path, name),
        name=name,
        start_point=start,
        end_point=end,
        source_code=_text(node, src),
        file_path=file_path,
    )


def _imported_modules(node, src: bytes) -> list[str]:
    """Return the module(s) an import / from-import statement depends on.

    Uses tree-sitter field names so ``from . import x`` / ``from .mod import y``
    resolve to the package (``.`` / ``.mod``) rather than the imported symbol.
    """
    if node.type == "import_from_statement":
        module = node.child_by_field_name("module_name")
        return [_text(module, src)] if module else []
    if node.type == "import_statement":
        modules = []
        for target in node.children_by_field_name("name"):
            dotted = (
                target if target.type == "dotted_name" else _child_of_type(target, "dotted_name")
            )
            if dotted:
                modules.append(_text(dotted, src))
        return modules
    return []


def parse_file(file_path: str, repository: Optional[Repository] = None) -> Optional[CodeFile]:
    """Parse a Python source file into a :class:`CodeFile` with its relationships.

    Returns ``None`` when tree-sitter is unavailable or the file cannot be read,
    so callers can skip it without special-casing.
    """
    if _PARSER is None:
        return None
    try:
        src = Path(file_path).read_bytes()
    except OSError:
        return None

    classes: list[ClassDefinition] = []
    functions: list[FunctionDefinition] = []
    imports: list[ImportStatement] = []

    def walk(node, class_name: Optional[str]) -> None:
        for child in node.children:
            if child.type == "class_definition":
                cls = _class(child, src, file_path)
                if cls:
                    classes.append(cls)
                    body = _child_of_type(child, "block")
                    if body:
                        walk(body, cls.name)
            elif child.type == "function_definition":
                fn = _function(child, src, file_path, class_name)
                if fn:
                    functions.append(fn)
            elif child.type in ("import_statement", "import_from_statement"):
                start, end = _points(child)
                for module in _imported_modules(child, src):
                    imports.append(
                        ImportStatement(
                            id=ImportStatement.id_for(file_path, module),
                            name=module,
                            module=module,
                            start_point=start,
                            end_point=end,
                            source_code=_text(child, src),
                            file_path=file_path,
                        )
                    )
            else:
                # decorated_definition, if/try wrappers, etc. — descend to reach
                # the definitions inside while keeping the enclosing class context.
                walk(child, class_name)

    walk(_PARSER.parse(src).root_node, None)

    return CodeFile(
        id=CodeFile.id_for(file_path),
        name=Path(file_path).name,
        file_path=file_path,
        language="python",
        part_of=repository,
        depends_on=imports,
        provides_class_definition=classes,
        provides_function_definition=functions,
    )
