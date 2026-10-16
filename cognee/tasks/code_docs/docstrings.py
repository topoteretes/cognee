"""Docstring readers over repository sources.

Pure functions with no graph or database imports: given a repository root and
a repo-relative file path they return the file's module docstring and every
def/class with its header lines, signature and docstring. Python is read with
the standard ``ast`` module; ``READERS`` is keyed by language so readers for
other languages can be registered without touching the extraction task.

Reading is guarded the way a server should guard a path it did not choose:
only files whose real path lies inside the repository's real path are opened
(a symlink pointing out of the repository is refused), files over
``MAX_SOURCE_BYTES`` are skipped, and a file that is not UTF-8 or does not
parse is reported, not raised.
"""

import ast
import os
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass, field

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SIGNATURE_CHARS = 300

# Extension -> language a reader exists for.
LANGUAGE_BY_EXTENSION = {".py": "python", ".pyi": "python"}

# FileDocs.status values.
STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported"  # no reader for the file's language
STATUS_OUTSIDE_REPO = "outside_repo"  # real path escapes the repository
STATUS_MISSING = "missing"  # no such file (or not a regular file)
STATUS_TOO_LARGE = "too_large"
STATUS_UNREADABLE = "unreadable"  # unreadable or not UTF-8
STATUS_PARSE_ERROR = "parse_error"

_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]},])")
# The line breaks the parser counts. str.splitlines() also cuts at form feeds
# and the Unicode separators, which would shift every ``lineno`` below one.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True)
class SymbolDoc:
    """One def or class of a source file: where it is, its header and its docstring."""

    # Dotted path inside the module, e.g. ``Order.total`` or ``outer.inner``.
    name: str
    # ``function``, ``method`` or ``class``.
    kind: str
    # The def/class line, 1-based; decorators excluded.
    line: int
    end_line: int | None
    # Every line an extractor may report for the symbol: the def line and
    # each decorator's line.
    header_lines: tuple[int, ...]
    signature: str | None
    docstring: str | None


@dataclass
class FileDocs:
    """What a reader found in one file, plus the lookups the extraction task needs."""

    status: str
    language: str | None = None
    module_doc: str | None = None
    # Source order.
    symbols: list[SymbolDoc] = field(default_factory=list)
    _by_line: dict[int, SymbolDoc] = field(default_factory=dict, repr=False)
    _by_name: dict[str, list[SymbolDoc]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for symbol in self.symbols:
            # The def line wins over a decorator line that another symbol
            # cannot share anyway; setdefault keeps the first claim.
            for line in symbol.header_lines:
                self._by_line.setdefault(line, symbol)
            self._by_name.setdefault(symbol.name, []).append(symbol)

    @property
    def readable(self) -> bool:
        return self.status == STATUS_OK

    def find(self, line: int | None, name: str | None) -> SymbolDoc | None:
        """The symbol at ``line`` (def or decorator line), else the one called ``name``.

        A documented candidate is preferred over an undocumented one at the
        same name, so an ``@overload`` stub reported by the extractor resolves
        to the implementation that carries the docstring. When the line now
        holds a differently named symbol and ``name`` is found elsewhere, the
        source has moved on since the snapshot and the name is trusted. When
        ``name`` is nowhere in the file, the symbol on the line stands in for
        it only if it is at least called the same (an extractor may qualify a
        nested def differently): a stale line that now holds some other def is
        not this symbol. Returns an undocumented symbol when that is all there
        is, and None when the file holds no such symbol.
        """
        line_match = self._by_line.get(line) if line is not None else None
        name_matches = list(self._by_name.get(name, ())) if name else []
        if (
            line_match is not None
            and name
            and not name_matches
            and _last_component(line_match.name) != _last_component(name)
        ):
            line_match = None
        if line_match is not None and name_matches and line_match.name != name:
            candidates = name_matches
        else:
            candidates = [line_match] if line_match is not None else []
            candidates.extend(symbol for symbol in name_matches if symbol is not line_match)
        for symbol in candidates:
            if symbol.docstring:
                return symbol
        return candidates[0] if candidates else None


def _last_component(dotted_name: str) -> str:
    return dotted_name.rsplit(".", 1)[-1]


def language_of(file_path: str) -> str | None:
    """The language a reader exists for, by file extension; None when there is none."""
    return LANGUAGE_BY_EXTENSION.get(posixpath.splitext(file_path)[1].lower())


def module_name_of(file_path: str) -> str:
    """The file path without its extension, ``/`` kept: ``pkg/mod.py`` -> ``pkg/mod``."""
    return posixpath.splitext(file_path)[0]


def symbol_dotted_name(qualified_name: str, file_path: str) -> str:
    """The path of a symbol inside its module, from an extractor's qualified name.

    Enola names Python symbols ``<module path>.<dotted path>`` with ``/`` in
    the module part, e.g. ``pkg/mod.Order.total`` for ``Order.total`` in
    ``pkg/mod.py``. A name without that prefix is returned as it is, minus any
    directory part.
    """
    module_name = module_name_of(file_path)
    if module_name and qualified_name.startswith(module_name + "."):
        return qualified_name[len(module_name) + 1 :]
    return qualified_name.rsplit("/", 1)[-1]


def resolve_source_path(repo_root: str, file_path: str) -> str | None:
    """The real path of ``file_path`` under ``repo_root``, or None when it escapes it.

    Real paths on both sides, so a symlink inside the repository that points
    outside it is refused as well as ``..`` segments and absolute paths.
    """
    if not file_path or os.path.isabs(file_path):
        return None
    root = os.path.realpath(repo_root)
    candidate = os.path.realpath(os.path.join(root, file_path))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        # Different drives (Windows) share no common path.
        return None
    return candidate


def read_source(path: str) -> tuple[str | None, str]:
    """The UTF-8 text of a regular file under the size cap, with a FileDocs status."""
    try:
        if not os.path.isfile(path):
            return None, STATUS_MISSING
        if os.path.getsize(path) > MAX_SOURCE_BYTES:
            return None, STATUS_TOO_LARGE
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None, STATUS_UNREADABLE
    try:
        return raw.decode("utf-8"), STATUS_OK
    except UnicodeDecodeError:
        return None, STATUS_UNREADABLE


def _normalize_signature(text: str) -> str:
    text = " ".join(text.split())
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_CLOSE.sub(r"\1", text)
    if len(text) > MAX_SIGNATURE_CHARS:
        text = text[: MAX_SIGNATURE_CHARS - 3] + "..."
    return text


def _signature(lines: list[str], node: ast.AST) -> str | None:
    """The def/class header on one line: from the def line to where the body starts.

    Decorators are excluded (``lineno`` is the def line since Python 3.8).
    Blank and comment-only lines between the header and the body are dropped;
    a body that starts on the header's own line is cut at its column.
    """
    body = getattr(node, "body", None)
    if not body or not hasattr(node, "lineno"):
        return None
    first = node.lineno - 1
    body_first = body[0]
    last = body_first.lineno - 1
    if last < first or last >= len(lines):
        return None
    if last == first:
        parts = [lines[first][: body_first.col_offset]]
    else:
        parts = lines[first:last]
        while parts and (not parts[-1].strip() or parts[-1].lstrip().startswith("#")):
            parts.pop()
        head = lines[last][: body_first.col_offset]
        if head.strip():
            parts.append(head)
    text = _normalize_signature(" ".join(parts))
    return text or None


def _statement_lists(node: ast.AST):
    """The statement lists directly inside a compound statement (if/try/with/for/match)."""
    for value in ast.iter_child_nodes(node):
        if isinstance(value, (ast.ExceptHandler, ast.match_case)):
            yield value.body
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            yield value


def _walk(statements, prefix: str, in_class: bool, lines: list[str], out: list[SymbolDoc]) -> None:
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            if isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "method" if in_class else "function"
            out.append(
                SymbolDoc(
                    name=name,
                    kind=kind,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", None),
                    header_lines=(
                        node.lineno,
                        *(decorator.lineno for decorator in node.decorator_list),
                    ),
                    signature=_signature(lines, node),
                    docstring=ast.get_docstring(node, clean=True) or None,
                )
            )
            _walk(node.body, name + ".", isinstance(node, ast.ClassDef), lines, out)
        elif isinstance(node, ast.stmt):
            # Defs under ``if TYPE_CHECKING:``, ``try:`` import shims and the
            # like belong to the enclosing scope, so the prefix is kept.
            for block in _statement_lists(node):
                _walk(block, prefix, in_class, lines, out)


def parse_python_docs(source: str) -> tuple[str | None, list[SymbolDoc]]:
    """The module docstring and every def/class of a Python source, in source order.

    Raises ``SyntaxError`` (or ``ValueError`` for null bytes) when the source
    does not parse; ``read_file_docs`` turns that into a status.
    """
    tree = ast.parse(source)
    lines = _LINE_BREAK.split(source)
    symbols: list[SymbolDoc] = []
    _walk(tree.body, "", False, lines, symbols)
    return ast.get_docstring(tree, clean=True) or None, symbols


# language -> reader of one file's source text.
READERS: dict[str, Callable[[str], tuple[str | None, list[SymbolDoc]]]] = {
    "python": parse_python_docs,
}


def read_file_docs(repo_root: str, file_path: str) -> FileDocs:
    """Read the docstrings of ``file_path`` (repo-relative) under ``repo_root``.

    Never raises for a bad file: the returned ``status`` says why nothing was
    read (unsupported language, path outside the repository, missing, too
    large, unreadable, parse error) so the caller can count it and fall back.
    """
    language = language_of(file_path)
    if language is None or language not in READERS:
        return FileDocs(status=STATUS_UNSUPPORTED, language=language)
    path = resolve_source_path(repo_root, file_path)
    if path is None:
        return FileDocs(status=STATUS_OUTSIDE_REPO, language=language)
    source, status = read_source(path)
    if source is None:
        return FileDocs(status=status, language=language)
    try:
        module_doc, symbols = READERS[language](source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return FileDocs(status=STATUS_PARSE_ERROR, language=language)
    return FileDocs(status=STATUS_OK, language=language, module_doc=module_doc, symbols=symbols)
