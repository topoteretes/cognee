"""The source docstring readers (cognee/tasks/code_docs/docstrings.py)."""

import os

import pytest

from cognee.tasks.code_docs.docstrings import (
    MAX_SIGNATURE_CHARS,
    MAX_SOURCE_BYTES,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_OUTSIDE_REPO,
    STATUS_PARSE_ERROR,
    STATUS_TOO_LARGE,
    STATUS_UNREADABLE,
    STATUS_UNSUPPORTED,
    language_of,
    module_name_of,
    parse_python_docs,
    read_file_docs,
    resolve_source_path,
    symbol_dotted_name,
)

SOURCE = '''"""Shop module.

    Indented second paragraph.
"""

import functools


def helper(a, b=2):
    """Add two numbers."""
    return a + b


async def fetch(url):
    """Fetch a URL."""
    return url


@functools.lru_cache(maxsize=None)
@staticmethod
def cached():
    """Cached helper."""
    return 1


def undocumented():
    return None


class Order:
    """An order."""

    def total(self):
        """Sum of the lines."""
        return 0

    @property
    def paid(self) -> bool:
        """Whether the order is paid."""
        return True

    class Line:
        """One order line."""


def outer():
    """Outer."""

    def inner():
        """Inner."""

    return inner


def multi(
    first: int,
    second: str = "x",
    *args,
    **kwargs,
) -> dict:
    """Multi-line header."""
    return {}


@overload
def convert(value: int) -> int: ...


@overload
def convert(value: str) -> str: ...


def convert(value):
    """Convert a value."""
    return value


if TYPE_CHECKING:

    def shim():
        """Only when type checking."""


def commented(a):
    # a comment between header and body
    """Commented."""


def one_liner(): "On the header line."
'''

LINES = SOURCE.splitlines()


def _line_of(text: str, nth: int = 1) -> int:
    """1-based line of the nth line that starts with ``text``."""
    hits = [index + 1 for index, line in enumerate(LINES) if line.startswith(text)]
    return hits[nth - 1]


def _write(tmp_path, relative: str, text: str) -> str:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def parsed():
    module_doc, symbols = parse_python_docs(SOURCE)
    return module_doc, symbols


@pytest.fixture
def file_docs(tmp_path):
    repo = _write(tmp_path, "shop/orders.py", SOURCE)
    return read_file_docs(repo, "shop/orders.py")


def test_module_docstring_is_cleaned(parsed):
    module_doc, _ = parsed
    assert module_doc == "Shop module.\n\nIndented second paragraph."


def test_kinds_and_dotted_names(parsed):
    _, symbols = parsed
    kinds = {symbol.name: symbol.kind for symbol in symbols}
    assert kinds["helper"] == "function"
    assert kinds["fetch"] == "function"
    assert kinds["cached"] == "function"
    assert kinds["Order"] == "class"
    assert kinds["Order.total"] == "method"
    assert kinds["Order.paid"] == "method"
    assert kinds["Order.Line"] == "class"
    assert kinds["outer"] == "function"
    assert kinds["outer.inner"] == "function"
    # A def under ``if TYPE_CHECKING:`` belongs to the module scope.
    assert kinds["shim"] == "function"
    # Source order, overloads included.
    assert [symbol.name for symbol in symbols].count("convert") == 3


def test_docstrings_are_dedented_and_absent_when_missing(parsed):
    _, symbols = parsed
    by_name = {symbol.name: symbol for symbol in symbols}
    assert by_name["helper"].docstring == "Add two numbers."
    assert by_name["fetch"].docstring == "Fetch a URL."
    assert by_name["Order.total"].docstring == "Sum of the lines."
    assert by_name["outer.inner"].docstring == "Inner."
    assert by_name["undocumented"].docstring is None


def test_lines_cover_the_def_and_its_decorators(parsed):
    _, symbols = parsed
    cached = next(symbol for symbol in symbols if symbol.name == "cached")
    def_line = _line_of("def cached")
    assert cached.line == def_line
    assert cached.header_lines == (def_line, def_line - 2, def_line - 1)
    assert cached.end_line == def_line + 2
    helper = next(symbol for symbol in symbols if symbol.name == "helper")
    assert helper.header_lines == (helper.line,)


def test_signatures_are_one_line_without_decorators(parsed):
    _, symbols = parsed
    by_name = {symbol.name: symbol for symbol in symbols}
    assert by_name["helper"].signature == "def helper(a, b=2):"
    assert by_name["fetch"].signature == "async def fetch(url):"
    assert by_name["cached"].signature == "def cached():"
    assert by_name["Order"].signature == "class Order:"
    assert by_name["Order.paid"].signature == "def paid(self) -> bool:"
    assert (
        by_name["multi"].signature
        == 'def multi(first: int, second: str = "x", *args, **kwargs,) -> dict:'
    )
    assert by_name["commented"].signature == "def commented(a):"
    assert by_name["commented"].docstring == "Commented."
    assert by_name["one_liner"].signature == "def one_liner():"
    assert by_name["one_liner"].docstring == "On the header line."


def test_line_breaks_follow_the_parser_not_splitlines():
    # A form feed (a GNU-style page break) and the Unicode line separator are
    # not line breaks to the parser; cutting the source at them would shift
    # every line below and yield another line's text as the signature.
    source = (
        'TEXT = "para break"\n'
        "\n"
        "\n"
        "def a():\n"
        '    """A."""\n'
        "    return 1\n"
        "\x0c\n"
        "def b(x, y):\n"
        '    """B."""\n'
        "    return x\n"
    )
    _, symbols = parse_python_docs(source)
    assert [(symbol.name, symbol.line, symbol.signature) for symbol in symbols] == [
        ("a", 4, "def a():"),
        ("b", 8, "def b(x, y):"),
    ]


def test_signature_is_capped_at_300_characters():
    params = ", ".join(f"parameter_{index}: int = {index}" for index in range(40))
    _, symbols = parse_python_docs(f"def wide({params}):\n    '''Doc.'''\n")
    signature = symbols[0].signature
    assert len(signature) == MAX_SIGNATURE_CHARS
    assert signature.endswith("...")
    assert signature.startswith("def wide(parameter_0: int = 0, ")


def test_find_by_def_line_and_by_decorator_line(file_docs):
    assert file_docs.status == STATUS_OK
    def_line = _line_of("def cached")
    assert file_docs.find(def_line, None).name == "cached"
    assert file_docs.find(def_line - 2, None).name == "cached"
    assert file_docs.find(def_line - 1, "cached").name == "cached"


def test_find_falls_back_to_the_name(file_docs):
    assert file_docs.find(None, "Order.total").docstring == "Sum of the lines."
    assert file_docs.find(9999, "Order.total").docstring == "Sum of the lines."
    assert file_docs.find(9999, None) is None
    assert file_docs.find(None, "Order.missing") is None
    # The line now holds another symbol: the name wins over the stale line.
    assert file_docs.find(_line_of("def helper"), "Order.total").name == "Order.total"
    # A line match is kept when the name is unknown to the reader (another
    # extractor's naming), and when it agrees with the name.
    assert file_docs.find(_line_of("def helper"), "nested.helper").name == "helper"
    assert file_docs.find(_line_of("def helper"), "helper").name == "helper"
    # The line holds some other def and the name is nowhere in the file: a
    # stale line, not this symbol, so another symbol's docstring is not
    # attached under its header.
    assert file_docs.find(_line_of("def helper"), "Order.gone") is None
    assert file_docs.find(_line_of("def helper"), "gone") is None


def test_find_prefers_the_documented_overload(file_docs):
    stub_line = _line_of("def convert", 1)
    assert file_docs.find(stub_line, "convert").docstring == "Convert a value."
    assert file_docs.find(None, "convert").docstring == "Convert a value."
    # An undocumented symbol is still found, so the caller can tell "no
    # docstring" from "no such symbol".
    found = file_docs.find(_line_of("def undocumented"), "undocumented")
    assert found is not None and found.docstring is None


def test_module_docstring_through_the_file_reader(file_docs):
    assert file_docs.language == "python"
    assert file_docs.module_doc == "Shop module.\n\nIndented second paragraph."


def test_syntax_error_file_is_reported_not_raised(tmp_path):
    repo = _write(tmp_path, "broken.py", "def broken(:\n    pass\n")
    docs = read_file_docs(repo, "broken.py")
    assert docs.status == STATUS_PARSE_ERROR
    assert docs.symbols == [] and docs.module_doc is None and not docs.readable


def test_non_utf8_file_is_unreadable(tmp_path):
    (tmp_path / "latin.py").write_bytes(b'"""caf\xe9"""\n')
    assert read_file_docs(str(tmp_path), "latin.py").status == STATUS_UNREADABLE


def test_missing_and_unsupported_files(tmp_path):
    _write(tmp_path, "main.go", "package main\n")
    assert read_file_docs(str(tmp_path), "nope.py").status == STATUS_MISSING
    docs = read_file_docs(str(tmp_path), "main.go")
    assert docs.status == STATUS_UNSUPPORTED and docs.language is None


def test_files_outside_the_repository_are_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text('"""Outside."""\n', encoding="utf-8")
    os.symlink(outside, repo / "linked.py")

    assert read_file_docs(str(repo), "../outside.py").status == STATUS_OUTSIDE_REPO
    assert read_file_docs(str(repo), str(outside)).status == STATUS_OUTSIDE_REPO
    # A symlink inside the repository that points out of it is not followed.
    assert read_file_docs(str(repo), "linked.py").status == STATUS_OUTSIDE_REPO
    assert resolve_source_path(str(repo), "linked.py") is None
    assert resolve_source_path(str(repo), "") is None

    inside = _write(repo, "pkg/mod.py", '"""Inside."""\n')
    assert resolve_source_path(inside, "pkg/mod.py") == os.path.realpath(
        os.path.join(inside, "pkg/mod.py")
    )
    assert read_file_docs(inside, "pkg/mod.py").module_doc == "Inside."


def test_files_over_the_size_cap_are_skipped(tmp_path):
    (tmp_path / "huge.py").write_text("#" * (MAX_SOURCE_BYTES + 1), encoding="utf-8")
    assert read_file_docs(str(tmp_path), "huge.py").status == STATUS_TOO_LARGE
    (tmp_path / "fits.py").write_text('"""Fits."""\n' + "#" * 1000, encoding="utf-8")
    assert read_file_docs(str(tmp_path), "fits.py").status == STATUS_OK


def test_path_helpers():
    assert language_of("a/b.py") == "python"
    assert language_of("stubs/types.PYI") == "python"
    assert language_of("main.go") is None
    assert module_name_of("pkg/mod.py") == "pkg/mod"
    assert module_name_of("app.py") == "app"
    assert symbol_dotted_name("pkg/mod.Order.total", "pkg/mod.py") == "Order.total"
    assert symbol_dotted_name("app.main", "app.py") == "main"
    assert symbol_dotted_name("other/thing.x", "pkg/mod.py") == "thing.x"
    assert symbol_dotted_name("bare", "pkg/mod.py") == "bare"
