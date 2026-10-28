"""Capture source beside freshly extracted facts, never from a later checkout."""

import hashlib
import re
import time
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.exceptions import CogneeValidationError

# Enola counts lines by "\n" only, so text is split the same way: splitlines() would also split
# on U+2028, form feed and similar, and shift every later symbol.
_LINE = re.compile(r"[^\n]*\n|[^\n]+\Z")
_MARKUP = frozenset({".md", ".markdown", ".rst", ".txt", ".adoc"})
# A line directly above a symbol that is a comment of it, and what starts an annotation,
# decorator or attribute (``@Override``, ``[Obsolete]``, ``#[Route]``).
_COMMENTS = ("#", "//", "/*", "*", "--")
_ANNOTATIONS = ("@", "[", "#[")
# Separators after a symbol's name that make a later name its member (a method, a nested
# type, a Markdown section, a C++ scope).
_MEMBER_SEPARATORS = ".#:"


def source_node_id(symbol_node_id: str | UUID) -> UUID:
    """The id of the node holding one symbol's source, a function of the symbol's id."""
    return uuid5(NAMESPACE_OID, f"code-source:{symbol_node_id}")


def capture_sources(facts: list[dict], root: Path, started_ns: int) -> dict[str, dict]:
    """Read referenced UTF-8 files that stayed unchanged during extraction.

    Imported snapshots have no trusted checkout and must not call this function.
    A path leaving the root (a symlink to elsewhere) and non-text files have no source
    evidence; their graph facts remain available. Each file also carries the line range of
    every symbol in it (see ``symbol_ranges``).
    """
    root = root.resolve()
    sources = {}
    symbols_by_file = defaultdict(list)
    for fact in facts:
        if (
            fact.get("kind") == "symbol"
            and isinstance(fact.get("file"), str)
            and type(fact.get("line")) is int
            and isinstance(fact.get("name"), str)
        ):
            symbols_by_file[fact["file"]].append((fact["line"], fact["name"]))
    for name in sorted({fact["file"] for fact in facts if isinstance(fact.get("file"), str)}):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue  # Outside the repository, or a virtual file location of some extractors.
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        # A rewrite shows in size and modification time; a chmod or a touch of metadata
        # alone does not change what was read. A file dated in the future cannot have been
        # modified after the extraction started.
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or (
            started_ns < before.st_mtime_ns <= time.time_ns()
        ):
            raise CogneeValidationError(
                message=f"Code source changed during extraction: {name!r}.",
                name="CodeSourceError",
                remediation="Run the index again once the files stop changing.",
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        lines = _LINE.findall(text)
        sources[name] = {
            "lines": lines,
            "ranges": symbol_ranges(symbols_by_file[name], lines, Path(name).suffix.lower()),
        }
    return sources


_LABEL = re.compile(r"(public|private|protected|internal)\s*:")
_OPENERS, _CLOSERS = "([{", ")]}"


def _depth_change(line: str) -> int:
    """Brackets a line opens minus the ones it closes, leaving out those inside strings."""
    for strings in (True, False):
        depth, quote, index = 0, None, 0
        while index < len(line):
            char = line[index]
            if quote:
                if char == "\\":
                    index += 1
                elif char == quote:
                    quote = None
            elif strings and char in "\"'":
                quote = char
            elif char in _OPENERS:
                depth += 1
            elif char in _CLOSERS:
                depth -= 1
            index += 1
        if quote is None:
            return depth
    return depth  # An apostrophe in prose opened a "string": count the brackets plainly.


def _skip_annotation(text: str, col: int, depth: int) -> tuple[int, int]:
    """Consume one annotation starting at ``col``, or the rest of one that is open at ``depth``.

    Returns where it ends and the bracket depth there: 0 when it closed on this line, more when
    its arguments go on over the next one.
    """
    index = col
    if depth == 0:
        index += 1 if text.startswith("#[", col) else 0
        if text.startswith("@", index):
            index += 1
            while index < len(text) and (text[index].isalnum() or text[index] in "_.:"):
                index += 1
            if index >= len(text) or text[index] != "(":
                return index, 0
    quote = None
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth == 0:
                return index + 1, 0
        index += 1
    return len(text), depth


def _consume_annotations(text: str, col: int, depth: int) -> tuple[int, int]:
    """Consume the annotations a line starts with (or continues): where they end, and the depth."""
    while True:
        if depth == 0:
            while col < len(text) and text[col] in " \t":
                col += 1
            if not text.startswith(_ANNOTATIONS, col):
                return col, 0
        col, depth = _skip_annotation(text, col, depth)
        if depth > 0 or col >= len(text):
            return col, depth


def _code_follows(text: str, col: int) -> bool:
    rest = text[col:].strip()
    return bool(rest) and not rest.startswith(("//", "/*", "#"))


def _is_annotation_line(text: str) -> bool:
    """A line that is annotations and nothing else (so not a list that starts with ``[``)."""
    stripped = text.lstrip()
    if not stripped.startswith(_ANNOTATIONS):
        return False
    col, depth = _consume_annotations(text, 0, 0)
    return depth == 0 and not _code_follows(text, col)


def _declaration_line(lines: list[str], line: int, end: int) -> int:
    """The line the declaration itself is on: past the annotations that start at ``line``.

    An annotation may span lines, may be followed by the declaration on the same line
    (``@Override public String toString() {``), or stand on lines of its own.
    """
    depth = 0
    for number in range(line, end + 1):
        col, depth = _consume_annotations(lines[number - 1], 0, depth)
        if depth == 0 and _code_follows(lines[number - 1], col):
            return number
    return line


def _continues_declaration(line: str, depth: int) -> bool:
    """A line at the declaration's own indent that still belongs to it.

    That is an access label (``public:``), a lone opening brace of the next-line style, and,
    while a bracket the declaration opened is still open (``depth``), the closing of it:
    brackets only, or the rest of a signature that starts with one (``) -> int:``).
    """
    stripped = line.strip(" \t\r\n")
    if _LABEL.fullmatch(stripped) or stripped == "{":
        return True
    closes = not stripped.strip(")]}>;,") or stripped.startswith((")", "]", "}"))
    return closes and depth > 0


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _decorator_start(lines: list[str], last: int, previous: int) -> int | None:
    """The line a decorator or attribute that ends with brackets on ``last`` starts on."""
    depth = 0
    for number in range(last, previous, -1):
        text = lines[number - 1]
        depth -= _depth_change(text)
        if depth <= 0:
            return number if text.lstrip().startswith(_ANNOTATIONS) else None
    return None


def symbol_ranges(
    symbols: list[tuple[int, str]], lines: list[str], suffix: str
) -> dict[tuple, tuple[int, int]]:
    """The first and last line of each ``(line, name)`` symbol of one file.

    The pinned extractor reports the line a symbol is declared on (the first annotation line
    for annotated Java, Kotlin and C#), not where it starts or ends, so both are derived from
    the text:

    - the start moves up over the decorators, attributes and comments directly above,
      including ones that span lines;
    - the end is the line before the next symbol that is not one of its members (a class
      includes its methods), or the end of the file, and is cut earlier at the first line
      that is less indented than the declaration (the enclosing scope closing) or equally
      indented and not a continuation of it (code that follows the symbol). The declaration
      is the first line from the reported one that has code after its annotations, which may
      span lines or share the line. Markup files
      have no indentation structure and skip that cut.

    Both are approximations of a parser, applied only when the extractor gives no range.
    """
    symbols = sorted(symbols)
    starts = []
    previous = 0
    for line, _name in symbols:
        start = line
        while 0 < start - 1 <= len(lines) and start - 1 > previous:
            above = lines[start - 2]
            if above.lstrip().startswith(_COMMENTS) or _is_annotation_line(above):
                start -= 1
                continue
            begin = (
                _decorator_start(lines, start - 1, previous)
                if above.lstrip().startswith(tuple(_CLOSERS))
                else None
            )
            if begin is None:
                break
            start = begin
        starts.append(start)
        previous = max(previous, line)
    ranges = {}
    for index, (line, name) in enumerate(symbols):
        end = len(lines)
        for later, (later_line, later_name) in enumerate(symbols[index + 1 :], start=index + 1):
            member = (
                later_name.startswith(name)
                and len(later_name) > len(name)
                and later_name[len(name)] in _MEMBER_SEPARATORS
            )
            if later_line > line and not member:
                end = starts[later] - 1
                break
        if suffix not in _MARKUP and 1 <= line <= len(lines):
            declaration = _declaration_line(lines, line, end)
            declared = _indent(lines[declaration - 1])
            depth = _depth_change(lines[declaration - 1])
            for number in range(declaration + 1, end + 1):
                text = lines[number - 1]
                if not text.strip():
                    continue
                indent = _indent(text)
                if indent < declared or (
                    indent == declared and not _continues_declaration(text, depth)
                ):
                    end = number - 1
                    break
                depth += _depth_change(text)
        while end > line and not lines[end - 1].strip():
            end -= 1  # Blank lines between symbols belong to none of them.
        ranges[(line, name)] = (starts[index], max(end, line))
    return ranges


def _usable(start, end, count) -> bool:
    return type(start) is int and type(end) is int and 1 <= start <= end <= count


def source_for_fact(fact: dict, sources: dict[str, dict]) -> dict | None:
    """The source lines of a symbol: the extractor's range, else the derived one.

    A symbol with no usable range keeps the whole file rather than losing its source.
    """
    source = sources.get(fact.get("file"))
    if source is None:
        return None
    lines = source["lines"]
    start, end = fact.get("line"), fact.get("end_line")
    if not _usable(start, end, len(lines)):
        start, end = source["ranges"].get((start, fact.get("name")), (None, None))
        if not _usable(start, end, len(lines)):
            start, end = 1, len(lines)
    text = "".join(lines[start - 1 : end])
    return {
        "text": text,
        "line": start,
        "end_line": end,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
