#!/usr/bin/env python3
"""List the uv workspace packages of cognee-connectors.

The workspace root ``pyproject.toml`` names its members (``[tool.uv.workspace]``). This
script expands those globs, reads every member's ``pyproject.toml`` and answers the
questions the CI workflows and ``scripts/clean_install.sh`` ask about them; it is also
the one place that derives a member's import name.

Usage:
    python scripts/packages.py --json                 # every package, as JSON records
    python scripts/packages.py --all-names            # one distribution name per line
    python scripts/packages.py --dir NAME             # the package directory
    python scripts/packages.py --module NAME          # the package's import name
    python scripts/packages.py --version NAME         # the version in its pyproject.toml
    python scripts/packages.py --install-order NAME   # directories to install, dependencies first

NAME is a distribution name (``cognee-connector-notion``) or a package directory
(``connectors/notion``). A ``--json`` record carries ``name``, ``dir`` (POSIX path
relative to the repository root), ``module``, ``has_tests`` (a ``tests/`` folder
exists) and ``depends_on_workspace`` (the distribution names of the workspace packages
it requires, in its dependencies or its extras).

Standard library only. Python 3.10 has no ``tomllib``; there a small TOML reader
(:class:`TomlSubsetReader`) covers what pyproject files use.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


class TomlError(ValueError):
    """The file is malformed, or uses TOML the fallback reader does not cover."""


class TomlSubsetReader:
    """A reader for the TOML this repository's pyproject files use.

    Tables, dotted and quoted keys, single-line basic and literal strings, booleans,
    integers, floats, arrays (nested, trailing commas, comments between items) and
    inline tables. Multi-line strings, dates and arrays of tables raise
    :class:`TomlError` naming the line; a file that needs them is read with
    ``tomllib`` (Python 3.11+), which :func:`load_toml` prefers whenever it exists.
    """

    _BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
    _NUMBER = re.compile(
        r"[+-]?(?:0x[0-9A-Fa-f_]+|0o[0-7_]+|0b[01_]+|inf|nan"
        r"|[0-9_]+(?:\.[0-9_]+)?(?:[eE][+-]?[0-9_]+)?)"
    )
    _ESCAPES = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}
    _VALUE_END = " \t\r\n,]}#"

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    # -- entry point ---------------------------------------------------------------

    def parse(self) -> dict[str, Any]:
        root: dict[str, Any] = {}
        current = root
        while True:
            self._skip(newlines=True)
            if self._at_end():
                return root
            if self.text.startswith("[[", self.pos):
                self._fail("arrays of tables ([[...]]) are not supported")
            if self._peek() == "[":
                self.pos += 1
                self._skip(newlines=False)
                keys = self._key_path()
                self._skip(newlines=False)
                self._expect("]")
                current = self._table(root, keys)
                self._end_of_line()
                continue
            keys = self._key_path()
            self._skip(newlines=False)
            self._expect("=")
            self._skip(newlines=False)
            value = self._value()
            parent = self._table(current, keys[:-1])
            if keys[-1] in parent:
                self._fail(f"duplicate key {'.'.join(keys)!r}")
            parent[keys[-1]] = value
            self._end_of_line()

    # -- structure -----------------------------------------------------------------

    def _table(self, base: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        table = base
        for key in keys:
            child = table.setdefault(key, {})
            if not isinstance(child, dict):
                self._fail(f"{key!r} is not a table")
            table = child
        return table

    def _key_path(self) -> list[str]:
        parts = [self._simple_key()]
        while True:
            saved = self.pos
            self._skip(newlines=False)
            if self._peek() != ".":
                self.pos = saved
                return parts
            self.pos += 1
            self._skip(newlines=False)
            parts.append(self._simple_key())

    def _simple_key(self) -> str:
        char = self._peek()
        if char == '"':
            return self._basic_string()
        if char == "'":
            return self._literal_string()
        match = self._BARE_KEY.match(self.text, self.pos)
        if not match:
            self._fail("expected a key")
        self.pos = match.end()
        return match.group()

    # -- values --------------------------------------------------------------------

    def _value(self) -> Any:
        char = self._peek()
        if char == '"':
            if self.text.startswith('"""', self.pos):
                self._fail("multi-line strings are not supported")
            return self._basic_string()
        if char == "'":
            if self.text.startswith("'''", self.pos):
                self._fail("multi-line strings are not supported")
            return self._literal_string()
        if char == "[":
            return self._array()
        if char == "{":
            return self._inline_table()
        for literal, value in (("true", True), ("false", False)):
            if self.text.startswith(literal, self.pos) and self._value_ends_at(
                self.pos + len(literal)
            ):
                self.pos += len(literal)
                return value
        match = self._NUMBER.match(self.text, self.pos)
        if match and self._value_ends_at(match.end()):
            self.pos = match.end()
            return self._number(match.group())
        self._fail("unsupported value (dates and multi-line strings are not supported)")
        raise AssertionError("unreachable")

    def _number(self, token: str) -> int | float:
        text = token.replace("_", "")
        lowered = text.lower().lstrip("+-")
        try:
            if lowered in ("inf", "nan") or (
                not lowered.startswith(("0x", "0o", "0b")) and any(c in lowered for c in ".e")
            ):
                return float(text)
            return int(text, 0)
        except ValueError:
            self._fail(f"invalid number {token!r}")
            raise AssertionError("unreachable")

    def _basic_string(self) -> str:
        self.pos += 1
        out: list[str] = []
        while True:
            if self._at_end() or self.text[self.pos] == "\n":
                self._fail("unterminated string")
            char = self.text[self.pos]
            if char == '"':
                self.pos += 1
                return "".join(out)
            if char == "\\":
                escape = self.text[self.pos + 1 : self.pos + 2]
                if escape in self._ESCAPES:
                    out.append(self._ESCAPES[escape])
                    self.pos += 2
                elif escape in ("u", "U"):
                    width = 4 if escape == "u" else 8
                    digits = self.text[self.pos + 2 : self.pos + 2 + width]
                    if len(digits) != width or not all(
                        c in "0123456789abcdefABCDEF" for c in digits
                    ):
                        self._fail("invalid unicode escape")
                    out.append(chr(int(digits, 16)))
                    self.pos += 2 + width
                else:
                    self._fail(f"unknown escape \\{escape}")
                continue
            out.append(char)
            self.pos += 1

    def _literal_string(self) -> str:
        self.pos += 1
        end = self.text.find("'", self.pos)
        newline = self.text.find("\n", self.pos)
        if end == -1 or (newline != -1 and newline < end):
            self._fail("unterminated string")
        value = self.text[self.pos : end]
        self.pos = end + 1
        return value

    def _array(self) -> list[Any]:
        self.pos += 1
        items: list[Any] = []
        while True:
            self._skip(newlines=True)
            if self._peek() == "]":
                self.pos += 1
                return items
            items.append(self._value())
            self._skip(newlines=True)
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "]":
                self.pos += 1
                return items
            self._fail("expected ',' or ']' in array")

    def _inline_table(self) -> dict[str, Any]:
        self.pos += 1
        table: dict[str, Any] = {}
        self._skip(newlines=True)
        if self._peek() == "}":
            self.pos += 1
            return table
        while True:
            self._skip(newlines=True)
            keys = self._key_path()
            self._skip(newlines=False)
            self._expect("=")
            self._skip(newlines=False)
            value = self._value()
            parent = self._table(table, keys[:-1])
            if keys[-1] in parent:
                self._fail(f"duplicate key {'.'.join(keys)!r}")
            parent[keys[-1]] = value
            self._skip(newlines=True)
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "}":
                self.pos += 1
                return table
            self._fail("expected ',' or '}' in inline table")

    # -- lexing helpers ------------------------------------------------------------

    def _skip(self, newlines: bool) -> None:
        while not self._at_end():
            char = self.text[self.pos]
            if char in " \t":
                self.pos += 1
            elif char in "\r\n":
                if not newlines:
                    return
                self.pos += 1
            elif char == "#":
                end = self.text.find("\n", self.pos)
                self.pos = len(self.text) if end == -1 else end
            else:
                return

    def _end_of_line(self) -> None:
        self._skip(newlines=False)
        if not self._at_end() and self.text[self.pos] not in "\r\n":
            self._fail("expected end of line")

    def _expect(self, char: str) -> None:
        if self._peek() != char:
            self._fail(f"expected {char!r}")
        self.pos += 1

    def _value_ends_at(self, index: int) -> bool:
        return index >= len(self.text) or self.text[index] in self._VALUE_END

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _at_end(self) -> bool:
        return self.pos >= len(self.text)

    def _fail(self, message: str) -> None:
        line = self.text.count("\n", 0, self.pos) + 1
        raise TomlError(f"line {line}: {message}")


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file: ``tomllib`` where it exists, the subset reader on Python 3.10."""
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            return TomlSubsetReader(text).parse()
        except TomlError as error:
            raise TomlError(f"{path}: {error}") from error
    return tomllib.loads(text)


def normalize_name(name: str) -> str:
    """PEP 503 normalization: case-insensitive, runs of ``-_.`` collapse to ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str | None:
    """The normalized distribution name a PEP 508 requirement string refers to."""
    match = _REQUIREMENT_NAME.match(requirement)
    return normalize_name(match.group(1)) if match else None


@dataclass
class Package:
    """One workspace member."""

    name: str
    dir: str
    module: str
    version: str
    has_tests: bool
    depends_on_workspace: list[str]

    def record(self) -> dict[str, Any]:
        """The ``--json`` shape the CI matrices consume."""
        return {
            "name": self.name,
            "dir": self.dir,
            "module": self.module,
            "has_tests": self.has_tests,
            "depends_on_workspace": list(self.depends_on_workspace),
        }


def workspace_member_dirs(root: Path = REPO_ROOT) -> list[Path]:
    """The member directories ``[tool.uv.workspace]`` names, in path order."""
    config = load_toml(root / "pyproject.toml")
    workspace = config.get("tool", {}).get("uv", {}).get("workspace", {})
    found: dict[str, Path] = {}
    for pattern in workspace.get("members", []):
        for match in glob.glob(pattern, root_dir=root):
            directory = root / match
            if (directory / "pyproject.toml").is_file():
                found[directory.as_posix()] = directory
    for pattern in workspace.get("exclude", []):
        for match in glob.glob(pattern, root_dir=root):
            found.pop((root / match).as_posix(), None)
    return [found[key] for key in sorted(found)]


def _module_name(config: dict[str, Any], name: str) -> str:
    wheel = (
        config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    )
    packages = wheel.get("packages") or []
    if packages:
        return Path(str(packages[0])).name
    return re.sub(r"[-.]+", "_", name).lower()


def discover_packages(root: Path = REPO_ROOT) -> list[Package]:
    """Every workspace member, with its workspace dependencies resolved, sorted by name."""
    configs: list[tuple[Path, dict[str, Any]]] = []
    for directory in workspace_member_dirs(root):
        configs.append((directory, load_toml(directory / "pyproject.toml")))
    names = {
        normalize_name(config["project"]["name"]): config["project"]["name"]
        for _, config in configs
    }

    packages: list[Package] = []
    for directory, config in configs:
        project = config["project"]
        requirements = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        depends_on = sorted(
            {
                names[dependency]
                for dependency in map(requirement_name, requirements)
                if dependency in names and dependency != normalize_name(project["name"])
            }
        )
        packages.append(
            Package(
                name=project["name"],
                dir=directory.relative_to(root).as_posix(),
                module=_module_name(config, project["name"]),
                version=str(project.get("version", "")),
                has_tests=(directory / "tests").is_dir(),
                depends_on_workspace=depends_on,
            )
        )
    return sorted(packages, key=lambda package: package.name)


def find_package(packages: list[Package], key: str, root: Path = REPO_ROOT) -> Package:
    """The package named ``key`` (a distribution name) or located at ``key`` (a directory)."""
    wanted_name = normalize_name(key.strip())
    wanted_dir = key.strip().rstrip("/").removeprefix("./")
    candidate = Path(wanted_dir)
    if candidate.is_absolute():
        try:
            wanted_dir = candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    for package in packages:
        if normalize_name(package.name) == wanted_name or package.dir == wanted_dir:
            return package
    known = ", ".join(f"{package.name} ({package.dir})" for package in packages) or "none"
    raise SystemExit(f"error: no workspace package named or located at {key!r}; known: {known}")


def install_order(packages: list[Package], target: Package) -> list[Package]:
    """``target`` preceded by its workspace dependencies, dependencies first (a topological order)."""
    by_name = {normalize_name(package.name): package for package in packages}
    ordered: list[Package] = []
    state: dict[str, str] = {}

    def visit(package: Package, chain: list[str]) -> None:
        marker = state.get(package.name)
        if marker == "done":
            return
        if marker == "active":
            cycle = " -> ".join([*chain, package.name])
            raise SystemExit(f"error: workspace dependency cycle: {cycle}")
        state[package.name] = "active"
        for dependency in package.depends_on_workspace:
            visit(by_name[normalize_name(dependency)], [*chain, package.name])
        state[package.name] = "done"
        ordered.append(package)

    visit(target, [])
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the uv workspace packages of cognee-connectors.",
        epilog=(
            "NAME is a distribution name (cognee-connector-notion) or a package directory "
            "(connectors/notion)."
        ),
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        "--json", action="store_true", help="print every package as a JSON array of records"
    )
    query.add_argument(
        "--all-names", action="store_true", help="print one distribution name per line"
    )
    query.add_argument("--dir", metavar="NAME", help="print the package's directory")
    query.add_argument("--module", metavar="NAME", help="print the package's import name")
    query.add_argument("--version", metavar="NAME", help="print the package's version")
    query.add_argument(
        "--install-order",
        metavar="NAME",
        help="print the directories to install, one per line, workspace dependencies first",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        packages = discover_packages(args.root)
    except (TomlError, KeyError, OSError) as error:
        print(f"error: cannot read the workspace: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([package.record() for package in packages], indent=2))
    elif args.all_names:
        for package in packages:
            print(package.name)
    elif args.dir:
        print(find_package(packages, args.dir, args.root).dir)
    elif args.module:
        print(find_package(packages, args.module, args.root).module)
    elif args.version:
        print(find_package(packages, args.version, args.root).version)
    elif args.install_order:
        target = find_package(packages, args.install_order, args.root)
        for package in install_order(packages, target):
            print(package.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
