#!/usr/bin/env python3
"""Validate inventory.yml against scripts/inventory.schema.json and the connectors/ tree.

Three layers of checks:

1. Shape: the schema. ``jsonschema`` is used when it is importable; otherwise a built-in
   validator applies the same rules (types, required keys, no extra keys, enums, patterns,
   minLength) straight from the schema file, so the rules live in one place.
2. Consistency within the file: unique slugs, packages, modules and directories, and the
   naming relations between them (``directory == connectors/<slug>``,
   ``module == cognee_connector_<slug with underscores>``, ``package == cognee-connector-<slug with dashes>``).
3. Consistency with the tree: every ``connectors/*`` directory has an entry, every entry has
   its directory unless its status is ``planned``, and an existing directory's
   ``pyproject.toml`` names the entry's package and ships the entry's module.

The file is read by a small block-YAML reader (standard library only) so the check runs
anywhere; when PyYAML is importable the two readings are compared, which keeps the file
inside the documented subset. ``load_inventory`` is also what render_inventory.py and
new_connector.py use.

Usage:
    python scripts/validate_inventory.py
    python scripts/validate_inventory.py --inventory path/to/inventory.yml --root path/to/repo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from packages import REPO_ROOT, TomlError, load_toml, normalize_name

SCHEMA_PATH = Path(__file__).resolve().parent / "inventory.schema.json"
DEFAULT_INVENTORY = REPO_ROOT / "inventory.yml"


class InventoryError(ValueError):
    """inventory.yml cannot be read as the documented block-YAML subset."""


# ---------------------------------------------------------------------------------------
# A reader for the block-YAML subset inventory.yml is written in.
# ---------------------------------------------------------------------------------------


class _YamlSubsetReader:
    """Block mappings and sequences, inline ``[...]``/``{...}`` collections, plain and quoted
    scalars, comments. No anchors, tags, multi-line scalars or multi-document streams."""

    _BOOLEANS = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "on": True,
        "off": False,
    }
    _NULLS = {"", "~", "null"}

    def __init__(self, text: str) -> None:
        self.lines: list[tuple[int, str, int]] = []
        for number, raw in enumerate(text.splitlines(), 1):
            content = self._strip_comment(raw).rstrip()
            if not content.strip():
                continue
            indent = len(content) - len(content.lstrip(" "))
            if "\t" in content[:indent]:
                raise InventoryError(f"line {number}: tabs are not allowed for indentation")
            self.lines.append((indent, content.strip(), number))

    def parse(self) -> Any:
        if not self.lines:
            return None
        value, index = self._block(0, self.lines[0][0])
        if index != len(self.lines):
            raise InventoryError(f"line {self.lines[index][2]}: unexpected indentation")
        return value

    def _block(self, index: int, indent: int) -> tuple[Any, int]:
        if self._is_item(self.lines[index][1]):
            return self._sequence(index, indent)
        return self._mapping(index, indent)

    def _mapping(self, index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(self.lines):
            line_indent, content, number = self.lines[index]
            if line_indent < indent:
                break
            if line_indent > indent:
                raise InventoryError(f"line {number}: unexpected indentation")
            if self._is_item(content):
                break
            key, rest = self._split_key(content, number)
            if key in result:
                raise InventoryError(f"line {number}: duplicate key {key!r}")
            index += 1
            if rest:
                result[key] = self._scalar_or_flow(rest, number)
                continue
            if index < len(self.lines):
                next_indent, next_content, _ = self.lines[index]
                if next_indent > indent:
                    result[key], index = self._block(index, next_indent)
                    continue
                if next_indent == indent and self._is_item(next_content):
                    result[key], index = self._sequence(index, indent)
                    continue
            result[key] = None
        return result, index

    def _sequence(self, index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(self.lines):
            line_indent, content, number = self.lines[index]
            if line_indent < indent or not self._is_item(content):
                break
            if line_indent > indent:
                raise InventoryError(f"line {number}: unexpected indentation")
            item = content[1:].strip()
            if not item:
                index += 1
                if index < len(self.lines) and self.lines[index][0] > indent:
                    value, index = self._block(index, self.lines[index][0])
                    result.append(value)
                else:
                    result.append(None)
                continue
            if self._looks_like_mapping(item):
                # "- key: value": a mapping whose first key sits on the item line and
                # whose other keys are indented to the same column (indent + 2).
                self.lines[index] = (indent + 2, item, number)
                value, index = self._mapping(index, indent + 2)
                result.append(value)
                continue
            result.append(self._scalar_or_flow(item, number))
            index += 1
        return result, index

    def _scalar_or_flow(self, text: str, number: int) -> Any:
        if text.startswith("["):
            if not text.endswith("]"):
                raise InventoryError(f"line {number}: inline list must close on the same line")
            inner = text[1:-1].strip()
            return [self._scalar(part, number) for part in self._split_flow(inner, number)]
        if text.startswith("{"):
            if not text.endswith("}"):
                raise InventoryError(f"line {number}: inline mapping must close on the same line")
            inner = text[1:-1].strip()
            mapping: dict[str, Any] = {}
            for part in self._split_flow(inner, number):
                key, rest = self._split_key(part, number)
                mapping[key] = self._scalar(rest, number)
            return mapping
        return self._scalar(text, number)

    def _scalar(self, text: str, number: int) -> Any:
        text = text.strip()
        if text.startswith('"'):
            if not text.endswith('"') or len(text) < 2:
                raise InventoryError(f"line {number}: unterminated string")
            return self._unescape(text[1:-1], number)
        if text.startswith("'"):
            if not text.endswith("'") or len(text) < 2:
                raise InventoryError(f"line {number}: unterminated string")
            return text[1:-1].replace("''", "'")
        lowered = text.lower()
        if lowered in self._NULLS:
            return None
        if lowered in self._BOOLEANS:
            return self._BOOLEANS[lowered]
        if re.fullmatch(r"[-+]?[0-9]+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
            return float(text)
        if text[0] in "&*!|>%@`":
            raise InventoryError(f"line {number}: {text[0]!r} syntax is outside the subset")
        return text

    def _unescape(self, text: str, number: int) -> str:
        escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "0": "\0"}
        out: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char != "\\":
                out.append(char)
                index += 1
                continue
            escape = text[index + 1 : index + 2]
            if escape in escapes:
                out.append(escapes[escape])
                index += 2
            elif escape == "u" and len(text) >= index + 6:
                out.append(chr(int(text[index + 2 : index + 6], 16)))
                index += 6
            else:
                raise InventoryError(f"line {number}: unknown escape \\{escape}")
        return "".join(out)

    @staticmethod
    def _split_flow(inner: str, number: int) -> list[str]:
        if not inner:
            return []
        parts: list[str] = []
        current: list[str] = []
        quote: str | None = None
        depth = 0
        for char in inner:
            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                continue
            if char in "\"'":
                quote = char
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if quote or depth:
            raise InventoryError(f"line {number}: unbalanced inline collection")
        parts.append("".join(current).strip())
        return [part for part in parts if part != ""]

    @staticmethod
    def _split_key(content: str, number: int) -> tuple[str, str]:
        if content.startswith(("'", '"')):
            quote = content[0]
            end = content.find(quote, 1)
            if end == -1 or not content[end + 1 :].lstrip().startswith(":"):
                raise InventoryError(f"line {number}: expected 'key: value'")
            key = content[1:end]
            rest = content[end + 1 :].lstrip()[1:]
        else:
            match = re.match(r"^([^:#]+?)\s*:(?:\s+|$)(.*)$", content)
            if not match:
                raise InventoryError(f"line {number}: expected 'key: value'")
            key, rest = match.group(1), match.group(2)
        return key.strip(), rest.strip()

    @staticmethod
    def _looks_like_mapping(text: str) -> bool:
        if text.startswith(("[", "{", "'", '"')):
            return False
        return re.match(r"^[^:#]+?\s*:(?:\s|$)", text) is not None

    @staticmethod
    def _is_item(content: str) -> bool:
        return content == "-" or content.startswith("- ")

    @staticmethod
    def _strip_comment(line: str) -> str:
        quote: str | None = None
        for index, char in enumerate(line):
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "#" and (index == 0 or line[index - 1] in " \t"):
                return line[:index]
        return line


def load_inventory(path: Path) -> Any:
    """Read ``inventory.yml``; when PyYAML is importable, insist both readers agree."""
    text = path.read_text(encoding="utf-8")
    try:
        parsed = _YamlSubsetReader(text).parse()
    except InventoryError as error:
        raise InventoryError(f"{path}: {error}") from error
    try:
        import yaml
    except ImportError:
        return parsed
    reference = yaml.safe_load(text)
    if reference != parsed:
        raise InventoryError(
            f"{path}: PyYAML and the built-in reader disagree; the file uses YAML outside the "
            "block subset described in its header"
        )
    return parsed


# ---------------------------------------------------------------------------------------
# Schema validation: jsonschema when available, the same rules by hand otherwise.
# ---------------------------------------------------------------------------------------

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _has_type(value: Any, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _validate(
    value: Any, schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError(f"unsupported $ref {reference!r}")
        target: Any = root
        for part in reference[2:].split("/"):
            target = target[part]
        _validate(value, target, path, root, errors)
        return
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_has_type(value, name) for name in names):
            errors.append(f"{path}: expected {' or '.join(names)}, got {type(value).__name__}")
            return
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
        return
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: must be at least {schema['minLength']} character(s)")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                _validate(child, properties[key], f"{path}.{key}", root, errors)
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unknown key {key!r}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]", root, errors)


def validate_schema(inventory: Any, schema: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        errors: list[str] = []
        _validate(inventory, schema, "inventory", schema, errors)
        return errors
    validator = jsonschema.Draft202012Validator(schema)
    found = sorted(validator.iter_errors(inventory), key=lambda error: list(error.absolute_path))
    return [
        "inventory"
        + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        + f": {error.message}"
        for error in found
    ]


# ---------------------------------------------------------------------------------------
# Consistency within the file and with the connectors/ tree.
# ---------------------------------------------------------------------------------------


def validate_consistency(inventory: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    entries = inventory.get("connectors") or []
    if not isinstance(entries, list):
        return errors

    for field in ("slug", "package", "module", "directory"):
        seen: dict[Any, int] = {}
        for index, entry in enumerate(entries):
            value = entry.get(field) if isinstance(entry, dict) else None
            if value is None:
                continue
            if value in seen:
                errors.append(
                    f"connectors[{index}]: duplicate {field} {value!r} (also connectors[{seen[value]}])"
                )
            seen.setdefault(value, index)

    connectors_dir = root / "connectors"
    existing = (
        {
            child.name
            for child in connectors_dir.iterdir()
            if child.is_dir() and not child.name.startswith((".", "_"))
        }
        if connectors_dir.is_dir()
        else set()
    )
    listed: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        label = f"connectors[{index}]"
        slug = entry.get("slug")
        directory = entry.get("directory")
        if not isinstance(slug, str) or not isinstance(directory, str):
            continue
        label = f"{label} ({slug})"
        expected_directory = f"connectors/{slug}"
        expected_module = "cognee_connector_" + slug.replace("-", "_")
        expected_package = "cognee-connector-" + slug.replace("_", "-")
        if directory != expected_directory:
            errors.append(f"{label}: directory must be {expected_directory!r}, found {directory!r}")
        if entry.get("module") != expected_module:
            errors.append(
                f"{label}: module must be {expected_module!r}, found {entry.get('module')!r}"
            )
        if entry.get("package") != expected_package:
            errors.append(
                f"{label}: package must be {expected_package!r}, found {entry.get('package')!r}"
            )

        listed.add(Path(directory).name)
        package_dir = root / directory
        status = entry.get("status")
        if not package_dir.is_dir():
            if status != "planned":
                errors.append(f"{label}: status is {status!r} but {directory} does not exist")
            continue
        pyproject = package_dir / "pyproject.toml"
        if not pyproject.is_file():
            errors.append(f"{label}: {directory} has no pyproject.toml")
            continue
        try:
            config = load_toml(pyproject)
        except (TomlError, OSError) as error:
            errors.append(f"{label}: cannot read {pyproject}: {error}")
            continue
        name = config.get("project", {}).get("name")
        if isinstance(entry.get("package"), str) and (
            not isinstance(name, str) or normalize_name(name) != normalize_name(entry["package"])
        ):
            errors.append(
                f"{label}: {pyproject} names the project {name!r}, inventory says {entry['package']!r}"
            )
        module = entry.get("module")
        if isinstance(module, str) and not (package_dir / module).is_dir():
            errors.append(f"{label}: {directory} has no {module}/ package directory")

    for name in sorted(existing - listed):
        errors.append(f"connectors/{name} exists but has no inventory entry")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate inventory.yml.")
    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_INVENTORY, help="the file to check"
    )
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="the JSON schema")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the repository root")
    args = parser.parse_args(argv)

    try:
        inventory = load_inventory(args.inventory)
    except (InventoryError, OSError) as error:
        print(f"error: {error}")
        return 1
    schema = json.loads(args.schema.read_text(encoding="utf-8"))

    errors = validate_schema(inventory, schema)
    if not errors and isinstance(inventory, dict):
        errors = validate_consistency(inventory, args.root)

    count = len(inventory.get("connectors") or []) if isinstance(inventory, dict) else 0
    if errors:
        print(
            f"{args.inventory}: {len(errors)} problem(s) in {count} connector entr{'y' if count == 1 else 'ies'}:\n"
        )
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"{args.inventory}: {count} connector entr{'y' if count == 1 else 'ies'}, valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
