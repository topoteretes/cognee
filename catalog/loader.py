"""Catalog loader for the Cognee Integrations Hub and Use-Case Gallery.

Reads every YAML file under ``catalog/entries/`` and returns a validated list of
:class:`CatalogEntry`. Validation happens in three passes:

1. Structural: entry conforms to ``catalog/schema.json``.
2. Naming: ``id`` matches the filename stem, and the entry lives under the
   subdirectory that matches its ``kind``.
3. Resolution: for entries pointing at ``topoteretes/cognee`` sources, the
   referenced files must exist in the current checkout. Cross-repo references
   (``topoteretes/cognee-community``, ``topoteretes/cognee-integrations``) are
   syntax-checked only; live resolution is deferred to ``inventory_sync.py``
   which fetches them via the GitHub API.

The loader is intentionally dependency-light: only ``pyyaml`` and
``jsonschema``, both already present in a standard cognee install, so the
catalog tooling adds nothing to the shipped package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_ROOT = Path(__file__).resolve().parent
ENTRIES_ROOT = CATALOG_ROOT / "entries"
SCHEMA_PATH = CATALOG_ROOT / "schema.json"

KIND_TO_SUBDIR = {
    "integration": "integrations",
    "use-case": "use-cases",
    "package": "packages",
}

LOCAL_REPO = "topoteretes/cognee"
EXTERNAL_REPOS = {"topoteretes/cognee-community", "topoteretes/cognee-integrations"}


class CatalogError(Exception):
    """Raised when the catalog fails validation.

    The ``errors`` attribute carries every discovered problem so a contributor
    fixes them in one pass instead of one CI cycle per typo.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("catalog validation failed:\n  - " + "\n  - ".join(errors))


@dataclass
class CatalogEntry:
    """A single loaded, validated catalog entry."""

    id: str
    title: str
    kind: str
    stack: str
    tags: list[str]
    summary: str
    what_youll_build: str
    quickstart: str
    expected_output: str
    difficulty: str
    source_path: Path
    repo: str | None = None
    path: str | None = None
    example_path: str | None = None
    inventory_slug: str | None = None
    docs_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path) -> CatalogEntry:
        return cls(
            id=data["id"],
            title=data["title"],
            kind=data["kind"],
            stack=data["stack"],
            tags=list(data["tags"]),
            summary=data["summary"],
            what_youll_build=data["what_youll_build"],
            quickstart=data["quickstart"],
            expected_output=data["expected_output"],
            difficulty=data["difficulty"],
            source_path=source_path,
            repo=data.get("repo"),
            path=data.get("path"),
            example_path=data.get("example_path"),
            inventory_slug=data.get("inventory_slug"),
            docs_url=data.get("docs_url"),
            raw=data,
        )


def _load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(source_path: Path) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise CatalogError(
            [f"{source_path}: top-level must be a mapping, got {type(loaded).__name__}"]
        )
    return loaded


def validate_entry(data: dict[str, Any], source_path: Path) -> list[str]:
    """Validate a single already-parsed entry.

    Returns a list of human-readable error strings. Empty list means valid.
    Kept public because both :func:`load_catalog` and the CI drift check use
    it, and future tooling (an editor plugin, a preview site) may want it too.
    """

    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors: list[str] = []

    for problem in validator.iter_errors(data):
        path = "/".join(str(segment) for segment in problem.absolute_path) or "<root>"
        errors.append(f"{source_path}: schema violation at {path}: {problem.message}")

    if not errors:
        entry_id = data.get("id")
        stem = source_path.stem
        if entry_id != stem:
            errors.append(f"{source_path}: id '{entry_id}' does not match filename stem '{stem}'")

        expected_subdir = KIND_TO_SUBDIR.get(data.get("kind", ""))
        if expected_subdir and source_path.parent.name != expected_subdir:
            errors.append(
                f"{source_path}: kind '{data.get('kind')}' expects the entry under "
                f"catalog/entries/{expected_subdir}/, found under "
                f"catalog/entries/{source_path.parent.name}/"
            )

        repo = data.get("repo")
        path = data.get("path")
        if repo == LOCAL_REPO and path:
            resolved = REPO_ROOT / path
            if not resolved.exists():
                errors.append(f"{source_path}: local path does not exist: {path}")
        elif repo is not None and repo not in EXTERNAL_REPOS and repo != LOCAL_REPO:
            errors.append(
                f"{source_path}: repo '{repo}' is not one of the known Cognee repos "
                f"({LOCAL_REPO}, {', '.join(sorted(EXTERNAL_REPOS))})"
            )

        example_path = data.get("example_path")
        if example_path and (repo in (None, LOCAL_REPO)):
            resolved = REPO_ROOT / example_path
            if not resolved.exists():
                errors.append(
                    f"{source_path}: example_path does not exist in the local checkout: {example_path}"
                )

    return errors


def load_catalog(entries_root: Path | None = None) -> list[CatalogEntry]:
    """Load every catalog entry, validate, and return a sorted list.

    Raises :class:`CatalogError` with the full list of problems on any
    validation failure. On success, entries are returned sorted by kind then id
    so downstream renderers get a stable order.
    """

    root = entries_root or ENTRIES_ROOT
    if not root.exists():
        raise CatalogError([f"catalog entries root does not exist: {root}"])

    all_errors: list[str] = []
    entries: list[CatalogEntry] = []
    seen_ids: dict[str, Path] = {}

    for source_path in sorted(root.rglob("*.yaml")):
        try:
            data = _load_yaml(source_path)
        except yaml.YAMLError as cause:
            all_errors.append(f"{source_path}: could not parse YAML: {cause}")
            continue
        except CatalogError as cause:
            all_errors.extend(cause.errors)
            continue

        errors = validate_entry(data, source_path)
        if errors:
            all_errors.extend(errors)
            continue

        entry_id = data["id"]
        prior = seen_ids.get(entry_id)
        if prior is not None:
            all_errors.append(f"{source_path}: duplicate id '{entry_id}' also seen at {prior}")
            continue
        seen_ids[entry_id] = source_path

        entries.append(CatalogEntry.from_dict(data, source_path))

    if all_errors:
        raise CatalogError(all_errors)

    return sorted(entries, key=lambda entry: (entry.kind, entry.id))


def main() -> int:
    """CLI entry point: ``python -m catalog.loader``.

    Returns exit code 0 on success, 1 on validation failure. Used by CI.
    """

    try:
        entries = load_catalog()
    except CatalogError as cause:
        print(str(cause))
        return 1

    print(f"loaded {len(entries)} catalog entries")
    by_kind: dict[str, int] = {}
    for entry in entries:
        by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
