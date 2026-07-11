"""Tests for the catalog loader's validation passes.

Each test writes YAML fixtures into a temporary entries root and asserts the
loader either returns the expected entries or raises ``CatalogError`` with a
message pointing at the specific problem. One smoke test loads the real shipped
catalog so a broken seed entry fails here too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from catalog.loader import CatalogError, load_catalog

# A minimal entry that passes every validation pass. ``repo`` points at an
# external repo so ``path`` is syntax-checked only (never resolved on disk).
VALID_INTEGRATION = {
    "id": "example-tool",
    "title": "Example tool integration",
    "kind": "integration",
    "stack": "framework",
    "tags": ["example", "python"],
    "summary": "A minimal valid integration entry used by the loader tests.",
    "what_youll_build": "Nothing real; this is a fixture for the loader tests.",
    "quickstart": "pip install example\npython run.py",
    "expected_output": "Some expected output text.",
    "difficulty": "easy",
    "repo": "topoteretes/cognee-community",
    "path": "packages/example",
}

SUBDIR = {"integration": "integrations", "use-case": "use-cases", "package": "packages"}


def _use_case(**overrides: object) -> dict:
    """A valid use-case entry whose example_path resolves against the repo root."""
    entry = {
        **VALID_INTEGRATION,
        "kind": "use-case",
        "example_path": "pyproject.toml",
    }
    entry.pop("repo")
    entry.pop("path")
    entry.update(overrides)
    return entry


def _write(root: Path, entry: dict, *, filename: str | None = None) -> Path:
    subdir = root / SUBDIR[entry["kind"]]
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / (filename or f"{entry['id']}.yaml")
    path.write_text(yaml.safe_dump(entry), encoding="utf-8")
    return path


def test_valid_entry_loads(tmp_path: Path) -> None:
    _write(tmp_path, VALID_INTEGRATION)
    entries = load_catalog(entries_root=tmp_path)
    assert [e.id for e in entries] == ["example-tool"]
    assert entries[0].kind == "integration"


def test_entries_sorted_by_kind_then_id(tmp_path: Path) -> None:
    _write(tmp_path, {**VALID_INTEGRATION, "id": "b-tool"})
    _write(tmp_path, {**VALID_INTEGRATION, "id": "a-tool"})
    _write(tmp_path, _use_case(id="a-case"))
    entries = load_catalog(entries_root=tmp_path)
    assert [(e.kind, e.id) for e in entries] == [
        ("integration", "a-tool"),
        ("integration", "b-tool"),
        ("use-case", "a-case"),
    ]


def test_missing_required_field_fails(tmp_path: Path) -> None:
    _write(tmp_path, {k: v for k, v in VALID_INTEGRATION.items() if k != "summary"})
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("summary" in e for e in exc.value.errors)


def test_id_must_match_filename_stem(tmp_path: Path) -> None:
    _write(tmp_path, VALID_INTEGRATION, filename="different-name.yaml")
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("does not match filename stem" in e for e in exc.value.errors)


def test_kind_must_match_subdirectory(tmp_path: Path) -> None:
    subdir = tmp_path / "use-cases"
    subdir.mkdir(parents=True)
    (subdir / "example-tool.yaml").write_text(yaml.safe_dump(VALID_INTEGRATION), encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("expects the entry under" in e for e in exc.value.errors)


def test_use_case_requires_example_path(tmp_path: Path) -> None:
    entry = _use_case()
    entry.pop("example_path")
    _write(tmp_path, entry)
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("example_path" in e for e in exc.value.errors)


def test_local_example_path_must_exist(tmp_path: Path) -> None:
    _write(tmp_path, _use_case(example_path="does/not/exist.py"))
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("does not exist" in e for e in exc.value.errors)


def test_duplicate_id_across_kinds_fails(tmp_path: Path) -> None:
    _write(tmp_path, VALID_INTEGRATION)
    _write(tmp_path, _use_case(id="example-tool"))
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert any("duplicate id" in e for e in exc.value.errors)


def test_all_errors_aggregated(tmp_path: Path) -> None:
    _write(tmp_path, {k: v for k, v in VALID_INTEGRATION.items() if k != "summary"})
    _write(tmp_path, _use_case(id="broken", example_path="nope.py"))
    with pytest.raises(CatalogError) as exc:
        load_catalog(entries_root=tmp_path)
    assert len(exc.value.errors) >= 2


def test_shipped_catalog_is_valid() -> None:
    entries = load_catalog()
    assert len(entries) >= 10
    assert {e.kind for e in entries} == {"integration", "use-case", "package"}
