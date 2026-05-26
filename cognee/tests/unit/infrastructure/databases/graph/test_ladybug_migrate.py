"""Tests for Ladybug migration helpers."""

import os
import struct
import subprocess
import tempfile

import pytest

from cognee.infrastructure.databases.graph.ladybug.ladybug_migrate import (
    ladybug_version_mapping,
    run_migration_step,
    read_ladybug_storage_version,
)


def _write_catalog_kz(dir_path: str, version_code: int) -> str:
    """Write a fake catalog.kz with the given storage version_code.

    Format mirrors what read_ladybug_storage_version expects:
      3-byte magic 'KUZ' + 1 byte padding + 8 bytes little-endian uint64.
    """
    os.makedirs(dir_path, exist_ok=True)
    catalog = os.path.join(dir_path, "catalog.kz")
    with open(catalog, "wb") as f:
        f.write(b"KUZ\x00")
        f.write(struct.pack("<Q", version_code))
    return catalog


def test_read_ladybug_storage_version_known_code(tmp_path):
    # Pick any code that's actually in the mapping.
    code, expected = next(iter(ladybug_version_mapping.items()))
    _write_catalog_kz(str(tmp_path), code)
    assert read_ladybug_storage_version(str(tmp_path)) == expected


def test_read_ladybug_storage_version_unknown_code_raises(tmp_path):
    # Anything outside the known table — e.g., a code emitted by a newer
    # ladybug release that hasn't been added yet.
    unknown = max(ladybug_version_mapping.keys()) + 100
    _write_catalog_kz(str(tmp_path), unknown)
    with pytest.raises(ValueError, match="Could not map version_code"):
        read_ladybug_storage_version(str(tmp_path))


def test_run_migration_step_isolates_legacy_import_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.getcwd())
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        "cognee_db_workers.ladybug_migrate.subprocess.run",
        fake_run,
    )

    run_migration_step("/tmp/legacy/bin/python", "kuzu", "relative_db", "MATCH (n) RETURN n")

    args, kwargs = calls[0]
    assert args[:2] == ["/tmp/legacy/bin/python", "-c"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["cwd"] == tempfile.gettempdir()
    assert "PYTHONPATH" not in kwargs["env"]
    assert f"Database({os.path.abspath('relative_db')!r})" in args[2]
    assert "conn.execute('MATCH (n) RETURN n')" in args[2]
