"""The docker 1.4.x → 1.6.x upgrade must not strand existing graph databases.

A database created under an older image has ``json`` registered in its
catalog (1.4.x ran ``INSTALL JSON;`` at runtime). On open, the engine
auto-loads every catalog-registered extension from
``~/.lbdb/extension/<version>/<platform>/json/`` — but an upgraded image
ships the binary only inside ``ladybug_extensions/``, so ``Database(...)``
itself dies with ``IO exception: Failed to load library ...`` before the
``load_json_extension`` ladder (which runs on a live connection, i.e. only
after a successful open) can help. Both open paths must recognize that one
error, materialize the bundled binary where the engine looks, and retry.

The worker test simulates the engine's open failure with a stub because
macOS wheels — where the unit tests run — statically link json and thus
cannot be made to fail for real; the stub raises the exact error text from
the issue and only for the upgraded database (a fresh/throwaway database
opens fine, which is why fresh installs are unaffected). The adapter test
runs the real local engine end to end; only the version/platform
announcement is canned, so tests stay deterministic whether or not the
local engine already has json installed.
"""

from pathlib import Path
from types import SimpleNamespace

import ladybug
import pytest

from cognee_db_workers import _kuzu_helpers, kuzu_worker
from cognee_db_workers.harness import HandleRegistry


def _canned_announcement(monkeypatch) -> tuple[str, str]:
    """Canned announcement so the test is deterministic whether or not the
    local engine already has json installed (a real engine's probe only
    errors while the extension is missing — the parse of that error shape is
    covered by ``test_probe_parses_real_error_shape``)."""
    requested = ("v9.9.9", "test_arch")
    monkeypatch.setattr(_kuzu_helpers, "_requested_extension_relpath", lambda execute: requested)
    return requested


def test_worker_open_recovers_missing_catalog_json_extension(monkeypatch, tmp_path):
    """Regression test for the issue's traceback site: the worker's
    ``_open_database`` must self-heal instead of dying on the first open."""
    announcement = ("v9.9.9", "test_arch")  # canned: the stubbed probe's answer

    bundle_dir = tmp_path / "bundle"
    bundled = bundle_dir / announcement[0] / announcement[1] / "libjson.lbug_extension"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"\x7fELF-fake-json-extension")

    engine_store = tmp_path / "home" / ".lbdb"
    expected_str = str(
        engine_store
        / "extension"
        / announcement[0].removeprefix("v")
        / announcement[1]
        / "json"
        / "libjson.lbug_extension"
    )
    upgraded_str = str(tmp_path / "user-graph.pkl")
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(bundle_dir))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(engine_store))

    def stub_init(self, *args, **kwargs):
        # Engine behavior: a database whose catalog registers json fails at
        # open while the binary is absent from the engine store; a fresh or
        # throwaway database (empty catalog) opens fine.
        if kwargs.get("database_path") == upgraded_str and not Path(expected_str).exists():
            raise RuntimeError(
                "IO exception: Failed to load library: "
                f"{expected_str} which is needed by extension: json."
            )

    def noop(self):
        pass

    Stub = type("Stub", (), {"__init__": stub_init, "init_database": noop, "close": noop})

    def conn_init(self, db):
        pass

    def conn_execute(self, statement):
        if statement.startswith("INSTALL JSON FROM"):
            raise RuntimeError(
                "IO exception: Failed to download extension: json at URL "
                f"{_kuzu_helpers._PROBE_REPO}{announcement[0]}/{announcement[1]}"
                "/json/libjson.lbug_extension (ERROR: Could not establish connection)"
            )

    ProbeConn = type(
        "ProbeConn", (), {"__init__": conn_init, "execute": conn_execute, "close": noop}
    )

    monkeypatch.setattr(ladybug, "Database", Stub)
    monkeypatch.setattr(ladybug, "Connection", ProbeConn)

    # The bug: the engine's own open fails while the store is empty.
    assert not Path(expected_str).exists()
    with pytest.raises(RuntimeError, match="Failed to load library"):
        Stub(database_path=upgraded_str)

    registry = HandleRegistry()
    result = kuzu_worker._open_database(
        registry, SimpleNamespace(kwargs={"database_path": upgraded_str})
    )

    assert registry.get(result.handle_id) is not None
    assert Path(expected_str).exists(), "bundled binary was not materialized"
    assert Path(expected_str).read_bytes() == bundled.read_bytes()


def test_adapter_warmup_materializes_bundled_binary(monkeypatch, tmp_path):
    """The local-mode adapter warms up before opening; with a bundled binary
    present that warm-up used to be a no-op, leaving ~/.lbdb empty for
    upgraded databases. It must leave the engine's store populated."""
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    version_dir, platform = _canned_announcement(monkeypatch)
    bundle_dir = tmp_path / "bundle"
    bundled = bundle_dir / version_dir / platform / "libjson.lbug_extension"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"\x7fELF-fake-json-extension")

    engine_store = tmp_path / "home" / ".lbdb"
    expected = (
        engine_store
        / "extension"
        / version_dir.removeprefix("v")
        / platform
        / "json"
        / "libjson.lbug_extension"
    )
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(bundle_dir))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(engine_store))

    adapter = LadybugAdapter(db_path=str(tmp_path / "graph"))

    assert adapter.connection is not None
    assert expected.exists(), "warm-up did not materialize the bundled binary"
    assert expected.read_bytes() == bundled.read_bytes()
