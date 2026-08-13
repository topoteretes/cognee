"""Tests for the bundled Ladybug JSON extension loader.

Ladybug's Linux/Windows wheels dynamic-load the JSON extension from a remote
repo at runtime; ``load_json_extension`` lets cognee ship the binary instead
and load it by absolute path, so no network is needed. The ladder must be:
by-name load → bundled binary by path → INSTALL from remote repo.
"""

import sys

import pytest

from cognee_db_workers import _kuzu_helpers
from cognee_db_workers._kuzu_helpers import (
    _ladybug_platform,
    bundled_json_extension_path,
    load_json_extension,
)

NOT_INSTALLED = RuntimeError(
    "Binder exception: Extension: json is an official extension and has not been installed."
)


class RecordingExecute:
    """Fake connection.execute that scripts per-statement outcomes.

    Each configured failure fires once — mirroring the real engine, where a
    successful INSTALL makes the subsequent by-name LOAD succeed.
    """

    def __init__(self, failures=()):
        self.statements = []
        self._failures = dict(failures)

    def __call__(self, statement):
        self.statements.append(statement)
        for prefix in list(self._failures):
            if statement.startswith(prefix):
                raise self._failures.pop(prefix)


def test_by_name_load_succeeds_first(monkeypatch):
    execute = RecordingExecute()
    load_json_extension(execute)
    assert execute.statements == ["LOAD EXTENSION JSON;"]


def test_bundled_path_used_when_not_installed(monkeypatch, tmp_path):
    bundled = tmp_path / "libjson.lbug_extension"
    bundled.write_bytes(b"\x7fELF")
    monkeypatch.setattr(_kuzu_helpers, "bundled_json_extension_path", lambda *a, **k: str(bundled))

    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": NOT_INSTALLED})
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        f"LOAD EXTENSION '{bundled}';",
    ]


def test_remote_install_when_no_bundle(monkeypatch):
    monkeypatch.setattr(_kuzu_helpers, "bundled_json_extension_path", lambda *a, **k: None)

    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": NOT_INSTALLED})
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_broken_bundle_falls_back_to_remote_install(monkeypatch, tmp_path):
    bundled = tmp_path / "libjson.lbug_extension"
    bundled.write_bytes(b"garbage")
    monkeypatch.setattr(_kuzu_helpers, "bundled_json_extension_path", lambda *a, **k: str(bundled))

    execute = RecordingExecute(
        failures={
            "LOAD EXTENSION JSON;": NOT_INSTALLED,
            f"LOAD EXTENSION '{bundled}';": RuntimeError("Failed to load library"),
        }
    )
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        f"LOAD EXTENSION '{bundled}';",
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_unrelated_load_error_is_raised(monkeypatch):
    error = RuntimeError("IO exception: database is locked")
    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": error})
    with pytest.raises(RuntimeError, match="database is locked"):
        load_json_extension(execute)
    assert execute.statements == ["LOAD EXTENSION JSON;"]


def test_bundled_path_resolution(monkeypatch, tmp_path):
    plat = _ladybug_platform()
    if plat is None:
        pytest.skip("unknown platform token")
    monkeypatch.setattr(_kuzu_helpers, "_EXTENSION_REPO_VERSIONS", {_ladybug_version(): "v9.9.9"})
    target = tmp_path / "v9.9.9" / plat / "libjson.lbug_extension"
    target.parent.mkdir(parents=True)

    # Missing file -> None; present file -> its absolute path.
    assert bundled_json_extension_path(bundled_dir=str(tmp_path)) is None
    target.write_bytes(b"\x7fELF")
    assert bundled_json_extension_path(bundled_dir=str(tmp_path)) == str(target)


def test_unknown_ladybug_version_resolves_to_none(monkeypatch, tmp_path):
    monkeypatch.setattr(_kuzu_helpers, "_EXTENSION_REPO_VERSIONS", {})
    assert bundled_json_extension_path(bundled_dir=str(tmp_path)) is None


def test_platform_token_shape():
    plat = _ladybug_platform()
    if plat is None:
        pytest.skip("unknown platform token")
    os_part, arch_part = plat.split("_")
    assert os_part in {"linux", "osx", "win"}
    assert arch_part in {"amd64", "arm64"}
    if sys.platform == "darwin":
        assert os_part == "osx"


def _ladybug_version() -> str:
    import ladybug

    return getattr(ladybug, "__version__", "")
