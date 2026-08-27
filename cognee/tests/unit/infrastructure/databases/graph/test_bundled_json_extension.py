"""Tests for the bundled Ladybug JSON extension loader.

Ladybug's Linux/Windows wheels dynamic-load the JSON extension from a remote
repo at runtime; ``load_json_extension`` lets cognee ship the binary instead
and load it by absolute path, so no network is needed. Which binary matches is
announced by the engine itself: ``INSTALL JSON FROM '<invalid path>'`` fails
instantly with the exact ``<version>/<platform>`` in the error (loading a
wrong-version binary can segfault, so nothing is ever guessed). The ladder
must be: by-name load → probe-announced bundled binary → remote INSTALL.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from cognee_db_workers import _kuzu_helpers
from cognee_db_workers._kuzu_helpers import (
    _PROBE_REPO,
    bundled_extensions_present,
    bundled_json_extension_path,
    load_json_extension,
)

REPO_ROOT = Path(__file__).resolve().parents[6]

NOT_INSTALLED = RuntimeError(
    "Binder exception: Extension: json is an official extension and has not been installed."
)


def probe_error(version="v9.9.9", platform="test_arch"):
    return RuntimeError(
        "IO exception: Failed to download extension: json at URL "
        f"{_PROBE_REPO}{version}/{platform}/json/libjson.lbug_extension "
        "(ERROR: Could not establish connection)"
    )


class RecordingExecute:
    """Fake connection.execute that scripts per-statement outcomes.

    Each configured failure fires once — mirroring the real engine, where a
    successful INSTALL makes the subsequent by-name LOAD succeed. The probe
    statement always fails (that is its purpose), so it re-raises repeatedly.
    """

    def __init__(self, failures=(), probe=None):
        self.statements = []
        self._failures = dict(failures)
        self._probe = probe if probe is not None else probe_error()

    def __call__(self, statement):
        self.statements.append(statement)
        if statement.startswith("INSTALL JSON FROM"):
            raise self._probe
        for prefix in list(self._failures):
            if statement.startswith(prefix):
                raise self._failures.pop(prefix)


PROBE_STMT = f"INSTALL JSON FROM '{_PROBE_REPO}';"


def _bundle(tmp_path, version="v9.9.9", platform="test_arch", content=b"\x7fELF"):
    target = tmp_path / version / platform / "libjson.lbug_extension"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    return target


def test_by_name_load_succeeds_first():
    execute = RecordingExecute()
    load_json_extension(execute)
    assert execute.statements == ["LOAD EXTENSION JSON;"]


def test_probe_announced_bundle_is_loaded(monkeypatch, tmp_path):
    target = _bundle(tmp_path)
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path))

    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": NOT_INSTALLED})
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        PROBE_STMT,
        f"LOAD EXTENSION '{target}';",
    ]


def test_remote_install_when_no_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path))

    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": NOT_INSTALLED})
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        PROBE_STMT,
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_remote_install_when_probe_is_unparseable(monkeypatch, tmp_path):
    _bundle(tmp_path)
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path))

    execute = RecordingExecute(
        failures={"LOAD EXTENSION JSON;": NOT_INSTALLED},
        probe=RuntimeError("some future error format"),
    )
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        PROBE_STMT,
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_broken_bundle_falls_back_to_remote_install(monkeypatch, tmp_path):
    target = _bundle(tmp_path, content=b"garbage")
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path))

    execute = RecordingExecute(
        failures={
            "LOAD EXTENSION JSON;": NOT_INSTALLED,
            f"LOAD EXTENSION '{target}';": RuntimeError("Failed to load library"),
        }
    )
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        PROBE_STMT,
        f"LOAD EXTENSION '{target}';",
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_unrelated_load_error_is_raised():
    error = RuntimeError("IO exception: database is locked")
    execute = RecordingExecute(failures={"LOAD EXTENSION JSON;": error})
    with pytest.raises(RuntimeError, match="database is locked"):
        load_json_extension(execute)
    assert execute.statements == ["LOAD EXTENSION JSON;"]


def test_probe_parses_real_error_shape(tmp_path):
    """The parser must handle the exact message ladybug emits (verified on
    0.16.0-0.18.2), including the no-trailing-slash concatenation quirk."""
    real = RuntimeError(
        "IO exception: Failed to download extension: json at URL "
        "/cognee-nonexistent-probev0.18.1/linux_arm64/json/libjson.lbug_extension "
        "(ERROR: Could not establish connection)"
    )
    execute = RecordingExecute(probe=real)
    assert _kuzu_helpers._requested_extension_relpath(execute) == ("v0.18.1", "linux_arm64")


def test_bundled_path_resolution(monkeypatch, tmp_path):
    execute = RecordingExecute(probe=probe_error("v1.2.3", "some_platform"))
    assert bundled_json_extension_path(execute, bundled_dir=str(tmp_path)) is None
    target = _bundle(tmp_path, "v1.2.3", "some_platform")
    assert bundled_json_extension_path(execute, bundled_dir=str(tmp_path)) == str(target)


def test_bundled_extensions_present(tmp_path):
    assert not bundled_extensions_present(str(tmp_path))
    _bundle(tmp_path)
    assert bundled_extensions_present(str(tmp_path))


# --- Source-of-truth guards -------------------------------------------------
#
# The ladybug constraint in pyproject.toml decides which extension versions
# get bundled; scripts/ladybug_extension_versions.py filters the extension
# repo's published dirs by it at fetch time. These tests pin that filter's
# semantics.

_spec = importlib.util.spec_from_file_location(
    "ladybug_extension_versions", REPO_ROOT / "scripts" / "ladybug_extension_versions.py"
)
resolver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resolver)

GHCR_LISTING = [
    "dataset",
    "v0.11.3",
    "v0.15.0",
    "v0.16.0",
    "v0.17.0",
    "v0.18.0",
    "v0.18.1",
    "v0.19.0",
    "vdev",
]


def _pyproject_requirement() -> str:
    return resolver.ladybug_requirement((REPO_ROOT / "pyproject.toml").read_text())


def test_filter_selects_range_from_real_listing():
    dirs = resolver.supported_extension_dirs(GHCR_LISTING, _pyproject_requirement())
    assert dirs, "current pyproject constraint selects no extension dirs"
    assert "vdev" not in dirs and "dataset" not in dirs
    lock_text = (REPO_ROOT / "uv.lock").read_text()
    locked = re.search(r'name = "ladybug"\nversion = "([^"]+)"', lock_text).group(1)
    # The locked version's dir (which may trail it) must be present: the
    # newest selected dir is <= the locked version and >= its true dir.
    assert any(d == f"v{locked}" or d < f"v{locked}" for d in dirs)


def test_filter_includes_trailing_dir_below_floor():
    # Floor 0.17.1 has no exact dir; its extension lives in v0.17.0, which
    # must be shipped even though 0.17.0 itself is outside the range.
    dirs = resolver.supported_extension_dirs(GHCR_LISTING, "ladybug>=0.17.1,<=0.18.1")
    assert dirs == ["v0.17.0", "v0.18.0", "v0.18.1"]


def test_filter_skips_below_floor_when_floor_has_exact_dir():
    dirs = resolver.supported_extension_dirs(GHCR_LISTING, "ladybug>=0.17.0,<=0.18.1")
    assert dirs == ["v0.17.0", "v0.18.0", "v0.18.1"]


def test_resolver_matches_packaging_semantics():
    """The stdlib-only comparator must agree with PEP 440 for plain versions."""
    from packaging.specifiers import SpecifierSet

    requirement = _pyproject_requirement()
    spec = SpecifierSet(requirement.removeprefix("ladybug").strip())
    for candidate in GHCR_LISTING:
        if not re.fullmatch(r"v\d+(\.\d+)*", candidate):
            continue
        version = candidate[1:]
        assert resolver.satisfies(version, requirement) == spec.contains(version), (
            f"resolver disagrees with packaging for ladybug {version}"
        )
