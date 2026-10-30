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


def _load_stmt(target) -> str:
    """The exact statement load_json_extension emits for a bundled path.

    Mirrors production quoting: backslashes become forward slashes on every
    platform, so a literal comparison also holds on Windows.
    """
    escaped = str(target).replace("\\", "/").replace("'", "''")
    return f"LOAD EXTENSION '{escaped}';"


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
        _load_stmt(target),
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
            _load_stmt(target): RuntimeError("Failed to load library"),
        }
    )
    load_json_extension(execute)
    assert execute.statements == [
        "LOAD EXTENSION JSON;",
        PROBE_STMT,
        _load_stmt(target),
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


# --- Materializing the bundled binary into the engine's store ----------------
#
# ``Database(...)`` auto-loads every extension a database's catalog registers,
# from ``~/.lbdb/extension/<version>/<platform>/json/``. Databases created
# under older cognee images carry ``json`` in their catalog; an upgraded image
# ships the binary only inside ``ladybug_extensions/``, so the open dies with
# "IO exception: Failed to load library ..." before any LOAD-by-path fallback
# can run (no connection exists yet). Materializing the bundled binary at the
# engine-announced location makes those databases openable again — the fix
# for the docker 1.4.x → 1.6.x upgrade break (issue #5736).

MISSING_JSON_LIBRARY_ERROR = (
    "IO exception: Failed to load library: /app/.lbdb/extension/0.19.0/linux_amd64/json/"
    "libjson.lbug_extension which is needed by extension: json."
)


def test_missing_json_extension_error_is_matched_narrowly():
    from cognee_db_workers._kuzu_helpers import is_missing_json_extension_error

    assert is_missing_json_extension_error(MISSING_JSON_LIBRARY_ERROR)
    # Unrelated open failures must never be classified as a missing extension:
    assert not is_missing_json_extension_error("IO exception: database is locked")
    assert not is_missing_json_extension_error("IO exception: Permission denied")
    # A bare "Failed to load library" (e.g. a corrupted bundled binary at
    # LOAD time) is the loader ladder's business, not an open-time recovery:
    assert not is_missing_json_extension_error("Failed to load library")
    assert not is_missing_json_extension_error("wal replay failed: checksum mismatch")


def test_materialize_targets_engine_layout(monkeypatch, tmp_path):
    from cognee_db_workers._kuzu_helpers import materialize_bundled_json_extension

    target_root = tmp_path / "home" / ".lbdb"
    bundle = _bundle(tmp_path / "bundle")
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(target_root))

    execute = RecordingExecute()
    result = materialize_bundled_json_extension(execute)

    # The probe's "v" prefix is dropped and the trailing json/ dir added —
    # exactly the layout the engine auto-loads catalog-registered extensions
    # from (verified against the engine's own error path on 0.19.0/linux_amd64).
    expected = target_root / "extension" / "9.9.9" / "test_arch" / "json" / "libjson.lbug_extension"
    assert result == str(expected)
    assert expected.read_bytes() == bundle.read_bytes()
    assert execute.statements == [PROBE_STMT]


def test_materialize_skips_when_bundle_does_not_cover_announcement(monkeypatch, tmp_path):
    from cognee_db_workers._kuzu_helpers import materialize_bundled_json_extension

    target_root = tmp_path / "home" / ".lbdb"
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(target_root))

    # Probe announces a version/platform the bundle does not carry.
    execute = RecordingExecute(probe=probe_error("v9.9.9", "unbundled_platform"))
    assert materialize_bundled_json_extension(execute) is None
    assert not target_root.exists()


def test_materialize_skips_when_probe_is_unparseable(monkeypatch, tmp_path):
    from cognee_db_workers._kuzu_helpers import materialize_bundled_json_extension

    target_root = tmp_path / "home" / ".lbdb"
    _bundle(tmp_path / "bundle")
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(target_root))

    execute = RecordingExecute(probe=RuntimeError("some future error format"))
    assert materialize_bundled_json_extension(execute) is None
    assert not target_root.exists()


def test_materialize_never_rewrites_existing_binary(monkeypatch, tmp_path):
    from cognee_db_workers._kuzu_helpers import materialize_bundled_json_extension

    target_root = tmp_path / "home" / ".lbdb"
    _bundle(tmp_path / "bundle")
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(target_root))

    expected = target_root / "extension" / "9.9.9" / "test_arch" / "json" / "libjson.lbug_extension"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"already-there")
    # A concurrently running worker may have the file open/mapped — the file
    # must never be rewritten in place.
    assert materialize_bundled_json_extension(RecordingExecute()) == str(expected)
    assert expected.read_bytes() == b"already-there"


def test_warmup_materializes_instead_of_noop(monkeypatch, tmp_path):
    """The warm-up used to return immediately when a bundled binary existed,
    leaving ~/.lbdb empty — the root cause of the 1.4.x → 1.6.x upgrade break.
    It must now materialize the bundled binary into the engine's store (and
    still fall back to the remote INSTALL when it can't). The announcement is
    canned so the test is deterministic whether or not the local engine has
    json installed already (a real engine's probe only errors while the
    extension is missing)."""
    import ladybug

    requested = ("v9.9.9", "test_arch")
    bundle = _bundle(tmp_path / "bundle", *requested)
    monkeypatch.setattr(_kuzu_helpers, "_BUNDLED_EXTENSIONS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(_kuzu_helpers, "_LBUG_EXTENSION_ROOT", str(tmp_path / "home" / ".lbdb"))
    monkeypatch.setattr(_kuzu_helpers, "_requested_extension_relpath", lambda execute: requested)

    _kuzu_helpers.install_json_extension_local(buffer_pool_size=64 * 1024 * 1024)

    expected = (
        tmp_path
        / "home"
        / ".lbdb"
        / "extension"
        / requested[0].removeprefix("v")
        / requested[1]
        / "json"
        / "libjson.lbug_extension"
    )
    assert expected.read_bytes() == bundle.read_bytes()


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


def _pyproject_requirements() -> list[str]:
    return resolver.ladybug_requirements((REPO_ROOT / "pyproject.toml").read_text())


def test_filter_selects_range_from_real_listing():
    dirs = resolver.bundled_extension_dirs(GHCR_LISTING, _pyproject_requirements())
    assert dirs, "current pyproject constraints select no extension dirs"
    assert "vdev" not in dirs and "dataset" not in dirs
    lock_text = (REPO_ROOT / "uv.lock").read_text()
    # The lock pins one ladybug per platform-marker line; every pinned
    # version's dir (which may trail it) must be present: some selected dir
    # is <= the locked version and >= its true dir.
    locked_versions = re.findall(r'name = "ladybug"\nversion = "([^"]+)"', lock_text)
    assert locked_versions, "uv.lock pins no ladybug version"
    for locked in locked_versions:
        assert any(d == f"v{locked}" or d < f"v{locked}" for d in dirs), (
            f"no bundled dir serves locked ladybug {locked}"
        )


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

    for requirement in _pyproject_requirements():
        spec = SpecifierSet(requirement.removeprefix("ladybug").strip())
        for candidate in GHCR_LISTING:
            if not re.fullmatch(r"v\d+(\.\d+)*", candidate):
                continue
            version = candidate[1:]
            assert resolver.satisfies(version, requirement) == spec.contains(version), (
                f"resolver disagrees with packaging for ladybug {version} vs {requirement!r}"
            )


def test_extension_repo_digest_pinned_and_in_sync():
    """Both consumers of the extension-repo image must pin the same digest.

    Pinning by digest (not tag) is what keeps a compromised :latest from
    injecting binaries into wheels or the Docker image; keeping the two pins
    identical means the offline e2e validates the exact bytes both ship.
    """
    digest_pattern = re.compile(r"ghcr\.io/ladybugdb/extension-repo@(sha256:[0-9a-f]{64})")
    script = (REPO_ROOT / "scripts" / "fetch_ladybug_json_extension.sh").read_text()
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()

    script_digests = digest_pattern.findall(script)
    dockerfile_digests = digest_pattern.findall(dockerfile)
    assert script_digests, "fetch script no longer pins the extension-repo image by digest"
    assert dockerfile_digests, "Dockerfile no longer pins the extension-repo image by digest"
    assert set(script_digests) == set(dockerfile_digests), (
        "fetch script and Dockerfile pin different extension-repo digests — "
        "refresh them together (see the comment in scripts/fetch_ladybug_json_extension.sh)"
    )
    # Only the executable references count — the fetch script's comment may
    # (and does) mention :latest in the digest-refresh instructions.
    assert not re.search(r"^IMAGE=.*extension-repo:latest", script, re.MULTILINE), (
        "fetch script IMAGE= reverted to the unpinned :latest tag"
    )
    assert not re.search(
        r"^FROM ghcr\.io/ladybugdb/extension-repo:latest", dockerfile, re.MULTILINE
    ), "Dockerfile FROM reverted to the unpinned :latest tag"
