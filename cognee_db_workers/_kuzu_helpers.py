"""Tiny stdlib-only helpers shared between the Ladybug worker and the
local-mode adapter. Importable from either side without dragging in
``harness`` or ``cognee``.

Keep this module stdlib-only (apart from a lazy ``import ladybug`` inside
the function body). It's imported by both the cognee adapter (which runs in
the parent process with cognee available) and by
``cognee_db_workers.kuzu_worker`` (which runs in a spawned subprocess that
must NOT pull cognee in). Adding a top-level cognee import here would
silently regress that invariant — the subprocess would re-import cognee's
full ~200 MB dependency graph at start. The ``test_worker_import_hygiene.py``
test enforces the no-cognee rule, but keeping it documented at the source
avoids surprising contributors.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
import tempfile
from typing import Callable, Optional


def _safe_close(obj) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass


# --- Bundled JSON extension -------------------------------------------------
#
# Ladybug's macOS wheels compile the JSON extension into the native library
# (plain ``LOAD EXTENSION JSON`` works offline), but the Linux and Windows
# wheels dynamic-load it: ``INSTALL JSON`` downloads a ``libjson.lbug_extension``
# binary from http://extension.ladybugdb.com at runtime. To keep cognee working
# without that network dependency, the matching binaries can be shipped inside
# the ``cognee_db_workers/ladybug_extensions/`` package directory and loaded by
# absolute path — ``LOAD EXTENSION '<path>'`` reads the file directly and never
# consults the remote repo. See ladybug_extensions/README.md for how the
# directory is populated.

# Maps the installed ladybug package version to the extension-repo version
# directory its binary requests (baked into the native lib; it can trail the
# package version). Discover a new entry by running ``INSTALL JSON;`` offline —
# the error message prints the exact URL, e.g.
# ``.../v0.18.1/linux_arm64/json/libjson.lbug_extension``.
_EXTENSION_REPO_VERSIONS = {
    "0.18.1": "v0.18.1",
    "0.18.2": "v0.18.1",
}

_BUNDLED_EXTENSIONS_DIR = os.path.join(os.path.dirname(__file__), "ladybug_extensions")


def _ladybug_platform() -> Optional[str]:
    """The platform token ladybug uses in extension paths, or None if unknown."""
    machine = _platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        return None
    if sys.platform.startswith("linux"):
        return f"linux_{arch}"
    if sys.platform == "darwin":
        return f"osx_{arch}"
    if sys.platform == "win32":
        return f"win_{arch}"
    return None


def bundled_json_extension_path(
    bundled_dir: str = _BUNDLED_EXTENSIONS_DIR,
) -> Optional[str]:
    """Absolute path of the bundled JSON extension for this ladybug install.

    Returns None when no binary is bundled for the installed ladybug version
    and platform (e.g. macOS, where the extension is statically linked, or a
    version/platform the bundle does not cover).
    """
    import ladybug

    repo_version = _EXTENSION_REPO_VERSIONS.get(getattr(ladybug, "__version__", ""))
    plat = _ladybug_platform()
    if repo_version is None or plat is None:
        return None
    path = os.path.join(bundled_dir, repo_version, plat, "libjson.lbug_extension")
    return path if os.path.isfile(path) else None


def load_json_extension(execute: Callable[[str], object]) -> None:
    """Load the JSON extension on a live connection without requiring network.

    Order: the by-name form first (succeeds on statically linked builds and
    when the extension is already installed), then the bundled binary by
    absolute path, then — only when neither applies — the classic
    INSTALL-from-remote-repo path. Raises when every applicable step fails.
    """
    try:
        execute("LOAD EXTENSION JSON;")
        return
    except Exception as error:
        if "not been installed" not in str(error):
            raise

    bundled = bundled_json_extension_path()
    if bundled is not None:
        try:
            escaped = bundled.replace("'", "''")
            execute(f"LOAD EXTENSION '{escaped}';")
            return
        except Exception as error:
            # A bundled binary that fails to dlopen (e.g. a glibc build on a
            # musl system) should not strand the user: fall through to the
            # remote install below, which serves the correct binary.
            print(
                f"[ladybug worker] bundled JSON extension failed to load: {error!r}",
                file=sys.stderr,
            )

    execute("INSTALL JSON;")
    execute("LOAD EXTENSION JSON;")


def install_json_extension_local(
    buffer_pool_size: int,
    max_db_size: Optional[int] = None,
) -> None:
    """Install Ladybug's JSON extension via a throwaway database.

    The extension must be installed against an empty Ladybug database before
    the real database is opened — otherwise queries that touch JSON fail
    with a confusing "extension not loaded" error. Best-effort: any failure
    is swallowed (already-installed and offline-machine cases both look
    like raises here).

    Uses ``TemporaryDirectory`` rather than ``NamedTemporaryFile`` so the
    path can be reopened by Ladybug on Windows, where an open
    ``NamedTemporaryFile`` cannot be reopened by another handle. Same
    pattern as
    ``cognee/infrastructure/databases/graph/ladybug/ladybug_migrate.py``.
    """
    import ladybug

    # A bundled binary makes the warm-up pointless: LOAD by absolute path
    # needs no pre-install and no network, so skip the throwaway database
    # (and its startup cost) entirely.
    if bundled_json_extension_path() is not None:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = os.path.join(tmp_dir, "ladybug-json-install")
        # Initialize handles to None so cleanup in ``finally`` works even if
        # ``Database(...)`` itself raises (e.g. invalid kwargs, OOM at init).
        # Without this, an outer-except-only flow would skip ``tmp_db.close()``
        # and leak the native object until GC.
        tmp_db = None
        conn = None
        try:
            kwargs = {"buffer_pool_size": buffer_pool_size}
            if max_db_size is not None:
                kwargs["max_db_size"] = max_db_size
            tmp_db = ladybug.Database(temp_db_path, **kwargs)
            tmp_db.init_database()
            conn = ladybug.Connection(tmp_db)
            try:
                conn.execute("INSTALL JSON;")
            except Exception as error:
                # Still best-effort (LOAD EXTENSION retries the install on
                # the live connection), but say why it failed — a silent
                # swallow here made "has not been installed" errors at LOAD
                # time impossible to diagnose from CI logs.
                print(
                    f"[ladybug worker] warm-up INSTALL JSON failed: {error!r}",
                    file=sys.stderr,
                )
        except Exception as error:
            # Best-effort install: missing/incompatible JSON extension and
            # init failures all surface here. The cleanup below still runs.
            print(
                f"[ladybug worker] warm-up JSON install setup failed: {error!r}",
                file=sys.stderr,
            )
        finally:
            _safe_close(conn)
            _safe_close(tmp_db)
