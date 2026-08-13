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
import re
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
# Ladybug's macOS wheels currently compile the JSON extension into the native
# library (plain ``LOAD EXTENSION JSON`` works offline), but the Linux and
# Windows wheels dynamic-load it: ``INSTALL JSON`` downloads a
# ``libjson.lbug_extension`` binary from http://extension.ladybugdb.com at
# runtime. To keep cognee working without that network dependency — and to
# cover macOS too should a future wheel stop embedding it — the official
# binaries are shipped inside the ``cognee_db_workers/ladybug_extensions/``
# package directory and loaded by absolute path: ``LOAD EXTENSION '<path>'``
# reads the file directly and never consults the remote repo. See
# ladybug_extensions/README.md for how the directory is populated.
#
# Which binary matches is decided by ladybug itself, not by a maintained
# mapping: ``INSTALL JSON FROM '<invalid local path>'`` fails instantly (the
# path is treated as an unreachable URL — no network involved) and the error
# message spells out the exact ``<version>/<platform>`` the installed binary
# requests. Never guess here: loading a wrong-version extension binary can
# segfault the process, so only the engine-announced path is ever loaded.

_BUNDLED_EXTENSIONS_DIR = os.path.join(os.path.dirname(__file__), "ladybug_extensions")

# Deliberately unreachable "repo": makes INSTALL fail instantly while its
# error reveals the version/platform path the binary wants (verified instant
# and offline on ladybug 0.16.0 through 0.18.2).
_PROBE_REPO = "/cognee-nonexistent-extension-repo/"

_EXTENSION_RELPATH_PATTERN = re.compile(r"(v[\d.]+)[/\\]([A-Za-z0-9_]+)[/\\]json[/\\]libjson")


def _requested_extension_relpath(
    execute: Callable[[str], object],
) -> Optional[tuple[str, str]]:
    """The ``(version_dir, platform)`` the installed ladybug requests, or None.

    Asks the engine itself via the failing-INSTALL probe, so there is nothing
    to maintain when ladybug versions change.
    """
    try:
        execute(f"INSTALL JSON FROM '{_PROBE_REPO}';")
    except Exception as error:
        match = _EXTENSION_RELPATH_PATTERN.search(str(error))
        if match:
            return match.group(1), match.group(2)
    return None


def bundled_extensions_present(bundled_dir: Optional[str] = None) -> bool:
    """True when any extension binary is bundled (cheap, connection-free)."""
    bundled_dir = bundled_dir or _BUNDLED_EXTENSIONS_DIR
    if not os.path.isdir(bundled_dir):
        return False
    for _root, _dirs, files in os.walk(bundled_dir):
        if any(name.endswith(".lbug_extension") for name in files):
            return True
    return False


def bundled_json_extension_path(
    execute: Callable[[str], object],
    bundled_dir: Optional[str] = None,
) -> Optional[str]:
    """Absolute path of the bundled JSON extension for this ladybug install.

    Returns None when the probe yields nothing or no binary is bundled for
    the announced version/platform (e.g. macOS while the extension is
    statically linked, or a version the bundle does not cover).
    """
    bundled_dir = bundled_dir or _BUNDLED_EXTENSIONS_DIR
    requested = _requested_extension_relpath(execute)
    if requested is None:
        return None
    version_dir, platform_token = requested
    path = os.path.join(bundled_dir, version_dir, platform_token, "libjson.lbug_extension")
    return path if os.path.isfile(path) else None


def load_json_extension(execute: Callable[[str], object]) -> None:
    """Load the JSON extension on a live connection without requiring network.

    Order: the by-name form first (succeeds on statically linked builds and
    when the extension is already installed), then the bundled binary the
    engine announces via the probe, then — only when neither applies — the
    classic INSTALL-from-remote-repo path. Raises when every applicable step
    fails.
    """
    try:
        execute("LOAD EXTENSION JSON;")
        return
    except Exception as error:
        if "not been installed" not in str(error):
            raise

    bundled = bundled_json_extension_path(execute)
    if bundled is not None:
        try:
            # Forward slashes work on every platform and keep Windows
            # backslashes from being read as escape sequences in the literal.
            escaped = bundled.replace("\\", "/").replace("'", "''")
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

    # Bundled binaries make the warm-up pointless: LOAD by absolute path
    # needs no pre-install and no network, so skip the throwaway database
    # (and its startup cost) entirely.
    if bundled_extensions_present():
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
