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
import tempfile
from typing import Optional


def _safe_close(obj) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass


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
            except Exception:
                pass
        except Exception:
            # Best-effort install: missing/incompatible JSON extension and
            # init failures all surface here. The cleanup below still runs.
            pass
        finally:
            _safe_close(conn)
            _safe_close(tmp_db)
