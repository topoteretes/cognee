"""Guardrail: the DB worker subprocesses must NOT import cognee.

This regressed silently in the past (old SubprocessGraphDBWrapper /
SubprocessVectorDBWrapper wrapped cognee adapter classes in the child,
dragging the entire ~200 MB cognee import graph into every child).

The new design puts worker code under the top-level ``cognee_db_workers``
package, which by construction imports only the native DB library (kuzu or
lancedb) + pyarrow + stdlib. This test enforces that invariant by spawning a
clean Python child and verifying ``cognee`` is absent from ``sys.modules``.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

import pytest


def _probe_kuzu_worker(result_q):
    # Import the worker; make sure ``cognee`` isn't dragged in.
    import cognee_db_workers.kuzu_worker  # noqa: F401

    cognee_modules = sorted(m for m in sys.modules if m == "cognee" or m.startswith("cognee."))
    result_q.put(cognee_modules)


def _probe_lancedb_worker(result_q):
    import cognee_db_workers.lancedb_worker  # noqa: F401

    cognee_modules = sorted(m for m in sys.modules if m == "cognee" or m.startswith("cognee."))
    result_q.put(cognee_modules)


def _probe_harness(result_q):
    import cognee_db_workers.harness  # noqa: F401

    cognee_modules = sorted(m for m in sys.modules if m == "cognee" or m.startswith("cognee."))
    result_q.put(cognee_modules)


def _run_probe(target) -> list:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    try:
        return q.get(timeout=30)
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)


def test_kuzu_worker_does_not_import_cognee():
    modules = _run_probe(_probe_kuzu_worker)
    assert modules == [], (
        f"Kuzu worker pulled cognee into the child's sys.modules: {modules}. "
        "Keep the worker module (cognee_db_workers.kuzu_worker) free of "
        "cognee imports."
    )


@pytest.mark.skipif(
    "lancedb" not in _run_probe.__globals__ and False,
    reason="lancedb is optional",
)
def test_lancedb_worker_does_not_import_cognee():
    # Skip gracefully if lancedb isn't installed in this environment.
    try:
        import lancedb  # noqa: F401
    except ImportError:
        pytest.skip("lancedb not installed")

    modules = _run_probe(_probe_lancedb_worker)
    assert modules == [], (
        f"LanceDB worker pulled cognee into the child's sys.modules: {modules}. "
        "Keep the worker module (cognee_db_workers.lancedb_worker) free of "
        "cognee imports."
    )


def test_harness_does_not_import_cognee():
    modules = _run_probe(_probe_harness)
    assert modules == [], (
        f"Harness pulled cognee into the child's sys.modules: {modules}. "
        "Keep cognee_db_workers.harness stdlib-only."
    )
