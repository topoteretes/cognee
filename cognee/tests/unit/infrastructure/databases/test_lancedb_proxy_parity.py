"""The subprocess proxy must expose every table method the adapter calls.

Subprocess mode is the DEFAULT for LanceDB
(``vector_db_subprocess_enabled = True``), so a table method the adapter uses
but the proxy does not forward works in local mode and raises AttributeError
in normal operation. That is how ``update_payload`` shipped calling
``collection.schema()`` against a proxy that had no ``schema``: the
incremental update's chunk-renumbering path failed on the default backend.

These are static checks — they neither spawn a worker nor need lancedb — so
the parity gap is caught wherever the suite runs.
"""

import ast
from pathlib import Path

import pytest

from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable

_ADAPTER = (
    Path(__file__).resolve().parents[4]
    / "infrastructure"
    / "databases"
    / "vector"
    / "lancedb"
    / "LanceDBAdapter.py"
)


def _methods_called_on(variable_names: set) -> set:
    """Attribute names the adapter calls on a table handle.

    Finds ``<name>.<attr>(...)`` for the locals the adapter binds a table to.
    """
    tree = ast.parse(_ADAPTER.read_text())
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in variable_names
        ):
            called.add(func.attr)
    return called


# Table methods the adapter calls only on its local-mode branch (guarded by
# ``not self._subprocess_mode`` in ``_compact_collection``): in subprocess mode
# the compaction crosses the boundary as ``compact_fragments`` /
# ``prune_versions`` and the worker calls these on the real table.
_LOCAL_ONLY_TABLE_METHODS = {"checkout_latest"}


def test_proxy_forwards_every_table_method_the_adapter_calls():
    # The adapter binds an open table to one of these before using it.
    called = _methods_called_on({"collection", "table"})
    assert called, "parsed no table calls — the adapter's naming changed"

    missing = {
        name
        for name in called
        if name not in _LOCAL_ONLY_TABLE_METHODS and not hasattr(RemoteLanceDBTable, name)
    }

    assert not missing, (
        f"LanceDBAdapter calls {sorted(missing)} on a table, but RemoteLanceDBTable "
        "does not forward it. Subprocess mode is the default, so this raises "
        "AttributeError in normal operation while passing in local mode."
    )


def test_optimize_mirrors_lancedb_signature():
    """``optimize`` is lancedb's method, so the proxy takes exactly its arguments.

    Code calling ``optimize(cleanup_older_than=...)`` must behave the same in
    local and subprocess mode; cognee's own compaction has separate methods.
    """
    import inspect

    lancedb_table = pytest.importorskip("lancedb.table")

    def parameters(method):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(method).parameters.values()
            if p.name != "self"
        ]

    assert parameters(RemoteLanceDBTable.optimize) == parameters(lancedb_table.AsyncTable.optimize)


def test_compaction_ops_are_wired_end_to_end():
    from cognee_db_workers.lancedb_protocol import (
        OP_TABLE_COMPACT_FRAGMENTS,
        OP_TABLE_OPTIMIZE,
        OP_TABLE_PRUNE_VERSIONS,
    )
    from cognee_db_workers.lancedb_worker import DISPATCH

    for op in (OP_TABLE_OPTIMIZE, OP_TABLE_COMPACT_FRAGMENTS, OP_TABLE_PRUNE_VERSIONS):
        assert op in DISPATCH


def test_schema_is_forwarded():
    """Pins the specific gap that broke update_payload's renumbering path."""
    assert hasattr(RemoteLanceDBTable, "schema")


def test_schema_op_is_wired_end_to_end():
    """Proxy op, worker handler, and dispatch entry must agree."""
    from cognee_db_workers.lancedb_protocol import OP_TABLE_SCHEMA
    from cognee_db_workers.lancedb_worker import DISPATCH

    assert OP_TABLE_SCHEMA in DISPATCH, "worker does not handle OP_TABLE_SCHEMA"


def test_op_codes_are_unique():
    """A duplicated op-code silently routes one operation to another's handler."""
    import cognee_db_workers.lancedb_protocol as protocol

    codes = {
        name: value
        for name, value in vars(protocol).items()
        if name.startswith("OP_") and isinstance(value, int)
    }
    duplicates = {code for code in codes.values() if list(codes.values()).count(code) > 1}
    assert not duplicates, f"duplicate op-codes {duplicates} in {codes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
