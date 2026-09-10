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


def test_proxy_forwards_every_table_method_the_adapter_calls():
    # The adapter binds an open table to one of these before using it.
    called = _methods_called_on({"collection", "table"})
    assert called, "parsed no table calls — the adapter's naming changed"

    missing = {name for name in called if not hasattr(RemoteLanceDBTable, name)}

    assert not missing, (
        f"LanceDBAdapter calls {sorted(missing)} on a table, but RemoteLanceDBTable "
        "does not forward it. Subprocess mode is the default, so this raises "
        "AttributeError in normal operation while passing in local mode."
    )


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
