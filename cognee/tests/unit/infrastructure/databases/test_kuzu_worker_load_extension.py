"""Unit tests for the Kuzu worker's LOAD EXTENSION retry-with-install path.

The warm-up ``INSTALL JSON`` on a throwaway database is best-effort and can
fail silently (e.g. a transient network error downloading the extension on a
fresh CI runner). ``_load_extension`` must then install on the live
connection and retry, instead of surfacing Ladybug's
"Extension: json ... has not been installed" binder error.

Pure tests: fake connection/registry, no ladybug, no subprocess.
"""

import pytest

from cognee_db_workers.harness import Request
from cognee_db_workers.kuzu_protocol import OP_LOAD_EXTENSION
from cognee_db_workers.kuzu_worker import _load_extension


class FakeConnection:
    def __init__(self, installed: bool):
        self.installed = installed
        self.executed = []

    def execute(self, query: str):
        self.executed.append(query)
        if query.startswith("LOAD EXTENSION") and not self.installed:
            raise RuntimeError(
                "Binder exception: Extension: json is an official extension and has "
                "not been installed.\nYou can install it by: install json."
            )
        if query.startswith("INSTALL"):
            self.installed = True


class FakeRegistry:
    def __init__(self, conn):
        self._conn = conn

    def get(self, hid):
        return self._conn


def _request():
    return Request(op=OP_LOAD_EXTENSION, handle_id=1, args=("JSON",))


def test_load_succeeds_directly_when_installed():
    conn = FakeConnection(installed=True)
    _load_extension(FakeRegistry(conn), _request())
    assert conn.executed == ["LOAD EXTENSION JSON;"]


def test_load_installs_and_retries_when_not_installed():
    conn = FakeConnection(installed=False)
    _load_extension(FakeRegistry(conn), _request())
    assert conn.executed == [
        "LOAD EXTENSION JSON;",
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_unrelated_load_errors_propagate_without_install():
    class BrokenConnection:
        def __init__(self):
            self.executed = []

        def execute(self, query: str):
            self.executed.append(query)
            raise RuntimeError("IO exception: database is locked")

    conn = BrokenConnection()
    with pytest.raises(RuntimeError, match="database is locked"):
        _load_extension(FakeRegistry(conn), _request())
    assert conn.executed == ["LOAD EXTENSION JSON;"]
