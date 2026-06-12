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


def test_install_retries_through_transient_failures(monkeypatch):
    """A transient network failure during INSTALL must not kill the load."""
    import cognee_db_workers.kuzu_worker as worker

    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)

    class FlakyConnection:
        def __init__(self, install_failures: int):
            self.install_failures = install_failures
            self.installed = False
            self.executed = []

        def execute(self, query: str):
            self.executed.append(query)
            if query.startswith("LOAD EXTENSION") and not self.installed:
                raise RuntimeError(
                    "Binder exception: Extension: json is an official extension and "
                    "has not been installed."
                )
            if query.startswith("INSTALL"):
                if self.install_failures > 0:
                    self.install_failures -= 1
                    raise RuntimeError("IO exception: download failed")
                self.installed = True

    conn = FlakyConnection(install_failures=2)
    _load_extension(FakeRegistry(conn), _request())
    assert conn.executed == [
        "LOAD EXTENSION JSON;",
        "INSTALL JSON;",
        "INSTALL JSON;",
        "INSTALL JSON;",
        "LOAD EXTENSION JSON;",
    ]


def test_install_gives_up_after_max_attempts(monkeypatch):
    import cognee_db_workers.kuzu_worker as worker

    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)

    class AlwaysFailingConnection:
        def __init__(self):
            self.install_attempts = 0

        def execute(self, query: str):
            if query.startswith("LOAD EXTENSION"):
                raise RuntimeError("Extension: json ... has not been installed.")
            if query.startswith("INSTALL"):
                self.install_attempts += 1
                raise RuntimeError("IO exception: download failed")

    conn = AlwaysFailingConnection()
    with pytest.raises(RuntimeError, match="download failed"):
        _load_extension(FakeRegistry(conn), _request())
    assert conn.install_attempts == worker.INSTALL_ATTEMPTS
