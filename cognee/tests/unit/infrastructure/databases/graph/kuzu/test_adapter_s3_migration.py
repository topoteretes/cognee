import importlib.util
import os
import sys
import types
from types import ModuleType


class _DBOpenError(RuntimeError):
    pass


class _FakeDatabase:
    """Fake kuzu.Database that fails first, then succeeds."""

    calls = 0

    def __init__(self, path: str, **kwargs):
        _FakeDatabase.calls += 1
        if _FakeDatabase.calls == 1:
            raise _DBOpenError("version mismatch")

    def init_database(self):
        pass


class _FakeConnection:
    def __init__(self, db):
        pass

    def execute(self, query: str, params=None):
        class _Res:
            def has_next(self):
                return False

            def get_next(self):
                return []

        return _Res()


def _install_stub(name: str, module: ModuleType | None = None) -> ModuleType:
    mod = module or ModuleType(name)
    # Mark as package so submodule imports succeed when needed
    if not hasattr(mod, "__path__"):
        mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _find_repo_root(start_path: str) -> str:
    """Walk up directories until we find pyproject.toml (repo root)."""
    cur = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not locate repository root from: " + start_path)
        cur = parent


def _load_adapter_with_stubs(monkeypatch):
    # Provide fake 'kuzu' and submodules used by adapter imports
    kuzu_mod = _install_stub("kuzu")
    kuzu_mod.__dict__["__version__"] = "0.11.0"

    # Placeholders to satisfy adapter's "from kuzu import Connection" and "from kuzu.database import Database"
    class _PlaceholderConn:
        pass

    kuzu_mod.Connection = _PlaceholderConn
    kuzu_db_mod = _install_stub("kuzu.database")

    class _PlaceholderDB:
        pass

    kuzu_db_mod.Database = _PlaceholderDB

    # Create minimal stub tree for required cognee imports to avoid executing package __init__
    root = _install_stub("cognee")
    infra = _install_stub("cognee.infrastructure")
    databases = _install_stub("cognee.infrastructure.databases")
    graph = _install_stub("cognee.infrastructure.databases.graph")
    kuzu_pkg = _install_stub("cognee.infrastructure.databases.graph.kuzu")

    # graph_db_interface stub
    gdi_mod = _install_stub("cognee.infrastructure.databases.graph.graph_db_interface")

    class _GraphDBInterface:  # bare minimum
        pass

    def record_graph_changes(fn):
        return fn

    gdi_mod.GraphDBInterface = _GraphDBInterface
    gdi_mod.record_graph_changes = record_graph_changes

    # engine.DataPoint stub
    engine_mod = _install_stub("cognee.infrastructure.engine")

    class _DataPoint:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    engine_mod.DataPoint = _DataPoint

    # files.storage.get_file_storage stub
    files_storage_pkg = _install_stub("cognee.infrastructure.files")
    storage_pkg = _install_stub("cognee.infrastructure.files.storage")
    storage_pkg.get_file_storage = lambda path: types.SimpleNamespace(
        ensure_directory_exists=lambda: None
    )

    # utils.run_sync stub
    utils_pkg = _install_stub("cognee.infrastructure.utils")
    run_sync_mod = _install_stub("cognee.infrastructure.utils.run_sync")
    run_sync_mod.run_sync = lambda coro: None

    # modules.storage.utils JSONEncoder stub
    modules_pkg = _install_stub("cognee.modules")
    storage_pkg2 = _install_stub("cognee.modules.storage")
    utils_mod2 = _install_stub("cognee.modules.storage.utils")
    utils_mod2.JSONEncoder = object

    # shared.logging_utils.get_logger stub
    shared_pkg = _install_stub("cognee.shared")
    logging_utils_mod = _install_stub("cognee.shared.logging_utils")

    class _Logger:
        def debug(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    logging_utils_mod.get_logger = lambda: _Logger()

    # Now load adapter.py by path
    repo_root = _find_repo_root(os.path.dirname(__file__))
    adapter_path = os.path.join(
        repo_root, "cognee", "infrastructure", "databases", "graph", "kuzu", "adapter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cognee.infrastructure.databases.graph.kuzu.adapter", adapter_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # Replace Database/Connection in the loaded module
    monkeypatch.setattr(mod, "Database", _FakeDatabase, raising=True)
    monkeypatch.setattr(mod, "Connection", _FakeConnection, raising=True)

    # Patch migration helpers inside the kuzu_migrate module used by adapter
    # Load kuzu_migrate similarly
    km_path = os.path.join(
        repo_root, "cognee", "infrastructure", "databases", "graph", "kuzu", "kuzu_migrate.py"
    )
    km_spec = importlib.util.spec_from_file_location("kuzu_migrate_under_test", km_path)
    km_mod = importlib.util.module_from_spec(km_spec)
    assert km_spec and km_spec.loader
    km_spec.loader.exec_module(km_mod)  # type: ignore[attr-defined]

    calls = {"migrated": False}

    def fake_read_version(_):
        return "0.9.0"

    def fake_migration(**kwargs):
        calls["migrated"] = True

    monkeypatch.setattr(km_mod, "read_kuzu_storage_version", fake_read_version)
    monkeypatch.setattr(km_mod, "kuzu_migration", fake_migration)

    # Ensure adapter refers to our loaded km_mod
    monkeypatch.setitem(
        sys.modules, "cognee.infrastructure.databases.graph.kuzu.kuzu_migrate", km_mod
    )

    return mod, calls


def test_adapter_s3_auto_migration(monkeypatch):
    mod, calls = _load_adapter_with_stubs(monkeypatch)

    # ensure pull/push do not touch real S3
    monkeypatch.setattr(mod.KuzuAdapter, "pull_from_s3", lambda self: None)
    monkeypatch.setattr(mod.KuzuAdapter, "push_to_s3", lambda self: None)

    adapter = mod.KuzuAdapter("s3://bucket/db")
    assert calls["migrated"] is True
