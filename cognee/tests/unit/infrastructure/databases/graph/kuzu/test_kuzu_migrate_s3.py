import io
import os
import struct
import importlib.util
from types import ModuleType
from typing import Dict

import pytest


class _FakeS3:
    """
    Minimal fake S3 client implementing the subset used by kuzu_migrate helpers.

    Store layout is a dict mapping string keys to bytes (files). Directories are
    implicit via key prefixes. Methods operate on s3:// style keys.
    """

    def __init__(self, initial: Dict[str, bytes] | None = None):
        self.store: Dict[str, bytes] = dict(initial or {})

    # Helpers
    def _norm(self, path: str) -> str:
        return path.rstrip("/")

    def _is_prefix(self, prefix: str, key: str) -> bool:
        p = self._norm(prefix)
        return key == p or key.startswith(p + "/")

    # API used by kuzu_migrate
    def exists(self, path: str) -> bool:
        p = self._norm(path)
        if p in self.store:
            return True
        # any key under this prefix implies existence as a directory
        return any(self._is_prefix(p, k) for k in self.store)

    def isdir(self, path: str) -> bool:
        p = self._norm(path)
        # A directory is assumed if there is any key with this prefix and that key isn't exactly the same
        return any(self._is_prefix(p, k) and k != p for k in self.store)

    def isfile(self, path: str) -> bool:
        p = self._norm(path)
        return p in self.store

    def open(self, path: str, mode: str = "rb"):
        p = self._norm(path)
        if "r" in mode:
            if p not in self.store:
                raise FileNotFoundError(p)
            return io.BytesIO(self.store[p])
        elif "w" in mode:
            buf = io.BytesIO()

            def _close():
                self.store[p] = buf.getvalue()

            # monkeypatch close so that written data is persisted on close
            orig_close = buf.close

            def close_wrapper():
                _close()
                orig_close()

            buf.close = close_wrapper  # type: ignore[assignment]
            return buf
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def copy(self, src: str, dst: str, recursive: bool = True):
        s = self._norm(src)
        d = self._norm(dst)
        if recursive:
            # copy all keys under src prefix to dst prefix
            to_copy = [k for k in self.store if self._is_prefix(s, k)]
            for key in to_copy:
                new_key = key.replace(s, d, 1)
                self.store[new_key] = self.store[key]
        else:
            if s not in self.store:
                raise FileNotFoundError(s)
            self.store[d] = self.store[s]

    def rm(self, path: str, recursive: bool = False):
        p = self._norm(path)
        if recursive:
            for key in list(self.store.keys()):
                if self._is_prefix(p, key):
                    del self.store[key]
        else:
            if p in self.store:
                del self.store[p]
            else:
                raise FileNotFoundError(p)


def _load_module_by_path(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kuzu_migrate_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_repo_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not locate repository root from: " + start_path)
        cur = parent


@pytest.fixture
def km_module(monkeypatch):
    # Load the kuzu_migrate module directly from file to avoid importing package __init__
    repo_root = _find_repo_root(os.path.dirname(__file__))
    target = os.path.join(
        repo_root, "cognee", "infrastructure", "databases", "graph", "kuzu", "kuzu_migrate.py"
    )
    mod = _load_module_by_path(target)
    return mod


@pytest.fixture
def patch_get_s3_client(monkeypatch, km_module):
    # Provide each test with its own fake client instance
    client = _FakeS3()
    monkeypatch.setattr(km_module, "_get_s3_client", lambda: client)
    return client


def _make_catalog_bytes(version_code: int) -> bytes:
    # 4 bytes header skipped + 8 bytes little-endian version code
    return b"KUZ\x00" + struct.pack("<Q", version_code) + b"padding"


def test_read_kuzu_storage_version_from_s3_directory(monkeypatch, patch_get_s3_client, km_module):
    s3 = patch_get_s3_client
    # Simulate a directory with catalog.kz
    dir_key = "s3://bucket/db"
    catalog_key = dir_key + "/catalog.kz"
    s3.store[catalog_key] = _make_catalog_bytes(39)  # maps to 0.11.0

    assert km_module.read_kuzu_storage_version(dir_key) == "0.11.0"


def test_s3_rename_file_backup(monkeypatch, patch_get_s3_client, km_module):
    s3 = patch_get_s3_client

    old_db = "s3://bucket/graph.db"
    new_db = "s3://bucket/graph_new.db"
    # seed store
    s3.store[old_db] = b"OLD"
    s3.store[new_db] = b"NEW"

    km_module._s3_rename_databases(old_db, "0.9.0", new_db, delete_old=False)

    # old is replaced with new
    assert s3.store.get(old_db) == b"NEW"
    # backup exists with version suffix
    backup = "s3://bucket/graph.db_old_0_9_0"
    assert s3.store.get(backup) == b"OLD"
    # staging removed
    assert new_db not in s3.store


def test_s3_rename_directory_delete_old(monkeypatch, patch_get_s3_client, km_module):
    s3 = patch_get_s3_client

    old_dir = "s3://bucket/graph_dir"
    new_dir = "s3://bucket/graph_dir_new"

    # Represent a directory by multiple keys under the prefix
    s3.store[old_dir + "/catalog.kz"] = b"OLD1"
    s3.store[old_dir + "/data.bin"] = b"OLD2"

    s3.store[new_dir + "/catalog.kz"] = b"NEW1"
    s3.store[new_dir + "/data.bin"] = b"NEW2"

    km_module._s3_rename_databases(old_dir, "0.9.0", new_dir, delete_old=True)

    # old dir contents replaced by new
    assert s3.store.get(old_dir + "/catalog.kz") == b"NEW1"
    assert s3.store.get(old_dir + "/data.bin") == b"NEW2"

    # no backup created when delete_old=True
    backup_prefix = os.path.dirname(old_dir) + "/" + os.path.basename(old_dir) + "_old_0_9_0"
    assert not any(k.startswith(backup_prefix) for k in s3.store)

    # staging removed
    assert not any(k.startswith(new_dir) for k in s3.store)
