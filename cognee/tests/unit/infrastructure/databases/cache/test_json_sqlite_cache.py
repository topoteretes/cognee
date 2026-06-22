"""Regression tests for the JSON-only SQLite cache store.

These guard the GHSA-w8v5-vhqr-4h9v remediation (issue #2957): the filesystem
session cache must NOT depend on ``diskcache`` (whose default serialization can
fall back to ``pickle``). The store below only ever persists JSON strings.
"""

import importlib
import json
import tempfile

import pytest

from cognee.infrastructure.databases.cache.fscache.json_sqlite_cache import JsonSqliteCache


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        inst = JsonSqliteCache(directory=tmpdir)
        yield inst
        inst.close()


def test_diskcache_not_imported_by_fs_adapter():
    """The FS cache adapter must not import the vulnerable ``diskcache`` package."""
    adapter_module = importlib.import_module(
        "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter"
    )
    source = importlib.util.find_spec(adapter_module.__name__).origin
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()
    assert "import diskcache" not in text
    assert "diskcache as dc" not in text


def test_diskcache_not_a_declared_dependency():
    """``diskcache`` must not be a declared runtime dependency in pyproject.toml."""
    from pathlib import Path

    pyproject = Path(__file__).resolve()
    # Walk up to the repo root that contains pyproject.toml.
    for parent in pyproject.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            content = candidate.read_text(encoding="utf-8")
            break
    else:  # pragma: no cover - defensive
        pytest.skip("pyproject.toml not found")

    # No dependency line should declare diskcache (e.g. "diskcache>=...").
    for line in content.splitlines():
        stripped = line.strip().strip(",").strip('"')
        assert not stripped.startswith("diskcache"), f"diskcache still declared: {line!r}"


def test_set_get_roundtrip(cache):
    cache.set("k", json.dumps({"a": 1}))
    assert json.loads(cache.get("k")) == {"a": 1}


def test_get_missing_returns_none(cache):
    assert cache.get("nope") is None


def test_delete(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False


def test_clear(cache):
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert cache.get("a") is None and cache.get("b") is None


def test_expire_evicts_elapsed_entries(cache):
    cache.set("fresh", "1", expire=1000)
    cache.set("stale", "1", expire=-1)  # already expired
    removed = cache.expire()
    assert removed == 1
    assert cache.get("fresh") == "1"
    assert cache.get("stale") is None


def test_get_treats_expired_entry_as_missing(cache):
    cache.set("k", "v", expire=-1)
    assert cache.get("k") is None


def test_transact_commits_on_success(cache):
    with cache.transact():
        cache.set("k", "v")
    assert cache.get("k") == "v"


def test_transact_rolls_back_on_error(cache):
    cache.set("k", "original")
    with pytest.raises(RuntimeError):
        with cache.transact():
            cache.set("k", "mutated")
            raise RuntimeError("boom")
    # The mutation inside the failed transaction must not persist.
    assert cache.get("k") == "original"


def test_persists_across_reopen():
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = JsonSqliteCache(directory=tmpdir)
        c1.set("k", "v")
        c1.close()

        c2 = JsonSqliteCache(directory=tmpdir)
        try:
            assert c2.get("k") == "v"
        finally:
            c2.close()
