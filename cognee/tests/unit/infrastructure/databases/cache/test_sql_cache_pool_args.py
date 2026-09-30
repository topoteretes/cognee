"""Regression tests: shared POOL_ARGS values reach the SQL cache engine intact.

POOL_ARGS is one env var consumed by both the relational adapter and the SQL
cache adapter. The relational adapter accepts poolclass="nullpool" and
normalizes the string to the NullPool class; the cache adapter previously
splatted the raw string into create_async_engine, which fails inside
SQLAlchemy with "'str' object has no attribute '__dict__'" and surfaces as
CacheConnectionError on startup.
"""

import pytest
from sqlalchemy import NullPool

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.relational.config import get_relational_config


@pytest.fixture
def nullpool_pool_args(monkeypatch):
    """POOL_ARGS as the config stores it: a tuple of key-value pairs."""
    monkeypatch.setattr(get_relational_config(), "pool_args", (("poolclass", "nullpool"),))


def test_sqlite_cache_engine_accepts_nullpool_string(nullpool_pool_args, tmp_path):
    adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db")
    assert isinstance(adapter.engine.pool, NullPool)


def test_cache_engine_without_pool_args_keeps_default_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(get_relational_config(), "pool_args", None)
    adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db")
    assert not isinstance(adapter.engine.pool, NullPool)
