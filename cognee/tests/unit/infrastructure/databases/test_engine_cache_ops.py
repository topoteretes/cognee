"""Tests for EngineCacheOps — the shared cache-management facade.

Each engine module exposes one public instance of this class
(graph_engine_cache / vector_engine_cache); these tests pin the routing contract against a fake
decorated factory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from cognee.infrastructure.databases.utils.engine_cache_ops import EngineCacheOps


def make_ops(**kwargs):
    factory = MagicMock()
    factory.cache_await_closed = AsyncMock()
    key_args = MagicMock(return_value=("k1", "k2"))
    ops = EngineCacheOps(factory, key_args, "db_name_field", **kwargs)
    return ops, factory, key_args


def test_evict_routes_by_force_close():
    ops, factory, key_args = make_ops()

    ops.evict(some="cfg")
    factory.cache_evict.assert_called_once_with("k1", "k2")
    factory.cache_evict_and_close.assert_not_called()
    key_args.assert_called_with({"some": "cfg"})

    ops.evict(force_close=True, some="cfg")
    factory.cache_evict_and_close.assert_called_once_with("k1", "k2")


def test_touch_and_is_cached_use_key_args():
    ops, factory, key_args = make_ops()

    ops.touch(some="cfg")
    factory.cache_touch.assert_called_once_with("k1", "k2")

    ops.is_cached(other="cfg")
    factory.cache_contains.assert_called_once_with("k1", "k2")
    assert [c.args[0] for c in key_args.call_args_list] == [{"some": "cfg"}, {"other": "cfg"}]


def test_evict_for_database_matches_on_configured_field():
    ops, factory, _ = make_ops()

    ops.evict_for_database("dataset-uuid")
    factory.cache_evict_matching.assert_called_once_with(db_name_field="dataset-uuid")


def test_evict_for_database_rejects_empty_name():
    ops, _, _ = make_ops()
    with pytest.raises(ValueError, match="db_name_field"):
        ops.evict_for_database("")


@pytest.mark.asyncio
async def test_aevict_for_database_awaits_in_flight_closes():
    ops, factory, _ = make_ops()
    factory.cache_evict_matching.return_value = 3

    assert await ops.aevict_for_database("dataset-uuid") == 3
    factory.cache_evict_matching.assert_called_once_with(db_name_field="dataset-uuid")
    factory.cache_await_closed.assert_awaited_once_with(db_name_field="dataset-uuid")


def test_evict_for_url_matches_on_configured_field():
    ops, factory, _ = make_ops(database_url_field="db_url_field")

    ops.evict_for_url("bolt://localhost:7799")
    factory.cache_evict_matching.assert_called_once_with(db_url_field="bolt://localhost:7799")


def test_evict_for_url_rejects_empty_url_and_missing_field():
    ops, _, _ = make_ops(database_url_field="db_url_field")
    with pytest.raises(ValueError, match="db_url_field"):
        ops.evict_for_url("")

    ops_without_url, _, _ = make_ops()
    with pytest.raises(RuntimeError, match="database_url_field"):
        ops_without_url.evict_for_url("bolt://localhost:7799")


@pytest.mark.asyncio
async def test_aevict_for_url_awaits_in_flight_closes():
    ops, factory, _ = make_ops(database_url_field="db_url_field")
    factory.cache_evict_matching.return_value = 2

    assert await ops.aevict_for_url("bolt://localhost:7799") == 2
    factory.cache_evict_matching.assert_called_once_with(db_url_field="bolt://localhost:7799")
    factory.cache_await_closed.assert_awaited_once_with(db_url_field="bolt://localhost:7799")
