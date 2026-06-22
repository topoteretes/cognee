from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.tasks.temporal_graph.models import Timestamp


@pytest.mark.asyncio
async def test_collect_time_ids_returns_list_for_unwind_params():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[["timestamp-1"], ["timestamp-2"]])

    ids = await adapter.collect_time_ids(time_to=Timestamp(year=1980))

    assert ids == ["timestamp-1", "timestamp-2"]


@pytest.mark.asyncio
async def test_collect_events_binds_ids_as_list():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[])

    await adapter.collect_events(ids=["timestamp-1", "timestamp-2"])

    _, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}


@pytest.mark.asyncio
async def test_collect_events_accepts_legacy_quoted_id_string():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[])

    await adapter.collect_events(ids="'timestamp-1', 'timestamp-2'")

    _, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}
