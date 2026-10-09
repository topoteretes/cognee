import json
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_properties",
    [None, "", "{not json", "null", "[1, 2]", '"a string"'],
    ids=["none", "empty", "malformed", "json-null", "json-list", "json-string"],
)
async def test_collect_events_tolerates_unusable_properties(bad_properties):
    # One Event node whose properties blob is missing or not a JSON object must
    # not abort the TEMPORAL search; its neighbour still comes back intact.
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    nodes = [
        {"id": "event-1", "name": "Bare event", "properties": bad_properties},
        {
            "id": "event-2",
            "name": "Described event",
            "properties": json.dumps({"description": "d2", "location": "Berlin"}),
        },
    ]
    adapter.query = AsyncMock(return_value=[[nodes]])

    result = await adapter.collect_events(ids=["timestamp-1"])

    assert result == [
        {
            "events": [
                {"id": "event-1", "name": "Bare event", "description": None},
                {
                    "id": "event-2",
                    "name": "Described event",
                    "description": "d2",
                    "location": "Berlin",
                },
            ]
        }
    ]
