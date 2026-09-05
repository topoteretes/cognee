"""Unit tests for the Ladybug temporal queries behind ``SearchType.TEMPORAL``.

``collect_time_ids`` / ``collect_events`` are the ``GraphDBInterface`` extension the
``TemporalRetriever`` relies on. These tests pin the contract: parameterised Cypher,
list results, and a flat event dict that carries the event's time anchors.
"""

import json
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.tasks.temporal_graph.models import Timestamp


def _adapter(query_side_effect):
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(side_effect=query_side_effect)
    return adapter


@pytest.mark.asyncio
async def test_collect_time_ids_returns_list_and_binds_bounds_as_params():
    adapter = _adapter([[["timestamp-1"], ["timestamp-2"]]])

    ids = await adapter.collect_time_ids(
        time_from=Timestamp(year=1900), time_to=Timestamp(year=1980)
    )

    assert ids == ["timestamp-1", "timestamp-2"]
    cypher, params = adapter.query.await_args.args
    assert set(params) == {"time_from", "time_to"}
    assert params["time_from"] < params["time_to"]
    # Bounds travel as parameters, never interpolated into the query text.
    assert str(params["time_from"]) not in cypher
    assert "t >= $time_from" in cypher and "t <= $time_to" in cypher


@pytest.mark.asyncio
async def test_collect_time_ids_single_bound_only_emits_that_condition():
    adapter = _adapter([[["timestamp-1"]]])

    ids = await adapter.collect_time_ids(time_to=Timestamp(year=1980))

    assert ids == ["timestamp-1"]
    cypher, params = adapter.query.await_args.args
    assert set(params) == {"time_to"}
    assert "$time_from" not in cypher


@pytest.mark.asyncio
async def test_collect_time_ids_without_bounds_does_not_query():
    adapter = _adapter([])

    assert await adapter.collect_time_ids() == []
    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_events_binds_ids_as_list_and_returns_empty_for_no_events():
    adapter = _adapter([[]])

    assert await adapter.collect_events(ids=["timestamp-1", "timestamp-2"]) == []

    _, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}


@pytest.mark.asyncio
async def test_collect_events_with_no_ids_does_not_query():
    adapter = _adapter([])

    assert await adapter.collect_events(ids=[]) == []
    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_events_returns_flat_events_with_time_anchors():
    event_rows = [
        ["ev-1", "Launch", json.dumps({"description": "Project launched", "location": "Oslo"})],
        ["ev-2", "Build phase", json.dumps({"description": "Building"})],
        ["ev-3", "Undated", json.dumps({"description": "No time link"})],
    ]
    anchor_rows = [
        ["ev-1", "at", "ts-1", "Timestamp", json.dumps({"time_at": 1000})],
        ["ev-2", "during", "iv-1", "Interval", json.dumps({})],
    ]
    bound_rows = [
        ["iv-1", "time_from", json.dumps({"time_at": 2000})],
        ["iv-1", "time_to", json.dumps({"time_at": 3000})],
    ]
    adapter = _adapter([event_rows, anchor_rows, bound_rows])

    events = {e["id"]: e for e in await adapter.collect_events(ids=["ts-1", "ts-2"])}

    assert events["ev-1"] == {
        "id": "ev-1",
        "name": "Launch",
        "description": "Project launched",
        "location": "Oslo",
        "time_at": 1000,
    }
    assert events["ev-2"] == {
        "id": "ev-2",
        "name": "Build phase",
        "description": "Building",
        "time_from": 2000,
        "time_to": 3000,
    }
    assert events["ev-3"] == {"id": "ev-3", "name": "Undated", "description": "No time link"}
    # Interval bounds are fetched only when an interval was actually linked.
    assert adapter.query.await_count == 3


@pytest.mark.asyncio
async def test_collect_events_skips_interval_lookup_without_intervals():
    event_rows = [["ev-1", "Launch", json.dumps({"description": "d"})]]
    anchor_rows = [["ev-1", "at", "ts-1", "Timestamp", json.dumps({"time_at": "1000"})]]
    adapter = _adapter([event_rows, anchor_rows])

    events = await adapter.collect_events(ids=["ts-1"])

    assert events[0]["time_at"] == 1000  # string time_at is coerced
    assert adapter.query.await_count == 2


@pytest.mark.asyncio
async def test_interface_default_raises_not_implemented():
    class BareAdapter(GraphDBInterface):
        pass

    BareAdapter.__abstractmethods__ = frozenset()
    adapter = BareAdapter()

    with pytest.raises(NotImplementedError, match="collect_time_ids"):
        await adapter.collect_time_ids(time_from=Timestamp(year=2000))
    with pytest.raises(NotImplementedError, match="collect_events"):
        await adapter.collect_events(["x"])
