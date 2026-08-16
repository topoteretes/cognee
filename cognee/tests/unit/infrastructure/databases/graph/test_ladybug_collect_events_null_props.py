"""Regression test: collect_events tolerates Event nodes with null properties.

``LadybugAdapter.collect_events`` parsed each event node's ``properties`` with an
unguarded ``json.loads(node["properties"])``. The local backend returns
``properties`` as a raw JSON string that can be ``None``/empty, so a single Event
node within 1-2 hops that has no properties raised
``TypeError: the JSON object must be str, bytes or bytearray, not NoneType`` and
aborted the entire TEMPORAL search. The parse is now guarded, matching every
other ``properties`` access in the adapter.

The method is exercised unbound with a stub ``self`` (no live Kuzu needed),
feeding a mocked query result whose first event node has ``properties=None``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


@pytest.mark.asyncio
async def test_collect_events_handles_null_properties():
    nodes = [
        {"id": "id-1", "name": "Event without props", "properties": None},
        {"id": "id-2", "name": "Event with props", "properties": json.dumps({"description": "d2"})},
    ]
    # collect_events reads result[0][0] as the list of event nodes.
    query_result = [[nodes]]

    stub = SimpleNamespace(
        _normalize_temporal_ids=lambda ids: ids,
        query=AsyncMock(return_value=query_result),
    )

    # On dev this raises TypeError from json.loads(None).
    result = await LadybugAdapter.collect_events(stub, ids=["id-1"])

    events = result[0]["events"]
    assert len(events) == 2
    by_id = {e["id"]: e for e in events}
    # Null-properties node is handled with an empty props dict.
    assert by_id["id-1"]["description"] is None
    assert by_id["id-2"]["description"] == "d2"
