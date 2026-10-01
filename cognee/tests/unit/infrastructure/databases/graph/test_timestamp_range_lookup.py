"""GraphDBInterface.get_timestamps_in_range: the overlap rule and the default scan (SDK-828)."""

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    timestamp_overlaps,
)

YEAR_1950 = {"time_at": -631152000000, "time_until": -599616000000}  # [1950-01-01, 1951-01-01)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (-631152000000, -599616000000, True),  # the year itself
        (-615000000000, -610000000000, True),  # a window inside the year
        (None, -599616000000, True),  # open start, ends when the year ends
        (-599616000000, None, False),  # starts exactly when the year ends: no overlap
        (None, -631152000000, False),  # ends exactly when the year starts: no overlap
        (-700000000000, -631152000000 + 1, True),  # ends one ms into the year
        (None, None, True),
    ],
)
def test_overlap_is_half_open_on_both_sides(start, end, expected):
    assert timestamp_overlaps({"id": "ts", **YEAR_1950}, start, end) is expected


def test_overlap_defaults_a_missing_time_until_to_one_second():
    node = {"id": "ts", "time_at": 0}
    assert timestamp_overlaps(node, 999, 5000) is True
    assert timestamp_overlaps(node, 1000, 5000) is False
    assert timestamp_overlaps({"id": "ts"}, None, None) is False


@pytest.mark.asyncio
async def test_default_implementation_scans_the_graph():
    class Adapter(GraphDBInterface):
        async def get_graph_data(self):
            return (
                [
                    ("ts_1950", {"type": "Timestamp", "timestamp_str": "1950", **YEAR_1950}),
                    (
                        "ts_1960",
                        {
                            "type": "Timestamp",
                            "timestamp_str": "1960",
                            "time_at": -315619200000,
                            "time_until": -284083200000,
                        },
                    ),
                    ("chunk", {"type": "DocumentChunk", "time_at": -631152000000}),
                ],
                [],
            )

    # Every other abstract method is irrelevant here.
    Adapter.__abstractmethods__ = frozenset()
    adapter = Adapter()

    found = await adapter.get_timestamps_in_range(-631152000000, -599616000000)
    assert [node["id"] for node in found] == ["ts_1950"]
    assert found[0]["timestamp_str"] == "1950"
    assert [node["id"] for node in await adapter.get_timestamps_in_range(None, None)] == [
        "ts_1950",
        "ts_1960",
    ]
    assert await adapter.get_timestamps_in_range(0, 1) == []
