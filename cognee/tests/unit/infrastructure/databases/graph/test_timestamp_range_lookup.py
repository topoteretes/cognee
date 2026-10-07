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


# --- get_temporal_anchors: the candidate-side default -----------------------------


def _node(node_id, type_name, **properties):
    return (node_id, {"type": type_name, **properties})


def _edge(source, target, relationship):
    return (source, target, relationship, {})


class _NeighborhoodAdapter(GraphDBInterface):
    """A graph read only through get_neighborhood, the way the default anchors walk it."""

    def __init__(self, nodes, edges):
        self._nodes = {node_id: props for node_id, props in nodes}
        self._edges = edges
        self.calls = []

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        self.calls.append((list(node_ids), depth, edge_types))
        seeds = set(node_ids)
        edges = [e for e in self._edges if e[0] in seeds or e[1] in seeds]
        touched = seeds | {e[0] for e in edges} | {e[1] for e in edges}
        return [(n, self._nodes[n]) for n in touched if n in self._nodes], edges


_NeighborhoodAdapter.__abstractmethods__ = frozenset()

_GRAPH_NODES = [
    _node("c_apollo", "DocumentChunk"),
    _node("c_curie", "DocumentChunk"),
    _node("c_mention", "DocumentChunk"),
    _node("eagle", "Entity"),
    _node("curie", "Entity"),
    _node(
        "ts_1969",
        "Timestamp",
        timestamp_str="1969-07-20",
        time_at=-14182940000,
        time_until=-14182939000,
    ),
    _node(
        "ts_1867",
        "Timestamp",
        timestamp_str="1867-11-07",
        time_at=-3222633600000,
        time_until=-3222547200000,
    ),
]
_GRAPH_EDGES = [
    _edge("c_apollo", "ts_1969", "contains"),
    _edge("c_apollo", "eagle", "contains"),
    _edge("eagle", "ts_1969", "landed_at"),
    _edge("c_mention", "eagle", "contains"),  # mentions Eagle, states no date itself
    _edge("c_curie", "ts_1867", "contains"),
    _edge("c_curie", "curie", "contains"),
    _edge("curie", "ts_1867", "born_at"),
]
YEAR_1969 = (-31536000000, 0)  # [1969-01-01, 1970-01-01) — the end is exactly 0


@pytest.mark.asyncio
async def test_default_anchors_read_from_the_candidate_side():
    adapter = _NeighborhoodAdapter(_GRAPH_NODES, _GRAPH_EDGES)

    anchors = await adapter.get_temporal_anchors(
        ["c_apollo", "c_curie", "c_mention"], ["curie"], *YEAR_1969
    )

    assert anchors["timestamp_ids"] == {"ts_1969"}
    # directly dated, and dated through the entity it mentions
    assert anchors["chunk_ids"] == {"c_apollo", "c_mention"}
    assert anchors["chunk_timestamps"] == {"c_apollo": {"ts_1969"}, "c_mention": {"ts_1969"}}
    assert anchors["entity_ids"] == {"eagle"}
    assert "c_curie" not in anchors["chunk_ids"] and "curie" not in anchors["entity_ids"]
    # two bounded hops: the candidates, then the entities the candidate chunks contain
    assert [call[0] for call in adapter.calls] == [
        ["c_apollo", "c_curie", "c_mention", "curie"],
        ["curie", "eagle"],
    ]


@pytest.mark.asyncio
async def test_default_anchors_cover_candidate_entities_and_empty_input():
    adapter = _NeighborhoodAdapter(_GRAPH_NODES, _GRAPH_EDGES)
    anchors = await adapter.get_temporal_anchors([], ["curie", "eagle"], None, -3000000000000)
    assert anchors == {
        "timestamp_ids": {"ts_1867"},
        "chunk_ids": set(),
        "entity_ids": {"curie"},
        "chunk_timestamps": {},
    }
    assert await adapter.get_temporal_anchors([], [], *YEAR_1969) == {
        "timestamp_ids": set(),
        "chunk_ids": set(),
        "entity_ids": set(),
        "chunk_timestamps": {},
    }
    assert adapter.calls[-1][0] != []  # the empty call never touched the graph


@pytest.mark.asyncio
async def test_default_anchors_bucket_a_dlt_row_as_the_chunk_it_was_asked_about():
    """A DLT row is a chunk of its own graph type; its date edge anchors it as a chunk."""
    nodes = _GRAPH_NODES + [_node("row_1", "DltRow"), _node("row_2", "DltRow")]
    edges = _GRAPH_EDGES + [
        _edge("row_1", "ts_1969", "order_date"),
        _edge("row_2", "ts_1867", "order_date"),
    ]
    adapter = _NeighborhoodAdapter(nodes, edges)

    anchors = await adapter.get_temporal_anchors(["row_1", "row_2", "c_curie"], [], *YEAR_1969)

    assert anchors["timestamp_ids"] == {"ts_1969"}
    assert anchors["chunk_ids"] == {"row_1"}
    assert anchors["chunk_timestamps"] == {"row_1": {"ts_1969"}}
    assert anchors["entity_ids"] == set()


def test_anchors_from_rows_bucket_by_the_requested_chunk_set():
    from cognee.infrastructure.databases.graph.graph_db_interface import (
        temporal_anchors_from_rows,
    )

    anchors = temporal_anchors_from_rows(
        [("row_1", "ts_a"), ("eagle", "ts_a")],
        [("c_1", "curie", "ts_b")],
        chunk_ids={"row_1", "c_1"},
    )
    assert anchors == {
        "timestamp_ids": {"ts_a", "ts_b"},
        "chunk_ids": {"row_1", "c_1"},
        "entity_ids": {"eagle", "curie"},
        "chunk_timestamps": {"row_1": {"ts_a"}, "c_1": {"ts_b"}},
    }
