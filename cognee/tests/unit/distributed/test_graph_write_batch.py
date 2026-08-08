import pytest

from distributed.graph_write_batch import apply_grouped_graph_writes, group_graph_writes


def test_group_graph_writes_groups_by_provenance_key():
    items = [
        (["n1"], [], "srA", "run1"),
        ([], ["e1"], "srA", "run1"),
        (["n2"], ["e2"], "srB", "run2"),
        (["n3"], [], "srA", "run1"),
    ]

    groups = group_graph_writes(items)

    assert groups[("srA", "run1")] == (["n1", "n3"], ["e1"])
    assert groups[("srB", "run2")] == (["n2"], ["e2"])
    # First-seen key order is preserved.
    assert list(groups.keys()) == [("srA", "run1"), ("srB", "run2")]


def test_group_graph_writes_keeps_unprovenanced_items_separate():
    items = [
        (["n1"], [], None, None),
        (["n2"], [], "srA", "run1"),
    ]

    groups = group_graph_writes(items)

    assert groups[(None, None)] == (["n1"], [])
    assert groups[("srA", "run1")] == (["n2"], [])


def test_group_graph_writes_empty():
    assert group_graph_writes([]) == {}


@pytest.mark.asyncio
async def test_apply_grouped_graph_writes_writes_all_nodes_before_any_edge():
    calls = []

    async def add_nodes(nodes, source_ref_key, pipeline_run_id):
        calls.append(("nodes", tuple(nodes), source_ref_key, pipeline_run_id))

    async def add_edges(edges, source_ref_key, pipeline_run_id):
        calls.append(("edges", tuple(edges), source_ref_key, pipeline_run_id))

    groups = {
        ("srA", "run1"): (["n1"], ["e1"]),
        ("srB", "run2"): (["n2"], ["e2"]),
    }

    await apply_grouped_graph_writes(groups, add_nodes, add_edges)

    # Every group's nodes are written before ANY edge, and each write carries its
    # own group's provenance.
    assert calls == [
        ("nodes", ("n1",), "srA", "run1"),
        ("nodes", ("n2",), "srB", "run2"),
        ("edges", ("e1",), "srA", "run1"),
        ("edges", ("e2",), "srB", "run2"),
    ]


@pytest.mark.asyncio
async def test_apply_grouped_graph_writes_skips_empty_batches():
    calls = []

    async def add_nodes(nodes, source_ref_key, pipeline_run_id):
        calls.append(("nodes", source_ref_key))

    async def add_edges(edges, source_ref_key, pipeline_run_id):
        calls.append(("edges", source_ref_key))

    groups = {
        ("srA", "run1"): (["n1"], []),  # nodes only
        ("srB", "run2"): ([], ["e2"]),  # edges only
    }

    await apply_grouped_graph_writes(groups, add_nodes, add_edges)

    assert calls == [("nodes", "srA"), ("edges", "srB")]
