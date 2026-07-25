"""Tests for the actor overlay merged into the main graph visualization.

build_actor_overlay() wires User —operates→ Agent —reads/writes→ Dataset
—contains→ document nodes, using only ids already present in the rendered
graph, so the overlay never dangles.
"""

import pytest

from cognee.api.v1.visualize.memory_provenance import build_actor_overlay


USER = {"id": "u1", "name": "vasilije@example.com"}
DATASETS = [{"id": "d1", "name": "business", "owner_id": "u1"}]
AGENTS = [
    {
        "id": "conn-1",
        "name": "business-copilot",
        "user_id": "u1",
        "datasets": [{"dataset_id": "d1", "role": "read_write"}],
    }
]


def _edge_set(edges):
    return {(e[0], e[1], e[2]) for e in edges}


def test_full_actor_chain():
    nodes, edges = build_actor_overlay(
        user=USER,
        agents=AGENTS,
        datasets=DATASETS,
        dataset_data_ids={"d1": ["doc-1", "doc-2"]},
        graph_node_ids={"doc-1", "doc-2"},
    )
    node_ids = {n[0] for n in nodes}
    assert {"user:u1", "agent:conn-1", "dataset:d1"} <= node_ids
    assert _edge_set(edges) == {
        ("user:u1", "dataset:d1", "owns"),
        ("dataset:d1", "doc-1", "contains"),
        ("dataset:d1", "doc-2", "contains"),
        ("user:u1", "agent:conn-1", "operates"),
        ("agent:conn-1", "dataset:d1", "reads"),
        ("agent:conn-1", "dataset:d1", "writes"),
    }


def test_contains_edges_only_for_rendered_nodes():
    """A Data row whose document is not in the rendered graph gets no edge —
    the overlay must never dangle."""
    _, edges = build_actor_overlay(
        user=USER,
        agents=[],
        datasets=DATASETS,
        dataset_data_ids={"d1": ["doc-1", "doc-absent"]},
        graph_node_ids={"doc-1"},
    )
    contains = [e for e in edges if e[2] == "contains"]
    assert [(e[0], e[1]) for e in contains] == [("dataset:d1", "doc-1")]


def test_agent_dataset_edges_require_known_dataset():
    """An agent scoped to a dataset that is not being rendered links nowhere."""
    _, edges = build_actor_overlay(
        user=USER,
        agents=[
            {
                "id": "conn-2",
                "name": "other-agent",
                "user_id": "u1",
                "datasets": [{"dataset_id": "d-unrendered", "role": "read"}],
            }
        ],
        datasets=DATASETS,
        dataset_data_ids={},
        graph_node_ids=set(),
    )
    assert not [e for e in edges if e[2] in ("reads", "writes")]


def test_read_role_gets_no_write_edge():
    _, edges = build_actor_overlay(
        user=USER,
        agents=[
            {
                "id": "conn-3",
                "name": "reader",
                "user_id": "u1",
                "datasets": [{"dataset_id": "d1", "role": "read"}],
            }
        ],
        datasets=DATASETS,
        dataset_data_ids={},
        graph_node_ids=set(),
    )
    kinds = {e[2] for e in edges if e[0] == "agent:conn-3"}
    assert kinds == {"reads"}


def test_empty_inputs_render_nothing():
    nodes, edges = build_actor_overlay(
        user=None, agents=[], datasets=[], dataset_data_ids={}, graph_node_ids=set()
    )
    assert nodes == [] and edges == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
