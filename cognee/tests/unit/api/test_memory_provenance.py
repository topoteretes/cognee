"""Unit tests for the memory-provenance graph builder (pure projection logic)."""

from importlib import import_module

# Import via importlib for consistency with the visualize submodule access pattern.
_module = import_module("cognee.api.v1.visualize.memory_provenance")
build_provenance_graph = _module.build_provenance_graph


def _scenario():
    return dict(
        tenants=[{"id": "T", "name": "Acme Logistics"}],
        users=[
            {"id": "U1", "name": "alice@x", "tenant_ids": ["T"]},
            {"id": "U2", "name": "bob@x", "tenant_ids": ["T"]},
        ],
        datasets=[
            {"id": "D1", "name": "Fleet Ops", "owner_id": "U1", "tenant_id": "T"},
            {"id": "D2", "name": "Carrier Intel", "owner_id": "U2", "tenant_id": "T"},
        ],
        files=[
            {"id": "F1", "name": "a.txt", "dataset_ids": ["D1"]},
            {"id": "F2", "name": "b.txt", "dataset_ids": ["D1", "D2"]},
        ],
        agents=[
            {
                "id": "A1",
                "name": "dispatch-copilot",
                "user_id": "U1",
                "session_id": "S1",
                "datasets": [
                    {"dataset_id": "D1", "role": "read_write"},
                    {"dataset_id": "D2", "role": "read"},
                ],
            }
        ],
        sessions=[
            {"id": "S1", "name": "sess1", "user_id": "U1", "dataset_id": "D1"},
            {"id": "S2", "name": "sess2", "user_id": "U2", "dataset_id": "D2", "agent_id": "A1"},
        ],
    )


def _build(**overrides):
    args = _scenario()
    args.update(overrides)
    nodes, edges = build_provenance_graph(**args)
    node_types = {nid: props["type"] for nid, props in nodes}
    edge_set = {(s, t, rel) for s, t, rel, _ in edges}
    return node_types, edge_set


def test_nodes_are_namespaced_with_correct_types():
    node_types, _ = _build()
    assert node_types["tenant:T"] == "Tenant"
    assert node_types["user:U1"] == "User"
    assert node_types["dataset:D1"] == "Dataset"
    assert node_types["file:F1"] == "TextDocument"  # files render in the Documents column
    assert node_types["agent:A1"] == "Agent"
    assert node_types["session:S1"] == "Session"


def test_ownership_and_membership_edges():
    _, edges = _build()
    assert ("tenant:T", "user:U1", "has_member") in edges
    assert ("tenant:T", "user:U2", "has_member") in edges
    assert ("user:U1", "dataset:D1", "owns") in edges
    assert ("user:U2", "dataset:D2", "owns") in edges
    # a file shared by two datasets is contained by both
    assert ("dataset:D1", "file:F2", "contains") in edges
    assert ("dataset:D2", "file:F2", "contains") in edges


def test_agent_read_write_roles():
    _, edges = _build()
    assert ("user:U1", "agent:A1", "operates") in edges
    # read_write produces BOTH reads and writes
    assert ("agent:A1", "dataset:D1", "reads") in edges
    assert ("agent:A1", "dataset:D1", "writes") in edges
    # read-only produces reads but NOT writes
    assert ("agent:A1", "dataset:D2", "reads") in edges
    assert ("agent:A1", "dataset:D2", "writes") not in edges


def test_session_linkage():
    _, edges = _build()
    # agent.session_id links the agent to S1; session.agent_id links it to S2
    assert ("agent:A1", "session:S1", "wrote") in edges
    assert ("agent:A1", "session:S2", "wrote") in edges
    # sessions are recorded into their dataset
    assert ("session:S1", "dataset:D1", "recorded_in") in edges
    assert ("session:S2", "dataset:D2", "recorded_in") in edges


def test_memory_layer_links_to_source_file():
    node_types, edges = _build(
        memory={
            "nodes": [("n1", {"type": "Entity", "name": "Carlos"})],
            "edges": [],
            "links": [{"node_id": "n1", "data_id": "F1", "dataset_id": "D1"}],
        }
    )
    assert node_types["n1"] == "Entity"
    assert ("file:F1", "n1", "mentions") in edges


def test_empty_input_is_safe():
    nodes, edges = build_provenance_graph()
    assert nodes == []
    assert edges == []


def test_dangling_references_are_skipped():
    # owner U9 / dataset D9 don't exist as nodes → no phantom edges
    _, edges = _build(
        datasets=[{"id": "D1", "name": "Fleet", "owner_id": "U9", "tenant_id": "T"}],
    )
    assert not any(t == "dataset:D1" and rel == "owns" for _, t, rel in edges)
