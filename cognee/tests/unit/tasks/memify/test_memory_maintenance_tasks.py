import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.memify.decay_weights import decay_weights
from cognee.tasks.memify.cross_connect import cross_connect, LinkPredictionResult
from cognee.tasks.memify.consolidate_merge import consolidate_merge, MergeDecision
from cognee.tasks.memify.reconcile_contradictions import (
    reconcile_contradictions,
    ReconciliationDecision,
)


@pytest.mark.asyncio
@patch("cognee.tasks.memify.decay_weights.get_graph_engine")
async def test_decay_weights_logic(mock_get_graph_engine):
    mock_engine = AsyncMock()
    mock_get_graph_engine.return_value = mock_engine

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    thirty_days_ago = now_ms - (30 * 24 * 60 * 60 * 1000)

    # 1. Mock get_filtered_graph_data: Node A (active/high-weight), Node B (stale/low-weight)
    mock_nodes = [
        (
            "node_active",
            {
                "name": "Active Node",
                "feedback_weight": 0.8,
                "frequency_weight": 2.0,
                "updated_at": now_ms,
            },
        ),
        (
            "node_stale",
            {
                "name": "Stale Node",
                "feedback_weight": 0.15,
                "frequency_weight": 0.0,
                "updated_at": thirty_days_ago,
            },
        ),
    ]
    mock_edges = [
        (
            "node_active",
            "node_stale",
            "connected",
            {"edge_object_id": "edge_1", "feedback_weight": 0.6, "updated_at": now_ms},
        )
    ]
    mock_engine.get_filtered_graph_data.return_value = (mock_nodes, mock_edges)

    # 2. Run decay
    result = await decay_weights(None, decay_rate=0.9, prune_threshold=0.15)

    # 3. Assertions
    assert result["nodes_processed"] == 2
    assert result["edges_processed"] == 1
    # Node stale was updated long ago, decay_factor makes it drop below threshold
    assert result["nodes_deleted"] == 1
    mock_engine.delete_nodes.assert_awaited_once_with(["node_stale"])
    mock_engine.set_node_feedback_weights.assert_awaited_once()


@pytest.mark.asyncio
@patch("cognee.tasks.memify.cross_connect.get_graph_engine")
@patch(
    "cognee.tasks.memify.cross_connect.LLMGateway.acreate_structured_output", new_callable=AsyncMock
)
async def test_cross_connect_logic(mock_llm, mock_get_graph_engine):
    mock_engine = AsyncMock()
    mock_get_graph_engine.return_value = mock_engine

    # 1. Mock unconnected pairs querying
    mock_engine.query.return_value = [
        [
            "id_a",
            "Entity A",
            "Desc A",
            {},
            "id_c",
            "Entity C",
            "Desc C",
            {},
            "id_b",
            "Entity B",
            "Desc B",
            {},
            "r1",
            "r2",
        ]
    ]

    # 2. Mock LLM response: they are related!
    mock_llm.return_value = LinkPredictionResult(
        is_related=True,
        relationship_name="collaborates_with",
        edge_text="Entity A collaborates with Entity C",
    )

    # 3. Run cross connect
    result = await cross_connect(None)

    # 4. Assertions
    assert result["connections_created"] == 1
    mock_engine.add_edges.assert_awaited_once()
    added_edges = mock_engine.add_edges.call_args[0][0]
    assert len(added_edges) == 1
    assert added_edges[0][0] == "id_a"
    assert added_edges[0][1] == "id_c"
    assert added_edges[0][2] == "collaborates_with"


@pytest.mark.asyncio
@patch("cognee.tasks.memify.consolidate_merge.get_graph_engine")
@patch("cognee.tasks.memify.consolidate_merge.get_node_edges")
@patch(
    "cognee.tasks.memify.consolidate_merge.LLMGateway.acreate_structured_output",
    new_callable=AsyncMock,
)
async def test_consolidate_merge_logic(mock_llm, mock_get_node_edges, mock_get_graph_engine):
    mock_engine = AsyncMock()
    mock_get_graph_engine.return_value = mock_engine

    # 1. Mock similar name nodes: Entity A and Entity A-prime
    mock_nodes = [
        ("id_a", {"name": "Acme Corp", "description": "Acme headquarters"}),
        ("id_b", {"name": "Acme Corp.", "description": "Acme main office"}),
    ]
    mock_engine.get_filtered_graph_data.return_value = (mock_nodes, [])

    # 2. Mock LLM response: duplicate! Primary is "id_a"
    mock_llm.return_value = MergeDecision(
        is_duplicate=True,
        primary_entity_id="id_a",
        consolidated_name="Acme Corp",
        consolidated_description="Merged Acme headquarters and main office",
    )

    # 3. Mock edge redirection
    mock_get_node_edges.return_value = (
        [("id_target", "has_part", "{}")],  # outgoing
        [("id_source", "member_of", "{}")],  # incoming
    )
    mock_engine.get_nodes.return_value = [{"id": "id_a", "properties": "{}"}]

    # 4. Run consolidate
    result = await consolidate_merge(None, similarity_threshold=0.7)

    # 5. Assertions
    assert result["nodes_merged"] == 1
    mock_engine.delete_nodes.assert_awaited_once_with(["id_b"])
    mock_engine.add_edges.assert_awaited_once()
    added_edges = mock_engine.add_edges.call_args[0][0]
    assert len(added_edges) == 2


@pytest.mark.asyncio
@patch("cognee.tasks.memify.reconcile_contradictions.get_graph_engine")
@patch(
    "cognee.tasks.memify.reconcile_contradictions.LLMGateway.acreate_structured_output",
    new_callable=AsyncMock,
)
async def test_reconcile_contradictions_logic(mock_llm, mock_get_graph_engine):
    mock_engine = AsyncMock()
    mock_get_graph_engine.return_value = mock_engine

    # 1. Mock contradictory claims: Company X founded in 2020 vs 2021
    mock_engine.query.return_value = [
        [
            "source_id",
            "Company X",
            "A company",
            "founded_in",
            "t1_id",
            "2020",
            "Year 2020",
            '{"edge_object_id": "edge_old"}',
            "t2_id",
            "2021",
            "Year 2021",
            '{"edge_object_id": "edge_new"}',
        ]
    ]

    # 2. Mock LLM response: contradiction! 2021 is the newer/superseding claim
    mock_llm.return_value = ReconciliationDecision(
        is_contradiction=True,
        superseding_target_id="t2_id",
        superseded_target_id="t1_id",
        explanation="2021 correction supersedes the old 2020 record",
    )

    # 3. Run reconciliation
    result = await reconcile_contradictions(None)

    # 4. Assertions
    assert result["contradictions_resolved"] == 1
    mock_engine.add_edges.assert_awaited_once()
    added_edges = mock_engine.add_edges.call_args[0][0]
    assert added_edges[0][0] == "t2_id"
    assert added_edges[0][1] == "t1_id"
    assert added_edges[0][2] == "supersedes"

    # Verify stale edge & node demotion is triggered
    mock_engine.set_edge_feedback_weights.assert_awaited_once_with({"edge_old": 0.1})
    mock_engine.set_node_feedback_weights.assert_awaited_once_with({"t1_id": 0.1})
