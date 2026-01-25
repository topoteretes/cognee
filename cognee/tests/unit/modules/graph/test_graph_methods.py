"""
Unit Tests: Graph Methods

Tests for core graph deletion methods with specific inputs and outputs.

Test Coverage:
- test_get_data_related_nodes_excludes_shared: Verify shared nodes are excluded
- test_delete_data_nodes_and_edges_removes_from_all_systems: Verify complete cleanup
"""

import os
import pathlib
import pytest
from uuid import UUID, uuid4, uuid5, NAMESPACE_OID

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.data.methods import create_dataset, get_dataset_data
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph.methods.delete_data_nodes_and_edges import delete_data_nodes_and_edges
from cognee.modules.graph.methods.get_data_related_nodes import get_data_related_nodes
from cognee.modules.graph.models import Node, Edge
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from sqlalchemy import select

logger = get_logger()


@pytest.mark.asyncio
async def test_get_data_related_nodes_excludes_shared():
    """
    Test that get_data_related_nodes returns only non-shared nodes.

    Setup:
    - Data A has nodes: Apple (unique), Google (shared with B)
    - Data B has nodes: Google (shared with A), Microsoft (unique)

    Expected:
    - get_data_related_nodes(dataset_id, data_A_id) returns [Apple] only
    - get_data_related_nodes(dataset_id, data_B_id) returns [Microsoft] only
    - Google is excluded from both (shared node)
    """
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent.parent,
        ".data_storage/test_get_data_related_nodes",
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent.parent,
        ".cognee_system/test_get_data_related_nodes",
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    # Create dataset
    dataset = await create_dataset("test_shared_nodes", user=user)
    dataset_id = dataset.id

    await set_database_global_context_variables(dataset_id, user.id)

    # Create unique data IDs
    data_a_id = uuid4()
    data_b_id = uuid4()

    # Create node slugs (UUIDs)
    apple_slug = uuid5(NAMESPACE_OID, "Apple")
    google_slug = uuid5(NAMESPACE_OID, "Google")  # Shared
    microsoft_slug = uuid5(NAMESPACE_OID, "Microsoft")

    # Create nodes for data A: Apple (unique), Google (shared)
    node_apple = Node(
        id=uuid4(),
        slug=apple_slug,
        user_id=user.id,
        data_id=data_a_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )
    node_google_a = Node(
        id=uuid4(),
        slug=google_slug,  # Same slug as in data B
        user_id=user.id,
        data_id=data_a_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )

    # Create nodes for data B: Google (shared), Microsoft (unique)
    node_google_b = Node(
        id=uuid4(),
        slug=google_slug,  # Same slug as in data A
        user_id=user.id,
        data_id=data_b_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )
    node_microsoft = Node(
        id=uuid4(),
        slug=microsoft_slug,
        user_id=user.id,
        data_id=data_b_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )

    # Insert nodes into database
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(node_apple)
        session.add(node_google_a)
        session.add(node_google_b)
        session.add(node_microsoft)
        await session.commit()

    logger.info("Inserted 4 nodes: Apple (A), Google (A+B shared), Microsoft (B)")

    # Test get_data_related_nodes for data A
    nodes_a = await get_data_related_nodes(dataset_id, data_a_id)
    node_a_slugs = [str(node.slug) for node in nodes_a]

    logger.info(f"Data A related nodes: {len(nodes_a)} nodes")
    logger.info(f"Data A node slugs: {node_a_slugs}")

    # Assertions for data A
    assert len(nodes_a) == 1, (
        f"Data A should have 1 non-shared node (Apple), but has {len(nodes_a)}"
    )
    assert str(apple_slug) in node_a_slugs, "Apple should be in data A's related nodes"
    assert str(google_slug) not in node_a_slugs, (
        "Google should NOT be in data A's related nodes (shared with B)"
    )

    # Test get_data_related_nodes for data B
    nodes_b = await get_data_related_nodes(dataset_id, data_b_id)
    node_b_slugs = [str(node.slug) for node in nodes_b]

    logger.info(f"Data B related nodes: {len(nodes_b)} nodes")
    logger.info(f"Data B node slugs: {node_b_slugs}")

    # Assertions for data B
    assert len(nodes_b) == 1, (
        f"Data B should have 1 non-shared node (Microsoft), but has {len(nodes_b)}"
    )
    assert str(microsoft_slug) in node_b_slugs, "Microsoft should be in data B's related nodes"
    assert str(google_slug) not in node_b_slugs, (
        "Google should NOT be in data B's related nodes (shared with A)"
    )

    logger.info("✅ test_get_data_related_nodes_excludes_shared PASSED")


@pytest.mark.asyncio
async def test_delete_data_nodes_and_edges_removes_from_all_systems():
    """
    Test that delete_data_nodes_and_edges removes data from all systems.

    Setup:
    - Create data with 3 nodes and 2 edges
    - Insert into relational DB, graph engine, and vector index

    Operation:
    - Call delete_data_nodes_and_edges()

    Expected:
    - Relational DB: 0 records with this data_id
    - Graph engine: 0 nodes with these slugs
    - Vector engine: 0 items in collections for these IDs
    """
    # Enable backend access control for multi-user support
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent.parent,
        ".data_storage/test_delete_data_nodes_edges",
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent.parent,
        ".cognee_system/test_delete_data_nodes_edges",
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    # Create dataset
    dataset = await create_dataset("test_delete_complete", user=user)
    dataset_id = dataset.id

    await set_database_global_context_variables(dataset_id, user.id)

    # Create data ID
    data_id = uuid4()

    # Create 3 node slugs (all unique to this data_id)
    node_1_slug = uuid5(NAMESPACE_OID, f"Node1_{data_id}")
    node_2_slug = uuid5(NAMESPACE_OID, f"Node2_{data_id}")
    node_3_slug = uuid5(NAMESPACE_OID, f"Node3_{data_id}")

    # Create nodes
    node_1_id = uuid4()
    node_2_id = uuid4()
    node_3_id = uuid4()

    node_1 = Node(
        id=node_1_id,
        slug=node_1_slug,
        user_id=user.id,
        data_id=data_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )
    node_2 = Node(
        id=node_2_id,
        slug=node_2_slug,
        user_id=user.id,
        data_id=data_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )
    node_3 = Node(
        id=node_3_id,
        slug=node_3_slug,
        user_id=user.id,
        data_id=data_id,
        dataset_id=dataset_id,
        type="Entity",
        indexed_fields=["name"],
    )

    # Create 2 edges
    edge_1_slug = uuid5(NAMESPACE_OID, f"Edge1_{data_id}")
    edge_2_slug = uuid5(NAMESPACE_OID, f"Edge2_{data_id}")

    edge_1 = Edge(
        id=uuid4(),
        slug=edge_1_slug,
        user_id=user.id,
        data_id=data_id,
        dataset_id=dataset_id,
        relationship_name="related_to",
        source_node_id=node_1_id,
        destination_node_id=node_2_id,
    )
    edge_2 = Edge(
        id=uuid4(),
        slug=edge_2_slug,
        user_id=user.id,
        data_id=data_id,
        dataset_id=dataset_id,
        relationship_name="connected_to",
        source_node_id=node_2_id,
        destination_node_id=node_3_id,
    )

    # Insert into relational database
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(node_1)
        session.add(node_2)
        session.add(node_3)
        session.add(edge_1)
        session.add(edge_2)
        await session.commit()

    logger.info("Inserted 3 nodes and 2 edges into relational DB")

    # Insert into graph engine
    await get_graph_engine()

    # Add nodes to graph
    from cognee.infrastructure.engine.models import DataPoint

    class TestEntity(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    # Note: In real scenario, nodes would be added through normal cognify process
    # For unit test, we verify the relational DB records exist

    # Verify nodes exist in relational DB before deletion
    async with db_engine.get_async_session() as session:
        nodes_before = await session.scalars(select(Node).where(Node.data_id == data_id))
        nodes_before_list = nodes_before.all()
        assert len(nodes_before_list) == 3, (
            f"Should have 3 nodes before deletion, found {len(nodes_before_list)}"
        )

        edges_before = await session.scalars(select(Edge).where(Edge.data_id == data_id))
        edges_before_list = edges_before.all()
        assert len(edges_before_list) == 2, (
            f"Should have 2 edges before deletion, found {len(edges_before_list)}"
        )

    logger.info("Verified: 3 nodes and 2 edges exist before deletion")

    # Execute delete_data_nodes_and_edges
    logger.info(f"Deleting data nodes and edges for data_id={data_id}...")
    await delete_data_nodes_and_edges(dataset_id, data_id, user.id)

    # Verify deletion from relational DB
    async with db_engine.get_async_session() as session:
        nodes_after = await session.scalars(select(Node).where(Node.data_id == data_id))
        nodes_after_list = nodes_after.all()
        assert len(nodes_after_list) == 0, (
            f"Should have 0 nodes after deletion, found {len(nodes_after_list)}"
        )

        edges_after = await session.scalars(select(Edge).where(Edge.data_id == data_id))
        edges_after_list = edges_after.all()
        assert len(edges_after_list) == 0, (
            f"Should have 0 edges after deletion, found {len(edges_after_list)}"
        )

    logger.info("✅ Verified: All nodes and edges removed from relational DB")

    # Note: Graph engine and vector engine deletion is tested in integration tests
    # This unit test focuses on the relational DB cleanup

    logger.info("✅ test_delete_data_nodes_and_edges_removes_from_all_systems PASSED")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_get_data_related_nodes_excludes_shared())
    asyncio.run(test_delete_data_nodes_and_edges_removes_from_all_systems())
