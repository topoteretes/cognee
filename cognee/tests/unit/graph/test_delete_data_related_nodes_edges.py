import uuid
import pytest
from sqlalchemy import select, func

from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_async_session,
)
from cognee.modules.graph.models import Node, Edge
from cognee.modules.graph.methods import (
    delete_data_related_nodes,
    delete_data_related_edges,
)


@pytest.mark.asyncio
async def test_delete_data_related_nodes_removes_only_target_data():
    await create_db_and_tables()

    dataset_id = uuid.uuid4()
    user_id = uuid.uuid4()
    data_a = uuid.uuid4()
    data_b = uuid.uuid4()

    node_a1 = Node(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_a,
        dataset_id=dataset_id,
        label="A1",
        type="TypeA",
        indexed_fields=["text"],
        attributes={"k": "v"},
    )
    node_a2 = Node(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_a,
        dataset_id=dataset_id,
        label="A2",
        type="TypeA",
        indexed_fields=["text"],
        attributes={"k2": "v2"},
    )
    node_b1 = Node(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_b,
        dataset_id=dataset_id,
        label="B1",
        type="TypeB",
        indexed_fields=["text"],
        attributes={"k3": "v3"},
    )

    async with get_async_session(auto_commit=True) as session:
        session.add_all([node_a1, node_a2, node_b1])

    await delete_data_related_nodes(data_a)

    async with get_async_session() as session:
        count_a = (
            await session.scalar(select(func.count()).select_from(Node).where(Node.data_id == data_a))
        )
        count_b = (
            await session.scalar(select(func.count()).select_from(Node).where(Node.data_id == data_b))
        )

        assert count_a == 0
        assert count_b == 1


@pytest.mark.asyncio
async def test_delete_data_related_edges_removes_only_target_data():
    await create_db_and_tables()

    dataset_id = uuid.uuid4()
    user_id = uuid.uuid4()
    data_a = uuid.uuid4()
    data_b = uuid.uuid4()

    # Nodes for referencing
    n1 = Node(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_a,
        dataset_id=dataset_id,
        label="N1",
        type="TypeA",
        indexed_fields=["text"],
        attributes={},
    )
    n2 = Node(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_b,
        dataset_id=dataset_id,
        label="N2",
        type="TypeB",
        indexed_fields=["text"],
        attributes={},
    )

    e_a = Edge(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_a,
        dataset_id=dataset_id,
        source_node_id=n1.id,
        destination_node_id=n1.id,
        relationship_name="REL_A",
        label="LA",
        attributes={},
    )
    e_b = Edge(
        id=uuid.uuid4(),
        slug=uuid.uuid4(),
        user_id=user_id,
        data_id=data_b,
        dataset_id=dataset_id,
        source_node_id=n2.id,
        destination_node_id=n2.id,
        relationship_name="REL_B",
        label="LB",
        attributes={},
    )

    async with get_async_session(auto_commit=True) as session:
        session.add_all([n1, n2, e_a, e_b])

    await delete_data_related_edges(data_a)

    async with get_async_session() as session:
        count_a = (
            await session.scalar(select(func.count()).select_from(Edge).where(Edge.data_id == data_a))
        )
        count_b = (
            await session.scalar(select(func.count()).select_from(Edge).where(Edge.data_id == data_b))
        )

        assert count_a == 0
        assert count_b == 1


@pytest.mark.asyncio
async def test_delete_data_related_nodes_edges_noop_on_empty_tables():
    await create_db_and_tables()

    data_x = uuid.uuid4()

    # Should not raise
    await delete_data_related_nodes(data_x)
    await delete_data_related_edges(data_x)


