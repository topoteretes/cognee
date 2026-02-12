import os
import pathlib
from typing import List
from uuid import UUID, uuid4
from pydantic import BaseModel

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.models import User
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import add_data_points

logger = get_logger()


async def main():
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_permission"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_permission"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    class Organization(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class ForProfit(Organization):
        name: str = "For-Profit"
        metadata: dict = {"index_fields": ["name"]}

    class NonProfit(Organization):
        name: str = "Non-Profit"
        metadata: dict = {"index_fields": ["name"]}

    class Person(DataPoint):
        name: str
        works_for: List[Organization]
        metadata: dict = {"index_fields": ["name"]}

    companyA = ForProfit(name="Company A")
    companyB = NonProfit(name="Company B")

    person1 = Person(name="John", works_for=[companyA, companyB])
    person2 = Person(name="Jane", works_for=[companyB])

    user1: User = await create_user(email="user1@example.com", password="password123")
    user2: User = await create_user(email="user2@example.com", password="password123")

    class CustomData(BaseModel):
        id: UUID

    dataset = await create_authorized_dataset(dataset_name="test_dataset", user=user1)

    data1 = CustomData(id=uuid4())
    data2 = CustomData(id=uuid4())

    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    await add_data_points(
        [person1],
        context={
            "user": user1,
            "dataset": dataset,
            "data": data1,
        },
    )

    await add_data_points(
        [person2],
        context={
            "user": user1,
            "dataset": dataset,
            "data": data2,
        },
    )

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 4 and len(edges) == 3, (
        "Nodes and edges are not correctly added to the graph."
    )

    is_permission_error_raised = False
    try:
        await datasets.delete_data(dataset.id, data1.id, user2)
    except UnauthorizedDataAccessError:
        is_permission_error_raised = True

    assert is_permission_error_raised, "PermissionDeniedError was not raised as expected."

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 4 and len(edges) == 3, "Graph is changed without permissions."

    await authorized_give_permission_on_datasets(user2.id, [dataset.id], "delete", user1.id)

    await datasets.delete_data(dataset.id, data1.id, user2)

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 2 and len(edges) == 1, "Nodes and edges are not deleted properly."

    await datasets.delete_data(dataset.id, data2.id, user2)

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 0 and len(edges) == 0, "Nodes and edges are not deleted."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
