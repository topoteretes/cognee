import os
import pathlib
from typing import List
from uuid import uuid4

import cognee
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models import Data, Dataset
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph.methods import delete_data_nodes_and_edges
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import add_data_points

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_delete_custom_graph")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_delete_custom_graph")
        ).resolve()
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

    user: User = await get_default_user()  # type: ignore

    dataset = Dataset(id=uuid4())
    data1 = Data(id=uuid4())
    data2 = Data(id=uuid4())

    await add_data_points(
        [person1],
        context={
            "user": user,
            "dataset": dataset,
            "data": data1,
        },
    )

    await add_data_points(
        [person2],
        context={
            "user": user,
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

    await delete_data_nodes_and_edges(dataset.id, data1.id)  # type: ignore

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 2 and len(edges) == 1, "Nodes and edges are not deleted properly."

    await delete_data_nodes_and_edges(dataset.id, data2.id)  # type: ignore

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 0 and len(edges) == 0, "Nodes and edges are not deleted."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
