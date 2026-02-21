import asyncio
import json
from typing import Any, Dict, List, Tuple
from pydantic import computed_field

from os import path
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.operations.setup import setup
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import index_data_points
from cognee.modules.pipelines import Task
from cognee.api.v1.visualize.visualize import visualize_graph


class Item(DataPoint):
    name: str
    quantity: int
    unit_price: float
    metadata: dict = {"index_fields": ["name"]}


class Bill(DataPoint):
    items: list[Item] = []
    total: float
    metadata: dict = {"index_fields": ["bill_id"]}

    @computed_field
    @property
    def bill_id(self) -> str:
        return str(self.id)


class Person(DataPoint):
    name: str
    bought: Bill = None
    # Make names searchable in the vector store
    metadata: Dict[str, Any] = {"index_fields": ["name"]}


async def leave_a_tip(_data=None, context: Dict[str, Any] | None = None) -> List[Tuple[str, dict]]:
    graph = await get_graph_engine()
    nodes, _edges = await graph.get_filtered_graph_data([{"type": ["Bill"]}])
    result = []
    for id, node in nodes:
        node["metadata"] = json.loads(node["metadata"])
        node["metadata"]["type"] = "Bill"
        node["total"] = node["total"] * 1.2
        result.append(node)
    return [Bill.from_json(json.dumps(node)) for node in result]


async def add_nodes_only(
    data_points: List[DataPoint],
    context: Dict[str, Any] | None = None,
) -> List[DataPoint]:
    """
    Update surface-level node attributes (total) in the graph and index.
    """
    graph_engine = await get_graph_engine()
    await graph_engine.add_nodes(data_points)
    await index_data_points(data_points)
    return data_points


async def main(text_data):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    await cognee.add(text_data)
    await cognee.cognify(graph_model=Person)

    tasks = [
        Task(leave_a_tip),  # input: text -> output: list[Bill]
        Task(
            add_nodes_only,
        ),
    ]

    await cognee.run_custom_pipeline(tasks=tasks)

    graph_visualization_path = path.join(path.dirname(__file__), "graph.html")
    await visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    text = "Mark bought 5 apples and 3 pears. One apple is 5$ and one pear is 8$."
    asyncio.run(main(text))
