import asyncio
from os import path
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.tasks.storage import add_data_points


class Clothes(DataPoint):
    name: str
    description: str


class Object(DataPoint):
    name: str
    description: str
    has_clothes: list[Clothes]


class Person(DataPoint):
    name: str
    description: str
    has_items: (Edge, list[Clothes])
    has_objects: list[Object]


async def main():
    # Create clothes items
    item1 = Clothes(name="Shirt", description="A shirt")
    item2 = Clothes(name="Pants", description="Pants")

    # Create object with weighted relationship to clothes
    object1 = Object(
        name="Closet", 
        description="A closet", 
        has_clothes=[item1, item2]
    )

    # Create person with weighted relationship to items
    person1 = Person(
        name="John",
        description="A person",
        has_items=(Edge(weight=0.8, relationship_type="owns"), [item1, item2]),
        has_objects=[object1]
    )

    all_data_points = [item1, item2, object1, person1]

    # Add data points to the graph
    await add_data_points(all_data_points)

    # Visualize the graph
    graph_visualization_path = path.join(path.dirname(__file__), "weighted_graph_visualization.html")
    await visualize_graph(graph_visualization_path)

    print("Graph with weighted edges has been created and visualized!")
    print(f"Visualization saved to: {graph_visualization_path}")


if __name__ == "__main__":
    asyncio.run(main()) 