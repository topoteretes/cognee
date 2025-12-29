import asyncio
from os import path
from typing import Any
from pydantic import SkipValidation
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.tasks.storage import add_data_points
import cognee


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
    has_items: SkipValidation[Any]  # (Edge, list[Clothes])
    has_objects: SkipValidation[Any]  # (Edge, list[Object])
    knows: SkipValidation[Any]  # (Edge, list["Person"])


async def main():
    # Clear the database for a clean state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Create clothes items
    item1 = Clothes(name="Shirt", description="A blue shirt")
    item2 = Clothes(name="Pants", description="Black pants")
    item3 = Clothes(name="Jacket", description="Leather jacket")

    # Create object with simple relationship to clothes
    object1 = Object(
        name="Closet", description="A wooden closet", has_clothes=[item1, item2, item3]
    )

    # Create people with various weighted relationships
    person1 = Person(
        name="John",
        description="A software engineer",
        # Single weight (backward compatible)
        has_items=(Edge(weight=0.8, relationship_type="owns"), [item1, item2]),
        # Simple relationship without weights
        has_objects=(Edge(relationship_type="stores_in"), [object1]),
        knows=[],
    )

    person2 = Person(
        name="Alice",
        description="A designer",
        # Multiple weights on edge
        has_items=(
            Edge(
                weights={
                    "ownership": 0.9,
                    "frequency_of_use": 0.7,
                    "emotional_attachment": 0.8,
                    "monetary_value": 0.6,
                },
                relationship_type="owns",
            ),
            [item3],
        ),
        has_objects=(Edge(relationship_type="uses"), [object1]),
        knows=[],
    )

    person3 = Person(
        name="Bob",
        description="A friend",
        # Mixed: single weight + multiple weights
        has_items=(
            Edge(
                weight=0.5,  # Default weight
                weights={"trust_level": 0.9, "communication_frequency": 0.6},
                relationship_type="borrows",
            ),
            [item1],
        ),
        has_objects=[],
        knows=[],
    )

    # Create relationships between people with multiple weights
    person1.knows = (
        Edge(
            weights={
                "friendship_strength": 0.9,
                "trust_level": 0.8,
                "years_known": 0.7,
                "shared_interests": 0.6,
            },
            relationship_type="friend",
        ),
        [person2, person3],
    )

    person2.knows = (
        Edge(
            weights={"professional_collaboration": 0.8, "personal_friendship": 0.6},
            relationship_type="colleague",
        ),
        [person1],
    )

    all_data_points = [item1, item2, item3, object1, person1, person2, person3]

    # Add data points to the graph
    await add_data_points(all_data_points)

    # Visualize the graph
    graph_visualization_path = path.join(
        path.dirname(__file__), "weighted_graph_visualization.html"
    )
    await visualize_graph(graph_visualization_path)

    print("Graph with multiple weighted edges has been created and visualized!")
    print(f"Visualization saved to: {graph_visualization_path}")
    print("\nFeatures demonstrated:")
    print("- Single weight edges (backward compatible)")
    print("- Multiple weights on single edges")
    print("- Mixed single + multiple weights")
    print("- Hover over edges to see all weight information")
    print("- Different visual styling for single vs. multiple weighted edges")


if __name__ == "__main__":
    asyncio.run(main())
