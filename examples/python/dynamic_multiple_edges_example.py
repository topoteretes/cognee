import asyncio
from os import path
from typing import Any
from pydantic import SkipValidation
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.tasks.storage import add_data_points
import cognee


class Employee(DataPoint):
    name: str
    role: str


class Company(DataPoint):
    name: str
    industry: str
    employs: SkipValidation[Any]  # Mixed list: employees with/without weights


async def main():
    # Clear the database for a clean state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Create employees
    michael = Employee(name="Michael", role="Regional Manager")
    dwight = Employee(name="Dwight", role="Assistant to the Regional Manager")
    jim = Employee(name="Jim", role="Sales Representative")
    pam = Employee(name="Pam", role="Receptionist")
    kevin = Employee(name="Kevin", role="Accountant")
    angela = Employee(name="Angela", role="Senior Accountant")
    oscar = Employee(name="Oscar", role="Accountant")
    stanley = Employee(name="Stanley", role="Sales Representative")
    phyllis = Employee(name="Phyllis", role="Sales Representative")

    # Create Dunder Mifflin with mixed employee relationships
    dunder_mifflin = Company(
        name="Dunder Mifflin Paper Company",
        industry="Paper Sales",
        employs=[
            # Manager with high authority weight
            (Edge(weight=0.9, relationship_type="manager"), michael),
            # Sales team with performance weights
            (
                Edge(weights={"sales_performance": 0.8, "loyalty": 0.9}, relationship_type="sales"),
                dwight,
            ),
            (
                Edge(
                    weights={"sales_performance": 0.7, "creativity": 0.8}, relationship_type="sales"
                ),
                jim,
            ),
            (
                Edge(
                    weights={"sales_performance": 0.6, "customer_service": 0.9},
                    relationship_type="sales",
                ),
                phyllis,
            ),
            (
                Edge(
                    weights={"sales_performance": 0.5, "experience": 0.8}, relationship_type="sales"
                ),
                stanley,
            ),
            # Accounting department as a group
            (
                Edge(
                    weights={"department_efficiency": 0.8, "team_cohesion": 0.9},
                    relationship_type="accounting",
                ),
                [oscar, kevin, angela],
            ),
            # Admin staff without weights (simple relationships)
            pam,
        ],
    )

    all_data_points = [
        michael,
        dwight,
        jim,
        pam,
        kevin,
        angela,
        oscar,
        stanley,
        phyllis,
        dunder_mifflin,
    ]

    # Add data points to the graph
    await add_data_points(all_data_points)

    # Visualize the graph
    graph_visualization_path = path.join(path.dirname(__file__), "dunder_mifflin_graph.html")
    await visualize_graph(graph_visualization_path)

    print("Dynamic multiple edges graph has been created and visualized!")
    print(f"Visualization saved to: {graph_visualization_path}")
    print("\nTechnical features demonstrated:")
    print("- Mixed list support: weighted and unweighted relationships in single field")
    print("- Single weight edges with relationship types")
    print("- Multiple weight edges with custom metrics")
    print("- Group relationships: single edge connecting multiple nodes")
    print("- Simple relationships without edge metadata")
    print("- Flexible edge extraction from heterogeneous data structures")


if __name__ == "__main__":
    asyncio.run(main())
