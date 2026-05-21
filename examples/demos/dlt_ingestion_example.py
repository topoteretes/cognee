import asyncio
import os
import cognee

try:
    import dlt
except ImportError:
    dlt = None

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.modules.visualization.cognee_network_visualization import cognee_network_visualization
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver


async def main():
    """Demonstrates all DLT-based data ingestion modes in Cognee."""

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # ── Mode 1: Explicit dlt resource with nested data (merge/upsert) ──

    print("\n=== Mode 1: Explicit dlt resource ===")

    data = [
        {
            "id": 1,
            "name": "Alice",
            "pets": [
                {"id": 1, "name": "Fluffy", "type": "cat"},
                {"id": 2, "name": "Spot", "type": "dog"},
            ],
        },
        {"id": 2, "name": "Bob", "pets": [{"id": 3, "name": "Fido", "type": "dog"}]},
        {"id": 3, "name": "Charlie", "pets": [{"id": 4, "name": "Klokan", "type": "kangaroo"}]},
    ]

    @dlt.resource()
    def users_and_pets():
        yield data

    await cognee.add(
        users_and_pets,
        dataset_name="users_and_pets",
        primary_key="id",
        incremental_loading=False,
    )
    await cognee.cognify()

    result = await cognee.search("Which pet does Alice have?")
    print("Mode 1 results:", result)

    # ── Mode 2: CSV auto-detection ──

    print("\n=== Mode 2: CSV auto-detection ===")

    csv_path = os.path.join(os.path.dirname(__file__), "test_data", "employees.csv")

    await cognee.add(
        csv_path,
        dataset_name="employees",
        primary_key="id",
        incremental_loading=False,
    )
    await cognee.cognify()

    result = await cognee.search("Who works in Engineering?")
    print("Mode 2 results:", result)

    # ── Mode 3: Write disposition - append (always insert, no dedup) ──

    print("\n=== Mode 3: Write disposition - append ===")

    batch_1 = [
        {"id": 1, "event": "login", "user": "Alice", "timestamp": "2025-01-01"},
        {"id": 2, "event": "purchase", "user": "Bob", "timestamp": "2025-01-02"},
    ]
    batch_2 = [
        {"id": 3, "event": "logout", "user": "Alice", "timestamp": "2025-01-03"},
        {"id": 4, "event": "signup", "user": "Diana", "timestamp": "2025-01-04"},
    ]

    @dlt.resource()
    def event_batch_1():
        yield batch_1

    @dlt.resource()
    def event_batch_2():
        yield batch_2

    # First batch
    await cognee.add(
        event_batch_1,
        dataset_name="events_append",
        primary_key="id",
        write_disposition="append",
        incremental_loading=False,
    )
    # Second batch appended (no dedup)
    await cognee.add(
        event_batch_2,
        dataset_name="events_append",
        primary_key="id",
        write_disposition="append",
        incremental_loading=False,
    )
    await cognee.cognify()

    result = await cognee.search("What events happened?")
    print("Mode 3 results:", result)

    # ── Mode 4: Write disposition - replace (drop & recreate each run) ──

    print("\n=== Mode 4: Write disposition - replace ===")

    old_inventory = [
        {"id": 1, "product": "Widget A", "stock": 100},
        {"id": 2, "product": "Widget B", "stock": 50},
    ]
    new_inventory = [
        {"id": 1, "product": "Widget A", "stock": 200},
        {"id": 3, "product": "Widget C", "stock": 75},
    ]

    @dlt.resource()
    def inventory_old():
        yield old_inventory

    @dlt.resource()
    def inventory_new():
        yield new_inventory

    # First load
    await cognee.add(
        inventory_old,
        dataset_name="inventory_replace",
        primary_key="id",
        write_disposition="replace",
        incremental_loading=False,
    )
    # Replace entirely with new data
    await cognee.add(
        inventory_new,
        dataset_name="inventory_replace",
        primary_key="id",
        write_disposition="replace",
        incremental_loading=False,
    )
    await cognee.cognify()

    # ── Mode 5: Adding some unstructured text about users and pets along with the dlt resource ──

    result = await cognee.search("What products are in inventory?")
    print("Mode 4 results:", result)

    text = """Alice has two pets: a cat named Fluffy and a dog named Spot.
    She often says Fluffy is calm in the mornings, while Spot gets excited whenever someone mentions a walk.
    Bob has a dog named Fido, who is friendly with both Fluffy and Spot. Charlie owns a kangaroo named Klokan, which makes Charlie’s household the most unusual in the neighborhood.
    Recently, a new user named Diana joined their pet group with her cat, Luna.
    Diana says Luna is playful and curious, and Luna quickly became friends with Fluffy during their first meetup."""

    await cognee.add(
        [text, users_and_pets],
        dataset_name="users_and_pets_with_text",
        primary_key="id",
        incremental_loading=False,
    )

    await cognee.cognify()

    result = await cognee.search("Who is Diana?")
    print("Mode 5 results:", result)

    # ── Mode 6: Adding a csv along with an ontology ──

    await cognee.add(
        csv_path,
        dataset_name="employees",
        primary_key="id",
        incremental_loading=False,
    )

    ontology_path = os.path.join(os.path.dirname(__file__), "test_data", "employees_ontology.owl")

    # Create full config structure manually
    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    await cognee.cognify(config=config)

    result = await cognee.search("Who works in Engineering and is female?")
    print("Mode 6 results:", result)

    # ── Visualize the final graph ──

    print("\n=== Generating visualization ===")
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()
    nodes, edges = graph_data
    print(f"Final graph: {len(nodes)} nodes, {len(edges)} edges")

    dest = os.path.join(os.path.dirname(__file__), "dlt_example_graph.html")
    await cognee_network_visualization(graph_data, dest)
    print(f"Visualization saved to {dest}")


if __name__ == "__main__":
    asyncio.run(main())
