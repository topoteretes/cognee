import asyncio
from pprint import pprint
from pydantic import BaseModel

import cognee
from cognee.shared.graph_model_utils import graph_schema_to_graph_model, graph_model_to_graph_schema
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.search import SearchType


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    text = (
        "Python is an interpreted, high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. "
        + "Python is widely used in data analysis, web development, and machine learning."
    )

    await cognee.add(text)

    # Define a custom graph model for programming languages.
    # Note: Models for generating graph schema can't inherit DataPoint directly, but will be set to inherit from
    # DataPoint in the graph_schema_to_model function later on
    class FieldType(BaseModel):
        name: str = "Field"
        metadata: dict = {"index_fields": ["name"]}

    class Field(BaseModel):
        name: str
        is_type: FieldType
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguageType(BaseModel):
        name: str = "Programming Language"
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguage(BaseModel):
        name: str
        used_in: list[Field] = []
        is_type: ProgrammingLanguageType
        metadata: dict = {"index_fields": ["name"]}

    # Transform the custom graph model to a JSON schema and then back to a Pydantic model class to ensure it is
    # properly formatted for cognee's graph engine
    graph_model_schema = graph_model_to_graph_schema(ProgrammingLanguage)

    graph_model = graph_schema_to_graph_model(graph_model_schema)

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify(graph_model=graph_model)

    query_text = "Tell me about Python and Rust"
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )

    print("Search results:")
    # Display results
    for result_text in search_results:
        pprint(result_text)

    # Generate interactive graph visualization
    print("\nGenerating graph visualization...")
    from cognee.api.v1.visualize import visualize_graph

    await visualize_graph()
    print("Visualization saved to ~/graph_visualization.html")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
