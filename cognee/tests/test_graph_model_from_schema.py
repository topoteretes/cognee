"""cognify() and remember() build the graph from a graph model given as its JSON schema.

End to end, with a real LLM: the schema dict a REST caller sends (or one stored with
``PUT /api/v1/datasets/{id}/schema``) goes straight to the SDK, which builds the model
class itself. A schema that cannot be built is refused before anything is stored.
"""

import asyncio
from pprint import pprint

from pydantic import BaseModel

import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.exceptions import ExternalSchemaReferenceError
from cognee.shared.graph_model_utils import graph_model_to_graph_schema
from cognee.shared.logging_utils import ERROR, setup_logging


# Models for generating a graph schema can't inherit DataPoint directly; the class
# cognee builds from the schema inherits from DataPoint.
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


PYTHON_TEXT = (
    "Python is an interpreted, high-level, general-purpose programming language. It was "
    "created by Guido van Rossum and first released in 1991. Python is widely used in data "
    "analysis, web development, and machine learning."
)
RUST_TEXT = (
    "Rust is a general-purpose programming language emphasizing performance and memory "
    "safety. It is used in systems programming, web browsers and embedded devices."
)


async def node_types_by_name() -> dict[str, set[str]]:
    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()
    found: dict[str, set[str]] = {}
    for _node_id, properties in nodes:
        found.setdefault(properties.get("type"), set()).add(str(properties.get("name", "")))
    return found


def assert_language_graph(found: dict[str, set[str]], language: str) -> None:
    """The schema's classes, not KnowledgeGraph's Entity nodes, hold the extraction."""
    languages = {name.lower() for name in found.get("ProgrammingLanguage", set())}
    assert any(language in name for name in languages), (
        f"no ProgrammingLanguage node for {language!r}; node types: {sorted(found)}"
    )
    assert "ProgrammingLanguageType" in found, sorted(found)


async def main():
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # The JSON schema as a REST caller would send it, passed to the SDK as a plain dict.
    graph_model_schema = graph_model_to_graph_schema(ProgrammingLanguage)

    # cognify() with the schema dict.
    await cognee.add(PYTHON_TEXT, dataset_name="schema_cognify")
    await cognee.cognify(datasets=["schema_cognify"], graph_model=graph_model_schema)
    assert_language_graph(await node_types_by_name(), "python")
    print("cognify(graph_model=<schema dict>) built ProgrammingLanguage nodes.")

    # remember() with the same dict, which it hands on to cognify() unchanged.
    await cognee.remember(
        RUST_TEXT,
        dataset_name="schema_remember",
        graph_model=graph_model_schema,
        self_improvement=False,
    )
    assert_language_graph(await node_types_by_name(), "rust")
    print("remember(graph_model=<schema dict>) built ProgrammingLanguage nodes.")

    # A schema that cannot be built is refused before remember() stores anything.
    refused = {
        "title": "Refused",
        "type": "object",
        "properties": {"nodes": {"type": "array", "items": {"$ref": "/etc/passwd"}}},
    }
    try:
        await cognee.remember(RUST_TEXT, dataset_name="schema_refused", graph_model=refused)
    except ExternalSchemaReferenceError:
        pass
    else:
        raise AssertionError("remember() accepted a schema with an external $ref")
    dataset_names = {dataset.name for dataset in await cognee.datasets.list_datasets()}
    assert "schema_refused" not in dataset_names, dataset_names
    print("An invalid schema was refused before anything was stored.")

    query_text = "Tell me about Python and Rust"
    print(f"Searching cognee for insights with query: '{query_text}'")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )
    assert search_results, "GRAPH_COMPLETION returned nothing over the schema-built graph"
    print("Search results:")
    for result_text in search_results:
        pprint(result_text)

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
