import os
import asyncio
from typing import Any, List
from pydantic import SkipValidation

from cognee import add, cognify, prune, visualize_graph
from cognee.low_level import DataPoint

CUSTOM_PROMPT = """
    Extract information as a simple, consistent knowledge graph.

    - Nodes are entities or concepts.
    - Edges are relationships between them.

    Rules:
    - Use general node labels like `Person`, not `Scientist` or `Mathematician`.
    - Put specific roles or categories in properties, such as `profession`.
    - Do not use vague labels like `Entity`.
    - Never use integers for node IDs.
    - Use human-readable IDs from the text.
    - Every node must include a `name` field.
    - Always use the most complete human-readable name.
    - Store properties as key-value pairs.
    - Do not use escaped quotes inside property values.
    - Use `snake_case` for relationship names, for example `acted_in`.
    - Use the most complete name consistently for the same entity.

    Goal:
    Keep the graph clear, simple, and easy to understand.
"""


# Define a custom graph model for programming languages.
class FieldType(DataPoint):
    name: str = "Field"


class Field(DataPoint):
    name: str
    is_type: FieldType
    metadata: dict = {"index_fields": ["name"]}


class ProgrammingLanguageType(DataPoint):
    name: str = "Programming Language"


class ProgrammingLanguage(DataPoint):
    name: str
    used_in: List[Field] = None
    is_type: ProgrammingLanguageType
    metadata: dict = {"index_fields": ["name"]}


async def visualize_data():
    graph_file_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "custom_graph_model_entity_schema_definition.html"
    )
    await visualize_graph(graph_file_path)


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    text = "The Python programming language is widely used in data analysis, web development, and machine learning."

    await add(text)
    await cognify(graph_model=ProgrammingLanguage, custom_prompt=CUSTOM_PROMPT)

    await visualize_data()


if __name__ == "__main__":
    asyncio.run(main())
