import os
import asyncio
from typing import List

from cognee import forget, remember, visualize_graph
from cognee.low_level import DataPoint

CUSTOM_PROMPT = (
    "Extract a simple graph containing Programming Language and Fields that it is used in."
)


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
    await forget(everything=True)

    text = "The Python programming language is widely used in data analysis, web development, and machine learning."

    await remember(
        text,
        graph_model=ProgrammingLanguage,
        custom_prompt=CUSTOM_PROMPT,
        self_improvement=False,
    )

    await visualize_data()


if __name__ == "__main__":
    asyncio.run(main())
