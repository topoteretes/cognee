import asyncio
import json
import cognee

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.graph_model_utils import graph_model_to_graph_schema


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
    used_in: list[Field] = []
    is_type: ProgrammingLanguageType
    metadata: dict = {"index_fields": ["name"]}


async def main():
    schema_dict = graph_model_to_graph_schema(ProgrammingLanguage)

    graph_model_schema_json = json.dumps(schema_dict)

    print(graph_model_schema_json)

    user_prompt = render_prompt(
        "custom_prompt_generation_user.txt", {"GRAPH_SCHEMA_JSON": graph_model_schema_json}
    )
    system_prompt = render_prompt("custom_prompt_generation_system.txt", {})

    custom_prompt = await LLMGateway.acreate_structured_output(
        text_input=user_prompt, system_prompt=system_prompt, response_model=str
    )

    print("Custom prompt generation complete")

    await cognee.add("""
        Python is a programming language widely used in machine learning, data analysis, and web development.
        Rust is a programming language used in systems programming, embedded software, and cybersecurity.
        SQL is a programming language used in database management and business intelligence reporting.
        JavaScript is a programming language used in frontend web development and interactive user interface design.
        Go is a programming language used in cloud infrastructure and backend microservices.
        R is a programming language used in statistics and academic research.
    """)

    await cognee.cognify(
        graph_model=ProgrammingLanguage,
        custom_prompt=custom_prompt,
    )

    print(custom_prompt)

    await cognee.visualize_graph()


if __name__ == "__main__":
    asyncio.run(main())
