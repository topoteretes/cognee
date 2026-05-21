import asyncio
import os
from typing import Any, Dict, List
from pydantic import BaseModel

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines import Task
from cognee.api.v1.visualize.visualize import visualize_graph
from uuid import uuid5, NAMESPACE_OID, UUID


class PersonLLM(BaseModel):
    """Lightweight Pydantic model for LLM extraction only."""

    name: str
    knows: List[str] = []  # Just names for now, we'll resolve to Person instances later


class PeopleLLM(BaseModel):
    """Lightweight Pydantic model for LLM extraction only."""

    persons: List[PersonLLM]


class Person(DataPoint):
    name: str
    # Optional relationships (we'll let the LLM populate this)
    knows: List["Person"] = []
    # Make names searchable in the vector store
    metadata: Dict[str, Any] = {"index_fields": ["name"]}


class LightweightData(DataPoint):
    """Lightweight DataPoint model for data ingestion only."""

    id: UUID
    text: str


def build_lightweight_data_object(text_data):
    return LightweightData(id=uuid5(NAMESPACE_OID, text_data), text=text_data)


async def extract_people(data: LightweightData) -> List[Person]:
    system_prompt = (
        "Extract people mentioned in the text. "
        "Return as `persons: Person[]` with each Person having `name` and optional `knows` relations. "
        "Infer ‘knows’ only when there is a clear interpersonal interaction in the text."
    )
    # Create a mapping of name -> Person DataPoint
    person_map: Dict[str, Person] = {}
    for data_item in data:
        people_llm = await LLMGateway.acreate_structured_output(
            data_item.text, system_prompt, PeopleLLM
        )

        for person_llm in people_llm.persons:
            person_map[person_llm.name] = Person(name=person_llm.name)

        # Resolve knows relationships
        for person_llm in people_llm.persons:
            person = person_map[person_llm.name]
            person.knows = [person_map[name] for name in person_llm.knows if name in person_map]

    return list(person_map.values())


async def main(text_data):
    await cognee.forget(everything=True)
    await setup()

    tasks = [
        Task(extract_people),  # input: text -> output: list[Person]
        Task(add_data_points),  # input: list[Person] -> output: list[Person]
    ]

    await cognee.run_custom_pipeline(
        tasks=tasks, data=build_lightweight_data_object(text_data), dataset="people_demo"
    )

    await cognee.cognify()

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "custom_tasks_and_pipelines.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    text = "Alice knows Mark. Mark had dinner with Bob and Alice. Bob knows Mary."
    asyncio.run(main(text))
