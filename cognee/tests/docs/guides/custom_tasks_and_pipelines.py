import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, SkipValidation

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines import Task, run_pipeline


class Person(DataPoint):
    name: str
    # Optional relationships (we'll let the LLM populate this)
    knows: List["Person"] = []
    # Make names searchable in the vector store
    metadata: Dict[str, Any] = {"index_fields": ["name"]}


class People(BaseModel):
    persons: List[Person]


async def extract_people(text: str) -> List[Person]:
    system_prompt = (
        "Extract people mentioned in the text. "
        "Return as `persons: Person[]` with each Person having `name` and optional `knows` relations. "
        "If the text says someone knows someone set `knows` accordingly. "
        "Only include facts explicitly stated."
    )
    people = await LLMGateway.acreate_structured_output(text, system_prompt, People)
    return people.persons


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    text = "Alice knows Bob."

    tasks = [
        Task(extract_people),  # input: text -> output: list[Person]
        Task(add_data_points),  # input: list[Person] -> output: list[Person]
    ]

    async for _ in run_pipeline(tasks=tasks, data=text, datasets=["people_demo"]):
        pass


if __name__ == "__main__":
    asyncio.run(main())
