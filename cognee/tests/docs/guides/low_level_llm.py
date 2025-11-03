import asyncio

from pydantic import BaseModel
from typing import List
from cognee.infrastructure.llm.LLMGateway import LLMGateway


class MiniEntity(BaseModel):
    name: str
    type: str


class MiniGraph(BaseModel):
    nodes: List[MiniEntity]


async def main():
    system_prompt = (
        "Extract entities as nodes with name and type. "
        "Use concise, literal values present in the text."
    )

    text = "Apple develops iPhone; Audi produces the R8."

    result = await LLMGateway.acreate_structured_output(text, system_prompt, MiniGraph)
    print(result)
    # MiniGraph(nodes=[MiniEntity(name='Apple', type='Organization'), ...])


if __name__ == "__main__":
    asyncio.run(main())
