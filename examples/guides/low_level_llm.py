"""Call LLMGateway.acreate_structured_output directly to get a Pydantic model back from the LLM.

No graph or database is involved: one sentence is parsed into a MiniGraph of typed entities and
the model instance is printed.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/low_level_llm.py
"""

import asyncio

from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway


class MiniEntity(BaseModel):
    name: str
    type: str


class MiniGraph(BaseModel):
    nodes: list[MiniEntity]


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
