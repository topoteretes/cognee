"""Two async methods calling the same downstream acreate path.

The script first prunes, adds one sentence, and runs cognify.
Then it asks: "What does cognee do?" using acreate structured output.
One method has memory enabled, the other does not.

Run:
    uv run python examples/demos/simple_agent_v2/acreate_structured_output_usage_example.py
"""

from __future__ import annotations

import asyncio

import cognee

from examples.demos.simple_agent_v2.agentic_context_trace import (
    agentic_trace_root,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def setup_memory() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Cognee is working on agentic use cases to extract traces.")
    await cognee.cognify()


async def downstream_acreate_call(question: str) -> str:
    """Shared downstream path that always calls LLMGateway.acreate_structured_output."""
    return await LLMGateway.acreate_structured_output(
        text_input=question,
        system_prompt="Answer briefly.",
        response_model=str,
    )


@agentic_trace_root(
    with_memory=True,
    task_query="How are agents related to Cognee",
)
async def with_memory_method(query="What does cognee do?") -> str:
    return await downstream_acreate_call(question=query)


@agentic_trace_root(with_memory=False)
async def without_memory_method(query='What does cognee do?') -> str:
    return await downstream_acreate_call(question=query)

async def main() -> None:
    await setup_memory()
    with_memory_result = await with_memory_method()
    without_memory_result = await without_memory_method()

    print("with_memory -> result:")
    print(with_memory_result)
    print("\nwithout_memory -> result:")
    print(without_memory_result)

if __name__ == "__main__":


    asyncio.run(main())
