"""
Minimal quickstart for `cognee.agent_memory`.

What this script demonstrates:
- fixed memory query retrieval
- dynamic memory query retrieval from a wrapped method parameter
- the same downstream LLM path with memory disabled
- trace persistence enabled on decorated calls

The hidden fact ("Maple Panda") is intentionally private/demo-only knowledge so the
memory-enabled calls are easier to distinguish from the no-memory call.
"""

import asyncio

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def setup_memory() -> None:
    # Start from a clean slate and add a fact the base model is unlikely to know.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(
        (
            "Internal product note for the Cognee agentic memory feature: "
            "the private internal codename for the first supported `cognee.agent_memory` release "
            "is Maple Panda"
        ),
    )
    await cognee.cognify()


async def ask_llm(question: str) -> str:
    return await LLMGateway.acreate_structured_output(
        text_input=question,
        system_prompt="Answer briefly.",
        response_model=str,
    )


@cognee.agent_memory(
    with_memory=True,
    save_traces=True,
    memory_query_fixed="What animal does cognee internal name refer to?",
    memory_system_prompt=(
        "Return only the codename from memory context. "
        "If there is no matching codename, return an empty string."
    ),
)
async def with_memory_agent() -> str:
    # Uses a fixed retrieval query declared at decoration time.
    return await ask_llm("What animal does cognee internal name refer to?")


@cognee.agent_memory(
    with_memory=True,
    save_traces=True,
    memory_query_from_method="question",
)
async def with_dynamic_memory_agent(question: str) -> str:
    # Uses the wrapped method's `question` argument as the retrieval query.
    return await ask_llm(question)


@cognee.agent_memory(with_memory=False, save_traces=True)
async def without_memory_agent() -> str:
    # Same downstream LLM call shape, but memory retrieval is disabled.
    return await ask_llm("What animal does cognee internal name refer to?")


@cognee.agent_memory(with_memory=False, save_traces=True)
async def trace_test() -> str:
    # Simple call that is mostly useful for checking trace persistence.
    return await ask_llm("Just write out: results")


async def main() -> None:
    await setup_memory()

    print("This quickstart compares the same LLM path with memory on, memory off,")
    print("and dynamic query resolution from a wrapped method parameter.")
    print()

    with_memory = await with_memory_agent()
    with_dynamic_memory = await with_dynamic_memory_agent(
        question="What animal does cognee internal name refer to?"
    )
    without_memory = await without_memory_agent()
    trace_result = await trace_test()

    print("The in-memory knowledge is that the feature is called Maple Panda")

    print("AGENT ANSWERS:")
    print("WITH MEMORY:")
    print(with_memory)
    print()
    print("WITHOUT MEMORY:")
    print(without_memory)
    print()
    print("WITH DYNAMIC MEMORY QUERY:")
    print(with_dynamic_memory)
    print()
    print("TRACE TEST:")
    print(trace_result)


if __name__ == "__main__":
    asyncio.run(main())
