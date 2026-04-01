import asyncio

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def setup_memory() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(
        (
            "Internal product note for the Cognee agentic memory feature: "
            "the private internal codename for the first supported `cognee.agent_memory` release "
            "is 'Maple Panda'"
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
    memory_query_fixed="What is the private internal codename for the first supported cognee.agent_memory release?",
)
async def with_memory_agent() -> str:
    return await ask_llm(
        "What is the private internal codename for the first supported cognee.agent_memory release?"
    )


@cognee.agent_memory(
    with_memory=True,
    save_traces=True,
    memory_query_from_method="question",
)
async def with_dynamic_memory_agent(question: str) -> str:
    return await ask_llm(question)


@cognee.agent_memory(with_memory=False, save_traces=True)
async def without_memory_agent() -> str:
    return await ask_llm(
        "What is the private internal codename for the first supported cognee.agent_memory release?"
    )

@cognee.agent_memory(with_memory=False, save_traces=True)
async def trace_test() -> str:
    return await ask_llm("Just write out: results")


async def main() -> None:
    await setup_memory()

    with_memory = await with_memory_agent()
    with_dynamic_memory = await with_dynamic_memory_agent(
        question="What animal does cognee internal name refer to?"
    )
    without_memory = await without_memory_agent()
    trace_result = await trace_test()

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
