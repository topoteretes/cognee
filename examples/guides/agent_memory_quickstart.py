"""
Minimal quickstart for `cognee.agent_memory`.

What this script demonstrates:
- fixed memory query retrieval
- dynamic memory query retrieval from a wrapped method parameter
- session-backed memory retrieval from recent trace feedback
- the same downstream LLM path with memory disabled
- trace persistence enabled on decorated calls

The hidden fact ("Maple Panda") is intentionally private/demo-only knowledge so the
memory-enabled calls are easier to distinguish from the no-memory call.
"""

import asyncio

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway


def pretty_print_section(title: str, body: str) -> None:
    print(f"\n[{title}]")
    print(body)


async def setup_memory() -> None:
    # Start from a clean slate and add a fact the base model is unlikely to know.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.remember(
        (
            "Internal product note for the Cognee agentic memory feature: "
            "the private internal codename for the first supported `cognee.agent_memory` release "
            "is Maple Panda"
        ),
        self_improvement=False,
    )


async def ask_llm(question: str) -> str:
    return await LLMGateway.acreate_structured_output(
        text_input=question,
        system_prompt="Answer briefly.",
        response_model=str,
    )


@cognee.agent_memory(
    with_memory=True,
    save_session_traces=True,
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
    save_session_traces=True,
    memory_query_from_method="question",
)
async def with_dynamic_memory_agent(question: str) -> str:
    # Uses the wrapped method's `question` argument as the retrieval query.
    return await ask_llm(question)


@cognee.agent_memory(with_memory=False, save_session_traces=False)
async def without_memory_agent() -> str:
    # Same downstream LLM call shape, but memory retrieval is disabled.
    return await ask_llm("What animal does cognee internal name refer to?")


@cognee.agent_memory(with_memory=False, save_session_traces=True)
async def trace_test() -> str:
    # Simple call that is mostly useful for checking trace persistence.
    return await ask_llm("Just write out: results")


@cognee.agent_memory(
    with_memory=False,
    with_session_memory=True,
    save_session_traces=True,
    session_memory_last_n=5,
)
async def with_session_memory_agent() -> str:
    # Uses recent session-backed trace feedback without dataset search retrieval.
    return await ask_llm("What animal does cognee internal name refer to?")


async def main() -> None:
    await setup_memory()

    print("Agent memory quickstart")
    print("======================")
    print("Prompt: What animal does cognee internal name refer to?")
    print("Stored memory: Maple Panda")

    with_memory = await with_memory_agent()
    with_dynamic_memory = await with_dynamic_memory_agent(
        question="What animal does cognee internal name refer to?"
    )
    with_session_memory = await with_session_memory_agent()
    without_memory = await without_memory_agent()
    trace_result = await trace_test()

    pretty_print_section("Memory enabled", with_memory)
    pretty_print_section("Session memory", with_session_memory)
    pretty_print_section("Memory disabled", without_memory)
    pretty_print_section("Dynamic memory query", with_dynamic_memory)
    pretty_print_section("Trace test", trace_result)


if __name__ == "__main__":
    asyncio.run(main())
