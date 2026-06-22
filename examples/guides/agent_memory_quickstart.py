"""
Two agents, two types of memory.

support_agent — remembers everything within the active session.
                Traces are saved to the knowledge graph over time.

faq_bot       — reads only from the knowledge graph.
                It learns only what support_agent has already filed.
"""

import asyncio
import os
import warnings

os.environ["LOG_LEVEL"] = "ERROR"
os.environ["COGNEE_LOG_FILE"] = "false"
warnings.filterwarnings("ignore")

import cognee  # noqa: E402
from cognee.infrastructure.llm.LLMGateway import LLMGateway  # noqa: E402

SESSION_ID = "ticket_001"
BUG = "Login fails with error XQ-99."
FIX = "Set XQ_TOKEN=1 in the .env file."
NO_INFO = "NO INFO AVAILABLE"


async def setup() -> None:
    await cognee.forget(everything=True)
    await cognee.remember(
        ["Our app is a web service. Users log in to access their account."], self_improvement=False
    )


async def ask_llm(question: str, system_prompt: str) -> str:
    return await LLMGateway.acreate_structured_output(
        text_input=question,
        system_prompt=system_prompt,
        response_model=str,
    )


@cognee.agent_memory(
    with_memory=False,
    with_session_memory=True,
    save_session_traces=True,
    session_id=SESSION_ID,
    session_memory_last_n=2,
    persist_session_trace_after=3,
)
async def support_agent(question: str, system_prompt: str) -> str:
    return await ask_llm(question, system_prompt)


@cognee.agent_memory(
    with_memory=True,
    with_session_memory=False,
    save_session_traces=False,
    memory_query_from_method="question",
)
async def faq_bot(question: str, system_prompt: str) -> str:
    return await ask_llm(question, system_prompt)


async def main() -> None:
    print("=== Agent Memory Quickstart ===\n")
    print("support_agent: session memory  — knows what happened in this conversation.")
    print("faq_bot:       knowledge graph — knows only what has been formally filed.\n")

    print("Setting up knowledge graph...")
    await setup()
    print("Ready.\n")

    recall_prompt = f"Answer based on the available context. If it is not available in the context, say exactly: {NO_INFO}"

    support_agent_q1 = f"A user just reported this: {BUG}"
    print(f"support_agent_q: {support_agent_q1}")
    support_agent_a1 = await support_agent(
        support_agent_q1, f"Confirm you received it. Say exactly: {BUG}"
    )
    print(f"support_agent_a: {support_agent_a1}\n")

    faq_bot_q = "How do I fix error XQ-99?"

    support_agent_q2 = "What bug was just reported?"
    print(f"support_agent_q: {support_agent_q2}")
    support_agent_a2 = await support_agent(
        support_agent_q2,
        f"Use your session memory. If you know, say: {BUG} If not, say: {NO_INFO}",
    )
    print(f"support_agent_a: {support_agent_a2}")

    print(f"faq_bot_q:       {faq_bot_q}")
    faq_bot_a_before = await faq_bot(faq_bot_q, recall_prompt)
    print(f"faq_bot_a:       {faq_bot_a_before}")

    print("\n^ support_agent recalled the bug from session. faq_bot had no context yet.\n")

    support_agent_q3 = f"Log this fix for the login crash: {FIX}"
    print(f"support_agent_q: {support_agent_q3}")
    support_agent_a3 = await support_agent(support_agent_q3, f"Confirm the fix. Say exactly: {FIX}")
    print(f"support_agent_a: {support_agent_a3}")
    print("(Session traces are now persisted to the knowledge graph.)\n")

    print(f"faq_bot_q:       {faq_bot_q}")
    faq_bot_a_after = await faq_bot(faq_bot_q, recall_prompt)
    print(f"faq_bot_a:       {faq_bot_a_after}")

    print("\n^ faq_bot now answered correctly — session traces reached the knowledge graph.")


if __name__ == "__main__":
    asyncio.run(main())
