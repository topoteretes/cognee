"""LangChain agent with Cognee memory via tool calling.

This example shows how to give a LangChain agent persistent memory
using cognee.tools. The agent gets `remember` and `recall` as
LangChain StructuredTool objects.

Prerequisites:
    pip install cognee langchain langchain-openai
    export LLM_API_KEY="sk-..."
"""

import asyncio
import os

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from cognee.tools import for_langchain


async def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("LLM_API_KEY"),
    )

    tools = for_langchain()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with persistent memory. "
                "Use the `remember` tool to save important facts the user shares. "
                "Use the `recall` tool to recall previously stored information.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    # Store some facts
    print("User: Remember that I prefer dark mode and use vim keybindings.")
    result = await executor.ainvoke(
        {
            "input": "Remember that I prefer dark mode and use vim keybindings.",
        }
    )
    print(f"Assistant: {result['output']}\n")

    # Retrieve them later
    print("User: What are my editor preferences?")
    result = await executor.ainvoke(
        {
            "input": "What are my editor preferences?",
        }
    )
    print(f"Assistant: {result['output']}\n")


if __name__ == "__main__":
    asyncio.run(main())
