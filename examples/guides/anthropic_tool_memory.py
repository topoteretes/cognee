"""Anthropic agent with Cognee memory via tool use.

This example shows how to give an Anthropic Claude model persistent memory
using cognee.tools. The model gets two tools — `remember` and
`recall` — and decides when to store or retrieve information.

Prerequisites:
    pip install cognee anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    export LLM_API_KEY="sk-..."   # used by Cognee
"""

import asyncio
import os

from anthropic import Anthropic
from cognee.tools import for_anthropic, handle_tool_call


def run_agent(user_message: str, client: Anthropic, model: str = "claude-sonnet-4-20250514"):
    """Single turn: send a message, let the model use tools, return final answer."""
    tools = for_anthropic()
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=(
                "You are a helpful assistant with persistent memory. "
                "Use the `remember` tool to save important facts the user shares. "
                "Use the `recall` tool to recall previously stored information."
            ),
            messages=messages,
            tools=tools,
        )

        # Collect the full response content
        messages.append({"role": "assistant", "content": response.content})

        # Check if the model wants to use tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = asyncio.run(handle_tool_call(block))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool use — extract final text
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""


def main():
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Store some facts
    print("User: Remember that I prefer dark mode and use vim keybindings.")
    answer = run_agent(
        "Remember that I prefer dark mode and use vim keybindings.",
        client,
    )
    print(f"Assistant: {answer}\n")

    # Retrieve them later
    print("User: What are my editor preferences?")
    answer = run_agent("What are my editor preferences?", client)
    print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
