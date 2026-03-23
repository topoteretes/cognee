"""OpenAI agent with Cognee memory via tool calling.

This example shows how to give an OpenAI chat model persistent memory
using cognee.tools. The model gets two tools — `remember` and
`recall` — and decides when to store or retrieve information.

Prerequisites:
    pip install cognee openai
    export LLM_API_KEY="sk-..."   # used by both OpenAI and Cognee
"""

import asyncio
import os

from openai import OpenAI
from cognee.tools import for_openai, handle_tool_call


def run_agent(user_message: str, client: OpenAI, model: str = "gpt-4o-mini"):
    """Single turn: send a message, let the model use tools, return final answer."""
    tools = for_openai()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with persistent memory. "
                "Use the `remember` tool to save important facts the user shares. "
                "Use the `recall` tool to recall previously stored information."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        choice = response.choices[0]

        # If the model wants to call tools, execute them and feed results back
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                result = asyncio.run(handle_tool_call(tool_call))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            # No more tool calls — return the final text
            return choice.message.content


def main():
    client = OpenAI(api_key=os.environ.get("LLM_API_KEY"))

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
