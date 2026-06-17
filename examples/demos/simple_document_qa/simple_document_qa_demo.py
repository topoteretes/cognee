# ruff: noqa: E402
import asyncio
import os

# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY, in case you did not set it in the .env file
# Set this before Cognee config is initialized so the example uses the intended value.
# os.environ["LLM_API_KEY"] = ""

import cognee


async def cognee_demo():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    await cognee.forget(everything=True)

    # Call Cognee to process document
    await cognee.remember(file_path, self_improvement=False)

    # Query Cognee for information from provided document
    answer = await cognee.recall("List me all the important characters in Alice in Wonderland.")
    print(answer)

    answer = await cognee.recall("How did Alice end up in Wonderland?")
    print(answer)

    answer = await cognee.recall("Tell me about Alice's personality.")
    print(answer)


# Cognee is an async library, it has to be called in an async context
if __name__ == "__main__":
    asyncio.run(cognee_demo())
