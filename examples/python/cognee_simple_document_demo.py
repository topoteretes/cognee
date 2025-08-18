import asyncio
import cognee

import os

# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY
os.environ["LLM_API_KEY"] = ""


async def cognee_demo():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent.parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    # Call Cognee to process document
    await cognee.add(file_path)
    await cognee.cognify()

    # Query Cognee for information from provided document
    answer = await cognee.search("List me all the important characters in Alice in Wonderland.")
    print(answer)

    answer = await cognee.search("How did Alice end up in Wonderland?")
    print(answer)

    answer = await cognee.search("Tell me about Alice's personality.")
    print(answer)


# Cognee is an async library, it has to be called in an async context
asyncio.run(cognee_demo())
