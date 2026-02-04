import asyncio
import cognee

import os
from os import path

from cognee import visualize_graph
from poc_cognify import cognify
# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY


async def cognee_demo():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Call Cognee to process document
    await cognee.add(file_path)
    await cognify()

    graph_visualization_path = path.join(path.dirname(__file__), "simple_example_result.html")

    await visualize_graph(graph_visualization_path)


# Cognee is an async library, it has to be called in an async context
asyncio.run(cognee_demo())
