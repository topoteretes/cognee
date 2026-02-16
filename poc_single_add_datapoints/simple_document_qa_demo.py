import asyncio
import cognee

import os
from os import path

from cognee import visualize_graph
from cognee.infrastructure.databases.graph import get_graph_engine
from poc_single_add_datapoints_pipeline import poc_cognify
# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY


async def main(use_poc):
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    graph_visualization_path = path.join(
        path.dirname(__file__), f"results/{'poc_' if use_poc else ''}simple_example_result.html"
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Call Cognee to process document
    await cognee.add(file_path)

    if use_poc:
        await poc_cognify(use_single_add_datapoints_poc=True)
    else:
        await cognee.cognify()

    await visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(use_poc=False))
        loop.run_until_complete(main(use_poc=True))

    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
