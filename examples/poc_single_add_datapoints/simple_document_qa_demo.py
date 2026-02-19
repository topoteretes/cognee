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
    file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "alice_in_wonderland.txt"
        )
    )

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


async def _run():
    await main(use_poc=False)
    await main(use_poc=True)


if __name__ == "__main__":
    asyncio.run(_run())
