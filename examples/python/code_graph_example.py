"""Build a code knowledge graph with enola + cognee, then query it.

What it shows:
    - Running the enola-backed code graph pipeline (extract -> load nodes -> load edges)
    - Searching the resulting architectural graph with GRAPH_COMPLETION

Requirements:
    - The enola binary installed (https://github.com/enola-labs/enola#installation)
      or ENOLA_PATH pointing at it
    - A `.env` file with a working LLM_API_KEY (copy `.env.template`)

Run it:
    CODE_GRAPH_REPO_PATH=/path/to/some/repo uv run python examples/python/code_graph_example.py
"""

import asyncio
import os

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.code_graph import get_code_graph_tasks


async def main():
    repo_path = os.getenv("CODE_GRAPH_REPO_PATH", os.getcwd())

    # Start clean so the example is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print(f"Extracting code graph from: {repo_path}")
    await cognee.run_custom_pipeline(
        tasks=get_code_graph_tasks(repo_path),
        data=repo_path,
        dataset="code_graph_demo",
        pipeline_name="code_graph_pipeline",
    )

    query_text = "What services, routes, and storage does this codebase have?"
    print(f"Searching the code graph with query: '{query_text}'")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=query_text,
        datasets=["code_graph_demo"],
    )

    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
