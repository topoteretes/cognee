"""Build a code knowledge graph with enola + cognee, then query it.

What it shows:
    - Running the enola-backed code graph pipeline (extract -> load nodes -> load edges)
    - Querying the resulting graph deterministically with SearchType.CODE

Requirements:
    - The enola binary — installed automatically on first run (pinned release,
      checksum-verified, placed in ~/.cognee/bin; opt out with
      ENOLA_AUTO_INSTALL=false), or install it yourself
      (https://github.com/enola-labs/enola#installation) / set ENOLA_PATH

SearchType.CODE does not require an LLM API key or embedding model.

For cross-repository paths, generate one Enola append/multi-repository snapshot
and ingest it into one dataset. Repositories indexed in separate datasets are
searched independently and cannot have graph paths between them.

Run it:
    CODE_GRAPH_REPO_PATH=/path/to/some/repo uv run python examples/python/code_graph_example.py
"""

import asyncio
import json
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
        # Pass index_vectors=True only if these facts should also be available
        # to semantic/LLM retrievers; SearchType.CODE does not need it.
        tasks=get_code_graph_tasks(repo_path),
        data=repo_path,
        dataset="code_graph_demo",
        pipeline_name="code_graph_pipeline",
    )

    print("Listing the first indexed code facts")
    search_results = await cognee.search(
        query_type=SearchType.CODE,
        query_text="",
        datasets=["code_graph_demo"],
        code_query={
            "operation": "query_facts",
            "kinds": ["module", "symbol", "route", "storage", "service"],
            "limit": 20,
        },
    )

    print(json.dumps(search_results, indent=2, default=str))

    # Other deterministic operations use the same API shape:
    # code_query={"operation": "explore", "id": "<fact id>", "max_depth": 2}
    # code_query={"operation": "traverse", "node_ids": ["<fact id>"], "direction": "reverse"}
    # code_query={"operation": "find_path", "source_id": "<id>", "target_id": "<id>"}
    # code_query={"operation": "impact_analysis", "id": "<fact id>", "max_depth": 3}


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
