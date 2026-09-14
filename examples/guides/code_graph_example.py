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

Prefer a one-liner? cognee.remember(repo_path_or_git_url, content_type="code")
runs this same pipeline in a single call (it also accepts a list of
repositories, and index_vectors=True to enable embeddings). This example
assembles the pipeline explicitly so each step stays visible.

For cross-repository paths, generate one Enola append/multi-repository snapshot
and ingest it into one dataset. Repositories indexed in separate datasets are
searched independently and cannot have graph paths between them.

Run it:
    CODE_GRAPH_REPO_PATH=/path/to/some/repo uv run python examples/guides/code_graph_example.py
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
        # This pipeline is deterministic (no LLM/embedding calls), so skip the
        # first-run LLM/embedding connection checks and stay truly keyless.
        skip_connection_test=True,
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

    print("Module-level architecture overview, drawn as a Mermaid diagram")
    architecture = await cognee.search(
        query_type=SearchType.CODE,
        query_text="",
        datasets=["code_graph_demo"],
        # Symbol-to-symbol edges are rolled up to the modules that declare
        # them; routes/storage/services hang off their modules. The result
        # includes deterministic Mermaid source (paste it into any Markdown
        # renderer that supports ```mermaid fences); "diagram": "dot" gives
        # Graphviz, and any other operation accepts the same option.
        code_query={"operation": "architecture", "max_nodes": 40},
    )
    # search() returns one {dataset_id, dataset_name, search_result} entry per
    # dataset; the CODE operation's result is the (single) search_result item.
    for entry in architecture:
        payload = entry.get("search_result") if isinstance(entry, dict) else None
        if isinstance(payload, list) and payload:
            payload = payload[0]
        diagram = payload.get("diagram") if isinstance(payload, dict) else None
        if diagram and diagram.get("source"):
            print(diagram["source"])

    print("Architecture findings enola's explainers produced (with their evidence facts)")
    insights = await cognee.search(
        query_type=SearchType.CODE,
        query_text="",
        datasets=["code_graph_demo"],
        # Structural findings (cycles, declared-layer violations) score 1.0;
        # heuristic ones (hotspots, god-class, complexity outliers) score below.
        code_query={"operation": "insights", "min_confidence": 0.5, "limit": 10},
    )
    print(json.dumps(insights, indent=2, default=str))

    # Other deterministic operations use the same API shape. Ids may be
    # cognee node ids or enola's own 32-hex fact ids (from facts.jsonl):
    # code_query={"operation": "explore", "id": "<fact id>", "max_depth": 2}
    # code_query={"operation": "traverse", "node_ids": ["<fact id>"], "direction": "reverse"}
    # code_query={"operation": "find_path", "source_id": "<id>", "target_id": "<id>"}
    # code_query={"operation": "impact_analysis", "id": "<fact id>", "max_depth": 3}
    # code_query={"operation": "query_facts", "kind": "dependency", "prop": "type",
    #             "prop_value": "package"}   # declared packages from manifests (purl names)
    # code_query={"operation": "delta"}    # last ingestion's changes + the snapshot receipt


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
