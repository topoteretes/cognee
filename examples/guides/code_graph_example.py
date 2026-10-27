"""Build a code knowledge graph with enola + cognee, then query it.

What it shows:
    - Running the enola-backed code graph pipeline (extract -> load nodes ->
      load edges -> store docstrings)
    - Querying the resulting graph deterministically with SearchType.CODE
    - Finding a docstring with SearchType.CHUNKS: the pipeline stores the
      repository's docstrings as knowledge-graph chunks linked to the code
      graph (on by default; CODE_GRAPH_DOCS=false switches it off)

Requirements:
    - The enola binary included with cognee, or an explicit binary set through
      ENOLA_PATH
    - For the docstring step: an embedding model. With no embedding provider
      and no LLM key configured, the local fastembed model is used (downloaded
      on first use), so the example stays keyless. No step needs an LLM key.

SearchType.CODE does not require an LLM API key or embedding model.

Prefer a one-liner? cognee.remember(repo_path_or_git_url, content_type="code")
runs this same pipeline in a single call (it also accepts a list of
repositories, and index_vectors=True to embed the code facts themselves too).
This example assembles the pipeline explicitly so each step stays visible.

For cross-repository paths, generate one Enola append/multi-repository snapshot
and ingest it into one dataset. Repositories indexed in separate datasets are
searched independently and cannot have graph paths between them.

Run it:
    CODE_GRAPH_REPO_PATH=/path/to/some/repo uv run python examples/guides/code_graph_example.py

    The code graph alone, no docstrings and no embedding at all:
    CODE_GRAPH_DOCS=false CODE_GRAPH_REPO_PATH=/path/to/some/repo uv run python examples/guides/code_graph_example.py
"""

import asyncio
import json
import os

import cognee
from cognee import SearchType
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.code_graph import get_code_graph_tasks


async def main():
    repo_path = os.getenv("CODE_GRAPH_REPO_PATH", os.getcwd())
    # The docstring step reads CODE_GRAPH_DOCS (on by default), the same
    # setting cognify()'s code routes follow; get_code_graph_tasks(repo_path,
    # code_docs=False) is the programmatic opt-out.
    store_docstrings = get_cognify_config().code_graph_docs

    # Start clean so the example is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print(f"Extracting code graph from: {repo_path}")
    await cognee.run_custom_pipeline(
        # Pass index_vectors=True only if the code facts themselves should also
        # be available to semantic/LLM retrievers; SearchType.CODE does not
        # need it, and the docstring chunks are embedded either way.
        tasks=get_code_graph_tasks(repo_path),
        data=repo_path,
        dataset="code_graph_demo",
        pipeline_name="code_graph_pipeline",
        # No task in this list calls the LLM, so skip the first-run connection
        # checks and stay keyless. The docstring step still embeds its chunks
        # (fastembed when no provider is configured); a misconfigured embedding
        # provider surfaces there rather than in the probe.
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

    # Docstrings as memory. The last task of the pipeline above stored each
    # documented symbol's docstring (the whole comment enola wrote to its
    # docs.jsonl) as a DocumentChunk whose text starts with "file:line" and the
    # symbol's name, linked to its CodeSymbol with a `describes` edge, so
    # recall / CHUNKS / GRAPH_COMPLETION answer "what does X do" from the
    # docstring and reach the code in one hop. Embed-only: no LLM call.
    # cognify_code_docs_pipeline(user, dataset=..., extract_entities=True) in
    # cognee.memify_pipelines.cognify_code_docs additionally runs cognify's LLM
    # entity extraction and summaries over the chunks (needs an LLM key).
    if store_docstrings:
        print("Docstring chunks answering a question (SearchType.CHUNKS, embeddings only)")
        docstring_hits = await cognee.search(
            query_type=SearchType.CHUNKS,
            query_text="what does the main function do",
            datasets=["code_graph_demo"],
            top_k=3,
        )
        print(json.dumps(docstring_hits, indent=2, default=str))

    # Other deterministic operations use the same API shape. Ids may be
    # cognee node ids or enola's own 32-hex fact ids (from facts.jsonl):
    # code_query={"operation": "explore", "id": "<fact id>", "max_depth": 2}
    # code_query={"operation": "traverse", "node_ids": ["<fact id>"], "direction": "reverse"}
    # code_query={"operation": "find_path", "source_id": "<id>", "target_id": "<id>"}
    # code_query={"operation": "impact_analysis", "id": "<fact id>", "max_depth": 3}
    # code_query={"operation": "query_facts", "kind": "dependency", "prop": "type",
    #             "prop_value": "package"}   # declared packages from manifests (purl names)
    # code_query={"operation": "delta"}    # last ingestion's changes + the snapshot receipt
    # code_query={"operation": "ask", "question": "How is an order placed?"}
    #     # seeds, flows, one-hop neighbourhood and file:line citations to read


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
