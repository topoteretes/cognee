"""
arXiv connector — runnable demo.

Ingests research papers from arXiv into Cognee's knowledge graph
and queries the result.

Requirements:
    pip install cognee httpx dlt
    export LLM_API_KEY="your-openai-key"
"""

import asyncio
import cognee
from cognee.tasks.ingestion.connectors.arxiv import arxiv_source


async def main():
    # Start fresh
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("=" * 60)
    print("arXiv → Cognee knowledge graph demo")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Ingest papers by keyword search
    # ------------------------------------------------------------------ #
    print("\n[1] Ingesting papers about 'attention mechanism transformer'...")
    await cognee.remember(
        arxiv_source(
            query="attention mechanism transformer",
            max_results=5,
            resource_name="arxiv_transformers",
        ),
        dataset_name="arxiv_demo",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    print("    Done.")

    # ------------------------------------------------------------------ #
    # 2. Ingest a specific paper by arXiv ID
    # ------------------------------------------------------------------ #
    print("\n[2] Ingesting paper 1706.03762 (Attention Is All You Need)...")
    await cognee.remember(
        arxiv_source(
            paper_ids=["1706.03762"],
            resource_name="arxiv_specific",
        ),
        dataset_name="arxiv_demo",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    print("    Done.")

    # ------------------------------------------------------------------ #
    # 3. Ingest papers from the cs.AI category
    # ------------------------------------------------------------------ #
    print("\n[3] Ingesting latest cs.AI papers...")
    await cognee.remember(
        arxiv_source(
            category="cs.AI",
            max_results=3,
            resource_name="arxiv_cs_ai",
        ),
        dataset_name="arxiv_demo",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    print("    Done.")

    # ------------------------------------------------------------------ #
    # 4. Query the knowledge graph
    # ------------------------------------------------------------------ #
    queries = [
        "What are the main components of the transformer architecture?",
        "Which authors have published about attention mechanisms?",
        "What research categories are represented in the knowledge base?",
    ]

    print("\n[4] Querying Cognee knowledge graph...")
    print("-" * 60)

    for q in queries:
        print(f"\nQ: {q}")
        results = await cognee.search("GRAPH_COMPLETION", query=q)
        if results:
            for r in results[:2]:
                print(f"A: {str(r)[:300]}")
        else:
            print("A: (no results)")

    print("\n" + "=" * 60)
    print("Done. Papers are stored in Cognee's knowledge graph.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
