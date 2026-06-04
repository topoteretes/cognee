"""Example: use HelixDB as a unified graph + vector backend for Cognee.

HelixDB v2 is a single engine that stores graph nodes/edges and native vector
indexes together, so one instance backs both Cognee's graph and vector stores
(via HelixHybridAdapter). Embeddings are computed client-side by Cognee.

Prerequisites:
  1. A running HelixDB v2 gateway, e.g.:
        docker run -p 6969:6969 ghcr.io/helixdb/database-dev
  2. LLM_API_KEY set in your environment / .env (used by cognify).

Run:  python examples/database_examples/helix_example.py
"""

import asyncio
import os
import pathlib

from dotenv import load_dotenv

import cognee
from cognee import SearchType

load_dotenv()

HELIX_URL = os.getenv("HELIX_URL", "http://localhost:6969")
HELIX_KEY = os.getenv("HELIX_KEY", "")  # Bearer token for Helix Cloud (empty for local)


async def main():
    # Point both the graph and vector stores at the SAME HelixDB instance.
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "helix",
            "graph_database_url": HELIX_URL,
            "graph_database_key": HELIX_KEY,
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "helix",
            "vector_db_url": HELIX_URL,
            "vector_db_key": HELIX_KEY,
        }
    )

    current_dir = pathlib.Path(__file__).parent
    cognee.config.data_root_directory(str(current_dir / "data_storage"))
    cognee.config.system_root_directory(str(current_dir / "cognee_system"))

    # Start from a clean slate (optional).
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    sample_text = """HelixDB is a unified graph and vector database. It stores graph nodes,
    edges, and vector indexes in a single engine, so a semantic search can flow directly into a
    graph traversal in one query. Cognee turns raw text into a knowledge graph using an
    Extract-Cognify-Load pipeline that combines vector search with graph reasoning."""

    await cognee.add(sample_text)
    await cognee.cognify()

    print("\n======== GRAPH_COMPLETION: 'What is HelixDB?' ========")
    for result in await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="What is HelixDB?"
    ):
        print(f"- {result}")

    print("\n======== CHUNKS: 'vector database' ========")
    for result in await cognee.search(query_type=SearchType.CHUNKS, query_text="vector database"):
        print(f"- {result}")


if __name__ == "__main__":
    asyncio.run(main())
