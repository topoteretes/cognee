"""Bounded subgraph visualization demo.

``visualize_graph`` renders a *bounded subgraph* by default (seed nodes plus
their k-hop neighborhood, capped at ``max_nodes``) instead of the whole graph.
This script builds a small graph and writes one HTML file per seeding mode.

It uses a dedicated ``subgraph_demo`` dataset and does not prune, so it will not
touch your other cognee data. Requires a working LLM/embedding configuration
(see the project README), same as any other cognee example.
"""

import asyncio
import os

import cognee
from cognee import visualize_graph

ARTIFACTS = os.path.join(os.path.dirname(__file__), ".artifacts", "subgraph_demo")
DATASET = "subgraph_demo"

TEXT = (
    "Python is a programming language. Guido van Rossum created Python. "
    "Django is a web framework written in Python. NLP is a subfield of AI. "
    "spaCy is an NLP library for Python."
)


async def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    # Build a small knowledge graph in a dedicated dataset.
    await cognee.add(TEXT, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    def out(name: str) -> str:
        return os.path.join(ARTIFACTS, f"{name}.html")

    # Default: bounded subgraph seeded by a query's nearest vector hits.
    await visualize_graph(out("query_seeded"), dataset=DATASET, query="What is Python used for?")

    # Bare call with no seed: highest-degree nodes seed a representative view.
    await visualize_graph(out("default_degree"), dataset=DATASET)

    # Legacy whole-graph render.
    await visualize_graph(out("full_graph"), dataset=DATASET, full=True)

    # Explicit seeds: pass node ids you already have (e.g. from a prior query or
    # recall result). Uncomment with real ids from your graph:
    #   await visualize_graph(out("explicit_seeds"), dataset=DATASET, seed_node_ids=[...])

    # "Subgraph behind an answer": pass a recall/search result whose provenance
    # (used_graph_element_ids) seeds the view:
    #   result = await cognee.recall("What is Python?", session_id="demo")
    #   await visualize_graph(out("recall_seeded"), dataset=DATASET, recall_result=result)

    print(f"Wrote subgraph visualizations to {ARTIFACTS}")
    print("Caps: neighborhood_depth=2, neighborhood_seed_top_k=10, max_nodes=500")


if __name__ == "__main__":
    asyncio.run(main())
