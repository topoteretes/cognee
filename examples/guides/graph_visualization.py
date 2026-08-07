"""Render a knowledge graph to an interactive HTML file.

``visualize_graph`` renders a *bounded subgraph* by default — seed nodes plus their
k-hop neighborhood, capped at ``max_nodes`` — instead of the whole graph. This guide
writes one HTML file per seeding mode so you can compare them:

  1. default   — highest-degree nodes seed a representative view
  2. query     — the query's nearest vector hits seed the view
  3. full      — legacy whole-graph render

Caps in effect: neighborhood_depth=2, neighborhood_seed_top_k=10, max_nodes=500.
"""

import asyncio
import os

import cognee
from cognee import visualize_graph

ARTIFACTS = os.path.join(os.path.dirname(__file__), ".artifacts", "graph_visualization")
DATASET = "graph_visualization_guide"

TEXT = [
    "Python is a programming language. Guido van Rossum created Python.",
    "Django is a web framework written in Python.",
    "NLP is a subfield of AI. spaCy is an NLP library for Python.",
]


async def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(TEXT, dataset_name=DATASET, self_improvement=False)

    def out(name: str) -> str:
        return os.path.join(ARTIFACTS, f"{name}.html")

    # 1. Bare call: highest-degree nodes seed a representative bounded subgraph.
    await visualize_graph(out("default_degree_seeded"), dataset=DATASET)

    # 2. Query-seeded: the query's nearest vector hits become the seeds.
    await visualize_graph(out("query_seeded"), dataset=DATASET, query="What is Python used for?")

    # 3. Whole graph, unbounded.
    await visualize_graph(out("full_graph"), dataset=DATASET, full=True)

    # Two more seeding options, if you already have node ids or a recall result:
    #   await visualize_graph(out("explicit_seeds"), dataset=DATASET, seed_node_ids=[...])
    #
    #   result = await cognee.recall("What is Python?", datasets=[DATASET])
    #   await visualize_graph(out("recall_seeded"), dataset=DATASET, recall_result=result)
    # The second seeds the view from the answer's provenance (used_graph_element_ids),
    # so you see the subgraph behind a specific answer.

    print(f"Wrote visualizations to {ARTIFACTS}")


if __name__ == "__main__":
    asyncio.run(main())
