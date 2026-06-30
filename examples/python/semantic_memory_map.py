"""Example: Semantic Memory Map visualization.

Stores a few facts, builds the knowledge graph, then renders an HTML
visualization where nodes are positioned by *semantic similarity* in
embedding space rather than graph topology.

Run:
    python examples/semantic_memory_map.py

The output file ``semantic_memory_map.html`` will open automatically in
your default browser.
"""

import asyncio
import os
import webbrowser

import cognee
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from cognee.modules.visualization.layouts.semantic_layout import (
    compute_semantic_positions,
)
from cognee.modules.visualization.preprocessor import preprocess


TEXTS = [
    "Neural networks are computing systems inspired by biological neurons. "
    "They learn by adjusting weights through backpropagation.",

    "The hippocampus converts short-term memories into long-term ones via "
    "synaptic plasticity and long-term potentiation.",

    "Transformers use self-attention to relate every token to every other "
    "token, enabling parallel processing of sequences.",

    "Gradient descent minimises the loss function by iteratively moving "
    "weights in the direction of the negative gradient.",

    "Dopamine acts as a reward signal in the brain, reinforcing neural "
    "pathways that led to positive outcomes.",
]


async def main() -> None:
    # Ingest texts into Cognee memory
    print("Storing texts in memory…")
    for text in TEXTS:
        await cognee.remember(text, dataset_name="semantic_map_demo")

    # Retrieve graph data
    from cognee.infrastructure.databases.graph import get_graph_engine
    engine = await get_graph_engine()
    graph_data = await engine.get_graph_data()

    # Preprocess into renderer-facing snapshot
    pre = preprocess(graph_data)

    # Compute semantic positions from embeddings
    print("Computing semantic layout…")
    positions = await compute_semantic_positions(pre)

    # Inject positions into each node so the renderer can seed D3
    pos_lookup = {nid: {"x": x, "y": y} for nid, (x, y) in positions.items()}
    import json
    positions_json = json.dumps(pos_lookup)

    # Render HTML — the orchestrator replaces __SEMANTIC_POSITIONS__
    output_path = os.path.join(os.path.dirname(__file__), "semantic_memory_map.html")
    html = await cognee_network_visualization(
        graph_data,
        destination_file_path=output_path,
    )

    # Patch in the semantic positions (until the orchestrator token is wired)
    html = html.replace('"__SEMANTIC_POSITIONS__"', positions_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved to {output_path}")
    webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    asyncio.run(main())
