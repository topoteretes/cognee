"""Demo: the Semantic Memory Map.

Runs a real cognee pipeline (add → cognify) and renders the knowledge graph
with ``visualize_graph``. The resulting HTML has a **Semantic** tab that lays
the graph out by *meaning*: every node is placed at the 2-D projection of its
embedding, so semantically similar nodes cluster together — a view the classic
topology layout can't show.

Nothing here patches the HTML. The semantic tab is produced by the production
render path itself:

    fetch_node_embeddings  (join graph nodes to their stored vectors)
        -> semantic_layout.compute_positions   (PCA, pinned)
        -> compute_clusters                     (k-means + nearest neighbors)
        -> cognee_network_visualization         (token substitution)

Requirements: an LLM + embedding key in the environment (e.g. ``LLM_API_KEY``),
exactly as ``cognify`` already needs. With no embeddings the tab simply shows a
friendly empty state — the classic render never breaks.

Run:
    python examples/python/semantic_memory_map.py
Then open the printed HTML and click the **Semantic** tab (or append
``#semantic`` to deep-link straight to it).
"""

import asyncio
import os

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph

DEST = os.path.join(os.path.expanduser("~"), "semantic_memory_map.html")

# A few short, deliberately multi-topic passages so distinct clusters emerge:
# computing pioneers, jazz, and ocean science.
TEXT = """
Ada Lovelace worked with Charles Babbage on the Analytical Engine in London.
Alan Turing formalized computation and broke ciphers at Bletchley Park.
Grace Hopper built the first compiler and worked on the Harvard Mark I.

Miles Davis recorded Kind of Blue, a landmark modal jazz album, in New York.
John Coltrane played saxophone with the Miles Davis Quintet before A Love Supreme.
Bill Evans, the pianist on Kind of Blue, shaped its impressionistic harmony.

Marine biologists study coral reefs, which host a quarter of all ocean species.
Rising sea temperatures cause coral bleaching, threatening reef ecosystems.
Phytoplankton in the ocean produce a large share of the planet's oxygen.
"""


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(TEXT)
    await cognee.cognify()

    html = await visualize_graph(destination_file_path=DEST)

    has_semantic = 'data-view="semantic"' in html
    has_positions = "window._semanticPositions = null" not in html
    print(f"\nSaved: {DEST}")
    print(f"Semantic tab present:   {has_semantic}")
    print(f"Semantic positions set: {has_positions}")
    print("\nOpen the file and click the Semantic tab (or append #semantic to the URL).")


if __name__ == "__main__":
    asyncio.run(main())
