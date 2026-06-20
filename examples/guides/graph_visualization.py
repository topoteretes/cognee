import asyncio
import os

import cognee
from cognee import visualize_graph


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(
        ["Alice knows Bob.", "NLP is a subfield of CS."],
        self_improvement=False,
    )

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "graph_after_remember.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    asyncio.run(main())
