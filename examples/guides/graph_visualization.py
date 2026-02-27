import asyncio
import cognee
import os
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    await cognee.add(["Alice knows Bob.", "NLP is a subfield of CS."])
    await cognee.cognify()

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "graph_after_cognify.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    asyncio.run(main())
