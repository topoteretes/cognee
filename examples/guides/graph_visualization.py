import asyncio
import cognee
from os import path
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    await cognee.add(["Alice knows Bob.", "NLP is a subfield of CS."])
    await cognee.cognify()

    graph_visualization_path = path.join(path.dirname(__file__), "graph_after_cognify.html")

    await visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    asyncio.run(main())
