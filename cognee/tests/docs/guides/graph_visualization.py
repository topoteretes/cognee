import asyncio
import cognee
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    await cognee.add(["Alice knows Bob.", "NLP is a subfield of CS."])
    await cognee.cognify()

    await visualize_graph("./graph_after_cognify.html")


asyncio.run(main())
