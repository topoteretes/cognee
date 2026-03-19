import asyncio
import cognee
import os
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    await cognee.add(
        "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
    )
    await cognee.add(
        "Alice moved to Paris in 2010, while Bob has always lived in New York.",
        importance_weight=0.7,
    )
    await cognee.add(
        "Andreas was born in Venice, but later settled in Lisbon.", importance_weight=0.3
    )
    await cognee.cognify()

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "importance_weight.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    asyncio.run(main())
