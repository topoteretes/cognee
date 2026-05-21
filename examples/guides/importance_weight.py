import asyncio
import cognee
import os
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    await cognee.forget(everything=True)

    await cognee.remember(
        "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
        dataset_name="importance_demo",
        self_improvement=False,
    )
    await cognee.remember(
        "Alice moved to Paris in 2010, while Bob has always lived in New York.",
        dataset_name="importance_demo",
        importance_weight=0.7,
        self_improvement=False,
    )
    await cognee.remember(
        "Andreas was born in Venice, but later settled in Lisbon.",
        dataset_name="importance_demo",
        importance_weight=0.3,
        self_improvement=False,
    )

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "importance_weight.html"
    )
    await visualize_graph(visualize_graph_path)

    result = await cognee.recall(
        "In which cities did Andreas live?",
        datasets=["importance_demo"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
