import os
import asyncio
import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.shared.logging_utils import setup_logging, ERROR

text_a = """
    AI is revolutionizing financial services through intelligent fraud detection
    and automated customer service platforms.
    """

text_b = """
    Advances in AI are enabling smarter systems that learn and adapt over time.
    """

text_c = """
    MedTech startups have seen significant growth in recent years, driven by innovation
    in digital health and medical devices.
    """

node_set_a = ["AI", "FinTech"]
node_set_b = ["AI"]
node_set_c = ["MedTech"]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(text_a, node_set=node_set_a)
    await cognee.add(text_b, node_set=node_set_b)
    await cognee.add(text_c, node_set=node_set_c)
    await cognee.cognify()

    visualization_path = os.path.join(
        os.path.dirname(__file__), "./.artifacts/graph_visualization.html"
    )
    await visualize_graph(visualization_path)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.run(main())
