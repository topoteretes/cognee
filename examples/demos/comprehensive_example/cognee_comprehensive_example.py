# ruff: noqa: E402
import os
import asyncio
from pathlib import Path

# provide your OpenAI key here
# Set these before Cognee config is initialized so the example uses the intended values.
os.environ["LLM_API_KEY"] = "your_api_key"

# create artifacts directory for storing visualization outputs
artifacts_path = ".artifacts"

developer_intro = (
    "Hi, I'm an AI/Backend engineer. "
    "I build FastAPI services with Pydantic, heavy asyncio/aiohttp pipelines, "
    "and production testing via pytest-asyncio. "
    "I've shipped low-latency APIs on AWS, Azure, and GoogleCloud."
)
data_dir = Path(__file__).resolve().parent / "data"
asset_paths = {
    "human_agent_conversations": str(data_dir / "copilot_conversations.json"),
    "python_zen_principles": str(data_dir / "zen_principles.md"),
    "ontology": str(data_dir / "basic_ontology.owl"),
}

human_agent_conversations = asset_paths["human_agent_conversations"]
python_zen_principles = asset_paths["python_zen_principles"]
ontology_path = asset_paths["ontology"]

# configure ontology file path for structured data processing
# Set these before Cognee config is initialized so the example uses the intended values.
os.environ["ONTOLOGY_FILE_PATH"] = ontology_path

import cognee  # noqa: E402


async def main():
    await cognee.forget(everything=True)

    await cognee.remember(developer_intro, node_set=["developer_data"], self_improvement=False)
    await cognee.remember(
        human_agent_conversations,
        node_set=["developer_data"],
        self_improvement=False,
    )
    await cognee.remember(
        python_zen_principles,
        node_set=["principles_data"],
        self_improvement=False,
    )

    # generate the initial graph visualization showing nodesets and ontology structure
    initial_graph_visualization_path = os.path.join(
        os.path.dirname(__file__), artifacts_path, "graph_visualization_nodesets_and_ontology.html"
    )
    await cognee.visualize_graph(initial_graph_visualization_path)

    # enhance the knowledge graph with memory consolidation for improved connections
    await cognee.memify()

    # generate the second graph visualization after memory enhancement
    enhanced_graph_visualization_path = os.path.join(
        os.path.dirname(__file__), artifacts_path, "graph_visualization_after_memify.html"
    )
    await cognee.visualize_graph(enhanced_graph_visualization_path)

    # demonstrate cross-document knowledge retrieval from multiple data sources
    results = await cognee.recall(
        query_text="How does my AsyncWebScraper implementation align with Python's design principles?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    print("Python Pattern Analysis:", results)

    # demonstrate filtered recall over a specific node set

    results = await cognee.recall(
        query_text="How should variables be named?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        node_name=["principles_data"],
    )
    print("Filtered search result:", results)


if __name__ == "__main__":
    asyncio.run(main())
