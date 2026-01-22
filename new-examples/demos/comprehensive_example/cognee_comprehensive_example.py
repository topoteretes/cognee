"""
Core Features Getting Started Example

Reference: https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing

"""

import os
import cognee
import asyncio
from cognee.modules.engine.models.node_set import NodeSet

# provide your OpenAI key here
os.environ["LLM_API_KEY"] = "your_api_key"

# create artifacts directory for storing visualization outputs
artifacts_path = "artifacts"

developer_intro = (
    "Hi, I'm an AI/Backend engineer. "
    "I build FastAPI services with Pydantic, heavy asyncio/aiohttp pipelines, "
    "and production testing via pytest-asyncio. "
    "I've shipped low-latency APIs on AWS, Azure, and GoogleCloud."
)

asset_paths = {
    "human_agent_conversations": "data/copilot_conversations.json",
    "python_zen_principles": "data/zen_principles.md",
    "ontology": "data/basic_ontology.owl",
}

human_agent_conversations = asset_paths["human_agent_conversations"]
python_zen_principles = asset_paths["python_zen_principles"]
ontology_path = asset_paths["ontology"]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(developer_intro, node_set=["developer_data"])
    await cognee.add(human_agent_conversations, node_set=["developer_data"])
    await cognee.add(python_zen_principles, node_set=["principles_data"])

    # configure ontology file path for structured data processing
    os.environ["ONTOLOGY_FILE_PATH"] = ontology_path

    # transform all the data in the cognee store into a knowledge graph backed by embeddings
    await cognee.cognify()

    # generate the initial graph visualization showing nodesets and ontology structure
    initial_graph_visualization_path = (
        artifacts_path + "/graph_visualization_nodesets_and_ontology.html"
    )
    await cognee.visualize_graph(initial_graph_visualization_path)

    # enhance the knowledge graph with memory consolidation for improved connections
    await cognee.memify()

    # generate the second graph visualization after memory enhancement
    enhanced_graph_visualization_path = artifacts_path + "/graph_visualization_after_memify.html"
    await cognee.visualize_graph(enhanced_graph_visualization_path)

    # demonstrate cross-document knowledge retrieval from multiple data sources
    results = await cognee.search(
        query_text="How does my AsyncWebScraper implementation align with Python's design principles?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    print("Python Pattern Analysis:", results)

    # demonstrate filtered search using NodeSet to query only specific subsets of memory

    results = await cognee.search(
        query_text="How should variables be named?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        node_type=NodeSet,
        node_name=["principles_data"],
    )
    print("Filtered search result:", results)

    # demonstrate interactive search with feedback mechanism for continuous improvement
    answer = await cognee.search(
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        query_text="What is the most zen thing about Python?",
        save_interaction=True,
    )
    print("Initial answer:", answer)

    # provide feedback on the previous search result to improve future retrievals
    # the last_k parameter specifies which previous answer to give feedback about
    await cognee.search(
        query_type=cognee.SearchType.FEEDBACK,
        query_text="Last result was useful, I like code that complies with best practices.",
        last_k=1,
    )

    feedback_enhanced_graph_visualization_path = (
        artifacts_path + "/graph_visualization_after_feedback.html"
    )

    await cognee.visualize_graph(feedback_enhanced_graph_visualization_path)


if __name__ == "__main__":
    asyncio.run(main())
