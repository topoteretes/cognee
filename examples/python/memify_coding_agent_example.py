import asyncio
import pathlib
import os

import cognee
from cognee import memify
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")
    print("Adding conversation about rules to cognee:\n")

    coding_rules_chat_from_principal_engineer = """
    We want code to be formatted by PEP8 standards.
    Typing and Docstrings must be added.
    Please also make sure to write NOTE: on all more complex code segments.
    If there is any duplicate code, try to handle it in one function to avoid code duplication.
    Susan should also always review new code changes before merging to main.
    New releases should not happen on Friday so we don't have to fix them during the weekend.
    """
    print(
        f"Coding rules conversation with principal engineer: {coding_rules_chat_from_principal_engineer}"
    )

    coding_rules_chat_from_manager = """
    Susan should always review new code changes before merging to main.
    New releases should not happen on Friday so we don't have to fix them during the weekend.
    """
    print(f"Coding rules conversation with manager: {coding_rules_chat_from_manager}")

    # Add the text, and make it available for cognify
    await cognee.add([coding_rules_chat_from_principal_engineer, coding_rules_chat_from_manager])
    print("Text added successfully.\n")

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    # Visualize graph after cognification
    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_only_cognify.html"
    )
    await visualize_graph(file_path)
    print(f"Open file to see graph visualization only after cognification: {file_path}\n")

    # After graph is created, create a second pipeline that will go through the graph and enchance it with specific
    # coding rule nodes

    # extract_subgraph_chunks is a function that returns all document chunks from specified subgraphs (if no subgraph is specifed the whole graph will be sent through memify)
    subgraph_extraction_tasks = [Task(extract_subgraph_chunks)]

    # add_rule_associations is a function that handles processing coding rules from chunks and keeps track of
    # existing rules so duplicate rules won't be created. As the result of this processing new Rule nodes will be created
    # in the graph that specify coding rules found in conversations.
    coding_rules_association_tasks = [
        Task(
            add_rule_associations,
            rules_nodeset_name="coding_agent_rules",
            task_config={"batch_size": 1},
        ),
    ]

    # Memify accepts these tasks and orchestrates forwarding of graph data through these tasks (if data is not specified).
    # If data is explicitely specified in the arguments this specified data will be forwarded through the tasks instead
    await memify(
        extraction_tasks=subgraph_extraction_tasks,
        enrichment_tasks=coding_rules_association_tasks,
    )

    # Find the new specific coding rules added to graph through memify (created based on chat conversation between team members)
    coding_rules = await cognee.search(
        query_text="List me the coding rules",
        query_type=cognee.SearchType.CODING_RULES,
        node_name=["coding_agent_rules"],
    )

    print("Coding rules created by memify:")
    for coding_rule in coding_rules:
        print("- " + coding_rule)

    # Visualize new graph with added memify context
    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_after_memify.html"
    )
    await visualize_graph(file_path)
    print(f"\nOpen file to see graph visualization after memify enhancment: {file_path}")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
