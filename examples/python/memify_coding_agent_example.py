import asyncio
import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.search import SearchType

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

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    coding_rules_text = """
    Code must be formatted by PEP8 standards.
    Typing and Docstrings must be added.
    """

    print("Adding text to cognee:")
    print(text.strip())
    # Add the text, and make it available for cognify
    await cognee.add(text)
    await cognee.add(coding_rules_text, node_set=["coding_rules"])
    print("Text added successfully.\n")

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    from cognee.api.v1.cognify.memify import memify

    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
    from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations
    from cognee.modules.pipelines.tasks.task import Task

    memify_tasks = [
        Task(CogneeGraph.resolve_edges_to_text, task_config={"batch_size": 10}),
        Task(
            add_rule_associations,
            rules_nodeset_name="coding_agent_rules",
            user_prompt_location="memify_coding_rule_association_agent_user.txt",
            system_prompt_location="memify_coding_rule_association_agent_system.txt",
        ),
    ]

    await memify(tasks=memify_tasks, node_name=["coding_rules"])

    import os
    import pathlib
    from cognee.api.v1.visualize.visualize import visualize_graph

    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization.html"
    )
    await visualize_graph(file_path)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
