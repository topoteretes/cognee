import asyncio
import cognee
from cognee import SearchType


async def main():
    # 1) Add two short chats and build a graph
    await cognee.add(
        [
            "We follow PEP8. Add type hints and docstrings.",
            "Releases should not be on Friday. Susan must review PRs.",
        ],
        dataset_name="rules_demo",
    )
    await cognee.cognify(datasets=["rules_demo"])  # builds graph

    # 2) Enrich the graph (uses default memify tasks)
    await cognee.memify(dataset="rules_demo")

    # 3) Query the new coding rules
    rules = await cognee.search(
        query_type=SearchType.CODING_RULES,
        query_text="List coding rules",
        node_name=["coding_agent_rules"],
    )
    print("Rules:", rules)


asyncio.run(main())
