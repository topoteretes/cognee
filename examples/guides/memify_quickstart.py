import asyncio
import cognee
from cognee.modules.search.types import SearchType
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    # 1) Remember two short chats (ingest + build the graph)
    await cognee.remember(
        [
            "We follow PEP8. Add type hints and docstrings.",
            "Releases should not be on Friday. Susan must review PRs.",
        ],
        dataset_name="rules_demo",
        self_improvement=False,
    )

    # 2) Enrich the graph with coding-rule extraction tasks.
    await cognee.improve(
        dataset="rules_demo",
        extraction_tasks=[Task(extract_subgraph_chunks)],
        enrichment_tasks=[
            Task(
                add_rule_associations,
                rules_nodeset_name="coding_agent_rules",
                task_config={"batch_size": 1},
            )
        ],
    )

    # 3) Query the new coding rules
    rules = await cognee.recall(
        query_type=SearchType.CODING_RULES,
        query_text="List coding rules",
        node_name=["coding_agent_rules"],
        datasets=["rules_demo"],
    )
    print("Rules:", rules)


if __name__ == "__main__":
    asyncio.run(main())
