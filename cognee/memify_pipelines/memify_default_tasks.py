from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.codingagents.coding_rule_associations import (
    add_rule_associations,
)


def get_default_memify_extraction_tasks():
    return [Task(extract_subgraph_chunks)]


def get_default_memify_enrichment_tasks():
    return [
        Task(
            add_rule_associations,
            rules_nodeset_name="coding_agent_rules",
            task_config={"batch_size": 1},
        )
    ]
