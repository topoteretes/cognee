"""Review dated CEO facts in a fresh dataset and inspect their stored conflicts.

Requires configured LLM/embedding providers and EDGE_EVIDENCE_ENABLED=true.
The dataset is kept for inspection; each run uses a new name.
"""

import asyncio
from uuid import uuid4

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph.utils.fact_conflicts import read_conflict_marks
from cognee.modules.improve.config import get_improve_config
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.data_item import DataItem


async def main():
    await setup()
    dataset_name = f"review_conflicts_{uuid4().hex[:8]}"
    user = await get_default_user()
    date_key = get_improve_config().effective_date_key
    for text, date, label in [
        ("Acme is led by CEO Alice.", "2020-05-01", "2020 directory"),
        ("Acme is led by CEO Bob.", "2026-06-01", "2026 board notice"),
    ]:
        await cognee.remember(
            DataItem(text, label=label, external_metadata={date_key: date}, literal_text=True),
            dataset_name=dataset_name,
            user=user,
            self_improvement=False,
            custom_prompt="Extract the company Acme and the named person. Connect Acme to its CEO with has_ceo.",
        )

    result = await cognee.improve(dataset=dataset_name, user=user, review_conflicts=True)
    print(dataset_name, result.stage("review_conflicts"))
    dataset = (
        await get_authorized_existing_datasets(
            user=user, datasets=[dataset_name], permission_type="read"
        )
    )[0]
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
        conflict_ids = set()
        for node_id, properties in nodes:
            if properties.get("type") == "FactConflict" and str(
                properties.get("dataset_id")
            ) == str(dataset.id):
                conflict_ids.add(str(node_id))
                print("Conflict:", properties)
        for source, target, relationship, properties in edges:
            if str(source) in conflict_ids:
                print("Link:", relationship, target, properties)
            marks = [
                mark
                for mark in read_conflict_marks(properties)
                if mark.get("conflict_id") in conflict_ids
            ]
            if marks:
                print("Fact:", properties.get("edge_text"), marks)

    repeat = await cognee.improve(dataset=dataset_name, user=user, review_conflicts=True)
    print("Unchanged repeat:", repeat.stage("review_conflicts"))


if __name__ == "__main__":
    asyncio.run(main())
