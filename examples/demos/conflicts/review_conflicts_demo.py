"""Follow dated CEO changes through review, stored marks, and hybrid recall.

Requires configured LLM/embedding credentials and edge evidence enabled (the default).
Creates a new dataset on each run and retains it for inspection.
Run: uv run python examples/demos/conflicts/review_conflicts_demo.py
"""

import asyncio
from uuid import uuid4

import cognee
from cognee import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph.utils.fact_conflicts import read_conflict_marks
from cognee.modules.improve.config import get_improve_config
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.data_item import DataItem


async def show_state(dataset, user, title):
    print(f"\n{title}")
    query = "Who is Acme's CEO?"
    for only_context in (True, False):
        result = await cognee.recall(
            query,
            query_type=SearchType.HYBRID_COMPLETION,
            datasets=[dataset],
            only_context=only_context,
        )
        print("Context:" if only_context else "Answer:", result)
    selected = (await get_authorized_existing_datasets([dataset], "read", user))[0]
    async with set_database_global_context_variables(selected.id, selected.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
    conflict_ids = set()
    for node_id, properties in nodes:
        if properties.get("type") == "FactConflict" and str(properties.get("dataset_id")) == str(
            selected.id
        ):
            conflict_ids.add(str(node_id))
            print("Conflict:", node_id, properties["status"], properties["text"])
    for source, target, relationship, properties in edges:
        marks = [
            mark
            for mark in read_conflict_marks(properties)
            if mark.get("conflict_id") in conflict_ids
        ]
        if marks:
            print("Fact:", source, relationship, target, properties.get("edge_text"), marks)


async def main():
    await setup()
    dataset = f"review_conflicts_demo_{uuid4().hex[:8]}"
    user = await get_default_user()
    date_key = get_improve_config().effective_date_key
    print("Dataset:", dataset)
    await cognee.remember(
        [
            DataItem(
                "Acme is led by CEO Alice.",
                external_metadata={date_key: "2020-05-01"},
                literal_text=True,
            ),
            DataItem(
                "Acme is led by CEO Bob.",
                external_metadata={date_key: "2026-01-10"},
                literal_text=True,
            ),
        ],
        dataset_name=dataset,
        self_improvement=False,
        custom_prompt="Extract the company Acme and the named person. Connect Acme to its CEO with has_ceo.",
    )
    await show_state(dataset, user, "Before review")
    print(await cognee.improve(dataset=dataset, review_conflicts=True))
    await show_state(dataset, user, "After review: Bob is the later dated value")
    await cognee.remember(
        DataItem(
            "Alice returned as Acme's CEO.",
            external_metadata={date_key: "2027-03-01"},
            literal_text=True,
        ),
        dataset_name=dataset,
        self_improvement=False,
        custom_prompt="Extract the company Acme and the named person. Connect Acme to its CEO with has_ceo.",
    )
    print(await cognee.improve(dataset=dataset, review_conflicts=True))
    await show_state(dataset, user, "After restoration: Alice has the newest support")


if __name__ == "__main__":
    asyncio.run(main())
