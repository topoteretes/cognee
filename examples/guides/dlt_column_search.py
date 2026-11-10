"""Query a DLT column through its table:column graph anchor.

Run: python examples/guides/dlt_column_search.py /path/to/Chinook_Sqlite.sqlite

DLT ingestion is LLM-free; embeddings still need a configured provider (or
the keyless local default). only_context previews GRAPH_COMPLETION without an
answering LLM call. Remove it to generate an answer with your configured LLM.

Anchors cover distinct materialized values; existing column selection, key
exclusions and null/blank handling still apply. Existing processed sources need
a memory rebuild (forget(memory_only=True), then cognify) to gain these nodes.
Identically named columns from multiple sources in one dataset have separate ids
but match the same name filter; use separate datasets to search one source alone.
"""

import argparse
import asyncio
from pathlib import Path

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.tasks.schema.models import SchemaColumn


async def main(database: Path) -> None:
    if not database.is_file():
        raise FileNotFoundError(database)
    dataset = "chinook_columns"
    await cognee.add(f"sqlite:///{database.resolve()}", dataset_name=dataset)
    await cognee.cognify(datasets=[dataset])

    # dlt normalizes SQL identifiers: Chinook's Genre.Name becomes genre:name.
    # The column links to distinct materialized values, not one node per row.
    # Key columns, null/blank cells and deselected values retain DLT's exclusions.
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([dataset], "read", user)
    async with set_database_global_context_variables(datasets[0].id, user.id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_nodeset_subgraph(SchemaColumn, ["genre:name"])

    if not edges:
        raise RuntimeError("The genre:name column has no materialized values.")
    values = [props["name"] for _, props in nodes if props.get("type") == "DltColumn"]
    print(f"genre:name: {len(values)} distinct values")
    print("\n".join(sorted(values)))

    # GRAPH_COMPLETION ranks top_k edges. Use the complete neighborhood's edge
    # count here so this small column fits; default top_k is not an exhaustive
    # column export, and a large column can exceed the answering model's context.
    results = await cognee.search(
        "List every genre in genre:name.",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[dataset],
        node_type=SchemaColumn,
        node_name=["genre:name"],
        top_k=len(edges),
        only_context=True,
        verbose=True,
    )
    for result in results:
        print(result["context_result"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    asyncio.run(main(parser.parse_args().database))
