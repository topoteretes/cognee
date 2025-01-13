import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.pipelines import Task, run_tasks
from cognee.tasks.temporal_awareness import (
    build_graph_with_temporal_awareness,
    search_graph_with_temporal_awareness,
)
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.tasks.storage.index_graph_edges import index_graphiti_nodes_and_edges

text_list = [
    "Kamala Harris is the Attorney General of California. She was previously "
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_relational_db_and_tables()

    for text in text_list:
        await cognee.add(text)

    tasks = [
        Task(build_graph_with_temporal_awareness, text_list=text_list),
    ]

    pipeline = run_tasks(tasks)

    async for result in pipeline:
        print(result)

    await index_graphiti_nodes_and_edges()


if __name__ == "__main__":
    asyncio.run(main())
