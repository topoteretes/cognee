import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.pipelines import Task, run_tasks
from cognee.tasks.temporal_awareness import (
    build_graph_with_temporal_awareness, search_graph_with_temporal_awareness)

text_list = [
    "Kamala Harris is the Attorney General of California. She was previously "
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]

async def main():

    tasks = [
        Task(build_graph_with_temporal_awareness, text_list=text_list),
        Task(search_graph_with_temporal_awareness, query='Who was the California Attorney General?')
    ]
    
    pipeline = run_tasks(tasks)
    
    async for result in pipeline:
        print(result)


if __name__ == '__main__':
    asyncio.run(main())
