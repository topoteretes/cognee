import asyncio

import cognee
import logging
from cognee.modules.pipelines import Task, run_tasks
from cognee.shared.utils import setup_logging
from cognee.tasks.temporal_awareness import build_graph_with_temporal_awareness
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.tasks.temporal_awareness.index_graphiti_objects import (
    index_and_transform_graphiti_nodes_and_edges,
)
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.completion.graph_query_completion import retrieved_edges_to_string
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client

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

    await index_and_transform_graphiti_nodes_and_edges()

    query = "When was Kamala Harris in office?"
    triplets = await brute_force_triplet_search(
        query=query,
        top_k=3,
        collections=["graphitinode_content", "graphitinode_name", "graphitinode_summary"],
    )

    args = {
        "question": query,
        "context": retrieved_edges_to_string(triplets),
    }

    user_prompt = render_prompt("graph_context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_simple_question_restricted.txt")

    llm_client = get_llm_client()
    computed_answer = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

    print(computed_answer)


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
