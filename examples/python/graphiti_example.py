import asyncio

import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.modules.pipelines import Task, run_tasks
from cognee.tasks.temporal_awareness import build_graph_with_temporal_awareness
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.tasks.temporal_awareness.index_graphiti_objects import (
    index_and_transform_graphiti_nodes_and_edges,
)
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.users.methods import get_default_user

text_list = [
    "Kamala Harris is the Attorney General of California. She was previously "
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_relational_db_and_tables()

    # Initialize default user
    user = await get_default_user()

    for text in text_list:
        await cognee.add(text)

    tasks = [
        Task(build_graph_with_temporal_awareness, text_list=text_list),
    ]

    pipeline = run_tasks(tasks, user=user)

    async for result in pipeline:
        print(result)

    await index_and_transform_graphiti_nodes_and_edges()

    query = "When was Kamala Harris in office?"
    triplets = await brute_force_triplet_search(
        query=query,
        user=user,
        top_k=3,
        collections=["graphitinode_content", "graphitinode_name", "graphitinode_summary"],
    )

    retriever = GraphCompletionRetriever()
    context = await retriever.resolve_edges_to_text(triplets)

    args = {
        "question": query,
        "context": context,
    }

    user_prompt = LLMGateway.render_prompt("graph_context_for_question.txt", args)
    system_prompt = LLMGateway.read_query_prompt("answer_simple_question_restricted.txt")

    computed_answer = await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

    print(computed_answer)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
