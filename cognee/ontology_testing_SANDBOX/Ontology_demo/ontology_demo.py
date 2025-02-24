import os
import cognee
from cognee.shared.utils import setup_logging
import logging
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.data.methods import get_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.pipelines import run_tasks

from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
import asyncio
from typing import Type, List

from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points


async def main():
    file_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),
        "ontology_test_input",
    )
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(file_path)
    await owl_testing_pipeline()


async def owl_testing_pipeline():
    user = await get_default_user()
    datasets = await get_datasets(user.id)

    for dataset in datasets:
        data_documents: list[Data] = await get_dataset_data(dataset_id=dataset.id)

        tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(extract_chunks_from_documents, max_chunk_tokens=get_max_chunk_tokens()),
            Task(
                owl_ontology_merging_layer,
                graph_model=KnowledgeGraph,
                task_config={"batch_size": 10},
            ),
        ]

        pipeline_run = run_tasks(tasks, dataset.id, data_documents, "cognify_pipeline")

        async for run_status in pipeline_run:
            pipeline_run_status = run_status
            print(pipeline_run_status)
        print()


async def owl_ontology_merging_layer(
    data_chunks: list[DocumentChunk], graph_model: Type[BaseModel]
) -> List[DocumentChunk]:
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )
    graph_engine = await get_graph_engine()

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        await add_data_points(chunk_graphs)
        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
        graph_engine,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks,
        chunk_graphs,
        existing_edges_map,
    )

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks


if __name__ == "__main__":
    setup_logging(logging.INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
