import pathlib
from os import path

from cognee.api.v1.add import add
from cognee.api.v1.prune import prune
from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods.get_datasets_by_name import get_datasets_by_name

from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents.classify_documents import classify_documents
from cognee.tasks.documents.extract_chunks_from_documents import extract_chunks_from_documents

from distributed.app import app
from distributed.queues import finished_jobs_queue, save_data_points_queue
from distributed.workers.data_point_saver_worker import data_point_saver_worker
from distributed.workers.graph_extraction_worker import graph_extraction_worker

logger = get_logger()


@app.local_entrypoint()
async def main():
    # Clear queues
    finished_jobs_queue.clear()
    save_data_points_queue.clear()

    dataset_name = "main"
    data_directory_name = ".data"
    data_directory_path = path.join(pathlib.Path(__file__).parent, data_directory_name)

    number_of_data_saving_workers = 1  # Total number of graph_extraction_worker functions to spawn
    document_batch_size = 50  # Batch size for producers

    results = []
    consumer_futures = []

    # Delete DBs and saved files from metastore
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    # Add files to the metastore
    await add(data=data_directory_path, dataset_name=dataset_name)

    user = await get_default_user()
    datasets = await get_datasets_by_name(dataset_name, user.id)
    documents = await get_dataset_data(dataset_id=datasets[0].id)

    print(f"We have {len(documents)} documents in the dataset.")

    # Start data_point_saver_worker functions
    for _ in range(number_of_data_saving_workers):
        worker_future = data_point_saver_worker.spawn(total_number_of_workers=len(documents))
        consumer_futures.append(worker_future)

    def process_chunks_remotely(document_chunks: list[DocumentChunk], document: Document):
        return graph_extraction_worker.spawn(
            user=user, document_name=document.name, document_chunks=document_chunks
        )

    # Produce chunks and spawn a graph_extraction_worker job for each batch of chunks
    for i in range(0, len(documents), document_batch_size):
        batch = documents[i : i + document_batch_size]

        producer_futures = []

        for item in batch:
            async for run_info in run_tasks(
                [
                    Task(classify_documents),
                    Task(
                        extract_chunks_from_documents,
                        max_chunk_size=get_max_chunk_tokens(),
                        chunker=TextChunker,
                    ),
                    Task(
                        process_chunks_remotely,
                        document=item,
                        task_config={"batch_size": 50},
                    ),
                ],
                data=[item],
                user=user,
                pipeline_name="chunk_processing",
            ):
                producer_futures.append(run_info)

        batch_results = []
        for producer_future in producer_futures:
            try:
                result = producer_future.get()
            except Exception as e:
                result = e
            batch_results.append(result)

        results.extend(batch_results)
        finished_jobs_queue.put(len(results))

    for consumer_future in consumer_futures:
        try:
            print("Finished but waiting")
            consumer_final = consumer_future.get()
            print(f"We got all futures {consumer_final}")
        except Exception as e:
            logger.error(e)

    print(results)
