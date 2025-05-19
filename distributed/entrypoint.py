import pathlib
from os import path

# from cognee.api.v1.add import add
from cognee.api.v1.prune import prune
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.operations.setup import setup
from cognee.modules.ingestion.get_text_content_hash import get_text_content_hash
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods.get_default_user import get_default_user
# from cognee.modules.data.methods.get_dataset_data import get_dataset_data
# from cognee.modules.data.methods.get_datasets_by_name import get_datasets_by_name

from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents.extract_chunks_from_documents import extract_chunks_from_documents

from distributed.app import app
from distributed.models.TextDocument import TextDocument
from distributed.queues import save_data_points_queue
from distributed.workers.data_point_saver_worker import data_point_saver_worker
from distributed.workers.graph_extraction_worker import graph_extraction_worker

logger = get_logger()


@app.local_entrypoint()
async def main():
    # Clear queues
    save_data_points_queue.clear()

    # dataset_name = "main"
    data_directory_name = ".data"
    data_directory_path = path.join(pathlib.Path(__file__).parent, data_directory_name)

    number_of_data_saving_workers = 1  # Total number of graph_extraction_worker functions to spawn
    document_batch_size = 50  # Batch size for producers

    results = []
    consumer_futures = []

    # Delete DBs and saved files from metastore
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await setup()

    # Add files to the metastore
    # await add(data=data_directory_path, dataset_name=dataset_name)

    user = await get_default_user()
    # datasets = await get_datasets_by_name(dataset_name, user.id)
    # documents = await get_dataset_data(dataset_id=datasets[0].id)

    import duckdb

    connection = duckdb.connect()
    dataset_file_name = "de-00000-of-00003-f8e581c008ccc7f2.parquet"
    dataset_file_path = path.join(data_directory_path, dataset_file_name)
    df = connection.execute(f"SELECT * FROM '{dataset_file_path}'").fetchdf()

    documents = []

    for _, row in df.iterrows():
        file_id = str(row["id"])
        content = row["text"]

        documents.append(
            TextDocument(
                name=file_id,
                content=content,
                raw_data_location=f"{dataset_file_name}_{file_id}",
                external_metadata="",
            )
        )

    documents: list[TextDocument] = documents[0:100]
    print(f"We have {len(documents)} documents in the dataset.")

    data_documents = [
        Data(
            id=document.id,
            name=document.name,
            raw_data_location=document.raw_data_location,
            extension="txt",
            mime_type=document.mime_type,
            owner_id=user.id,
            content_hash=get_text_content_hash(document.content),
            external_metadata=document.external_metadata,
            node_set=None,
            token_count=-1,
        )
        for document in documents
    ]

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add_all(data_documents)
        await session.commit()

    # Start data_point_saver_worker functions
    for _ in range(number_of_data_saving_workers):
        worker_future = data_point_saver_worker.spawn()
        consumer_futures.append(worker_future)

    producer_futures = []

    def process_chunks_remotely(document_chunks: list[DocumentChunk], document: Document):
        producer_future = graph_extraction_worker.spawn(
            user=user, document_name=document.name, document_chunks=document_chunks
        )
        producer_futures.append(producer_future)
        return producer_future

    # Produce chunks and spawn a graph_extraction_worker job for each batch of chunks
    for i in range(0, len(documents), document_batch_size):
        batch = documents[i : i + document_batch_size]

        for item in batch:
            async for worker_feature in run_tasks(
                [
                    Task(
                        extract_chunks_from_documents,
                        max_chunk_size=2000,
                        chunker=TextChunker,
                    ),
                    Task(
                        process_chunks_remotely,
                        document=item,
                        task_config={"batch_size": 10},
                    ),
                ],
                data=[item],
                user=user,
                pipeline_name="chunk_processing",
            ):
                pass

        batch_results = []

        for producer_future in producer_futures:
            try:
                result = producer_future.get()
            except Exception as e:
                result = e

            batch_results.append(result)

        print(f"Number of documents processed: {len(results)}")
        results.extend(batch_results)

    # Push empty tuple into the queue to signal the end of data.
    save_data_points_queue.put(())

    for consumer_future in consumer_futures:
        try:
            print("Finished but waiting for saving worker to finish.")
            consumer_final = consumer_future.get()
            print(f"All workers are done: {consumer_final}")
        except Exception as e:
            logger.error(e)

    print(results)
