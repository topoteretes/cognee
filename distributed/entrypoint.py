import os
import asyncio

import cognee
from cognee.api.v1.prune import prune
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.operations.setup import setup

from distributed.app import app
from distributed.queues import add_nodes_and_edges_queue, add_data_points_queue
from distributed.workers.graph_saving_worker import graph_saving_worker
from distributed.workers.data_point_saving_worker import data_point_saving_worker
from distributed.signal import QueueSignal

logger = get_logger()


os.environ["COGNEE_DISTRIBUTED"] = "True"


@app.local_entrypoint()
async def main():
    # Clear queues
    await add_nodes_and_edges_queue.clear.aio()
    await add_data_points_queue.clear.aio()

    number_of_graph_saving_workers = 1  # Total number of graph_saving_worker to spawn (MAX 1)
    number_of_data_point_saving_workers = (
        10  # Total number of graph_saving_worker to spawn (MAX 10)
    )

    consumer_futures = []

    await prune.prune_data()  # This prunes the data from the file storage
    # Delete DBs and saved files from metastore
    await prune.prune_system(metadata=True)

    await setup()

    # Start graph_saving_worker functions
    for _ in range(number_of_graph_saving_workers):
        worker_future = graph_saving_worker.spawn()
        consumer_futures.append(worker_future)

    # Start data_point_saving_worker functions
    for _ in range(number_of_data_point_saving_workers):
        worker_future = data_point_saving_worker.spawn()
        consumer_futures.append(worker_future)

    """ Example: Setting and adding S3 path as input
    s3_bucket_path = os.getenv("S3_BUCKET_PATH")
    s3_data_path = "s3://" + s3_bucket_path

    await cognee.add(s3_data_path, dataset_name="s3-files")
    """
    await cognee.add(
        [
            "Audi is a German car manufacturer",
            "The Netherlands is next to Germany",
            "Berlin is the capital of Germany",
            "The Rhine is a major European river",
            "BMW produces luxury vehicles",
        ],
        dataset_name="s3-files",
    )

    await cognee.cognify(datasets=["s3-files"])

    # Put Processing end signal into the queues to stop the consumers
    await add_nodes_and_edges_queue.put.aio(QueueSignal.STOP)
    await add_data_points_queue.put.aio(QueueSignal.STOP)

    for consumer_future in consumer_futures:
        try:
            print("Finished but waiting for saving workers to finish.")
            consumer_final = consumer_future.get()
            print(f"All workers are done: {consumer_final}")
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    asyncio.run(main())
