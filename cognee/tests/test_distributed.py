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
    await add_nodes_and_edges_queue.clear.aio()
    await add_data_points_queue.clear.aio()

    number_of_graph_saving_workers = 1
    number_of_data_point_saving_workers = (
        1
    )

    consumer_futures = []

    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await setup()

    for _ in range(number_of_graph_saving_workers):
        worker_future = graph_saving_worker.spawn()
        consumer_futures.append(worker_future)

    for _ in range(number_of_data_point_saving_workers):
        worker_future = data_point_saving_worker.spawn()
        consumer_futures.append(worker_future)

    await cognee.add(
        [
            "Audi is a German car manufacturer",
            "The Netherlands is next to Germany",
            "Berlin is the capital of Germany",
            "The Rhine is a major European river",
            "BMW produces luxury vehicles",
        ],
        dataset_name="ci-cd-modal-test",
    )

    await cognee.cognify(datasets=["ci-cd-modal-test"])

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

