import modal
import asyncio


from distributed.app import app
from distributed.modal_image import image
from distributed.queues import add_data_points_queue

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine


logger = get_logger("data_point_saving_worker")


@app.function(
    image=image,
    timeout=86400,
    max_containers=5,
    secrets=[modal.Secret.from_name("distributed_cognee")],
)
async def data_point_saving_worker():
    print("Started processing of data points; starting vector engine queue.")
    vector_engine = get_vector_engine()

    while True:
        if await add_data_points_queue.len.aio() != 0:
            try:
                add_data_points_request = await add_data_points_queue.get.aio(block=False)
            except modal.exception.DeserializationError as error:
                logger.error(f"Deserialization error: {str(error)}")
                continue

            if len(add_data_points_request) == 0:
                print("Finished processing all data points; stopping vector engine queue.")
                return True

            if len(add_data_points_request) == 2:
                (collection_name, data_points) = add_data_points_request

                print(f"Adding {len(data_points)} data points to '{collection_name}' collection.")

                await vector_engine.create_data_points(
                    collection_name, data_points, distributed=False
                )

                print("Finished adding data points.")

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
