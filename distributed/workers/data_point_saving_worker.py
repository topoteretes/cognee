import os
import modal
import asyncio
from sqlalchemy.exc import OperationalError, DBAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from distributed.app import app
from distributed.signal import QueueSignal
from distributed.modal_image import image
from distributed.queues import add_data_points_queue

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine

logger = get_logger("data_point_saving_worker")


class VectorDatabaseDeadlockError(Exception):
    message = "A deadlock occurred while trying to add data points to the vector database."


def is_deadlock_error(error):
    # SQLAlchemy-wrapped asyncpg
    try:
        import asyncpg

        if isinstance(error.orig, asyncpg.exceptions.DeadlockDetectedError):
            return True
    except ImportError:
        pass

    # PostgreSQL: SQLSTATE 40P01 = deadlock_detected
    if hasattr(error.orig, "pgcode") and error.orig.pgcode == "40P01":
        return True

    # SQLite: It doesn't support real deadlocks but may simulate them as "database is locked"
    if "database is locked" in str(error.orig).lower():
        return True

    return False


secret_name = os.environ.get("MODAL_SECRET_NAME", "distributed_cognee")


@app.function(
    retries=3,
    image=image,
    timeout=86400,
    max_containers=10,
    secrets=[modal.Secret.from_name(secret_name)],
)
async def data_point_saving_worker():
    print("Started processing of data points; starting vector engine queue.")
    vector_engine = get_vector_engine()
    # Defines how many data packets do we glue together from the modal queue before embedding call and ingestion
    BATCH_SIZE = 25
    stop_seen = False

    while True:
        if stop_seen:
            print("Finished processing all data points; stopping vector engine queue consumer.")
            return True

        if await add_data_points_queue.len.aio() != 0:
            try:
                print("Remaining elements in queue:")
                print(await add_data_points_queue.len.aio())

                # collect batched requests
                batched_points = {}
                for _ in range(min(BATCH_SIZE, await add_data_points_queue.len.aio())):
                    add_data_points_request = await add_data_points_queue.get.aio(block=False)

                    if not add_data_points_request:
                        continue

                    if add_data_points_request == QueueSignal.STOP:
                        await add_data_points_queue.put.aio(QueueSignal.STOP)
                        stop_seen = True
                        break

                    if len(add_data_points_request) == 2:
                        collection_name, data_points = add_data_points_request
                        if collection_name not in batched_points:
                            batched_points[collection_name] = []
                        batched_points[collection_name].extend(data_points)
                    else:
                        print("NoneType or invalid request detected.")

                if batched_points:
                    for collection_name, data_points in batched_points.items():
                        print(
                            f"Adding {len(data_points)} data points to '{collection_name}' collection."
                        )

                        @retry(
                            retry=retry_if_exception_type(VectorDatabaseDeadlockError),
                            stop=stop_after_attempt(3),
                            wait=wait_exponential(multiplier=2, min=1, max=6),
                        )
                        async def add_data_points():
                            try:
                                await vector_engine.create_data_points(
                                    collection_name, data_points, distributed=False
                                )
                            except DBAPIError as error:
                                if is_deadlock_error(error):
                                    raise VectorDatabaseDeadlockError()
                            except OperationalError as error:
                                if is_deadlock_error(error):
                                    raise VectorDatabaseDeadlockError()

                        await add_data_points()
                        print(f"Finished adding data points to '{collection_name}'.")

            except modal.exception.DeserializationError as error:
                logger.error(f"Deserialization error: {str(error)}")
                continue

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
