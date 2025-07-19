import modal
import asyncio
from sqlalchemy.exc import OperationalError, DBAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from distributed.app import app
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


@app.function(
    retries=3,
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

                print("Finished adding data points.")

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
