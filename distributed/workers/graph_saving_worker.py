import os
import modal
import asyncio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from distributed.app import app
from distributed.signal import QueueSignal
from distributed.modal_image import image
from distributed.queues import add_nodes_and_edges_queue

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_config

logger = get_logger("graph_saving_worker")


class GraphDatabaseDeadlockError(Exception):
    message = "A deadlock occurred while trying to add data points to the vector database."


def is_deadlock_error(error):
    graph_config = get_graph_config()

    if graph_config.graph_database_provider == "neo4j":
        # Neo4j
        from neo4j.exceptions import TransientError

        if isinstance(error, TransientError) and (
            error.code == "Neo.TransientError.Transaction.DeadlockDetected"
        ):
            return True

    # Kuzu
    if "deadlock" in str(error).lower() or "cannot acquire lock" in str(error).lower():
        return True

    return False


secret_name = os.environ.get("MODAL_SECRET_NAME", "distributed_cognee")


@app.function(
    retries=3,
    image=image,
    timeout=86400,
    max_containers=1,
    secrets=[modal.Secret.from_name(secret_name)],
)
async def graph_saving_worker():
    print("Started processing of nodes and edges; starting graph engine queue.")
    graph_engine = await get_graph_engine()
    # Defines how many data packets do we glue together from the queue before ingesting them into the graph database
    BATCH_SIZE = 25
    stop_seen = False

    while True:
        if stop_seen:
            print("Finished processing all data points; stopping graph engine queue consumer.")
            return True

        if await add_nodes_and_edges_queue.len.aio() != 0:
            try:
                print("Remaining elements in queue:")
                print(await add_nodes_and_edges_queue.len.aio())

                all_nodes, all_edges = [], []
                for _ in range(min(BATCH_SIZE, await add_nodes_and_edges_queue.len.aio())):
                    nodes_and_edges = await add_nodes_and_edges_queue.get.aio(block=False)

                    if not nodes_and_edges:
                        continue

                    if nodes_and_edges == QueueSignal.STOP:
                        await add_nodes_and_edges_queue.put.aio(QueueSignal.STOP)
                        stop_seen = True
                        break

                    if len(nodes_and_edges) == 2:
                        nodes, edges = nodes_and_edges
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                    else:
                        print("None Type detected.")

                if all_nodes or all_edges:
                    print(f"Adding {len(all_nodes)} nodes and {len(all_edges)} edges.")

                    @retry(
                        retry=retry_if_exception_type(GraphDatabaseDeadlockError),
                        stop=stop_after_attempt(3),
                        wait=wait_exponential(multiplier=2, min=1, max=6),
                    )
                    async def save_graph_nodes(new_nodes):
                        try:
                            await graph_engine.add_nodes(new_nodes, distributed=False)
                        except Exception as error:
                            if is_deadlock_error(error):
                                raise GraphDatabaseDeadlockError()

                    @retry(
                        retry=retry_if_exception_type(GraphDatabaseDeadlockError),
                        stop=stop_after_attempt(3),
                        wait=wait_exponential(multiplier=2, min=1, max=6),
                    )
                    async def save_graph_edges(new_edges):
                        try:
                            await graph_engine.add_edges(new_edges, distributed=False)
                        except Exception as error:
                            if is_deadlock_error(error):
                                raise GraphDatabaseDeadlockError()

                    if all_nodes:
                        await save_graph_nodes(all_nodes)

                    if all_edges:
                        await save_graph_edges(all_edges)

                    print("Finished adding nodes and edges.")

            except modal.exception.DeserializationError as error:
                logger.error(f"Deserialization error: {str(error)}")
                continue

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
