import modal
import asyncio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from distributed.app import app
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


@app.function(
    retries=3,
    image=image,
    timeout=86400,
    max_containers=5,
    secrets=[modal.Secret.from_name("distributed_cognee")],
)
async def graph_saving_worker():
    print("Started processing of nodes and edges; starting graph engine queue.")
    graph_engine = await get_graph_engine()

    while True:
        if await add_nodes_and_edges_queue.len.aio() != 0:
            try:
                nodes_and_edges = await add_nodes_and_edges_queue.get.aio(block=False)
            except modal.exception.DeserializationError as error:
                logger.error(f"Deserialization error: {str(error)}")
                continue

            if len(nodes_and_edges) == 0:
                print("Finished processing all nodes and edges; stopping graph engine queue.")
                return True

            if len(nodes_and_edges) == 2:
                print(
                    f"Adding {len(nodes_and_edges[0])} nodes and {len(nodes_and_edges[1])} edges."
                )
                nodes = nodes_and_edges[0]
                edges = nodes_and_edges[1]

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

                if nodes:
                    await save_graph_nodes(nodes)

                if edges:
                    await save_graph_edges(edges)

                print("Finished adding nodes and edges.")

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
