import modal
import asyncio


from distributed.app import app
from distributed.modal_image import image
from distributed.queues import save_data_points_queue

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine


logger = get_logger("data_point_saver_worker")


@app.function(
    image=image,
    timeout=86400,
    max_containers=100,
    secrets=[modal.Secret.from_name("distributed_cognee")],
)
async def data_point_saver_worker():
    print("Started processing of nodes and edges; starting graph engine queue.")
    graph_engine = await get_graph_engine()

    while True:
        if await save_data_points_queue.len.aio() != 0:
            try:
                nodes_and_edges = await save_data_points_queue.get.aio(block=False)
            except modal.exception.DeserializationError as error:
                logger.error(f"Deserialization error: {str(error)}")
                continue

            if len(nodes_and_edges) == 0:
                print("Finished processing all nodes and edges; stopping graph engine queue.")
                return True

            if len(nodes_and_edges) == 2:
                print(
                    f"Processing {len(nodes_and_edges[0])} nodes and {len(nodes_and_edges[1])} edges."
                )
                nodes = nodes_and_edges[0]
                edges = nodes_and_edges[1]

                if nodes:
                    await graph_engine.add_nodes(nodes, distributed=False)

                if edges:
                    await graph_engine.add_edges(edges, distributed=False)
                print("Finished processing nodes and edges.")

        else:
            print("No jobs, go to sleep.")
            await asyncio.sleep(5)
