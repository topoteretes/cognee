import asyncio

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

from distributed.app import app
from distributed.modal_image import image
from distributed.queues import finished_jobs_queue, save_data_points_queue


@app.function(image=image, timeout=86400, max_containers=100)
async def data_point_saver_worker(total_number_of_workers: int):
    graph_engine = await get_graph_engine()

    while True:
        if save_data_points_queue.len() != 0:
            nodes_and_edges = save_data_points_queue.get(block=False)
            if nodes_and_edges and len(nodes_and_edges) == 2:
                await graph_engine.add_nodes(nodes_and_edges[0])
                await graph_engine.add_edges(nodes_and_edges[1])
            else:
                print(f"Nodes and edges are: {nodes_and_edges}")
        else:
            await asyncio.sleep(5)

            number_of_finished_jobs = finished_jobs_queue.get(block=False)

            if number_of_finished_jobs == total_number_of_workers:
                # We put it back for the other consumers to see that we finished
                finished_jobs_queue.put(number_of_finished_jobs)

                print("Finished processing all nodes and edges; stopping graph engine queue.")
                return True
