import asyncio


from distributed.app import app
from distributed.modal_image import image
from distributed.queues import save_data_points_queue
from cognee.infrastructure.databases.graph import get_graph_engine


@app.function(image=image, timeout=86400, max_containers=100)
async def data_point_saver_worker():
    print("Started processing of nodes and edges; starting graph engine queue.")
    graph_engine = await get_graph_engine()

    while True:
        if save_data_points_queue.len() != 0:
            nodes_and_edges = save_data_points_queue.get(block=False)

            if len(nodes_and_edges) == 0:
                print("Finished processing all nodes and edges; stopping graph engine queue.")
                return True

            if len(nodes_and_edges) == 2:
                print(f"Processing {len(nodes_and_edges[0])} nodes and {len(nodes_and_edges[1])} edges.")
                nodes = nodes_and_edges[0]
                edges = nodes_and_edges[1]

                if nodes:
                    await graph_engine.add_nodes(nodes)

                if edges:
                    await graph_engine.add_edges(edges)
                print(f"Finished processing nodes and edges.")

        else:
            print(f"No jobs, go to sleep.")
            await asyncio.sleep(5)
