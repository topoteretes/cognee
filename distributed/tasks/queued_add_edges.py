async def queued_add_edges(edge_batch):
    from ..queues import add_nodes_and_edges_queue

    try:
        await add_nodes_and_edges_queue.put.aio(([], edge_batch))
    except Exception:
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        await add_nodes_and_edges_queue.put.aio(([], first_half))
        await add_nodes_and_edges_queue.put.aio(([], second_half))
