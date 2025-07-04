async def queued_add_nodes(node_batch):
    from ..queues import add_nodes_and_edges_queue

    try:
        await add_nodes_and_edges_queue.put.aio((node_batch, []))
    except Exception:
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        await add_nodes_and_edges_queue.put.aio((first_half, []))
        await add_nodes_and_edges_queue.put.aio((second_half, []))
