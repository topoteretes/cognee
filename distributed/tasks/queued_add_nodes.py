async def queued_add_nodes(node_batch):
    from ..queues import save_data_points_queue

    try:
        await save_data_points_queue.put.aio((node_batch, []))
    except Exception:
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        await save_data_points_queue.put.aio((first_half, []))
        await save_data_points_queue.put.aio((second_half, []))
