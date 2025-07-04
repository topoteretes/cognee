async def queued_add_edges(edge_batch):
    from ..queues import save_data_points_queue

    try:
        await save_data_points_queue.put.aio(([], edge_batch))
    except Exception:
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        await save_data_points_queue.put.aio(([], first_half))
        await save_data_points_queue.put.aio(([], second_half))
