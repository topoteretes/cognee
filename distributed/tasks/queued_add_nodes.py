async def queued_add_nodes(node_batch):
    from grpclib import GRPCError
    from ..queues import add_nodes_and_edges_queue

    try:
        await add_nodes_and_edges_queue.put.aio((node_batch, []))
    except GRPCError:
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        await queued_add_nodes(first_half)
        await queued_add_nodes(second_half)
