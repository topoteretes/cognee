async def queued_add_edges(edge_batch):
    from grpclib import GRPCError
    from ..queues import add_nodes_and_edges_queue

    try:
        await add_nodes_and_edges_queue.put.aio(([], edge_batch))
    except GRPCError:
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        await queued_add_edges(first_half)
        await queued_add_edges(second_half)
