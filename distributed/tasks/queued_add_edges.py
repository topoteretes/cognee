from typing import Optional


async def queued_add_edges(
    edge_batch,
    source_ref_key: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
):
    from grpclib import GRPCError
    from ..queues import add_nodes_and_edges_queue

    if source_ref_key is not None or pipeline_run_id is not None:
        raise NotImplementedError(
            "Distributed graph edge writes do not support graph provenance payloads yet."
        )

    try:
        await add_nodes_and_edges_queue.put.aio(([], edge_batch))
    except GRPCError:
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        await queued_add_edges(first_half)
        await queued_add_edges(second_half)
