from typing import Optional


async def queued_add_nodes(
    node_batch,
    source_ref_key: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
):
    from grpclib import GRPCError
    from ..queues import add_nodes_and_edges_queue

    if source_ref_key is not None or pipeline_run_id is not None:
        raise NotImplementedError(
            "Distributed graph node writes do not support graph provenance payloads yet."
        )

    try:
        await add_nodes_and_edges_queue.put.aio((node_batch, []))
    except GRPCError:
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        await queued_add_nodes(first_half)
        await queued_add_nodes(second_half)
