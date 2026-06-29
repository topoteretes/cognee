from typing import Optional


def _is_grpc_error(error: Exception) -> bool:
    try:
        from grpclib import GRPCError
    except ModuleNotFoundError:
        return False

    return isinstance(error, GRPCError)


async def queued_add_edges(
    edge_batch,
    source_ref_key: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
):
    from ..queues import add_nodes_and_edges_queue

    if source_ref_key is not None or pipeline_run_id is not None:
        raise NotImplementedError(
            "Distributed graph edge writes do not support graph provenance payloads yet."
        )

    try:
        await add_nodes_and_edges_queue.put.aio(([], edge_batch))
    except Exception as error:
        if not _is_grpc_error(error):
            raise
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        await queued_add_edges(first_half)
        await queued_add_edges(second_half)
