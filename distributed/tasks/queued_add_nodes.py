from typing import Optional


def _is_grpc_error(error: Exception) -> bool:
    try:
        from grpclib import GRPCError
    except ModuleNotFoundError:
        return False

    return isinstance(error, GRPCError)


async def queued_add_nodes(
    node_batch,
    source_ref_key: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
):
    from ..queues import add_nodes_and_edges_queue

    # The provenance stamp rides along in the queue payload so the
    # graph_saving_worker can fold it per data item (source_ref_key / run id are
    # None for non-provenance writes). Payload shape:
    # (node_batch, edge_batch, source_ref_key, pipeline_run_id).
    try:
        await add_nodes_and_edges_queue.put.aio((node_batch, [], source_ref_key, pipeline_run_id))
    except Exception as error:
        if not _is_grpc_error(error):
            raise
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        await queued_add_nodes(first_half, source_ref_key, pipeline_run_id)
        await queued_add_nodes(second_half, source_ref_key, pipeline_run_id)
