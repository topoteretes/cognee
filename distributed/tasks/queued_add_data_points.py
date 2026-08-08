def _is_grpc_error(error: Exception) -> bool:
    try:
        from grpclib import GRPCError
    except ModuleNotFoundError:
        return False

    return isinstance(error, GRPCError)


async def queued_add_data_points(collection_name, data_points_batch):
    from ..queues import add_data_points_queue

    try:
        await add_data_points_queue.put.aio((collection_name, data_points_batch))
    except Exception as error:
        if not _is_grpc_error(error):
            raise
        first_half, second_half = (
            data_points_batch[: len(data_points_batch) // 2],
            data_points_batch[len(data_points_batch) // 2 :],
        )
        await queued_add_data_points(collection_name, first_half)
        await queued_add_data_points(collection_name, second_half)
