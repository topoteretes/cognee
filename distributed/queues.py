from modal import Queue


# Create (or get) queues:
# - save_data_points_queue: Stores messages produced by the producer functions.

save_data_points_queue = Queue.from_name("save_data_points_queue", create_if_missing=True)
