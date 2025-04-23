from modal import Queue


# Create (or get) two queues:
# - save_data_points_queue: Stores messages produced by the producer functions.
# - finished_jobs_queue: Keeps track of the number of finished producer jobs.

save_data_points_queue = Queue.from_name("save_data_points_queue", create_if_missing=True)

finished_jobs_queue = Queue.from_name("finished_jobs_queue", create_if_missing=True)
