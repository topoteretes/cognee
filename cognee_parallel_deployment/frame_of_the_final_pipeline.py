import time
import random
from modal import App, Queue

# ------------------------------------------------------------------------------
# App and Queue Initialization
# ------------------------------------------------------------------------------

# Initialize the Modal application
app = App("queue_example")

# Create (or get) two queues:
# - graph_nodes_and_edges: Stores messages produced by the producer functions.
# - finished_producers: Keeps track of the number of finished producer jobs.
graph_nodes_and_edges = Queue.from_name("graph_nodes_and_edges", create_if_missing=True)
finished_producers = Queue.from_name("finished_producers", create_if_missing=True)

graph_nodes_and_edges.clear()

# ------------------------------------------------------------------------------
# Producer Function
# ------------------------------------------------------------------------------


@app.function(timeout=86400, max_containers=100)
def producer(producer_id: int):
    if producer_id == 42:
        raise ValueError("Item cannot be 42!")

    # COGNEE STEPS: EXTRACT GRAPH FROM DATA, SUMMARIZE TEXT + ADD DATAPOINTS (EXCEPT GRAPH INGESTION)

    # Simulate variable processing time
    time.sleep(random.randint(2, 15))

    # Put the result into the queue
    result_message = f"result of producer {producer_id}"
    graph_nodes_and_edges.put(result_message)

    # Return the id as confirmation of work done
    return producer_id


# ------------------------------------------------------------------------------
# Consumer Function
# ------------------------------------------------------------------------------


@app.function(timeout=86400, max_containers=100)
def consumer(number_of_files: int):
    while True:
        # If there are messages in the queue, process them

        if graph_nodes_and_edges.len() != 0:
            # COGNEE STEPS GRAPH INGESTION
            message = graph_nodes_and_edges.get()
            print(f"Consumer received: {message} with queue len: {graph_nodes_and_edges.len()}")
        else:
            number_of_finished_jobs = finished_producers.get()
            time.sleep(10)
            if number_of_finished_jobs == number_of_files:
                # We put it back for the other consumers to see that we finished
                finished_producers.put(number_of_finished_jobs)
                print("Finished processing all input elements; stopping consumers.")
                return True


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    finished_producers.clear()
    graph_nodes_and_edges.clear()
    total_items = list(range(50))  # input list
    number_of_consumers = 3  # Total number of consumer functions to spawn
    batch_size = 10  # Batch size for producers
    results = []

    # Start consumer functions
    for _ in range(number_of_consumers):
        consumer.spawn(number_of_files=len(total_items))

    # Process producer jobs in batches
    for i in range(0, len(total_items), batch_size):
        batch = total_items[i : i + batch_size]
        futures = []
        for item in batch:
            # COGNEE STEPS: CLASSIFY_DOCS, CHECK_PERMISSIONS, GET_CHUNKS
            future = producer.spawn(item)
            futures.append(future)

        batch_results = []
        for future in futures:
            try:
                result = future.get()
            except Exception as e:
                result = e
            batch_results.append(result)

        results.extend(batch_results)
        finished_producers.put(len(results))

    print(results)
