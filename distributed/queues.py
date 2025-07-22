from modal import Queue


add_nodes_and_edges_queue = Queue.from_name("add_nodes_and_edges_queue", create_if_missing=True)
add_data_points_queue = Queue.from_name("add_data_points_queue", create_if_missing=True)
