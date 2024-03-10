




async def process_items(grouped_data, unique_layer_uuids):
    results_to_check = []  # This will hold results excluding self comparisons
    tasks = []  # List to hold all tasks
    task_to_info = {}  # Dictionary to map tasks to their corresponding group id and item info

    # Iterate through each group in grouped_data
    for group_id, items in grouped_data.items():
        # Filter unique_layer_uuids to exclude the current group_id
        target_uuids = [uuid for uuid in unique_layer_uuids if uuid != group_id]

        # Process each item in the group
        for item in items:
            # For each target UUID, create an async task for the item's embedding retrieval
            for target_id in target_uuids:
                task = asyncio.create_task \
                    (async_get_embedding_with_backoff(item['description'], "text-embedding-3-large"))
                tasks.append(task)
                # Map the task to the target id, item's node_id, and description for later retrieval
                task_to_info[task] = (target_id, item['node_id'], group_id, item['description'])

    # Await all tasks to complete and gather results
    results = await asyncio.gather(*tasks)

    # Process the results, associating them with their target id, node id, and description
    for task, embedding in zip(tasks, results):

        target_id, node_id ,group_id, description = task_to_info[task]
        results_to_check.append([target_id, embedding, description, node_id, group_id])

    return results_to_check

async def graph_ready_output(results):
    relationship_dict ={}

    for result_tuple in results:

        uuid, scored_points_list, desc, node_id = result_tuple
        # Unpack the tuple

        # Ensure there's a list to collect related items for this uuid
        if uuid not in relationship_dict:
            relationship_dict[uuid] = []

        for scored_points in scored_points_list:  # Iterate over the list of ScoredPoint lists
            for scored_point in scored_points:  # Iterate over each ScoredPoint object
                if scored_point.score > 0.9:  # Check the score condition
                    # Append a new dictionary to the list associated with the uuid
                    relationship_dict[uuid].append({
                        'collection_name_uuid': uuid,
                        'searched_node_id': scored_point.id,
                        'score': scored_point.score,
                        'score_metadata': scored_point.payload,
                        'original_id_for_search': node_id,
                    })
    return relationship_dict

async def connect_nodes_in_graph(graph, relationship_dict):
    """
    For each relationship in relationship_dict, check if both nodes exist in the graph based on node attributes.
    If they do, create a connection (edge) between them.

    :param graph: A NetworkX graph object
    :param relationship_dict: A dictionary containing relationships between nodes
    """
    for id, relationships in relationship_dict.items():
        for relationship in relationships:
            searched_node_attr_id = relationship['searched_node_id']
            print(searched_node_attr_id)
            score_attr_id = relationship['original_id_for_search']
            score = relationship['score']


            # Initialize node keys for both searched_node and score_node
            searched_node_key, score_node_key = None, None

            # Find nodes in the graph that match the searched_node_id and score_id from their attributes
            for node, attrs in graph.nodes(data=True):
                if 'id' in attrs:  # Ensure there is an 'id' attribute
                    if attrs['id'] == searched_node_attr_id:
                        searched_node_key = node
                    elif attrs['id'] == score_attr_id:
                        score_node_key = node

                # If both nodes are found, no need to continue checking other nodes
                if searched_node_key and score_node_key:
                    break

            # Check if both nodes were found in the graph
            if searched_node_key is not None and score_node_key is not None:
                print(searched_node_key)
                print(score_node_key)
                # If both nodes exist, create an edge between them
                # You can customize the edge attributes as needed, here we use 'score' as an attribute
                graph.add_edge(searched_node_key, score_node_key, weight=score, score_metadata=relationship.get('score_metadata'))

    return graph
