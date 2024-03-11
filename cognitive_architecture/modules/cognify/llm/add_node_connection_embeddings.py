import asyncio


async def process_items(grouped_data, unique_layer_uuids, llm_client):
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
                task = asyncio.create_task(llm_client.async_get_embedding_with_backoff(item['description'], "text-embedding-3-large"))
                tasks.append(task)
                # Map the task to the target id, item's node_id, and description for later retrieval
                task_to_info[task] = (target_id, item['node_id'], group_id, item['description'])

    # Await all tasks to complete and gather results
    results = await asyncio.gather(*tasks)


    # Process the results, associating them with their target id, node id, and description
    for task, embedding in zip(tasks, results):
        target_id, node_id, group_id, description = task_to_info[task]
        results_to_check.append([target_id, embedding, description, node_id, group_id])

    return results_to_check


if __name__ == '__main__':

    process_items()

