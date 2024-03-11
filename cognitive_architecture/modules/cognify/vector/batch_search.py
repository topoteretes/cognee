
from cognitive_architecture.infrastructure.databases.vector.get_vector_database import get_vector_database

async def adapted_qdrant_batch_search(results_to_check,vector_client):
    search_results_list = []

    for result in results_to_check:
        id = result[0]
        embedding = result[1]
        node_id = result[2]
        target = result[3]
        b= result[4]

        # Assuming each result in results_to_check contains a single embedding
        limits = [3] * len(embedding)  # Set a limit of 3 results for this embedding

        try:
            #Perform the batch search for this id with its embedding
            #Assuming qdrant_batch_search function accepts a single embedding and a list of limits
            #qdrant_batch_search
            id_search_results = await vector_client.batch_search(collection_name = id, embeddings= embedding, with_vectors=limits)
            search_results_list.append((id, id_search_results, node_id, target))
        except Exception as e:
            print(f"Error during batch search for ID {id}: {e}")
            continue

    return search_results_list


if __name__ == '__main__':

    client = get_vector_database()