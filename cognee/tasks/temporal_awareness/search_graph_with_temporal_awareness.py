async def search_graph_with_temporal_awareness(graphiti, query):
    search_result = await graphiti.search(query)
    await graphiti.close()
    return search_result
