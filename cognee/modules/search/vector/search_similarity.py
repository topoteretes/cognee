
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.cognify.graph.add_node_connections import extract_node_descriptions


async def search_similarity(query:str ,graph):

    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))
    # print(node_descriptions)

    unique_layer_uuids = set(node["layer_decomposition_uuid"] for node in node_descriptions)

    client = get_llm_client()
    out = []
    query = await client.async_get_embedding_with_backoff(query)
    # print(query)
    for id in unique_layer_uuids:
        from cognee.infrastructure.databases.vector.get_vector_database import get_vector_database
        vector_client = get_vector_database()

        result = await vector_client.search(id, query,10)

        if result:
            result_ = [ result_.id for result_ in result]
            score_ = [ result_.score for result_ in result]

            out.append([result_, score_])

    relevant_context = []

    for proposition_id in out[0][0]:
        print(proposition_id)
        for n,attr in graph.nodes(data=True):
            if proposition_id in n:
                for n_, attr_ in graph.nodes(data=True):
                    relevant_layer = attr['layer_uuid']

                    if attr_.get('layer_uuid') == relevant_layer:
                        print(attr_['description'])
                        relevant_context.append(attr_['description'])

    return relevant_context