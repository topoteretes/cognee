
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.modules.cognify.graph.add_node_connections import extract_node_descriptions


async def search_similarity(query:str ,graph):

    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))
    # print(node_descriptions)

    unique_layer_uuids = set(node["layer_decomposition_uuid"] for node in node_descriptions)

    client = get_llm_client()
    out = []
    query = await client.async_get_embedding_with_backoff(query)
    # print(query)
    for id in unique_layer_uuids:
        result = client.search(id, query[0])

        if result:
            result_ = [ result_.id for result_ in result]
            score_ = [ result_.score for result_ in result]

            out.append([result_, score_])

    return out