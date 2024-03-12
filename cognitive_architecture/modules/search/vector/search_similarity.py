
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client

async def search_similarity(query ,unique_layer_uuids):

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