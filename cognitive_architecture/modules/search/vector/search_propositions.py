
async def find_relevant_chunks(query ,unique_layer_uuids):
    out = []
    query = await get_embeddings(query)
    # print(query)
    for id in unique_layer_uuids:
        result = qdrant_search(id, query[0])

        if result:
            result_ = [ result_.id for result_ in result]
            score_ = [ result_.score for result_ in result]

            out.append([result_, score_])

    return out