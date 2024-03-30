
from cognee.modules.cognify.graph.add_node_connections import extract_node_descriptions
from cognee.infrastructure import infrastructure_config


async def search_similarity(query: str, graph, other_param: str = None):
    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))

    unique_layer_uuids = set(node["layer_decomposition_uuid"] for node in node_descriptions)

    out = []

    for id in unique_layer_uuids:
        vector_engine = infrastructure_config.get_config()["vector_engine"]

        result = await vector_engine.search(id, query_text = query, limit = 10)

        if result:
            result_ = [ result_.id for result_ in result]
            score_ = [ result_.score for result_ in result]

            out.append([result_, score_])

    relevant_context = []

    if len(out) == 0:
        return []

    for proposition_id in out[0][0]:
        for n, attr in graph.nodes(data = True):
            if str(proposition_id) in str(n):
                for n_, attr_ in graph.nodes(data=True):
                    relevant_layer = attr["layer_uuid"]

                    if attr_.get("layer_uuid") == relevant_layer:
                        relevant_context.append(attr_["description"])

    def deduplicate_list(original_list):
        seen = set()
        deduplicated_list = [x for x in original_list if not (x in seen or seen.add(x))]
        return deduplicated_list

    relevant_context = deduplicate_list(relevant_context)

    return relevant_context
