from typing import List, cast

from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.utils.transform_context_to_graph import transform_context_to_graph


async def prepare_search_result(search_result):
    results, context, datasets = search_result

    graphs = None
    result_graph = None
    context_texts = {}

    if isinstance(context, List) and len(context) > 0 and isinstance(context[0], Edge):
        context_graph = transform_context_to_graph(context)

        graphs = {
            "*": context_graph,
        }
        context_texts = {
            "*": await resolve_edges_to_text(context),
        }
    elif isinstance(context, str):
        context_texts = {
            "*": context,
        }
    elif isinstance(context, List) and len(context) > 0 and isinstance(context[0], str):
        context_texts = {
            "*": "\n".join(cast(List[str], context)),
        }

    if isinstance(results, List) and len(results) > 0 and isinstance(results[0], Edge):
        result_graph = transform_context_to_graph(results)

    return {
        "result": result_graph or results[0] if len(results) == 1 else results,
        "graphs": graphs,
        "context": context_texts,
        "datasets": datasets,
    }
