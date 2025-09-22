from typing import List, cast
from uuid import uuid5, NAMESPACE_OID

from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.types.SearchResult import SearchResultDataset
from cognee.modules.search.utils.transform_context_to_graph import transform_context_to_graph
from cognee.modules.search.utils.transform_insights_to_graph import transform_insights_to_graph


async def prepare_search_result(search_result):
    results, context, datasets = search_result

    graphs = None
    result_graph = None
    context_texts = {}

    if isinstance(datasets, list) and len(datasets) == 0:
        datasets = [
            SearchResultDataset(
                id=uuid5(NAMESPACE_OID, "*"),
                name="all available datasets",
            )
        ]

    if (
        isinstance(context, List)
        and len(context) > 0
        and isinstance(context[0], tuple)
        and context[0][1].get("relationship_name")
    ):
        context_graph = transform_insights_to_graph(context)
        graphs = {
            ", ".join([dataset.name for dataset in datasets]): context_graph,
        }
        results = None
    elif isinstance(context, List) and len(context) > 0 and isinstance(context[0], Edge):
        context_graph = transform_context_to_graph(context)

        graphs = {
            ", ".join([dataset.name for dataset in datasets]): context_graph,
        }
        context_texts = {
            ", ".join([dataset.name for dataset in datasets]): await resolve_edges_to_text(context),
        }
    elif isinstance(context, str):
        context_texts = {
            ", ".join([dataset.name for dataset in datasets]): context,
        }
    elif isinstance(context, List) and len(context) > 0 and isinstance(context[0], str):
        context_texts = {
            ", ".join([dataset.name for dataset in datasets]): "\n".join(cast(List[str], context)),
        }

    if isinstance(results, List) and len(results) > 0 and isinstance(results[0], Edge):
        result_graph = transform_context_to_graph(results)

    return {
        "result": result_graph or results[0] if results and len(results) == 1 else results,
        "graphs": graphs,
        "context": context_texts,
        "datasets": datasets,
    }
