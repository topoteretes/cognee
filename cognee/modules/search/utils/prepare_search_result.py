import json
from typing import Any, List, Tuple, cast
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.types.SearchResult import SearchResultDataset
from cognee.modules.search.utils.transform_context_to_graph import transform_context_to_graph
from cognee.modules.search.utils.transform_insights_to_graph import transform_insights_to_graph


def _normalize_tuple_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """Convert tuple rows returned by graph queries into JSON dict tuples."""
    normalized: List[Tuple[Any, ...]] = []
    for row in rows:
        normalized_row: list[Any] = []
        for column in row:
            if isinstance(column, dict):
                normalized_row.append(column)
            elif isinstance(column, str):
                try:
                    normalized_row.append(json.loads(column))
                except json.JSONDecodeError:
                    normalized_row.append({"value": column})
            else:
                normalized_row.append(column)
        normalized.append(tuple(normalized_row))
    return normalized


async def prepare_search_result(search_result):
    results, context, datasets = search_result

    if isinstance(context, list) and context and isinstance(context[0], tuple):
        context = _normalize_tuple_rows(context)
    if isinstance(results, list) and results and isinstance(results[0], tuple):
        results = _normalize_tuple_rows(results)

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
        and len(context[0]) > 1
        and isinstance(context[0][1], dict)
        and context[0][1].get("relationship_name")
    ):
        context_graph = transform_insights_to_graph(context)
        graphs = {
            ", ".join([dataset.name for dataset in datasets]): context_graph,
        }
        results = None
    elif isinstance(context, List) and len(context) > 0 and isinstance(context[0], Edge):
        edge_context = cast(List[Edge], context)
        context_graph = transform_context_to_graph(edge_context)

        graphs = {
            ", ".join([dataset.name for dataset in datasets]): context_graph,
        }
        context_texts = {
            ", ".join([dataset.name for dataset in datasets]): await resolve_edges_to_text(
                edge_context
            ),
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
        edge_results = cast(List[Edge], results)
        result_graph = transform_context_to_graph(edge_results)

    return {
        "result": result_graph or results[0] if results and len(results) == 1 else results,
        "graphs": graphs,
        "context": context_texts,
        "datasets": datasets,
    }
