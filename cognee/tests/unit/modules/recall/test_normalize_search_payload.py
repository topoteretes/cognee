from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.recall.types.SearchResultItem import SearchResultKind
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def test_hybrid_completion_normalizes_as_graph_completion():
    payload = SearchResultPayload(
        completion=["answer"],
        search_type=SearchType.HYBRID_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.GRAPH_COMPLETION
