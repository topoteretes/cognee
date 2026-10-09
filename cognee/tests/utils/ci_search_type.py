"""Search type for tests whose search is incidental to what they check."""

from cognee.modules.preflight import llm_available
from cognee.modules.search.types import SearchType


def completion_or_chunks(search_type: SearchType = SearchType.GRAPH_COMPLETION) -> SearchType:
    """``search_type`` when an LLM can answer, else ``SearchType.CHUNKS``.

    For tests that search only to show ingested data is searchable (results come
    back, deleted data is gone, permissions hold), not to test the completion.
    With an LLM key the test runs exactly as before. Without one, as on fork PRs
    where CI has no secrets, it still runs, on the LLM-free CHUNKS search.
    """
    return search_type if llm_available() else SearchType.CHUNKS
