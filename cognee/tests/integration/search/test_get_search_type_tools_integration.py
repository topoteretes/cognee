import pytest

from cognee.modules.search.types import SearchType


class _DummyCompletionContextRetriever:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_completion(self, *args, **kwargs):
        return None

    def get_context(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_community_registry_is_consulted(monkeypatch):
    """
    This test covers the dynamic import + lookup of community retrievers in
    cognee.modules.retrieval.registered_community_retrievers.
    """
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval import registered_community_retrievers as registry

    monkeypatch.setattr(
        registry,
        "registered_community_retrievers",
        {SearchType.NATURAL_LANGUAGE: _DummyCompletionContextRetriever},
    )

    tools = await mod.get_search_type_tools(SearchType.NATURAL_LANGUAGE, query_text="q", top_k=9)

    assert len(tools) == 2
    assert tools[0].__self__.kwargs["top_k"] == 9
    assert tools[1].__self__.kwargs["top_k"] == 9
