"""NATURAL_LANGUAGE rows must survive the result payload.

The retriever hands the graph rows its generated Cypher returned straight to the
caller. ``SearchResultPayload.context`` is typed as text, so returning those rows as
the context failed validation on every search that found something: the only
NATURAL_LANGUAGE searches that completed were the ones with no rows. The rows belong
in ``result_object`` only, the shape the Cypher retriever already uses.
"""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)

# What a graph engine returns for ``MATCH (n) ... RETURN n.name``: one tuple per row.
ROWS = [("2017-05-12",), ("2021-09-03",), (None,)]


class _FakeGraphEngine:
    async def is_empty(self):
        return False


async def _rows(self, query=None, **kwargs):
    return ROWS


@pytest.mark.asyncio
async def test_rows_are_not_offered_as_text_context_or_completion():
    retriever = NaturalLanguageRetriever()

    assert await retriever.get_context_from_objects("q", ROWS) is None
    assert await retriever.get_completion_from_context("q", ROWS, context=None) is None


@pytest.mark.asyncio
async def test_natural_language_rows_reach_the_caller_through_the_payload():
    """A search that finds rows used to raise a pydantic ValidationError here."""
    retriever = NaturalLanguageRetriever()
    with (
        patch.object(NaturalLanguageRetriever, "get_retrieved_objects", _rows),
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
    ):
        payload = await get_retriever_output(SearchType.NATURAL_LANGUAGE, "which dates exist?")

    assert isinstance(payload, SearchResultPayload)
    assert payload.result_object == ROWS
    assert payload.context is None
    assert payload.completion is None
    # ``result`` is what search() hands back as search_result.
    assert payload.result == ROWS
    # And the payload still serializes for the HTTP response, one entry per row
    # (``result_object``'s serializer renders non-JSON rows as text, as it does for
    # CYPHER; the shape is the Cypher retriever's, not a new one).
    dumped = payload.model_dump(mode="json")
    assert len(dumped["result_object"]) == len(ROWS)
