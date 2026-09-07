"""Non-generative retrievers must not run a pre-retrieval LLM turn analysis.

`BaseRetriever.supports_session_turn_preparation` exists so that "search types whose
contract is explicitly non-generative" can skip `prepare_session_turn_for_retrieval`,
which may call an LLM before retrieval. The retrievers whose own
`get_completion_from_context` docstring says they "do not generate a completion, we just
return the payloads" belong in that set; otherwise a sub-second deterministic lookup pays
for a chat completion whose rewritten query it never benefits from.
"""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever
from cognee.modules.retrieval.lexical_retriever import LexicalRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType

# The package __init__ re-exports `get_retriever_output` under the same name as the
# submodule, so a dotted-string patch target resolves to the function. Patch the module
# object instead, as the sibling test for this path already does.
get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)

NON_GENERATIVE_RETRIEVERS = [
    ChunksRetriever,
    SummariesRetriever,
    LexicalRetriever,
    BM25ChunksRetriever,
    JaccardChunksRetriever,
]


class _FakeGraphEngine:
    async def is_empty(self):
        return False


@pytest.mark.parametrize("retriever_class", NON_GENERATIVE_RETRIEVERS, ids=lambda c: c.__name__)
def test_non_generative_retrievers_opt_out_of_session_turn_preparation(retriever_class):
    assert retriever_class.supports_session_turn_preparation is False


def test_generative_retrievers_still_opt_in():
    assert CompletionRetriever.supports_session_turn_preparation is True


async def _search_with(retriever, search_type):
    """Run a real search through `get_retriever_output`, stubbing only I/O.

    Returns the mock standing in for `prepare_session_turn_for_retrieval`, so a test can
    assert whether the pre-retrieval LLM analysis ran.
    """
    prepare = AsyncMock(
        return_value=SessionTurnPreparation(should_answer=True, effective_query="q")
    )
    with (
        patch.object(type(retriever), "prepare_session_turn_for_retrieval", prepare),
        patch.object(type(retriever), "get_retrieved_objects", AsyncMock(return_value=[])),
        patch.object(type(retriever), "get_context_from_objects", AsyncMock(return_value="")),
        patch.object(type(retriever), "get_completion_from_context", AsyncMock(return_value=[])),
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
        await get_retriever_output(search_type, "q")
    return prepare


@pytest.mark.asyncio
async def test_chunks_search_does_not_prepare_a_session_turn():
    """The regression from #4439: a CHUNKS recall paid for an LLM call before retrieval."""
    prepare = await _search_with(ChunksRetriever(), SearchType.CHUNKS)
    prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_summaries_search_does_not_prepare_a_session_turn():
    prepare = await _search_with(SummariesRetriever(), SearchType.SUMMARIES)
    prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_lexical_chunks_search_does_not_prepare_a_session_turn():
    prepare = await _search_with(BM25ChunksRetriever(), SearchType.CHUNKS_LEXICAL)
    prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_generative_search_still_prepares_a_session_turn():
    """Guard the other direction: generative search types keep the conversational turn."""
    prepare = await _search_with(CompletionRetriever(), SearchType.RAG_COMPLETION)
    prepare.assert_awaited_once()
