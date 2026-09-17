"""``only_context`` end to end on the default local databases (Ladybug + LanceDB).

The graph is built from typed DataPoints through ``add_data_points``, so building it needs
an embedding provider but no LLM. The LLM gateway is patched to fail if anything calls it:
an only_context search must retrieve from the real graph and vector store, render the full
LLM input, and never generate a completion.
"""

import logging
import pathlib
from unittest.mock import patch

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.low_level import DataPoint, setup
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType
from cognee.tasks.storage import add_data_points

logger = logging.getLogger(__name__)

QUESTION = "Who works at Figma?"


def _no_llm():
    return patch.object(
        LLMGateway,
        "acreate_structured_output",
        side_effect=AssertionError("only_context must not call the LLM"),
    )


async def _reset(name: str):
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    cognee.config.system_root_directory(str(base_dir / f".cognee_system/{name}"))
    cognee.config.data_root_directory(str(base_dir / f".data_storage/{name}"))
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()


async def _teardown(name: str):
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring teardown exception in %s", name, exc_info=True)


@pytest_asyncio.fixture
async def small_company_graph():
    name = "test_only_context_prompt_small_graph"
    await _reset(name)

    class Company(DataPoint):
        name: str
        description: str

    class Person(DataPoint):
        name: str
        description: str
        works_for: Company

    figma = Company(name="Figma", description="Figma is a company")
    canva = Company(name="Canva", description="Canva is a company")
    people = [
        Person(name="Steve Rodger", description="Designer at Figma", works_for=figma),
        Person(name="Ike Loma", description="Engineer at Figma", works_for=figma),
        Person(name="Mike Broski", description="Marketer at Canva", works_for=canva),
    ]
    await add_data_points([figma, canva, *people])

    yield

    await _teardown(name)


@pytest_asyncio.fixture
async def empty_graph():
    name = "test_only_context_prompt_empty_graph"
    await _reset(name)
    yield
    await _teardown(name)


@pytest.mark.asyncio
async def test_only_context_returns_the_full_llm_input_over_the_real_graph(small_company_graph):
    with _no_llm():
        payload = await get_retriever_output(
            SearchType.GRAPH_COMPLETION, QUESTION, only_context=True
        )

    assert payload.only_context is True
    assert payload.completion is None

    # The bare retrieval context is still addressable, and it came from the real graph.
    assert isinstance(payload.context, str)
    assert "Steve Rodger --[works_for]--> Figma" in payload.context

    # The result is the user prompt: the question and the retrieved context verbatim.
    # The system prompt (task instructions, session layer) travels separately.
    result = payload.result
    assert isinstance(result, str)
    assert result is payload.user_prompt
    assert QUESTION in result
    assert payload.context in result
    assert isinstance(payload.system_prompt, str) and payload.system_prompt
    assert "Steve Rodger --[works_for]--> Figma" not in payload.system_prompt


@pytest.mark.asyncio
async def test_only_context_on_an_empty_graph_raises_instead_of_building_a_prompt(empty_graph):
    """An empty graph is a state problem, not a query miss (SDK-270), so the retriever
    raises before retrieval returns and no prompt is ever built. only_context is not
    exempt: it reports the same 404 a completion call would.

    The neighbouring guarantee — an empty *retrieval* over a populated graph yields the
    bare context and never a prompt wrapped around nothing — cannot be staged here
    (graph retrieval is nearest-neighbour, so a populated graph always returns
    something). It is pinned deterministically in the unit tests instead:
    ``test_only_context_prompt.py::test_empty_retrieval_gets_no_prompt`` and
    ``test_get_retriever_output.py::test_only_context_falls_back_to_the_bare_context_when_no_prompt_is_built``.
    """
    with _no_llm(), pytest.raises(NoDataError):
        await get_retriever_output(SearchType.GRAPH_COMPLETION, QUESTION, only_context=True)
