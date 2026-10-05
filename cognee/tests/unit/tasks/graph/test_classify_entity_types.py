import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.graph.classify_entity_types import (
    NAMES_PER_CALL,
    EntityTypeCategories,
    classify_entity_type_names,
)

classify_module = importlib.import_module("cognee.tasks.graph.classify_entity_types")


def _answers(**categories: str) -> EntityTypeCategories:
    return EntityTypeCategories.model_validate(
        {"answers": [{"name": name, "category": value} for name, value in categories.items()]}
    )


@pytest.fixture
def llm():
    """Fake only the LLM call; the prompt, the response model and the matching are real."""
    with patch.object(classify_module.LLMGateway, "acreate_structured_output", AsyncMock()) as call:
        yield call


@pytest.mark.asyncio
async def test_each_name_gets_the_category_the_model_answered(llm):
    llm.return_value = _answers(country="place", company="organization")

    result = await classify_entity_type_names(["country", "company"])

    assert result == {"country": "place", "company": "organization"}
    assert llm.await_args.kwargs["text_input"] == "country\ncompany"


@pytest.mark.asyncio
async def test_a_label_outside_the_taxonomy_becomes_other(llm):
    """One bad label must not fail the call and lose the other answers."""
    llm.return_value = _answers(country="place", animal="creature")

    result = await classify_entity_type_names(["country", "animal"])

    assert result == {"country": "place", "animal": "other"}


@pytest.mark.asyncio
async def test_names_the_model_left_out_or_made_up_are_not_classified(llm):
    """A missing name stays None for a later run, which is not the same as other."""
    llm.return_value = _answers(country="place", invented="person")

    result = await classify_entity_type_names(["country", "company"])

    assert result == {"country": "place"}


@pytest.mark.asyncio
async def test_no_names_makes_no_call(llm):
    assert await classify_entity_type_names([]) == {}

    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_call_leaves_its_names_unclassified_and_the_rest_go_on(llm):
    """Classification must not fail the run, and one bad call must not cost the others."""
    names = [f"type_{index}" for index in range(NAMES_PER_CALL + 1)]
    llm.side_effect = [RuntimeError("provider down"), _answers(type_200="concept")]

    result = await classify_entity_type_names(names)

    assert result == {"type_200": "concept"}
    assert llm.await_count == 2
